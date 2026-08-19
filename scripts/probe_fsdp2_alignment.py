#!/usr/bin/env python3
"""FSDP2 分片对齐探针（E18 / 上游包③）：FSDP2 在 world=3 下产出的 all_gather
载荷到底对不对齐 —— veScale 论文说 "FSDP1 and FSDP2 suffer"，这里拿我们自己的数。

做三件事（3 卡，~1 分钟）：
  ① 真实 Qwen3-4B 的一层 decoder（bf16）交给 fully_shard(mesh=(3,))，
     monkeypatch dist.all_gather_into_tensor **捕获真实调用**的载荷尺寸与指针对齐；
     用 FSDPModule.unshard() 触发 all-gather（不需要真 forward）。
  ② 与逐参数分片数学的预测对账（padded_dim0=ceil(d0/3)*3，shard=padded_dim0/3×其余维）
     —— 错位应当来自 1-D 参数（norm [2560]、q/k_norm [128]），大权重按行切天然对齐。
  ③ 在捕获到的真实尺寸上做 bf16 all_gather 微基准：原尺寸 vs 补齐到 16B —— 悬崖还是不是 12×。

产物：_audit/infra/e18_fsdp2_alignment.json
"""
from __future__ import annotations
import json, math, os, time
import torch, torch.distributed as dist, torch.multiprocessing as mp

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORLD = 3


def bench_allgather(rank: int, world: int, per_elems: int, iters: int = 15) -> float:
    """bf16 all_gather busbw（GB/s），口径与 probe_alignment_cliff 相同。"""
    s = torch.ones(per_elems, dtype=torch.bfloat16, device=rank)
    g = torch.zeros(per_elems * world, dtype=torch.bfloat16, device=rank)
    for _ in range(3):
        dist.all_gather_into_tensor(g, s)
    torch.cuda.synchronize(rank)
    t0 = time.perf_counter()
    for _ in range(iters):
        dist.all_gather_into_tensor(g, s)
    torch.cuda.synchronize(rank)
    dt = (time.perf_counter() - t0) / iters
    total = per_elems * 2 * world
    return total * (world - 1) / world / dt / 1e9


def worker(rank: int, world: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"; os.environ["MASTER_PORT"] = "30051"
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world)

    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard
    from transformers import AutoConfig
    from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer

    cfg = AutoConfig.from_pretrained(os.path.join(_REPO, "models/Qwen3-4B"))
    torch.manual_seed(0)
    layer = Qwen3DecoderLayer(cfg, layer_idx=0).to(dtype=torch.bfloat16, device=f"cuda:{rank}")

    # 逐参数分片数学（预测）
    pred = []
    for name, p in layer.named_parameters():
        d0 = p.shape[0]
        rest = int(math.prod(p.shape[1:])) if p.dim() > 1 else 1
        shard_elems = (math.ceil(d0 / world) * world // world) * rest
        b = shard_elems * 2
        pred.append({"param": name, "shape": list(p.shape), "shard_elems": shard_elems,
                     "shard_bytes": b, "mod16": b % 16})
    pred_total = sum(x["shard_bytes"] for x in pred)

    # 捕获真实 all_gather
    captured = []
    orig = dist.all_gather_into_tensor

    def spy(output, input, group=None, async_op=False):  # noqa: A002
        captured.append({
            "in_elems": input.numel(), "dtype": str(input.dtype),
            "in_bytes": input.numel() * input.element_size(),
            "in_bytes_mod16": (input.numel() * input.element_size()) % 16,
            "in_ptr_mod16": input.data_ptr() % 16,
            "out_ptr_mod16": output.data_ptr() % 16,
        })
        return orig(output, input, group=group, async_op=async_op)

    dist.all_gather_into_tensor = spy
    mesh = init_device_mesh("cuda", (world,), mesh_dim_names=["fsdp"])
    fully_shard(layer, mesh=mesh)
    layer.unshard()          # ★ 触发真实的 foreach_all_gather
    torch.cuda.synchronize(rank)
    dist.all_gather_into_tensor = orig

    if rank == 0:
        assert captured, "没捕获到 all_gather —— unshard 没触发？"
        c = captured[0]
        print(f"\n  预测 per-rank 载荷 = {pred_total:,} B（%16 = {pred_total % 16}）")
        print(f"  实测 per-rank 载荷 = {c['in_bytes']:,} B（%16 = {c['in_bytes_mod16']}）"
              f"  dtype={c['dtype']}  in_ptr%16={c['in_ptr_mod16']}  out_ptr%16={c['out_ptr_mod16']}")
        bad = [x for x in pred if x["mod16"]]
        print(f"  错位来源（逐参数 %16≠0 的）：")
        for x in bad:
            print(f"    {x['param']:<44}{str(x['shape']):<16}shard {x['shard_bytes']:>7,} B  %16={x['mod16']}")
    dist.barrier()

    # ③ 在真实尺寸上量悬崖（bf16）：原尺寸 vs 补到 8 元素（16 B）倍数
    per = captured[0]["in_elems"]
    per_aligned = math.ceil(per / 8) * 8
    bw_raw = bench_allgather(rank, world, per)
    bw_ali = bench_allgather(rank, world, per_aligned)
    if rank == 0:
        print(f"\n  bf16 all_gather @ FSDP2 真实 per-rank 尺寸：")
        print(f"    {per:,} elems（{per*2:,} B，%16={per*2%16}）      → {bw_raw:6.2f} GB/s")
        print(f"    {per_aligned:,} elems（{per_aligned*2:,} B，%16=0，+{(per_aligned-per)*2} B) → {bw_ali:6.2f} GB/s")
        print(f"    比值 = {bw_ali/bw_raw:.2f}×")
        out = {
            "experiment": "E18 · FSDP2 (world=3) 分片对齐探针",
            "script": "scripts/probe_fsdp2_alignment.py",
            "model_layer": "Qwen3-4B Qwen3DecoderLayer(bf16)",
            "world_size": world,
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0),
            "predicted_per_rank_bytes": pred_total,
            "captured": captured,
            "per_param_shards": pred,
            "bench": {"raw_elems": per, "raw_bytes": per * 2, "raw_mod16": per * 2 % 16,
                      "raw_GBps": round(bw_raw, 2),
                      "aligned_elems": per_aligned, "aligned_pad_bytes": (per_aligned - per) * 2,
                      "aligned_GBps": round(bw_ali, 2),
                      "ratio": round(bw_ali / bw_raw, 2)},
        }
        p = os.path.join(_REPO, "_audit", "infra", "e18_fsdp2_alignment.json")
        json.dump(out, open(p, "w"), ensure_ascii=False, indent=2)
        print(f"\n  产物已写：{os.path.relpath(p, _REPO)}")
    dist.destroy_process_group()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    mp.spawn(worker, args=(WORLD,), nprocs=WORLD, join=True)
