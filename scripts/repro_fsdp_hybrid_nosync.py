#!/usr/bin/env python3
"""E21 最小复现：FSDP `HYBRID_SHARD` + 网格 `(N, 1)` 下，梯度到底同不同步。

★ 为什么要脱离 verl 做这个（E21 §5）：
我们在 verl 上观测到「三个 trainer rank 的梯度不同」，但那是在一整套框架里观测的。
**要判断这是 PyTorch 的行为还是 verl 的接线问题，必须把 verl 拿掉。**
这也是「能不能提上游」的前提。

复现的配置照抄 verl（`workers/engine/fsdp/utils.py:40`）：
    fsdp_size=1, world_size=3
    ⇒ mesh_shape=(world_size // fsdp_size, fsdp_size) = **(3, 1)**，维名 ["ddp", "fsdp"]
    ⇒ 二维网格 ⇒ sharding_strategy = HYBRID_SHARD
    ⇒ **分片维大小 = 1（等于不分片），复制维 = 3**

四个变体，一次跑完（**每个 rank 喂不同的数据**，正是 DDP 的场景）：
    A  mesh(3,1) HYBRID_SHARD · use_orig_params=True · **只有部分参数可训**  ← 我们的配置
    B  同 A，但**全部参数可训**            —— 分离"部分可训"这个变量
    C  mesh(3,)  FULL_SHARD                —— 真分片时同不同步
    D  纯 DDP                              —— **对照组，必须同步**

判据：反向之后，把各 rank 的梯度范数收集起来比。
      **同步 ⇒ 三个数逐位相同（DDP 是 all-reduce 求平均）。不同 ⇒ 没同步。**
"""
from __future__ import annotations
import os
import torch, torch.nn as nn, torch.distributed as dist, torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
from torch.nn.parallel import DistributedDataParallel as DDP


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.frozen = nn.Linear(256, 256, bias=False)   # 模拟冻结的基座
        self.train_a = nn.Linear(256, 32, bias=False)   # 模拟 lora_A
        self.train_b = nn.Linear(32, 256, bias=False)   # 模拟 lora_B
        nn.init.zeros_(self.train_b.weight)             # ★ B 零初始化，和 LoRA 一样
    def forward(self, x):
        return self.train_b(self.train_a(self.frozen(x)))


def run_variant(rank: int, world: int, tag: str, model: nn.Module) -> tuple[str, list[float]]:
    torch.manual_seed(1234)                       # 所有 rank 同一个初始化
    x = torch.randn(4, 256, device=rank) * (rank + 1)   # ★ 每个 rank 喂不同的数据
    # ⚠️ 必须用「和目标的差」当损失：B 零初始化 ⇒ output=0 ⇒ 若用 output² 则 dL/dB 也恒为 0
    #    （这正是真实系统 step1 时 lora_A 梯度为 0 的同一个数学，第一版测试就栽在这）
    target = torch.ones(4, 256, device=rank)
    loss = (model(x) - target).square().mean()
    loss.backward()
    g, seen = None, []
    for n, p in model.named_parameters():
        seen.append((n, p.requires_grad, p.grad is not None))
        if p.requires_grad and "train_b" in n and p.grad is not None:
            g = p.grad.detach().float().norm().item()
            break
    if g is None:                       # 兜底：FSDP 可能只把梯度挂在 flat param 上
        tot = 0.0
        for p in model.parameters():
            if p.grad is not None:
                tot += p.grad.detach().float().norm().item() ** 2
        for mod in model.modules():
            fp = getattr(mod, "_flat_param", None)
            if fp is not None and fp.grad is not None:
                tot += fp.grad.detach().float().norm().item() ** 2
        g = tot ** 0.5 if tot > 0 else None
    if g is None and rank == 0:
        print(f"  ⚠️ [{tag}] 一个梯度都没读到。参数清单（名字, 可训, 有梯度）：")
        for it in seen[:8]:
            print("     ", it)
    buf = [None] * world
    dist.all_gather_object(buf, g)
    return tag, buf


def worker(rank: int, world: int) -> None:
    # ★ REPRO_APPLY_FIX=1 时在**子进程内**装上 E21 的修复补丁，
    #   于是同一个脚本既是复现、又是验证（spawn 的子进程不会继承父进程打的补丁）
    if os.environ.get("REPRO_APPLY_FIX") == "1":
        import sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from syncopate.train.verl_patches import _patch_fsdp_degenerate_mesh
        _patch_fsdp_degenerate_mesh()
    os.environ["MASTER_ADDR"] = "127.0.0.1"; os.environ["MASTER_PORT"] = "30021"
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world)
    results = []

    # A · 我们的配置：mesh(3,1) + HYBRID_SHARD + use_orig_params + 部分可训
    m = Tiny().cuda(rank)
    for p in m.frozen.parameters(): p.requires_grad_(False)
    mesh = init_device_mesh("cuda", (world, 1), mesh_dim_names=["ddp", "fsdp"])
    fa = FSDP(m, device_mesh=mesh, sharding_strategy=ShardingStrategy.HYBRID_SHARD,
              use_orig_params=True, sync_module_states=True, device_id=rank)
    results.append(run_variant(rank, world, "A · mesh(N,1) HYBRID+部分可训（我们的）", fa))

    # B · 同 A，但全部可训
    m2 = Tiny().cuda(rank)
    mesh2 = init_device_mesh("cuda", (world, 1), mesh_dim_names=["ddp2", "fsdp2"])
    fb = FSDP(m2, device_mesh=mesh2, sharding_strategy=ShardingStrategy.HYBRID_SHARD,
              use_orig_params=True, sync_module_states=True, device_id=rank)
    results.append(run_variant(rank, world, "B · mesh(N,1) HYBRID+全部可训", fb))

    # C · 真分片
    m3 = Tiny().cuda(rank)
    for p in m3.frozen.parameters(): p.requires_grad_(False)
    mesh3 = init_device_mesh("cuda", (world,), mesh_dim_names=["fsdp3"])
    fc = FSDP(m3, device_mesh=mesh3, sharding_strategy=ShardingStrategy.FULL_SHARD,
              use_orig_params=True, sync_module_states=True, device_id=rank)
    results.append(run_variant(rank, world, "C · mesh(N,) FULL_SHARD+部分可训", fc))

    # E · **候选修法**：fsdp_size=1 时不要建 (N,1) 二维网格，
    #     直接用 FSDP(NO_SHARD) + 默认进程组（= 全部 3 张卡）
    m5 = Tiny().cuda(rank)
    for p in m5.frozen.parameters(): p.requires_grad_(False)
    fe = FSDP(m5, sharding_strategy=ShardingStrategy.NO_SHARD,
              use_orig_params=True, sync_module_states=True, device_id=rank)
    results.append(run_variant(rank, world, "E · NO_SHARD 无 mesh（候选修法）", fe))

    # D · 纯 DDP 对照组（必须同步）
    m4 = Tiny().cuda(rank)
    for p in m4.frozen.parameters(): p.requires_grad_(False)
    fd = DDP(m4, device_ids=[rank])
    results.append(run_variant(rank, world, "D · 纯 DDP（对照组）", fd))

    if rank == 0:
        print(f"\n  {'变体':<38}{'三个 rank 的梯度范数':<46}判定")
        print("  " + "-" * 96)
        for tag, buf in results:
            if any(b is None for b in buf):
                print(f"  {tag:<38}{'（读不到梯度）':<46}⚠️ 无法判定")
                continue
            same = max(buf) - min(buf) < 1e-9 * max(1.0, max(buf))
            verdict = "✅ 同步" if same else "🔴 **没同步**"
            print(f"  {tag:<38}{str([round(b, 8) for b in buf]):<46}{verdict}")
        print("\n  判据：DDP 会把梯度 all-reduce 成平均值 ⇒ 同步时三个数**逐位相同**。")
    dist.destroy_process_group()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    mp.spawn(worker, args=(3,), nprocs=3, join=True)
