#!/usr/bin/env python3
"""E21 复现矩阵：FSDP1 `HYBRID_SHARD` + 网格 `(N,1)` 静默不同步梯度 —— 复现 + 三种修法对照。

★ 为什么要脱离 verl 做这个（E21 §5）：
我们在 verl 上观测到「三个 trainer rank 的梯度不同」，但那是在一整套框架里观测的。
**要判断这是 PyTorch 的行为还是 verl 的接线问题，必须把 verl 拿掉。**
这也是「能不能提上游」的前提。

复现的配置照抄 verl（`workers/engine/fsdp/utils.py:40`；仓库已迁 verl-project/verl，该函数逐字未变）：
    fsdp_size=1, world_size=3
    ⇒ mesh_shape=(world_size // fsdp_size, fsdp_size) = **(3, 1)**，维名 ["ddp", "fsdp"]
    ⇒ 二维网格 ⇒ get_sharding_strategy 选 HYBRID_SHARD
    ⇒ **分片维大小 = 1（等于不分片），复制维 = 3**

七个变体一次跑完（**每个 rank 喂不同的数据**，正是数据并行的场景）：
    A  mesh(3,1) HYBRID_SHARD · 部分参数可训（我们的配置）        ⇒ 预期 🔴 不同步
    B  同 A，全部可训（分离「部分可训」这个变量）                  ⇒ 预期 🔴 不同步
    C  mesh(3,)  FULL_SHARD 真分片                                ⇒ 梯度按 rank 分片，无可比值（无结论）
    E  FSDP1 NO_SHARD + **不传 mesh**（我们已上线的补丁形态）      ⇒ 预期 ✅
    G  FSDP1 NO_SHARD + **原样保留 (3,1) mesh**（拟提 verl 的 3 行修法）⇒ 预期 ✅
    F  FSDP2 fully_shard + **同一个 (3,1) mesh**（上游推荐的迁移路径） ⇒ 预期 ✅
    D  纯 DDP（对照组：证明测试装置本身是对的）                    ⇒ 必须 ✅

判据一（梯度）：反向后各 rank 的 train_b 梯度范数**逐位相同** ⇔ 同步。
判据二（后果）：一步 SGD 后各 rank 的 train_b 权重范数逐位相同 ⇔ 没有发散。
★ 2026-08-19 起每个模型创建前显式 manual_seed(1234) ⇒ **全脚本确定性**：
  所有「同步」变体应打出同一个梯度范数（跨变体也相同），任何人可逐位复验。
  （08-18 首版靠 sync_module_states 对齐起点，数值与本版不同，判定相同。）

产物（rank0 写）：_audit/infra/e21_grad_sync_matrix{_fixmode}.json ——
  环境指纹 / 逐变体两组范数 / 构造期捕获的 UserWarning **原文** /
  A 的内部状态取证（state.world_size=1 + 被遗弃的 3-rank 复制组）/
  E、G 两种修法形态的 SHARDED_STATE_DICT 探针（决定 verl PR 要不要动 ckpt 路径）。

用法：
    python scripts/infra/repro_fsdp_hybrid_nosync.py                    # 复现 + 修法对照
    REPRO_APPLY_FIX=1 python scripts/infra/repro_fsdp_hybrid_nosync.py  # 验证我们的补丁把 A/B 变绿
"""
from __future__ import annotations

import datetime
import json
import os
import warnings

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardedStateDictConfig, ShardingStrategy, StateDictType
from torch.nn.parallel import DistributedDataParallel as DDP

try:  # FSDP2：torch>=2.6 在 torch.distributed.fsdp 下导出
    from torch.distributed.fsdp import fully_shard
except ImportError:  # 老版本没有，矩阵里那一行记「不可用」
    fully_shard = None

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.frozen = nn.Linear(256, 256, bias=False)   # 模拟冻结的基座
        self.train_a = nn.Linear(256, 32, bias=False)   # 模拟 lora_A
        self.train_b = nn.Linear(32, 256, bias=False)   # 模拟 lora_B
        nn.init.zeros_(self.train_b.weight)             # ★ B 零初始化，和 LoRA 一样

    def forward(self, x):
        return self.train_b(self.train_a(self.frozen(x)))


def _grad_norm_of_train_b(model) -> tuple[float | None, str]:
    """读 train_b 的梯度范数。DTensor 感知；FSDP1 flat-param 兜底。"""
    from torch.distributed.tensor import DTensor

    for n, p in model.named_parameters():
        if "train_b" in n and p.requires_grad and p.grad is not None:
            if isinstance(p.grad, DTensor):
                # ★ 必须读 **local** 分片：判「复制维上各 rank 是否一致」，
                #   而 DTensor 的全局 norm 会把「应当一致」当成前提替我们抹平 —— 判据要绕开它。
                return p.grad.to_local().detach().float().norm().item(), "dtensor_local"
            return p.grad.detach().float().norm().item(), "plain"
    tot = 0.0                              # 兜底：梯度只挂在 flat param 上（如 FULL_SHARD）
    for mod in model.modules():
        fp = getattr(mod, "_flat_param", None)
        if fp is not None and fp.grad is not None:
            tot += fp.grad.detach().float().norm().item() ** 2
    return (tot ** 0.5, "flat_param") if tot > 0 else (None, "missing")


def _param_norm_of_train_b(model) -> float | None:
    """一步优化器之后读 train_b 的权重范数（判「发散了没有」）。"""
    from torch.distributed.tensor import DTensor

    def find():
        for n, p in model.named_parameters():
            if "train_b" in n:
                return p
        return None

    p = find()
    if p is not None and isinstance(p, DTensor):
        return p.to_local().detach().float().norm().item()
    if isinstance(model, FSDP):
        with FSDP.summon_full_params(model):
            q = find()
            return q.detach().float().norm().item() if q is not None else None
    return p.detach().float().norm().item() if p is not None else None


def _fsdp1_state_info(model: FSDP) -> dict:
    """A 的取证核心：钳完之后 state.world_size / 归约用的组 / 被遗弃的复制组各是多大。"""
    info = {
        "sharding_strategy_after_init": str(model.sharding_strategy),
        "state_world_size": model.world_size,
        "process_group_size": model.process_group.size(),
    }
    inp = getattr(model, "_inter_node_pg", None)
    if inp is not None:
        info["orphaned_inter_node_pg_size"] = inp.size()
    return info


def _fsdp2_state_info(model) -> dict:
    from torch.distributed.tensor import DTensor

    for n, p in model.named_parameters():
        if "train_b" in n and isinstance(p, DTensor):
            return {
                "param_type": "DTensor",
                "placements": str(p.placements),
                "mesh_shape": str(tuple(p.device_mesh.shape)),
            }
    return {"param_type": type(next(model.parameters())).__name__}


def _probe_sharded_state_dict(model) -> dict:
    """verl 的 fsdp ckpt 走 SHARDED_STATE_DICT（fsdp_checkpoint_manager.py:173）⇒
    修法若改变这条路的行为，PR 必须带 save/load 测试。这个探针先把「会不会炸、产出什么类型」带回来。"""
    try:
        with FSDP.state_dict_type(
            model, StateDictType.SHARDED_STATE_DICT, ShardedStateDictConfig(offload_to_cpu=False)
        ):
            sd = model.state_dict()
        return {"ok": True, "n_entries": len(sd), "value_types": sorted({type(v).__name__ for v in sd.values()})}
    except Exception as e:  # 探针的职责就是把失败原样带回来
        return {"ok": False, "error": f"{type(e).__name__}: {e}"[:400]}


def run_variant(rank: int, world: int, key: str, tag: str, model, param_check: bool = True) -> dict:
    torch.manual_seed(1234)
    x = torch.randn(4, 256, device=rank) * (rank + 1)   # ★ 每个 rank 喂不同的数据
    # ⚠️ 必须用「和目标的差」当损失：B 零初始化 ⇒ output=0 ⇒ 若用 output² 则 dL/dB 也恒为 0
    #    （这正是真实系统 step1 时 lora_A 梯度为 0 的同一个数学，第一版测试就栽在这）
    target = torch.ones(4, 256, device=rank)
    loss = (model(x) - target).square().mean()
    loss.backward()
    g, gnote = _grad_norm_of_train_b(model)
    torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.1).step()
    w = _param_norm_of_train_b(model) if param_check else None

    gbuf: list = [None] * world
    dist.all_gather_object(gbuf, g)
    wbuf: list = [None] * world
    dist.all_gather_object(wbuf, w)

    def judge(buf):
        if any(b is None for b in buf):
            return None
        return bool(max(buf) - min(buf) < 1e-9 * max(1.0, max(buf)))

    return {
        "key": key, "tag": tag,
        "grad_norms": gbuf, "grad_read": gnote, "grad_synced": judge(gbuf),
        "param_norms_after_step": wbuf if param_check else None,
        "param_identical": judge(wbuf) if param_check else None,
    }


def build_and_run(rank, world, results, key, tag, build_fn, param_check=True, probe_sd=False):
    torch.manual_seed(1234)                      # ★ 起点确定性：build_fn 第一步创建模型
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always")
        model = build_fn()
    r = run_variant(rank, world, key, tag, model, param_check=param_check)
    r["warnings_at_construction"] = sorted({str(w.message) for w in wlist})
    r["state_info"] = _fsdp1_state_info(model) if isinstance(model, FSDP) else _fsdp2_state_info(model)
    if probe_sd:
        r["sharded_state_dict_probe"] = _probe_sharded_state_dict(model)
    results.append(r)


def worker(rank: int, world: int) -> None:
    fix_mode = os.environ.get("REPRO_APPLY_FIX") == "1"
    if fix_mode:
        # ★ 在**子进程内**装上 E21 的修复补丁：同一个脚本既是复现、又是验证
        #   （spawn 的子进程不会继承父进程打的补丁）
        import sys
        sys.path.insert(0, _REPO)
        from syncopate.train.verl_patches import _patch_fsdp_degenerate_mesh
        _patch_fsdp_degenerate_mesh()
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "30021"
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world)
    results: list[dict] = []

    def tiny(partial: bool = True) -> Tiny:
        m = Tiny().cuda(rank)
        if partial:
            for p in m.frozen.parameters():
                p.requires_grad_(False)
        return m

    # A · 我们的配置：mesh(3,1) + HYBRID_SHARD + use_orig_params + 部分可训
    def build_a():
        mesh = init_device_mesh("cuda", (world, 1), mesh_dim_names=["ddp", "fsdp"])
        return FSDP(tiny(), device_mesh=mesh, sharding_strategy=ShardingStrategy.HYBRID_SHARD,
                    use_orig_params=True, sync_module_states=True, device_id=rank)
    build_and_run(rank, world, results, "A", "A · mesh(N,1) HYBRID+部分可训（我们的）", build_a)

    # B · 同 A，但全部可训
    def build_b():
        mesh = init_device_mesh("cuda", (world, 1), mesh_dim_names=["ddp2", "fsdp2"])
        return FSDP(tiny(partial=False), device_mesh=mesh, sharding_strategy=ShardingStrategy.HYBRID_SHARD,
                    use_orig_params=True, sync_module_states=True, device_id=rank)
    build_and_run(rank, world, results, "B", "B · mesh(N,1) HYBRID+全部可训", build_b)

    # C · 真分片（梯度按 rank 分片 ⇒ 本判据读不到可比值，历来「无结论」，保留作完整性）
    def build_c():
        mesh = init_device_mesh("cuda", (world,), mesh_dim_names=["fsdp3"])
        return FSDP(tiny(), device_mesh=mesh, sharding_strategy=ShardingStrategy.FULL_SHARD,
                    use_orig_params=True, sync_module_states=True, device_id=rank)
    build_and_run(rank, world, results, "C", "C · mesh(N,) FULL_SHARD+部分可训", build_c, param_check=False)

    # E · 我们已上线的补丁形态：NO_SHARD + 不传 mesh（默认进程组）
    def build_e():
        return FSDP(tiny(), sharding_strategy=ShardingStrategy.NO_SHARD,
                    use_orig_params=True, sync_module_states=True, device_id=rank)
    build_and_run(rank, world, results, "E", "E · NO_SHARD 无 mesh（我们的补丁形态）", build_e, probe_sd=True)

    # G · ★ 拟提给 verl 的 3 行修法：网格**原样保留** (N,1)，只把策略换成 NO_SHARD
    #     ⇒ 非 hybrid 路径取 mesh_dim=0（复制维，N 个 rank）当归约组（_init_utils.py:119）
    def build_g():
        mesh = init_device_mesh("cuda", (world, 1), mesh_dim_names=["ddp7", "fsdp7"])
        return FSDP(tiny(), device_mesh=mesh, sharding_strategy=ShardingStrategy.NO_SHARD,
                    use_orig_params=True, sync_module_states=True, device_id=rank)
    build_and_run(rank, world, results, "G", "G · mesh(N,1) NO_SHARD（拟提 verl 的修法）", build_g, probe_sd=True)

    # F · ★ FSDP2：同一个 (N,1) 网格交给 fully_shard —— 上游推荐的迁移路径
    #     （fully_shard 没有 sync_module_states ⇒ 靠 build_and_run 的 manual_seed 保证各 rank 起点一致）
    if fully_shard is not None:
        def build_f():
            mesh = init_device_mesh("cuda", (world, 1), mesh_dim_names=["ddp6", "fsdp6"])
            m = tiny()
            fully_shard(m, mesh=mesh)
            return m
        build_and_run(rank, world, results, "F", "F · mesh(N,1) FSDP2 fully_shard", build_f)
    else:
        results.append({"key": "F", "tag": "F · FSDP2（本 torch 版本不可用）", "grad_norms": None,
                        "grad_synced": None, "param_norms_after_step": None, "param_identical": None})

    # D · 纯 DDP 对照组（必须同步 —— 它证明测试装置本身是对的）
    def build_d():
        return DDP(tiny(), device_ids=[rank])
    build_and_run(rank, world, results, "D", "D · 纯 DDP（对照组）", build_d)

    if rank == 0:
        def v_grad(r):
            return {True: "✅ 同步", False: "🔴 **没同步**", None: "⚠️ 无可比值"}[r.get("grad_synced")]

        def v_param(r):
            return {True: "✅ 一致", False: "🔴 **已发散**", None: "—"}[r.get("param_identical")]

        def fmt(buf):
            if buf is None:
                return "（不可用）"
            return str([None if b is None else round(b, 8) for b in buf])

        print(f"\n  {'变体':<40}{'梯度范数（三 rank）':<44}{'一步 SGD 后权重范数':<40}判定")
        print("  " + "-" * 136)
        for r in results:
            print(f"  {r['tag']:<40}{fmt(r['grad_norms']):<44}{fmt(r['param_norms_after_step']):<40}"
                  f"{v_grad(r)} / {v_param(r)}")
        print("\n  判据：同步 ⇒ all-reduce 求平均 ⇒ 三个数**逐位相同**；一步后权重仍逐位相同 ⇒ 没发散。")

        a = next(r for r in results if r["key"] == "A")
        print("\n  A 的内部状态取证（bug 的结构性证据）：")
        print(f"    {a['state_info']}")
        print(f"    构造期 warning：{a['warnings_at_construction']}")
        for key in ("E", "G"):
            r = next(x for x in results if x["key"] == key)
            print(f"  {key} 的 SHARDED_STATE_DICT 探针：{r.get('sharded_state_dict_probe')}")

        artifact = os.path.join(_REPO, "_audit", "infra",
                                f"e21_grad_sync_matrix{'_fixmode' if fix_mode else ''}.json")
        payload = {
            "experiment": "E21 · FSDP1 HYBRID_SHARD (N,1) 梯度同步矩阵（复现 + 三种修法对照）",
            "script": "scripts/infra/repro_fsdp_hybrid_nosync.py",
            "date": datetime.datetime.now().isoformat(timespec="seconds"),
            "fix_mode": fix_mode,
            "env": {
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0),
                "world_size": world,
                "NCCL_CUMEM_ENABLE": os.environ.get("NCCL_CUMEM_ENABLE"),
            },
            "criteria": "同步 ⇔ 三 rank 梯度范数逐位相同（all-reduce 求平均）；一步 SGD 后权重范数逐位相同 ⇔ 未发散",
            "variants": results,
        }
        os.makedirs(os.path.dirname(artifact), exist_ok=True)
        with open(artifact, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\n  产物已写：{os.path.relpath(artifact, _REPO)}")
    dist.destroy_process_group()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    mp.spawn(worker, args=(3,), nprocs=3, join=True)
