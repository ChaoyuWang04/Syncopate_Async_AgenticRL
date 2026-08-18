#!/usr/bin/env python3
"""0-A 第一级 · 梯度归约的**口径**：3 卡合起来的梯度，等不等于 1 卡在全部数据上的梯度？

★ 为什么需要这个（E21 之后的下一个问题）：
E21 修好的是「三个 rank 的梯度**有没有**碰面」（此前没有，各训各的）。
**但"碰了面"不等于"合对了"。** 数据并行的定义只有一句话：

    3 卡跑出来的梯度，必须等于 1 卡在那 48 条上跑出来的梯度。

而"平均还是求和"**不是自由选项** —— 它由「每张卡本地怎么归一化」决定，两者必须配套：

    甲 · 本地平均       loss_r = Σ_{i∈r} l_i / n_r
        ⇒ 归约必须**求平均**才等于全局平均。**隐含前提：每张卡的样本/token 数一样多**
    乙 · verl 的口径    loss_r = Σ_{i∈r} l_i / N_total × world_size
        ⇒ 预先乘了 world ⇒ 归约除以 world 时两者抵消，**且各卡数量不等也成立**
        （verl/trainer/ppo/core_algos.py:1172 `masked_sum(...) / batch_num_tokens * dp_size`）

⇒ verl 走的是乙，它**硬依赖「底层会除以 dp_size」这个前提**。本脚本就是去量这个前提。

⚠️ 本脚本**照 verl 的方式构造 FSDP**（mesh(3,1) + HYBRID_SHARD）**再装上 E21 的补丁**，
   量的是真实路径，不是"理想配置"。

四个变体（判据：与单进程参考的相对差）：
    ①  等量数据 + 甲（本地平均）      预测 ✅ 一致（数量相等时甲乙等价）
    ②  等量数据 + 乙（verl 口径）      预测 ✅ 一致 ← **这是我们真实的口径**
    ③  不等量数据 + 甲                 预测 🔴 不一致（经典的 mean-of-means 坑）
    ④  不等量数据 + 乙                 预测 ✅ 一致 ← 说明 verl 这段设计确实在保护我们

参考值由**同一进程内的非 FSDP 模型**在拼接后的全部数据上算出，权重逐位相同（同一个 seed）。
比较两个量：‖g‖（幅度）与 g·u（方向，u 是固定伪随机向量）—— 只比范数会漏掉方向错误。
"""
from __future__ import annotations

import os
import sys

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy

WORLD = 3
DIM = 256
# 每个 rank 的样本数：等量档 / 不等量档（不等量才是我们真实的形状 —— 变长序列）
EQUAL_COUNTS = [4, 4, 4]
UNEQUAL_COUNTS = [2, 4, 6]


class Tiny(nn.Module):
    """模拟 LoRA：冻结基座 + 两个可训的小矩阵。这里 B 不做零初始化，
    这样 A 和 B 在第一步就都有梯度（零初始化时 dL/dA ≡ 0，会少测一半参数）。"""

    def __init__(self) -> None:
        super().__init__()
        self.frozen = nn.Linear(DIM, DIM, bias=False)
        self.train_a = nn.Linear(DIM, 32, bias=False)
        self.train_b = nn.Linear(32, DIM, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.train_b(self.train_a(self.frozen(x)))


def make_model(device: int) -> Tiny:
    torch.manual_seed(1234)  # 所有 rank + 参考模型共用一个初始化
    m = Tiny().cuda(device)
    for p in m.frozen.parameters():
        p.requires_grad_(False)
    return m


def make_data(device: int, counts: list[int]) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """在**每个 rank 上都生成全部数据**（同一个 seed ⇒ 逐位相同），再各取各的切片。
    这样 rank0 才能在同一份数据上算出单进程参考值。"""
    torch.manual_seed(999)
    xs, ts = [], []
    for r, n in enumerate(counts):
        xs.append((torch.randn(n, DIM, device=device) * (r + 1)))
        ts.append(torch.ones(n, DIM, device=device) * (r + 1) * 0.1)
    return xs, ts


def per_sample_loss(model: nn.Module, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """每条样本一个标量损失（类比"每条序列的 token 损失和"）。"""
    return (model(x) - t).square().sum(dim=1)


def flat_grad(model: nn.Module, device: int) -> torch.Tensor:
    """把可训参数的梯度拼成一个一维向量（按参数名排序，保证两边顺序一致）。"""
    named = sorted(
        (n.replace("_fsdp_wrapped_module.", ""), p)
        for n, p in model.named_parameters()
        if p.requires_grad
    )
    chunks = [p.grad.detach().float().reshape(-1) for _, p in named if p.grad is not None]
    if not chunks:  # 兜底：梯度可能只挂在 flat param 上
        for mod in model.modules():
            fp = getattr(mod, "_flat_param", None)
            if fp is not None and fp.grad is not None:
                chunks.append(fp.grad.detach().float().reshape(-1))
    return torch.cat(chunks) if chunks else torch.zeros(1, device=device)


def probe(g: torch.Tensor) -> tuple[float, float]:
    """返回（幅度, 方向探针）。方向探针 = 与固定伪随机向量的内积，只比范数会漏掉方向错误。"""
    gen = torch.Generator(device=g.device).manual_seed(4242)
    u = torch.randn(g.numel(), generator=gen, device=g.device, dtype=g.dtype)
    return g.norm().item(), torch.dot(g, u).item()


def reference(device: int, counts: list[int]) -> tuple[float, float]:
    """单进程参考：同样的权重，在**拼接后的全部数据**上算 Σl_i / N_total。"""
    m = make_model(device)
    xs, ts = make_data(device, counts)
    losses = torch.cat([per_sample_loss(m, x, t) for x, t in zip(xs, ts)])
    (losses.sum() / sum(counts)).backward()
    return probe(flat_grad(m, device))


def distributed_run(rank: int, counts: list[int], convention: str) -> tuple[float, float]:
    """3 卡：照 verl 的方式构造 FSDP（补丁已装 ⇒ 实际落到 NO_SHARD + 默认进程组）。"""
    m = make_model(rank)
    mesh = init_device_mesh("cuda", (WORLD, 1), mesh_dim_names=[f"ddp_{convention}", f"fsdp_{convention}"])
    f = FSDP(m, device_mesh=mesh, sharding_strategy=ShardingStrategy.HYBRID_SHARD,
             use_orig_params=True, sync_module_states=True, device_id=rank)
    xs, ts = make_data(rank, counts)
    losses = per_sample_loss(f, xs[rank], ts[rank])
    if convention == "local_mean":                      # 甲 · 本地平均
        loss = losses.sum() / counts[rank]
    else:                                               # 乙 · verl：本地和 / 全局数 × world
        loss = losses.sum() / sum(counts) * WORLD
    loss.backward()
    return probe(flat_grad(f, rank))


def worker(rank: int) -> None:
    # spawn 的子进程不继承父进程打的补丁 ⇒ 必须在这里装（E21 §4.6.1 踩过这个坑）
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from syncopate.train.verl_patches import _patch_fsdp_degenerate_mesh

    _patch_fsdp_degenerate_mesh()
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "30031"
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=WORLD)

    cases = [
        ("① 等量数据 + 甲(本地平均)", EQUAL_COUNTS, "local_mean"),
        ("② 等量数据 + 乙(verl 口径)", EQUAL_COUNTS, "verl"),
        ("③ 不等量数据 + 甲(本地平均)", UNEQUAL_COUNTS, "local_mean"),
        ("④ 不等量数据 + 乙(verl 口径)", UNEQUAL_COUNTS, "verl"),
    ]
    rows = []
    for tag, counts, conv in cases:
        got = distributed_run(rank, counts, conv)
        ref = reference(rank, counts) if rank == 0 else None
        rows.append((tag, counts, got, ref))

    if rank == 0:
        print(f"\n  {'变体':<30}{'各卡样本数':<14}{'3 卡 ‖g‖':<14}{'1 卡 ‖g‖':<14}{'比值':<10}判定")
        print("  " + "-" * 104)
        for tag, counts, (gn, gd), ref in rows:
            rn, rd = ref
            ratio = gn / rn if rn else float("nan")
            dir_ok = abs(gd - rd) <= 1e-4 * max(1.0, abs(rd))
            same = abs(ratio - 1.0) < 1e-4 and dir_ok
            verdict = "✅ 等于单卡" if same else f"🔴 **不等**（方向{'一致' if dir_ok else '也不一致'}）"
            print(f"  {tag:<30}{str(counts):<14}{gn:<14.8f}{rn:<14.8f}{ratio:<10.6f}{verdict}")
        print("\n  判据：比值 ≈ 1.000 ⇒ 归约是**求平均**，口径对。")
        print("        比值 ≈ 3.000 ⇒ 归约是**求和**，等效 lr 系统性大 3 倍。")
        print("  ⚠️ ③ 若不等而 ④ 相等 ⇒ verl 那套「按全局 token 数归一 × dp_size」确实在保护我们。")
    dist.destroy_process_group()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    mp.spawn(worker, nprocs=WORLD, join=True)
