# 提交件 · verl issue + PR 正文（英文，GitHub 可直接粘贴）

> 状态：**草稿完成，等 Chaoyu 过目后提交**（2026-08-19）。
> 中文分析与证据链 → [`analysis.md`](analysis.md)；
> PyTorch 侧背景 → [`pytorch-background.md`](pytorch-background.md)。
> 提交目标仓库：**verl-project/verl**（不是旧的 volcengine/verl）。
> 顺序：先提 issue，PR 在 issue 里预告、随后提交并互相引用。

---

## 1 · Issue

**Title:**

```
[fsdp] fsdp_size=1 on multiple GPUs silently disables gradient synchronization
(degenerate (N,1) device mesh + FSDP1 HYBRID_SHARD)
```

**Body:**

````markdown
## Summary

With the FSDP1 backend (`strategy: fsdp`), setting `fsdp_config.fsdp_size=1` on a
multi-GPU trainer — the natural configuration for "data parallel, no sharding"
(small models, LoRA) — **silently disables gradient synchronization across ranks**.

Each rank trains its own independent copy of the model. Training runs to completion,
loss goes down, grad_norm / entropy / KL all look normal. Nothing fails. The
checkpointed rank-0 copy has effectively seen `1/world_size` of the data.

In our case this ran undetected for two months of RL experiments (3 trainer ranks:
every update used 16 of the 48 sampled sequences).

## Root cause

`create_device_mesh` expresses `fsdp_size=1` as a 2-D mesh, and the strategy is
chosen from the mesh's *rank count*, not its *shape*:

```python
# verl/workers/engine/fsdp/utils.py
mesh_shape=(world_size // fsdp_size, fsdp_size)   # fsdp_size=1  =>  (N, 1)
# get_sharding_strategy: ndim == 2  =>  HYBRID_SHARD
```

PyTorch FSDP1 then does the rest (all line refs at torch 2.9.0; `main` is identical):

```
_init_utils.py:152-153   HYBRID_SHARD branch: _inter_node_pg  = mesh.get_group(0)  # replicate dim, N ranks
                                              process_group   = mesh.get_group(1)  # shard dim, 1 rank
_init_utils.py:127       state.world_size = state.process_group.size()   ->  1
_init_core_state         world_size == 1  ->  UserWarning + clamp to NO_SHARD
_runtime_utils.py:936    dist.all_reduce(flat_param.grad, group=state.process_group)
                         -> all-reduce over a size-1 group  ->  no-op
                         (_inter_node_pg is only used on the hybrid branch, which
                          is no longer taken after the clamp — the N-rank replicate
                          group is created and then orphaned)
```

The only signal is:

```
UserWarning: FSDP is switching to use `NO_SHARD` instead of
ShardingStrategy.HYBRID_SHARD since the world size is 1.
```

It says "I changed strategy", not "your gradients are no longer synchronized".
Note this exact warning also fires for a *benign* and much more common reason —
a misconfigured cluster where world size really is 1 (e.g. #2478, where the answer
was "check `ray status`") — so users who see it have no reason to suspect silent
gradient divergence.

## Why this needs to be fixed in verl

PyTorch already confirmed the underlying behavior as a bug and declined to fix it:
[pytorch/pytorch#154888](https://github.com/pytorch/pytorch/issues/154888) —
maintainer: *"this is a bug ... we might be slow in fixing fsdp1"* — closed as
`not_planned` (FSDP1 is in maintenance mode). So the framework that builds the
degenerate mesh is the only place left to stop it.

## Minimal reproduction (pure PyTorch, no verl, deterministic)

```python
import os, torch, torch.nn as nn, torch.distributed as dist, torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
from torch.nn.parallel import DistributedDataParallel as DDP

def worker(rank, world):
    os.environ["MASTER_ADDR"] = "127.0.0.1"; os.environ["MASTER_PORT"] = "29531"
    torch.cuda.set_device(rank); dist.init_process_group("nccl", rank=rank, world_size=world)

    def run(tag, wrap):
        torch.manual_seed(0)
        model = wrap(nn.Linear(256, 256, bias=False).cuda(rank))
        x = torch.randn(4, 256, device=rank) * (rank + 1)      # different data per rank
        model(x).square().mean().backward()
        g = next(p.grad for p in model.parameters() if p.grad is not None)
        buf = [None] * world; dist.all_gather_object(buf, g.detach().norm().item())
        if rank == 0:
            same = max(buf) - min(buf) < 1e-9 * max(buf)
            print(f"{tag:<48} {[f'{b:.8f}' for b in buf]}  {'OK' if same else 'NOT SYNCED'}")

    mesh = init_device_mesh("cuda", (world, 1), mesh_dim_names=["ddp", "fsdp"])
    run("(N,1) + HYBRID_SHARD (= fsdp_size=1 today)",
        lambda m: FSDP(m, device_mesh=mesh, sharding_strategy=ShardingStrategy.HYBRID_SHARD,
                       use_orig_params=True, sync_module_states=True, device_id=rank))
    mesh2 = init_device_mesh("cuda", (world, 1), mesh_dim_names=["ddp2", "fsdp2"])
    run("(N,1) + NO_SHARD (= proposed fix)",
        lambda m: FSDP(m, device_mesh=mesh2, sharding_strategy=ShardingStrategy.NO_SHARD,
                       use_orig_params=True, sync_module_states=True, device_id=rank))
    run("plain DDP (control)", lambda m: DDP(m, device_ids=[rank]))
    dist.destroy_process_group()

if __name__ == "__main__":
    mp.spawn(worker, args=(min(torch.cuda.device_count(), 3),),
             nprocs=min(torch.cuda.device_count(), 3), join=True)
```

Measured (3× RTX 5090, torch 2.9.0+cu128, fully deterministic — reproducible bit-for-bit):

| variant | grad norms across 3 ranks | verdict |
|---|---|---|
| `(3,1)` mesh + `HYBRID_SHARD` (= `fsdp_size=1` today) | `[0.04565846, 0.09131692, 0.13697541]` | 🔴 not synced |
| `(3,1)` mesh + `NO_SHARD` (= proposed fix, **same mesh**) | `0.27395082` ×3 | ✅ |
| `(3,1)` mesh + FSDP2 `fully_shard` | `0.27395082` ×3 | ✅ (fsdp2 backend unaffected) |
| plain DDP (control) | `0.27395082` ×3 | ✅ |

Two details worth noting:

- The broken variant's three numbers are exactly `[g, 2g, 3g]` — each rank holds the
  gradient of *its own data only* (inputs are scaled by `rank+1`), pre-divided by 3
  by FSDP in anticipation of an all-reduce that never happens. Their mean equals the
  correct value `0.27395082` to 4.5e-8.
- Internal state of the broken variant: `state.world_size == 1`, reduction group size 1,
  and an **orphaned replicate group of size 3** that is created but never used.

## Evidence from real training (verl 0.8.0, Qwen3-4B + LoRA r32, 3 trainer ranks)

LoRA `B` matrices are zero-initialized, so at step 1 all ranks have *bit-identical
weights* — yet their gradients already differ (probe hooked before `optimizer_step`):

```
step=1  rank=0  lora_B  weight_norm=0.000000   grad_norm=2.209380e-05
step=1  rank=1  lora_B  weight_norm=0.000000   grad_norm=2.565470e-05   <- 16% apart
step=4  rank=0  lora_B  grad=2.906737e-04
step=4  rank=2  lora_B  grad=8.142963e-05                               <- 3.6x apart
```

Identical starting point + different gradients ⇒ the only possible cause is a missing
all-reduce. After 15 updates, the cross-rank relative difference of the trained LoRA
converges to ≈ √2 (1.4136 / 1.4132 / 1.4139 across four independent runs) — the
distance between equal-length random vectors, i.e. the three ranks learned
*statistically unrelated* things. Adam `exp_avg_sq` differs by 99% across ranks.

## Proposed fix

Choose `NO_SHARD` when the shard dim is degenerate — the mesh itself stays exactly
as it is; FSDP's non-hybrid path then uses `mesh_dim=0` (the replicate dim, N ranks)
as the reduction group (`_init_utils.py:119`):

```python
# verl/workers/engine/fsdp/utils.py :: get_sharding_strategy
elif device_mesh.ndim == 2:
    if device_mesh.size(1) == 1:
        sharding_strategy = ShardingStrategy.NO_SHARD
    else:
        sharding_strategy = hsdp_strategy
```

Verified: gradients become bit-identical across ranks (table above, and a real
verl training run with only this diff applied). Checkpoint format is unchanged —
with `NO_SHARD`, `SHARDED_STATE_DICT` already short-circuits to full tensors
*both before and after* the fix (the current broken config is clamped to `NO_SHARD`
internally anyway), so resume compatibility is unaffected. Will send a PR.

Also worth considering as a follow-up: a one-time post-first-step assertion that
gradients actually match across data-parallel ranks. This failure class (silent
gradient divergence) is invisible to every training metric; a single `all_gather`
of one gradient norm at step 1 would have caught this — and any future variant —
immediately.

**Workarounds for affected users** until then: use `strategy: fsdp2` (verified
unaffected), or pin `fsdp_size` to the full world size (accepting sharding cost).

## Environment

- verl 0.8.0 (`create_device_mesh` / `get_sharding_strategy` unchanged on current `main`)
- torch 2.9.0+cu128, NCCL 2.27.5
- 3–4 × RTX 5090 (also reproduced by others on 2 GPUs: pytorch#154888, forum #220486)
````

---

## 2 · PR

**Title:**

```
[fsdp] fix: fsdp_size=1 silently disables gradient sync — use NO_SHARD for degenerate (N,1) mesh
```

**Diff（对 verl main，函数只有这一处调用点 `transformer_impl.py:375`）：**

```diff
--- a/verl/workers/engine/fsdp/utils.py
+++ b/verl/workers/engine/fsdp/utils.py
@@ def get_sharding_strategy(device_mesh, zero3_enable=True):
     elif device_mesh.ndim == 2:
-        sharding_strategy = ShardingStrategy.HYBRID_SHARD
-        sharding_strategy = hsdp_strategy
+        if device_mesh.size(1) == 1:
+            # fsdp_size=1 degenerates the shard dim to size 1. FSDP1 clamps
+            # HYBRID_SHARD to NO_SHARD but keeps gradient reduction on the
+            # size-1 shard group, so gradients are never synchronized across
+            # the replicate dim -- silently (pytorch/pytorch#154888, closed
+            # as not_planned). Explicit NO_SHARD makes FSDP's non-hybrid path
+            # use mesh_dim=0 (the replicate dim) as the reduction group.
+            sharding_strategy = ShardingStrategy.NO_SHARD
+        else:
+            sharding_strategy = ShardingStrategy.HYBRID_SHARD
+            sharding_strategy = hsdp_strategy
```

**Body:**

````markdown
Fixes #<issue-number>.

## What

`fsdp_size=1` on a multi-GPU trainer (FSDP1 backend) builds a `(N, 1)` device mesh
and selects `HYBRID_SHARD`. PyTorch FSDP1 clamps that to `NO_SHARD` (shard dim has
1 rank) but leaves gradient reduction on the size-1 *shard* group — a no-op — so the
N ranks silently train N independent models. PyTorch confirmed the behavior as a bug
and closed it as not_planned (FSDP1 maintenance mode): pytorch/pytorch#154888.

This PR selects `NO_SHARD` explicitly when the shard dim is degenerate. The mesh is
untouched; FSDP's non-hybrid path then reduces gradients over `mesh_dim=0` — the
replicate dim, which is exactly the intended "data parallel, no sharding" semantics.

## Why this shape of fix

- `get_sharding_strategy` has a single call site (`transformer_impl.py`); no config
  or signature changes.
- Mesh shape and `mesh_dim_names` are unchanged ⇒ `model_merger`'s
  `assert mesh_dim_names in (("fsdp",), ("ddp", "fsdp"))` and existing checkpoints
  are unaffected.
- Checkpoint format is unchanged: under `NO_SHARD`, `SHARDED_STATE_DICT` already
  short-circuits to full tensors both before this PR (the strategy was clamped to
  `NO_SHARD` internally) and after (it is now `NO_SHARD` explicitly). Verified by
  probing `state_dict()` value types in both configurations: identical.
- Non-degenerate configs (`fsdp_size>1`, and 1-D meshes) take the exact same path
  as before.
- The only behavioral change is the gradient reduction group: size-1 shard group →
  N-rank replicate group. That is precisely the bug.

Note: FSDP1's `NO_SHARD` emits a deprecation `FutureWarning` pointing to DDP; within
FSDP1 it is nevertheless the only correct strategy for this topology. The `fsdp2`
backend is unaffected (verified: same `(N,1)` mesh under `fully_shard` syncs
correctly).

## Validation

1. Deterministic 3-GPU matrix (pure PyTorch): broken config shows per-rank gradients
   `[g, 2g, 3g]` (each rank's own data only); with this fix all ranks produce
   bit-identical `0.27395082`, matching a plain-DDP control bit-for-bit.
2. Real RL training (Qwen3-4B + LoRA r32, 3 trainer ranks, 4 updates) with only this
   diff applied: per-rank gradient norms match exactly at runtime, and the three
   saved rank checkpoints are **bit-identical for all 504/504 trainable tensors**,
   optimizer state included (they converge to a cross-rank relative difference of
   ≈ √2 — statistically unrelated — without the fix).

## Test

Added `test_fsdp_size_1_gradients_synchronized` (2 GPUs, skipped when unavailable):
builds the mesh via `create_device_mesh(world, fsdp_size=1)`, feeds each rank
different data, asserts post-backward gradients are identical across ranks.
Fails before this PR, passes after.
````

**测试文件草稿**（放哪个目录按维护者惯例调整；2 卡即可触发）：

```python
# tests/workers/engine/fsdp/test_degenerate_mesh_grad_sync.py
import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn


def _worker(rank, world, port, q):
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    from verl.workers.engine.fsdp.utils import create_device_mesh, get_sharding_strategy

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world)
    torch.manual_seed(0)
    mesh = create_device_mesh(world_size=world, fsdp_size=1)     # (N, 1)
    strategy = get_sharding_strategy(mesh)                        # NO_SHARD after the fix
    model = FSDP(nn.Linear(64, 64, bias=False).cuda(rank), device_mesh=mesh,
                 sharding_strategy=strategy, use_orig_params=True,
                 sync_module_states=True, device_id=rank)
    x = torch.randn(2, 64, device=rank) * (rank + 1)              # different data per rank
    model(x).square().mean().backward()
    g = next(p.grad for p in model.parameters() if p.grad is not None)
    buf = [None] * world
    dist.all_gather_object(buf, g.detach().norm().item())
    if rank == 0:
        q.put(buf)
    dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs >= 2 GPUs")
def test_fsdp_size_1_gradients_synchronized():
    world = 2
    ctx = mp.get_context("spawn")
    q = ctx.SimpleQueue()
    mp.spawn(_worker, args=(world, 29532, q), nprocs=world, join=True)
    norms = q.get()
    assert max(norms) - min(norms) < 1e-9 * max(norms), (
        f"gradients differ across ranks: {norms} — fsdp_size=1 is not synchronizing"
    )
```

---

## 3 · 提交时的注意事项

- [ ] issue 先提，拿到编号后 PR 里填 `Fixes #<n>`
- [ ] PR 分支基于 **verl-project/verl main**（两个函数逐字未变，diff 直接适用）
- [ ] 检查 verl 的 CONTRIBUTING / CI 要求（DCO sign-off？pre-commit？测试目录惯例？）
- [ ] PyTorch 侧动作（可选、独立）：在 pytorch#154888 评论补真实训练证据 + 本 issue 链接
- [ ] 提交前把仓库私有信息核一遍（机器路径、内部实验名不出现在正文里 —— 目前正文已清洁）
