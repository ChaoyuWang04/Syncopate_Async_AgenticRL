# 包① · GitHub Bug Report 模板逐字段填写指南（2026-08-20）

> ⚠️ 他们的 Bug report 是**结构化模板**，不能整篇粘贴。下面按字段分好，逐个复制。
> 带 ✳ 的是必填。System Info 的内容是**今天在上游 main（2eaaa8f）实跑 `scripts/diagnose.py`** 得到的。

---

## 字段 1 ✳ System Info

> 整段复制（含 ``` 三引号）

```
----------Python Info----------
Version      : 3.12.14
Compiler     : Clang 22.1.3 
Build        : ('main', 'Aug 14 2026 15:34:45')
Arch         : ('64bit', 'ELF')
------------Pip Info-----------
No corresponding pip install for current python.
vllm	     : 0.12.0
sglang	     : not found.
ray	     : 2.57.0
torch	     : 2.9.0
----------verl Info-----------
Version      : 0.10.0.dev
Directory    : /workspace/_upstream/verl/verl
Commit Hash  : 2eaaa8f42b22e478c1f4d7e49d2694b78f176b67
----------Platform Info----------
Platform     : Linux-6.8.0-85-generic-x86_64-with-glibc2.39
system       : Linux
node         : 4f2688afd23b
release      : 6.8.0-85-generic
version      : #85~22.04.1-Ubuntu SMP PREEMPT_DYNAMIC Fri Sep 19 16:18:59 UTC 2
----------Environment----------
CUDA Runtime : 12.8
CUDA Compiler : Cuda compilation tools, release 12.8, V12.8.93
----------System Info----------
CPU Memory	: 503.42 GB
GPU Count	: 4
GPU 1	Type    : NVIDIA GeForce RTX 5090
GPU 1	Memory  : 31.84 GB
GPU 2	Type    : NVIDIA GeForce RTX 5090
GPU 2	Memory  : 31.84 GB
GPU 3	Type    : NVIDIA GeForce RTX 5090
GPU 3	Memory  : 31.84 GB
GPU 4	Type    : NVIDIA GeForce RTX 5090
GPU 4	Memory  : 31.84 GB
```

Originally observed on verl **0.8.0** in production (3 trainer + 1 rollout GPU, `fully_async`).
The two functions involved (`create_device_mesh`, `get_sharding_strategy`) are byte-identical
between 0.8.0 and the `main` checkout above, so the report applies to both.

---

## 字段 2 · Information（复选框，非必需）

勾选：**☑ My own modified scripts**
（不勾 official example scripts —— 我们跑的是自己的启动脚本）

---

## 字段 3 · Tasks（复选框，非必需）

勾选：**☑ My own task or dataset (give details below)**

---

## 字段 4 ✳ Reproduction

> 整段复制，从下一行开始到「字段 5」之前

## What happens

With the FSDP1 backend (`strategy: fsdp`), setting `actor.fsdp_config.fsdp_size=1` on a
multi-GPU trainer — the natural way to say "data parallel, no sharding" (small models, LoRA) —
**silently disables gradient synchronization across ranks**. Every rank trains its own
independent copy of the model. Training runs to completion, loss goes down, `grad_norm` /
entropy / KL all look normal, nothing errors. The checkpointed rank-0 copy has effectively
seen `1/world_size` of the data. In our case this ran undetected for two months of RL
experiments (3 trainer ranks: every update used 16 of the 48 sampled sequences).

## Config that triggers it

```yaml
actor_rollout_ref.actor.strategy: fsdp            # FSDP1 backend
actor_rollout_ref.actor.fsdp_config.fsdp_size: 1  # "do not shard"
trainer.n_gpus_per_node: 3                        # any world_size > 1
```

## Root cause

`create_device_mesh` expresses `fsdp_size=1` as a 2-D mesh, and the strategy is chosen from
the mesh's *rank count*, not its *shape*:

```python
# verl/workers/engine/fsdp/utils.py
mesh_shape=(world_size // fsdp_size, fsdp_size)   # fsdp_size=1  =>  (N, 1)
# get_sharding_strategy: ndim == 2  =>  HYBRID_SHARD
```

PyTorch FSDP1 then does the rest (line refs at torch 2.9.0; `main` is identical):

```
_init_utils.py:152-153   HYBRID_SHARD branch: _inter_node_pg = mesh.get_group(0)  # replicate dim, N ranks
                                              process_group  = mesh.get_group(1)  # shard dim, 1 rank
_init_utils.py:127       state.world_size = state.process_group.size()   ->  1
_init_core_state         world_size == 1  ->  UserWarning + clamp to NO_SHARD
_runtime_utils.py:936    dist.all_reduce(flat_param.grad, group=state.process_group)
                         -> all-reduce over a size-1 group -> no-op
                         (_inter_node_pg is only used on the hybrid branch, which is no longer
                          taken after the clamp -- the N-rank replicate group is created, then orphaned)
```

The only signal is:

```
UserWarning: FSDP is switching to use `NO_SHARD` instead of
ShardingStrategy.HYBRID_SHARD since the world size is 1.
```

It says "I changed strategy", not "your gradients are no longer synchronized". Note this exact
warning also fires for a *benign* and much more common reason — a misconfigured cluster where
world size really is 1 (e.g. #2478, answered with "check `ray status`") — so users who see it
have no reason to suspect silent gradient divergence.

## Minimal reproduction (pure PyTorch, no verl, seeded)

```python
import os, time, torch, torch.nn as nn, torch.distributed as dist, torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy, fully_shard
from torch.distributed.tensor import DTensor
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
        # FSDP2 exposes DTensor grads; read the LOCAL shard -- a global .norm()
        # would reduce across ranks and mask the very difference we test for.
        g = g.to_local() if isinstance(g, DTensor) else g
        buf = [None] * world; dist.all_gather_object(buf, g.detach().norm().item())
        if rank == 0:
            same = max(buf) - min(buf) < 1e-9 * max(buf)
            print(f"{tag:<46} {[f'{b:.8f}' for b in buf]}  {'OK' if same else 'NOT SYNCED'}")

    mesh = init_device_mesh("cuda", (world, 1), mesh_dim_names=["ddp", "fsdp"])
    run("(N,1) + HYBRID_SHARD (= fsdp_size=1 today)",
        lambda m: FSDP(m, device_mesh=mesh, sharding_strategy=ShardingStrategy.HYBRID_SHARD,
                       use_orig_params=True, sync_module_states=True, device_id=rank))
    mesh2 = init_device_mesh("cuda", (world, 1), mesh_dim_names=["ddp2", "fsdp2"])
    run("(N,1) + NO_SHARD (= proposed fix, same mesh)",
        lambda m: FSDP(m, device_mesh=mesh2, sharding_strategy=ShardingStrategy.NO_SHARD,
                       use_orig_params=True, sync_module_states=True, device_id=rank))
    mesh3 = init_device_mesh("cuda", (world, 1), mesh_dim_names=["ddp3", "fsdp3"])
    def as_fsdp2(m):
        fully_shard(m, mesh=mesh3); return m
    run("(N,1) + FSDP2 fully_shard (same mesh)", as_fsdp2)
    run("plain DDP (control)", lambda m: DDP(m, device_ids=[rank]))
    dist.destroy_process_group()

if __name__ == "__main__":
    mp.spawn(worker, args=(min(torch.cuda.device_count(), 3),),
             nprocs=min(torch.cuda.device_count(), 3), join=True)
```

Output on 3x RTX 5090, torch 2.9.0+cu128 (seeded, reproduces bit-for-bit):

```
(N,1) + HYBRID_SHARD (= fsdp_size=1 today)     ['0.18393682', '0.73574728', '1.65543127']  NOT SYNCED
(N,1) + NO_SHARD (= proposed fix, same mesh)   ['2.57511520', '2.57511520', '2.57511520']  OK
(N,1) + FSDP2 fully_shard (same mesh)          ['2.57511520', '2.57511520', '2.57511520']  OK
plain DDP (control)                            ['2.57511520', '2.57511520', '2.57511520']  OK
```

Three details worth noting:

- The broken variant's numbers are exactly `[g, 4g, 9g]`: each rank holds the gradient of *its
  own data only* (inputs scaled by `rank+1`, loss quadratic in the input), pre-divided by
  `world_size` by FSDP in anticipation of an all-reduce that never happens. Undoing that,
  `0.18393682 x 14 = 2.5751155` — the value every synchronized variant reports.
- **FSDP2 on the same mesh is correct**, so this is specific to FSDP1's clamp path, not to the
  mesh shape or the way it is built.
- Internal state of the broken variant, read out of the FSDP instance: `state.world_size == 1`,
  reduction group size 1, and an **orphaned replicate group of size 3**, created but never used.

## Evidence from real training (verl 0.8.0, Qwen3-4B + LoRA r32, 3 trainer ranks)

LoRA `B` matrices are zero-initialized, so at step 1 all ranks hold *bit-identical weights* —
yet their gradients already differ (probe placed before `optimizer_step`):

```
step=1  rank=0  lora_B  weight_norm=0.000000   grad_norm=2.209380e-05
step=1  rank=1  lora_B  weight_norm=0.000000   grad_norm=2.565470e-05   <- 16% apart
step=4  rank=0  lora_B  grad=2.906737e-04
step=4  rank=2  lora_B  grad=8.142963e-05                               <- 3.6x apart
```

Identical starting point + different gradients => the only possible cause is a missing
all-reduce. After 15 updates the cross-rank relative difference of the trained LoRA converges
to ~sqrt(2) (1.4136 / 1.4132 / 1.4139 across four independent runs) — the distance between
equal-length random vectors, i.e. the three ranks learned *statistically unrelated* things.
Adam `exp_avg_sq` differs by 99% across ranks.

---

## 字段 5 ✳ Expected behavior

> 整段复制

`fsdp_size=1` means "replicate the model, do not shard it", so gradients should be all-reduced
across the `world_size` data-parallel ranks — the same semantics as DDP. Instead they are
reduced over a size-1 group, which is a no-op, and no error or dedicated warning is raised.

Concretely, the expectation is either of:

1. **gradients get synchronized** (our preference), or
2. **verl fails loudly** if this configuration is not meant to be supported.

Silently training `world_size` divergent copies of the model is the worst of the three, because
it is invisible to every training metric.

### Why this has to be fixed in verl rather than upstream

PyTorch already acknowledged the FSDP1 behavior as a bug and declined to fix it:
[pytorch/pytorch#154888](https://github.com/pytorch/pytorch/issues/154888) — maintainer:
*"this is a bug ... we might be slow in fixing fsdp1"* — closed as `not_planned` (FSDP1 is in
maintenance mode). The framework that builds the degenerate mesh is the only place left to
stop it.

### Proposed fix

Select `NO_SHARD` when the shard dim is degenerate. The mesh stays exactly as it is; FSDP's
non-hybrid path then uses `mesh_dim=0` — the replicate dim — as the reduction group
(`_init_utils.py:119`):

```python
# verl/workers/engine/fsdp/utils.py :: get_sharding_strategy
elif device_mesh.ndim == 2:
    if device_mesh.size(1) == 1:
        sharding_strategy = ShardingStrategy.NO_SHARD
    else:
        sharding_strategy = hsdp_strategy
```

Verified: gradients become bit-identical across ranks (output above), and a real verl training
run with only this diff applied produces three rank checkpoints that are bit-identical across
all 504/504 trainable tensors, optimizer state included.

Checkpoint format is unchanged. Probing `state_dict()` under `SHARDED_STATE_DICT` on both
configurations returns the same thing (`n_entries=1`, `value_types=['Tensor']`, identical
shapes), because the current config is already clamped to `NO_SHARD` internally and `NO_SHARD`
short-circuits sharded state dicts to full tensors. Resume compatibility is therefore
unaffected.

I have a PR ready with this fix plus a 2-rank regression test under
`tests/special_distributed/` (registered in `run_all.sh`); it fails before the change
(`[0.313, 1.254]`) and passes after (bitwise identical). Will open it right after this issue.

**Workarounds** until then: use `strategy: fsdp2` (verified unaffected), or set `fsdp_size` to
the full world size (accepting the sharding cost).
