标题（复制这一行）：
[BUG][FSDP1] fsdp_size=1 on multiple GPUs silently disables gradient synchronization

--- 正文从下一行开始，全选复制 ---

## Summary

With the FSDP1 backend (`strategy: fsdp`), setting `actor.fsdp_config.fsdp_size=1` on a multi-GPU trainer — the natural way to say "data parallel, no sharding" (small models, LoRA) — **silently disables gradient synchronization across ranks**. Every rank trains its own independent copy of the model. Training runs to completion, loss goes down, `grad_norm` / entropy / KL all look normal, nothing fails. The checkpointed rank-0 copy has effectively seen `1/world_size` of the data.

In our case this ran undetected for two months of RL experiments (3 trainer ranks: every update used 16 of the 48 sampled sequences).

## Root cause

`create_device_mesh` expresses `fsdp_size=1` as a 2-D mesh, and the strategy is chosen from the mesh's *rank count*, not its *shape*:

```python
# verl/workers/engine/fsdp/utils.py
mesh_shape=(world_size // fsdp_size, fsdp_size)   # fsdp_size=1  =>  (N, 1)
# get_sharding_strategy: ndim == 2  =>  HYBRID_SHARD
```

PyTorch FSDP1 then does the rest (line refs at torch 2.9.0; `main` is identical):

```
_init_utils.py:152-153   HYBRID_SHARD branch: _inter_node_pg  = mesh.get_group(0)  # replicate dim, N ranks
                                              process_group   = mesh.get_group(1)  # shard dim, 1 rank
_init_utils.py:127       state.world_size = state.process_group.size()   ->  1
_init_core_state         world_size == 1  ->  UserWarning + clamp to NO_SHARD
_runtime_utils.py:936    dist.all_reduce(flat_param.grad, group=state.process_group)
                         -> all-reduce over a size-1 group  ->  no-op
                         (_inter_node_pg is only used on the hybrid branch, which is no longer
                          taken after the clamp -- the N-rank replicate group is created, then orphaned)
```

The only signal is:

```
UserWarning: FSDP is switching to use `NO_SHARD` instead of
ShardingStrategy.HYBRID_SHARD since the world size is 1.
```

It says "I changed strategy", not "your gradients are no longer synchronized". Note this exact warning also fires for a *benign* and much more common reason — a misconfigured cluster where world size really is 1 (e.g. #2478, where the answer was "check `ray status`") — so users who see it have no reason to suspect silent gradient divergence.

## Why this needs to be fixed in verl

PyTorch already confirmed the underlying behavior as a bug and declined to fix it: [pytorch/pytorch#154888](https://github.com/pytorch/pytorch/issues/154888) — maintainer: *"this is a bug ... we might be slow in fixing fsdp1"* — closed as `not_planned` (FSDP1 is in maintenance mode). So the framework that builds the degenerate mesh is the only place left to stop it.

## Minimal reproduction (pure PyTorch, no verl, deterministic)

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

Output on 3x RTX 5090, torch 2.9.0+cu128 (seeded, so these values reproduce bit-for-bit):

```
(N,1) + HYBRID_SHARD (= fsdp_size=1 today)     ['0.18393682', '0.73574728', '1.65543127']  NOT SYNCED
(N,1) + NO_SHARD (= proposed fix, same mesh)   ['2.57511520', '2.57511520', '2.57511520']  OK
(N,1) + FSDP2 fully_shard (same mesh)          ['2.57511520', '2.57511520', '2.57511520']  OK
plain DDP (control)                            ['2.57511520', '2.57511520', '2.57511520']  OK
```

Three details worth noting:

- The broken variant's numbers are exactly `[g, 4g, 9g]`: each rank holds the gradient of *its own data only* (inputs are scaled by `rank+1`, and the loss is quadratic in the input), pre-divided by `world_size` by FSDP in anticipation of an all-reduce that never happens. Undoing that, `0.18393682 x 14 = 2.5751155` — the value every synchronized variant reports.
- **FSDP2 on the same mesh is correct**, so this is specific to FSDP1's clamp path, not to the mesh shape or the way it is built.
- Internal state of the broken variant, read out of the FSDP instance: `state.world_size == 1`, reduction group size 1, and an **orphaned replicate group of size 3** that is created but never used.

## Evidence from real training (verl 0.8.0, Qwen3-4B + LoRA r32, 3 trainer ranks)

LoRA `B` matrices are zero-initialized, so at step 1 all ranks have *bit-identical weights* — yet their gradients already differ (probe hooked before `optimizer_step`):

```
step=1  rank=0  lora_B  weight_norm=0.000000   grad_norm=2.209380e-05
step=1  rank=1  lora_B  weight_norm=0.000000   grad_norm=2.565470e-05   <- 16% apart
step=4  rank=0  lora_B  grad=2.906737e-04
step=4  rank=2  lora_B  grad=8.142963e-05                               <- 3.6x apart
```

Identical starting point + different gradients => the only possible cause is a missing all-reduce. After 15 updates, the cross-rank relative difference of the trained LoRA converges to ~sqrt(2) (1.4136 / 1.4132 / 1.4139 across four independent runs) — the distance between equal-length random vectors, i.e. the three ranks learned *statistically unrelated* things. Adam `exp_avg_sq` differs by 99% across ranks.

## Proposed fix

Choose `NO_SHARD` when the shard dim is degenerate — the mesh itself stays exactly as it is; FSDP's non-hybrid path then uses `mesh_dim=0` (the replicate dim, N ranks) as the reduction group (`_init_utils.py:119`):

```python
# verl/workers/engine/fsdp/utils.py :: get_sharding_strategy
elif device_mesh.ndim == 2:
    if device_mesh.size(1) == 1:
        sharding_strategy = ShardingStrategy.NO_SHARD
    else:
        sharding_strategy = hsdp_strategy
```

Verified: gradients become bit-identical across ranks (output above), and a real verl training run with only this diff applied produces three rank checkpoints that are bit-identical across all 504/504 trainable tensors, optimizer state included.

Checkpoint format is unchanged. Probing `state_dict()` under `SHARDED_STATE_DICT` on both configurations returns the same thing — `n_entries=1`, `value_types=['Tensor']`, identical shapes — because the current config is already clamped to `NO_SHARD` internally, and `NO_SHARD` short-circuits sharded state dicts to full tensors. Resume compatibility is therefore unaffected. PR follows.

Also worth considering as a follow-up: a one-time post-first-step assertion that gradients actually match across data-parallel ranks. This failure class is invisible to every training metric; a single `all_gather` of one gradient norm at step 1 would have caught this — and any future variant — immediately.

**Workarounds for affected users** until then: use `strategy: fsdp2` (verified unaffected), or pin `fsdp_size` to the full world size (accepting sharding cost).

## Environment

- verl 0.8.0 (`create_device_mesh` / `get_sharding_strategy` unchanged on current `main`)
- torch 2.9.0+cu128, NCCL 2.27.5
- 3-4 x RTX 5090 (also reproduced by others on 2 GPUs: pytorch#154888, [PyTorch forum #220486](https://discuss.pytorch.org/t/potential-bug-with-hybrid-shard-and-n-1-device-mesh-falling-back-to-no-shard/220486))
