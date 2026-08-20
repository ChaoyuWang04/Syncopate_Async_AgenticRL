# 提交件 · issue + PR（已提交，**已被 wontfix 关闭**，留档）

> issue [#7493](https://github.com/verl-project/verl/issues/7493) · PR [#7494](https://github.com/verl-project/verl/pull/7494)
> 结果与理由见 [`2-case.md`](2-case.md) 顶部。以下为当时提交的正文，原样留档。

---

# ① ISSUE

> ⚠️ 他们的 Bug report 是**结构化模板**，不能整篇粘贴。下面按字段分好，逐个复制。
>
> ## 标题（复制这一行）
>
> ```
> [BUG][FSDP1] fsdp_size=1 on multiple GPUs silently disables gradient synchronization
> ```
>
> `[BUG]` + `[FSDP1]` 两个标记都有先例（`[Bug][SFT][FSDP]` / `[BUG] AgentLoopOutput...`）；
> ★ `[FSDP1]` 尤其值得留 —— **FSDP2 在同一个 mesh 上是对的**，点明版本能提前拦掉
> 「是不是你配置写错了」这个最常见的误判。`fsdp_size` 必须小写（它是字面 config key）。
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


---

# ② PULL REQUEST

[fsdp] fix: fsdp_size=1 silently disables gradient synchronization

--- 正文从下一行开始，全选复制（issue 编号 #7493 已填好，无需再改） ---

### What does this PR do?

Fixes #7493.

`create_device_mesh(world_size, fsdp_size=1)` builds a `(world_size, 1)` mesh whose shard dim is degenerate. `get_sharding_strategy` selected `HYBRID_SHARD` for it, which FSDP1 clamps to `NO_SHARD` (the shard group holds a single rank) **while still reducing gradients over that size-1 shard group**. The replicate-dim ranks therefore never synchronize: every rank trains its own copy of the model, with no error and plausible metrics. PyTorch acknowledged this FSDP1 behavior as a bug and closed it as `not_planned` (FSDP1 is in maintenance mode): [pytorch/pytorch#154888](https://github.com/pytorch/pytorch/issues/154888).

This PR selects `NO_SHARD` explicitly when the shard dim is degenerate. The mesh is untouched; FSDP's non-hybrid path then reduces gradients over `mesh_dim=0` — the replicate dim, which is exactly the intended "data parallel, no sharding" semantics.

### Checklist Before Starting

- [x] Search for similar PRs. Paste at least one query link here: [`is:pr fsdp_size`](https://github.com/verl-project/verl/pulls?q=is%3Apr+fsdp_size) (34 hits, none about degenerate-mesh gradient sync), [`is:pr HYBRID_SHARD`](https://github.com/verl-project/verl/pulls?q=is%3Apr+HYBRID_SHARD) (0 hits). Closest existing issue is #2478 — same FSDP warning, different and benign cause (world size genuinely 1).
- [x] Format the PR title as `[{modules}] {type}: {description}`

### Test

Adds `tests/special_distributed/test_fsdp_degenerate_mesh_grad_sync.py` (2 ranks, registered in `tests/special_distributed/run_all.sh`, which `model.yml` already runs). It builds the mesh through `create_device_mesh(world_size, fsdp_size=1)`, feeds each rank different data, and asserts post-backward gradients are identical across ranks; it also pins the selected strategy.

```
before this PR:  AssertionError: gradients are not synchronized across ranks:
                 [0.3134024441242218, 1.2536097764968872] (sharding_strategy=ShardingStrategy.HYBRID_SHARD)
after this PR:   [fsdp_size=1] gradient norms across 2 ranks:
                 [1.5670123100280762, 1.5670123100280762] -- synchronized
```

Additionally validated outside CI:

- Seeded 3-GPU script (pure PyTorch, in the issue): the broken config yields per-rank gradients `['0.18393682', '0.73574728', '1.65543127']` (each rank's own data only, scaled by its input); with this fix all ranks report `2.57511520`, matching both a plain-DDP control and FSDP2 `fully_shard` on the same `(3,1)` mesh bit-for-bit — i.e. this is specific to FSDP1's clamp path, not to the mesh.
- Real RL training (Qwen3-4B + LoRA r32, 3 trainer ranks, fully_async) with only this diff applied: the three saved rank checkpoints are bit-identical for all 504/504 trainable tensors, optimizer state included. Without the fix, cross-rank relative difference converges to ~sqrt(2) (statistically unrelated).

### API and Usage Example

No API, config or checkpoint-format change. `fsdp_size=1` keeps its meaning ("do not shard") and now actually synchronizes gradients:

```bash
# unchanged usage; previously a silent 1/world_size data loss per update
actor_rollout_ref.actor.fsdp_config.fsdp_size=1 trainer.n_gpus_per_node=3
```

Checkpoint format is unchanged: probing `state_dict()` under `SHARDED_STATE_DICT` before and after this PR returns the same thing (`n_entries=1`, `value_types=['Tensor']`, identical shapes), because the current config is already clamped to `NO_SHARD` internally and `NO_SHARD` short-circuits sharded state dicts to full tensors. Resume compatibility is unaffected.

### Design & Code Changes

- `verl/workers/engine/fsdp/utils.py::get_sharding_strategy`: return `NO_SHARD` when `device_mesh.size(1) == 1`; all other cases unchanged.
  - Single call site (`workers/engine/fsdp/transformer_impl.py`), no signature change.
  - Mesh shape and `mesh_dim_names` unchanged, so `model_merger`'s `assert mesh_dim_names in (("fsdp",), ("ddp", "fsdp"))` and existing checkpoints are unaffected.
  - Non-degenerate configs (`fsdp_size>1`, 1-D meshes) take the exact same path as before. The only behavioral change is the gradient reduction group: size-1 shard group -> N-rank replicate group.
- `tests/special_distributed/test_fsdp_degenerate_mesh_grad_sync.py`: new regression test.
- `tests/special_distributed/run_all.sh`: register it.

Note: FSDP1's `NO_SHARD` emits a deprecation `FutureWarning` pointing to DDP; within FSDP1 it is nevertheless the only correct strategy for this topology. The `fsdp2` backend is unaffected (verified: same `(N,1)` mesh under `fully_shard` synchronizes correctly).

### Checklist Before Submitting

- [x] Read the Contribute Guide.
- [x] Apply pre-commit checks — `pre-commit run` on the changed files: all 14 hooks pass (ruff, ruff-format, mypy, autogen-trainer-cfg, license, docstrings, naming, test-structure, device-API, DataProto, compile-all).
- [ ] Add / Update the documentation — not needed: `fsdp_size` appears in `docs/examples/config.rst` only as a config listing (`fsdp_size: -1`) with no prose contradicting this fix, and `docs/advance/ppo_lora.rst` uses `fsdp_size=8`, which is unaffected. This PR makes the runtime behavior match the documented meaning of `fsdp_size=1`.
- [x] Add unit or end-to-end test(s) to the CI workflow.
- [ ] Once your PR is ready for CI, send a message in the `ci-request` channel.
- [x] Not related to the `recipe` submodule.


---

# ③ CI 申请（飞书群）

> ⚠️ Slack 的 `ci-request` 频道**进不去**：它限定邮箱域名（anyscale.com / bytedance.com / together.ai）。
> ⇒ 走飞书群：https://applink.larkoffice.com/client/chat/chatter/add_by_link?link_token=772jd4f1-cd91-441e-a820-498c6614126a
> 格式照抄群里既有的 CI 申请（一段说明 + 链接 + 一句请求）。

## 正式版（推荐，直接复制）

大家好，我提交了一个 FSDP1 梯度同步相关的修复：多卡下设置 `fsdp_size=1`（本意是"不分片、纯数据并行"）会构造出 `(world_size, 1)` 的退化 device mesh 并选中 `HYBRID_SHARD`，FSDP1 随后把它钳成 `NO_SHARD`，但梯度归约仍留在那个只有 1 个 rank 的分片组上——结果是各 rank 的梯度从不同步，而且不报错、loss 会降、指标全正常。现在改为在分片维退化时显式选 `NO_SHARD`，让 FSDP 在复制维（`mesh_dim=0`）上归约；同时补了一个 2 卡回归测试（`tests/special_distributed/`，已注册进 `run_all.sh`，修复前会红）。

Issue： https://github.com/verl-project/verl/issues/7493
PR： https://github.com/verl-project/verl/pull/7494

麻烦帮忙触发一下 CI，谢谢！

---

## 备用：更短的一版（如果群里习惯短消息）

大家好，提交了一个 FSDP1 的修复：多卡 `fsdp_size=1` 会造出退化的 `(N,1)` mesh，FSDP1 钳成 `NO_SHARD` 后仍在 size-1 的分片组上归约梯度 ⇒ **各 rank 梯度静默不同步**（不报错、指标正常）。改为退化时显式选 `NO_SHARD`，并补了 2 卡回归测试。
PR： https://github.com/verl-project/verl/pull/7494 （Issue #7493）
麻烦帮忙触发一下 CI，谢谢！

---

## 备用：英文版（若群里以英文沟通）

Hi all, I submitted a fix for silent gradient desync in FSDP1: with `fsdp_size=1` on multiple GPUs, verl builds a degenerate `(world_size, 1)` device mesh and selects `HYBRID_SHARD`; FSDP1 clamps that to `NO_SHARD` but keeps reducing gradients over the size-1 shard group, so ranks never synchronize — with no error and normal-looking metrics. The fix selects `NO_SHARD` explicitly for a degenerate shard dim so FSDP reduces over the replicate dim (`mesh_dim=0`), plus a 2-rank regression test under `tests/special_distributed/` (registered in `run_all.sh`, fails before the fix).

Issue: https://github.com/verl-project/verl/issues/7493
PR: https://github.com/verl-project/verl/pull/7494

Could someone help trigger CI? Thanks!

---

## 发完之后

1. 回 PR 页面把最后一条 checklist 勾上：
   `- [ ] Once your PR is ready for CI, send a message in the ci-request channel.`
2. ⚠️ **先签 CLA 再发**（或同时）：https://cla-assistant.io/verl-project/verl?pullRequest=7494
   —— CLA 未签时维护者一般不会 review，CI 也可能不给触发。
3. 相关 CI：本 PR 的测试需要**多卡**，落在 `model.yml`（它跑 `tests/special_distributed/run_all.sh`），
   触发路径 `verl/**/*.py` 与 `tests/special_distributed/run_all.sh` 我们都改到了，会自动匹配。
