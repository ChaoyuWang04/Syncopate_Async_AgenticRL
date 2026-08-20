# 提交件 · issue + PR（可直接粘贴）

> 顺序：① 提 issue（结构化表单，逐字段复制）→ ② 开 PR（`<ISSUE>` 换成编号）→ ③ 签 CLA → ④ 飞书群申请 CI
> 开 PR 直达：`https://github.com/verl-project/verl/compare/main...ChaoyuWang04:verl:fix/disaggregated-lora-adapter-sync?expand=1`

---

# ① ISSUE

> ⚠️ Bug report 是**结构化表单**，不能整篇粘贴。下面按字段分好，逐个复制。
>
> ## 标题（复制这一行）
>
> ```
> [BUG][FSDP][LoRA] Disaggregated weight sync never delivers the LoRA adapter — rollout keeps sampling from the initial policy
> ```
>
> `[BUG]` + 组件标记有仓库先例（`[Bug][SFT][FSDP]`）。点明 `LoRA` 与 `FSDP`
> 能让不用 LoRA 的人不必点进来，也提前说明适用范围。

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
Commit Hash  : 9326156b5ec1b8989fa55b508643703b8f2054bc
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

Originally observed on verl **0.8.0** in production (3 trainer + 1 rollout GPU, `fully_async`,
NCCL checkpoint engine, Qwen3-4B + LoRA r32). The two code paths involved are byte-identical
between 0.8.0 and the `main` checkout above.

---

## 字段 2 · Information（复选框，非必需）

勾选：**☑ My own modified scripts**

## 字段 3 · Tasks（复选框，非必需）

勾选：**☑ My own task or dataset (give details below)**

---

## 字段 4 ✳ Reproduction

> 整段复制，到「字段 5」之前

## What happens

**This is the default path for async LoRA RL — no unusual flags are involved:**
`model.lora.merge=False` is the default, and every disaggregated recipe sets a non-`naive`
checkpoint-engine backend. Anyone running LoRA with `fully_async` / `one_step_off` is affected.

With LoRA (`model.lora.merge=False`, the default) and a disaggregated trainer/rollout
deployment (`rollout.checkpoint_engine.backend != "naive"`, i.e. the async RL recipes),
**every weight sync pushes the unmodified frozen base model. The LoRA adapter is never
transferred.** The rollout engine samples from the initial policy for the entire run.

Training completes normally — loss falls, rewards move, `grad_norm` / entropy / ESS all look
plausible, no warning is emitted. The RL loop is silently severed: the policy gradient is
computed for the current policy while all data forever comes from the initial one, and the
mismatch grows monotonically with training.

The colocated path (`backend: naive`) is correct. Same config, different mode.

## Config that triggers it

```yaml
actor_rollout_ref.model.lora_rank: 32                        # LoRA
actor_rollout_ref.model.lora.merge: false                    # default
actor_rollout_ref.rollout.checkpoint_engine.backend: nccl    # any non-"naive" backend
# i.e. any async recipe where trainer and rollout sit on different GPUs
```

## Root cause

The disaggregated branch runs only half of the two-phase LoRA sync protocol, and discards the
signal that would tell the receiving side an adapter exists:

```python
# verl/workers/engine_workers.py
if effective_mode != "naive":
    per_tensor_param, _ = self.actor.engine.get_per_tensor_param()   # (1) no args, peft_config dropped
    metrics = await self.checkpoint_engine.send_weights(per_tensor_param, global_steps=global_steps)
    return metrics or {}                                            # (2) no second call, ever
```

1. `get_per_tensor_param()` defaults to `base_sync_done=False` — the "push the base first" half.
2. With LoRA + `merge=False` that lands in `collect_lora_params(base_sync_done=False)`
   (`verl/utils/fsdp_utils.py`), which **explicitly skips every `lora_` tensor** and returns the
   frozen base, renamed by `replace_lora_wrapper` to `...base_layer.weight` form — precisely so
   that a LoRA-enabled vLLM accepts it. That rename is why the failure is silent: the push
   *succeeds*.
3. The protocol expects a follow-up call with `base_sync_done=True` (push the adapter). The
   colocated branch does exactly that two-call dance. The disaggregated branch returns after the
   first half — every time.
4. Downstream, `CheckpointEngineWorker.update_weights(self, global_steps=None)`
   (`verl/checkpoint_engine/base.py`) has no parameter to receive `peft_config` anyway, while
   everything *below* it already supports adapters:
   `server_adapter.update_weights(..., peft_config=...)` → `update_weights_from_ipc` →
   `TensorLoRARequest(lora_tensors=...)` + `add_lora` — the exact machinery the colocated path
   exercises on every sync.

The code itself flags this as known-incomplete: the flag initialization carries the comment
*"base_sync_done is unused in merge-only mode but **kept for Phase 2 adapter path**"*
(`engine_workers.py`). This report is that Phase 2, with a fix.

## Evidence

**Offline, no Ray** (2-layer Qwen3 + LoRA r=32, FSDP-wrapped, calling verl's own
`collect_lora_params`):

| branch | tensors | bytes | of which `lora_` |
|---|---|---|---|
| `base_sync_done=False` (what disaggregated always runs) | 25 | 74.8 MiB | **0** |
| `base_sync_done=True` (what colocated also runs) | 28 | 0.2 MiB | **28** |

**Real training** (verl 0.8.0, Qwen3-4B + LoRA r32, 3 trainer + 1 rollout GPU, `fully_async`;
probe at `NCCLCheckpointEngine.send_weights`, streaming, no behavior change). Four independent
runs, two syncs each:

```
every sync:  399 tensors / 8,414.1 MiB / 0 of them lora_
watched layer  model.layers.0.self_attn.q_proj.base_layer.weight
  pushed ||W||           = 75.377708   (identical across all runs and syncs)
  on-disk initial model  = 75.377708   (bit-identical)
```

A policy being trained cannot push bit-identical weights on every sync — the payload is the
frozen initial model. Meanwhile the vLLM server runs with `--enable_lora` and an initialized
`PunicaWrapperGPU`: **`list_loras()` stays `[]` for the whole run.** The adapter slot exists
and is never filled.

**The failure masquerades as staleness**, which is what makes it expensive in practice — all of
these "look like" async-RL staleness while actually measuring a policy pinned at the initial
checkpoint (we spent two months on that wrong trail):

| observation | looks like | actually is |
|---|---|---|
| `rollout_corr/kl` climbs monotonically (36x) at fixed `sync_every` | staleness accumulating | distance to the initial policy = cumulative displacement |
| ESS collapses along `lr x steps`, insensitive to `sync_every` | odd | ESS is a function of displacement from the initial policy |
| staleness-queue knobs (6x more stale trajectories) don't move ESS | insensitive knob | the knob is connected to nothing |
| sync duration independent of expected payload (8 GB vs 132 MB) | fixed overheads | it is always 8 GB |

## Why `model.lora.merge=True` is not a good workaround

Mechanically it works (the pushed weights then change with training). But the merge is performed
in bf16, and an RL-scale LoRA delta (||dW||/||W|| ~ 0.05%) falls into bf16 rounding: on a fixed
probe set, merging shifts per-token logprobs by a **median of 1.7e-2 — about 50% of the
adapter's own effect, and ~50x the vLLM/FSDP numerical floor**. It also keeps paying the
full-model transfer cost (8.4 GB per sync) for a 252 MiB adapter.

---

## 字段 5 ✳ Expected behavior

> 整段复制

A weight sync should deliver **the policy the trainer currently holds**. With LoRA and
`merge=False` that means the adapter has to reach the rollout engine; today the rollout engine
receives the frozen base on every sync, so the sampling policy never changes.

Concretely, the expectation is either of:

1. **the adapter gets synced** (our preference), or
2. **verl fails loudly** if this combination is not meant to be supported.

Silently pushing a base-only payload is the worst of the three: it severs the RL loop while
every metric keeps looking healthy, and the resulting symptoms are indistinguishable from
ordinary staleness.

Related history: #2048 (closed) recorded that async + LoRA used to raise an explicit error;
#3654's fix brought the combination into the support surface — where it now fails silently
instead. A silent wrong-weights sync is strictly worse than the old error: the error cost one
launch, this costs entire training runs. #7287 / #3907 fixed adapter-sync defects on the
*colocated* SGLang/vLLM paths; the disaggregated checkpoint-engine path is the remaining gap.

### Proposed fix (PR to follow)

Two small, wire-compatible changes; no protocol or backend changes:

1. **Trainer side** (`engine_workers.py`): run the same two-phase protocol as the colocated
   branch — initialize `base_sync_done` from `"dummy" not in rollout.load_format` (with real
   weights loaded the rollout engine never needs a base push at all), collect with it, and flip
   it once the engine returns a `peft_config`.
2. **Rollout side** (`checkpoint_engine/base.py`): adapter pushes are self-describing — they
   consist *only* of `lora_` tensors, while base / full / merged pushes contain none. Peek at
   the first tensor name (the stream is re-chained, nothing is consumed twice); if it is an
   adapter push, rebuild the minimal `peft_config` dict from the worker's own `model_config`
   (vLLM's `PEFTHelper.from_dict` needs only `r` / `lora_alpha` / `target_modules`) and forward
   `peft_config` + `base_sync_done=True`. This covers every checkpoint-engine backend speaking
   the named-tensors wire format and leaves the delta engines untouched.

Measured with exactly this fix applied to the source tree (no monkeypatching), same setup as
above, 3 syncs:

```
payload            every sync: 252 MiB, 504/504 tensors lora_ -- the 8,414 MiB frozen-base
                   push never occurs at all (load_format is real, so the first sync is
                   already adapter-only, matching colocated semantics)
rollout side       recognised as an adapter push on every sync
rollout_corr/kl    6.4e-05 / 2.9e-04 -- at the numerical floor
                   (broken baseline at the same point in training: 3e-3 and climbing)
```

Longer runs with the functionally equivalent patch (60 steps + an independent 12-step rerun):
`list_loras()` `[]` -> `[<id>]`; kl mean 3.9e-3 -> 8e-4 across 15 sync periods; end-to-end
logprob agreement at the engine floor (~3-6e-4, i.e. scaling / module targeting / tensor
contents all verified, not just transfer); `param_sync` 6.25 s -> 0.97 s (6.4x).

### Also worth considering

A defensive invariant, independent of this fix: after every sync, compare one layer's norm
between trainer and rollout (two scalars). "The thing pushed is not the thing held" — in any
backend — should be a hard failure, not something users discover by diffing checkpoints two
months later.

**Workarounds** until then: use `backend: naive` (colocated, unaffected), or accept the bf16
merge cost with `model.lora.merge=True`.


---

# ② PULL REQUEST

[rollout, fsdp] fix: sync LoRA adapters to disaggregated rollout engines

--- 正文从下一行开始，全选复制；⚠️ 提交前把 <ISSUE> 换成 issue 编号 ---

### What does this PR do?

Fixes #<ISSUE>.

This is the default path for async LoRA RL: `model.lora.merge=False` is the default and every
disaggregated recipe uses a non-`naive` checkpoint-engine backend. In that configuration, every
weight sync pushed the unmodified frozen base model and never transferred the adapter — the rollout engine kept sampling from the initial
policy for the whole run, silently (evidence and impact in the issue).

The disaggregated branch called `get_per_tensor_param()` with its defaults, pinning
`base_sync_done=False`. For a LoRA model that lands in `collect_lora_params(base_sync_done=False)`,
which skips every `lora_` tensor and returns the frozen base renamed to `...base_layer.weight` — a
payload a LoRA-enabled engine happily accepts, which is why nothing ever failed. The colocated
branch already runs both phases (base, then adapters); only the disaggregated one stopped after
the first half. Downstream, `CheckpointEngineWorker.update_weights` had no parameter to pass
`peft_config` on, even though the server adapters below it already support adapter loads.

### Checklist Before Starting

- [x] Search for similar PRs. Paste at least one query link here: [`is:pr base_sync_done`](https://github.com/verl-project/verl/pulls?q=is%3Apr+base_sync_done), [`is:pr collect_lora_params`](https://github.com/verl-project/verl/pulls?q=is%3Apr+collect_lora_params), [`is:pr TensorLoRARequest`](https://github.com/verl-project/verl/pulls?q=is%3Apr+TensorLoRARequest) — recent adapter-sync fixes (#7287, #3907) all target the *colocated* paths; the disaggregated checkpoint-engine path is untouched.
- [x] Format the PR title as `[{modules}] {type}: {description}`

### Test

`tests/checkpoint_engine/test_disaggregated_lora_sync_on_cpu.py` — 7 CPU tests (no GPU, no Ray),
picked up automatically by `cpu_unit_tests.yml` (`tests/**/test_*_on_cpu.py`).

```
before this PR:  3 failed, 4 passed
  test_adapter_push_is_annotated                          assert None is True
                                                          (kwargs = {'wire_format': 'named_tensors'};
                                                           the rollout side is never told it is an adapter)
  test_real_load_format_pushes_adapters_from_the_first_sync   assert [False, False] == [True, True]
  test_dummy_load_format_pushes_base_first                    assert [False, False] == [False, True]
                                                          (the trainer never switches to adapters)
after this PR:   7 passed
```

The four that already pass before the change are the no-regression cases (base pushes,
full fine-tuning, delta wire format, and the flag staying put without a `peft_config`) — they
guard the paths this PR must leave alone. The tests build a real `CheckpointEngineWorker` via
`object.__new__` rather than a hand-wired stub, so the failures above land on behavioral
assertions rather than on a missing helper.

Additionally validated outside CI, with exactly this diff applied to the source tree (verl 0.8.0,
Qwen3-4B + LoRA r32, 3 trainer + 1 rollout GPU, `fully_async`, NCCL backend, 3 syncs):

- payload per sync: 8,414 MiB (frozen base, 0 `lora_` tensors) → **252 MiB, 504/504 `lora_`
  tensors**; the base push never occurs at all, since `load_format` is real
- `rollout_corr/kl` at the numerical floor (6.4e-05 / 2.9e-04) instead of climbing (3e-3 and
  rising at the same point in training) — i.e. the rollout policy now tracks the trainer policy
- longer runs with the functionally equivalent patch (60 steps + an independent 12-step rerun):
  `list_loras()` `[]` → `[<id>]`, end-to-end logprob agreement at the engine floor (~3-6e-4, so
  scaling / module targeting / tensor contents are verified, not just transfer), `param_sync`
  6.25 s → 0.97 s (6.4x)

### API and Usage Example

No API, config or wire-format change. The same async recipe now actually delivers the adapter:

```bash
# unchanged usage; previously the rollout engine kept the initial policy for the whole run
actor_rollout_ref.model.lora_rank=32 \
actor_rollout_ref.model.lora.merge=False \
actor_rollout_ref.rollout.checkpoint_engine.backend=nccl
```

Behavior is keyed off what the engine actually returns, so full-parameter training,
`model.lora.merge=True` and base pushes are byte-for-byte unaffected: their payloads contain no
`lora_` tensor, so they take today's exact path.

### Design & Code Changes

- `verl/workers/engine_workers.py` — the disaggregated branch now runs the same two-phase
  protocol as the colocated branch below it: `base_sync_done` initialized from
  `"dummy" not in rollout.load_format` (with real weights loaded, no base push is needed at all —
  the first sync is already adapter-only), collection called with it, flag flipped once the
  engine returns a `peft_config`.
- `verl/checkpoint_engine/base.py` — `CheckpointEngineWorker.update_weights` peeks at the first
  tensor name (stream re-chained, nothing consumed twice): adapter pushes consist only of `lora_`
  tensors, so they are self-describing. For an adapter push it rebuilds the minimal `peft_config`
  dict from its own `model_config` and forwards `peft_config` + `base_sync_done=True` to the
  server adapter, which already supports them (`update_weights_from_ipc` → `TensorLoRARequest` +
  `add_lora`).
- `tests/checkpoint_engine/test_disaggregated_lora_sync_on_cpu.py` — new CPU regression tests.

Why this shape:

- **No wire change**: nothing new is serialized, so every backend speaking the named-tensors wire
  format is covered (nccl / nixl / mooncake / kimi all go through this worker). The peek is
  guarded on `wire_format == "named_tensors"`; `delta_sharded` owns its own sync state machine
  and is untouched.
- `peft_config` is rebuilt worker-side from `model_config` (both processes are constructed from
  the same config) rather than serialized across the wire; vLLM's `PEFTHelper.from_dict` requires
  only `r` / `lora_alpha` / `target_modules`. Field-level equality with the trainer-side config
  was verified explicitly (r=32 / alpha=64 / scaling=2.0 on both sides).

### Checklist Before Submitting

- [x] Read the Contribute Guide.
- [x] Apply pre-commit checks — `pre-commit run` on the changed files: all 14 hooks pass (ruff, ruff-format, mypy, autogen-trainer-cfg, license, docstrings, naming, test-structure, device-API, DataProto, compile-all).
- [ ] Add / Update the documentation — not needed: this restores the documented behavior of an existing option (`model.lora.merge=False` with a disaggregated backend); no config or API surface changes.
- [x] Add unit or end-to-end test(s) to the CI workflow — CPU tests, auto-collected by `cpu_unit_tests.yml`.
- [ ] Once your PR is ready for CI, send a message in the `ci-request` channel.
- [x] Not related to the `recipe` submodule.


---

# ③ CI 申请（飞书群；Slack 限 anyscale/bytedance/together.ai 域名，进不去）

> 把 `<PR>` 与 `<ISSUE>` 换成实际编号后发群里。

大家好，我提交了一个 disaggregated 模式下 LoRA 权重同步的修复：用 LoRA（`lora.merge=False`，
**默认值**）且 `checkpoint_engine.backend != "naive"`（即各种异步 recipe）时，每次权重同步推给
rollout 的都是**未经修改的冻结基座**，adapter 一个字节都没推过去 —— rollout 全程用起点策略采样，
而训练看起来完全正常（loss 会降、指标正常、无任何告警）。根因是 disaggregated 分支只跑了两段式
协议的前半段（`get_per_tensor_param()` 用默认参数 ⇒ `base_sync_done=False` ⇒ 只收集基座、跳过
所有 `lora_` 张量），colocate 分支是对的。现在让 disaggregated 走同样的两段式协议，并让 rollout
侧从载荷自身判别 adapter 推送（只含 `lora_` 张量）后透传 `peft_config`；补了 7 条 CPU 回归测试
（修复前 3 条红在行为断言上）。

Issue： https://github.com/verl-project/verl/issues/<ISSUE>
PR： https://github.com/verl-project/verl/pull/<PR>

麻烦帮忙触发一下 CI，谢谢！

**发完之后**：回 PR 勾上最后一条 checklist · 签 CLA `https://cla-assistant.io/verl-project/verl?pullRequest=<PR>` ·
把目录改名 `READY-` → `OPEN-` 并更新 `2-case.md` 顶部状态块。
