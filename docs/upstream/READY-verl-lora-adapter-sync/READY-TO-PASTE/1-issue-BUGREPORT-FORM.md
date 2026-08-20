# 包② · GitHub Bug Report 表单逐字段填写指南（2026-08-20）

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
