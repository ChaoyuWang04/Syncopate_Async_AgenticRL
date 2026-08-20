# 提交件 · verl issue + PR 正文（英文，GitHub 可直接粘贴）

> 状态：**草稿完成，等 Chaoyu 过目后提交**（2026-08-19）。
> 中文分析与证据链 → [`analysis.md`](analysis.md)。目标仓库 **verl-project/verl**。
> 顺序：先 issue，PR 里填 `Fixes #<n>`；与包①（`../OPEN-verl-fsdp-size-1/`）同批提交、互相引用。
> ⚠️ 验证数字标注：〔源码树验证跑〕的数字以 `logs/e22_verl_fix_20260819.log` 为准回填。

---

## 1 · Issue

**Title:**

```
[rollout, fsdp] Disaggregated weight sync with LoRA silently pushes the frozen
base model — the adapter never reaches the rollout engine
```

**Body:**

````markdown
## Summary

With LoRA (`model.lora.merge=False`, the default) and a disaggregated
trainer/rollout deployment (`rollout.checkpoint_engine.backend != "naive"`, i.e.
the async RL recipes), **every weight sync pushes the unmodified frozen base
model. The LoRA adapter is never transferred.** The rollout engine samples from
the initial policy π₀ for the entire run.

Training completes normally — loss falls, rewards move, grad_norm / entropy /
ESS all look plausible, no warning is emitted. The RL loop is silently severed:
the policy gradient is computed for the current policy, while all data forever
comes from the initial one, and the mismatch grows monotonically with training.

The colocated path (`backend: naive`) is correct. Same config, different mode.

## Root cause

The disaggregated branch runs only half of the two-phase LoRA sync protocol,
and discards the signal that would tell the receiving side an adapter exists:

```python
# verl/workers/engine_workers.py (main)
if effective_mode != "naive":
    per_tensor_param, _ = self.actor.engine.get_per_tensor_param()   # (1) no args, peft_config discarded
    metrics = await self.checkpoint_engine.send_weights(...)
    return metrics or {}                                             # (2) no second call, ever
```

1. `get_per_tensor_param()` defaults to `base_sync_done=False` — the "push the
   base first" half of the protocol.
2. With LoRA + `merge=False`, that lands in `collect_lora_params(base_sync_done=False)`
   (`verl/utils/fsdp_utils.py`), which **explicitly skips every `lora_` tensor**
   and returns the frozen base, renamed via `replace_lora_wrapper` to
   `...base_layer.weight` form — precisely so that a LoRA-enabled vLLM accepts it.
   That rename is why the failure is silent: the push *succeeds*.
3. The protocol expects a follow-up `base_sync_done=True` call (push the
   adapter). The colocated branch does exactly that two-call dance. The
   disaggregated branch returns after the first half — every time.
4. Downstream, `CheckpointEngineWorker.update_weights(self, global_steps=None)`
   (`verl/checkpoint_engine/base.py`) has no way to receive `peft_config`
   anyway, while everything *below* it already supports adapters:
   `server_adapter.update_weights(..., peft_config=...)` →
   `update_weights_from_ipc` → `TensorLoRARequest(lora_tensors=...)` + `add_lora`
   — the exact machinery the colocated path exercises on every sync.

The code itself hints this was known-incomplete: the flag initialization carries
the comment *"base_sync_done is unused in merge-only mode but **kept for Phase 2
adapter path**"* (`engine_workers.py`). This report is that Phase 2, with a fix.

## Evidence

**Offline, no Ray** (2-layer Qwen3 + LoRA r=32, FSDP-wrapped, calling verl's own
`collect_lora_params`):

| branch | tensors | bytes | of which `lora_` |
|---|---|---|---|
| `base_sync_done=False` (what disaggregated always runs) | 25 | 74.8 MiB | **0** |
| `base_sync_done=True` (what colocated also runs) | 28 | 0.2 MiB | **28** |

**Real training** (verl 0.8.0, Qwen3-4B + LoRA r32, 3 trainer + 1 rollout GPU,
fully_async, probe at `NCCLCheckpointEngine.send_weights` — streaming, no
behavior change). Four independent runs, two syncs each:

```
every sync:  399 tensors / 8,414.1 MiB / 0 of them lora_
watched layer  model.layers.0.self_attn.q_proj.base_layer.weight
  pushed ‖W‖             = 75.377708   (identical across all runs and syncs)
  on-disk initial model  = 75.377708   (bit-identical)
```

A policy being trained cannot push bit-identical weights on every sync — the
payload is the frozen initial model. Meanwhile the vLLM server runs with
`--enable_lora` and an initialized `PunicaWrapperGPU`: **`list_loras()` stays
`[]` for the whole run.** The adapter slot exists and is never filled.

**The failure masquerades as staleness**, which is what makes it expensive in
practice — all of these "look like" async-RL staleness while actually measuring
a policy pinned at π₀ (we spent two months on that wrong trail):

| observation | looks like | actually is |
|---|---|---|
| `rollout_corr/kl` climbs monotonically (36×) at fixed `sync_every` | staleness accumulating | distance to π₀ = cumulative displacement |
| ESS collapses along `lr × steps`, insensitive to `sync_every` | odd | ESS is a function of displacement from π₀ |
| staleness-queue knobs (6× more stale trajectories) don't move ESS | insensitive knob | the knob is connected to nothing |
| sync duration independent of expected payload (8 GB vs 132 MB) | fixed overheads | it is always 8 GB |

## Why `model.lora.merge=True` is not a good workaround

Mechanically it works (the pushed weights then change with training). But the
merge is performed in bf16, and an RL-scale LoRA delta (‖ΔW‖/‖W‖ ~ 0.05%) falls
into bf16 rounding: on a fixed probe set, merging shifts per-token logprobs by a
**median of 1.7e-2 — about 50% of the adapter's own effect, and ~50× the
vLLM↔FSDP numerical floor**. It also keeps paying the full-model transfer cost
(8.4 GB per sync) for a 252 MiB adapter.

## Proposed fix (PR to follow)

Two small, wire-compatible changes; no protocol or backend changes:

1. **Trainer side** (`engine_workers.py`): run the same two-phase protocol as
   the colocated branch — initialize `base_sync_done` from
   `"dummy" not in rollout.load_format` (colocated semantics: with real weights
   loaded, the rollout engine never needs a base push at all), then collect with
   `base_sync_done` and flip it once a `peft_config` is returned. Full-parameter
   and `merge=True` runs return `peft_config=None` / no `lora_` tensors and are
   unaffected.
2. **Rollout side** (`checkpoint_engine/base.py`): adapter pushes are
   self-describing — they consist *only* of `lora_` tensors, while base/full/
   merged pushes contain none. Peek at the first tensor name (the stream is
   re-chained, nothing is consumed twice); if it is an adapter push, rebuild the
   minimal `peft_config` dict from the worker's own `model_config` (vLLM's
   `PEFTHelper.from_dict` needs only `r` / `lora_alpha` / `target_modules`) and
   forward `peft_config` + `base_sync_done=True` to the server adapter. This
   works for every checkpoint-engine backend that speaks the named-tensors wire
   format and leaves the delta engines untouched.

Measured with exactly this fix applied to the source tree (no monkeypatching),
same setup as above, 3 syncs:

```
payload            every sync: 252 MiB, 504/504 tensors lora_ -- the 8,414 MiB
                   frozen-base push never occurs at all (load_format is real, so
                   the first sync is already adapter-only, colocated semantics)
rollout side       "loading as adapter" on every sync (self-describing detection)
rollout_corr/kl    6.4e-05 / 2.9e-04 -- at the numerical floor
                   (broken baseline at the same point in training: 3e-3 and climbing)
```

Longer runs with the functionally equivalent patch (60 steps + an independent
12-step rerun): `list_loras()` `[]` -> `[<id>]`; kl mean 3.9e-3 -> 8e-4 across
15 sync periods; end-to-end logprob agreement at the engine floor; `param_sync`
6.25 s -> 0.97 s (6.4x).

End-to-end numeric check: with the adapter push in place, vLLM-returned
logprobs match trainer-recomputed logprobs to the engine numerical floor
(`log_ppl_diff` ~ 3-6e-4) — i.e. scaling, module targeting and tensor contents
are all correct, not just "something was pushed".

## Also worth considering

A defensive invariant, independent of this fix: after every sync, compare one
layer's norm between trainer and rollout (two scalars). "The thing pushed is
not the thing held" — in any backend — should be a hard failure, not something
users discover by diffing checkpoints two months later.

## Related

- #2048 (closed): async + LoRA used to raise an explicit error. #3654's fix
  brought the combination into the support surface — where it now fails
  silently instead. A silent wrong-weights sync is strictly worse than the old
  error: the error cost one launch; this costs entire training runs.
- #7287 / #3907 fixed adapter-sync defects on the *colocated* SGLang/vLLM
  paths; the disaggregated checkpoint-engine path is the remaining gap.

## Environment

- verl 0.8.0 (breakpoints re-verified unchanged on current `main`)
- torch 2.9.0+cu128, vLLM 0.12.0 (`--enable_lora`), NCCL checkpoint engine
- Qwen3-4B + LoRA r32, 3 trainer + 1 rollout GPU (RTX 5090), fully_async
````

---

## 2 · PR

**Title:**

```
[rollout, fsdp] fix: sync LoRA adapters to disaggregated rollout (two-phase
protocol + self-describing adapter detection)
```

**Body:**

````markdown
Fixes #<issue-number>.

## What

Disaggregated weight sync (`checkpoint_engine.backend != "naive"`) with LoRA
(`merge=False`, default) pushed the frozen base model on every sync and never
transferred the adapter — the rollout policy stayed at the initial checkpoint
for the whole run, silently (see issue for evidence and impact).

Two changes:

1. `verl/workers/engine_workers.py` — the disaggregated branch now runs the
   same two-phase protocol as the colocated branch below it: `base_sync_done`
   initialized from `"dummy" not in rollout.load_format` (with real weights
   loaded, no base push is needed at all — first sync is already adapter-only),
   collection called with `base_sync_done`, flag flipped once the engine
   returns a `peft_config`.
2. `verl/checkpoint_engine/base.py` — `CheckpointEngineWorker.update_weights`
   peeks at the first tensor name (stream re-chained, nothing consumed twice):
   adapter pushes consist only of `lora_` tensors, so they are self-describing.
   For an adapter push it rebuilds the minimal `peft_config` dict from its own
   `model_config` and forwards `peft_config` + `base_sync_done=True` to the
   server adapter — which already supports them (`update_weights_from_ipc` →
   `TensorLoRARequest` + `add_lora`, the colocated machinery).

## Why this shape

- **No wire change**: nothing new is serialized; works unchanged for every
  backend that speaks the named-tensors wire format (nccl / nixl / mooncake /
  kimi all go through this worker). The peek is guarded on
  `wire_format == "named_tensors"`; delta engines (`delta_sharded`) own their
  sync state machine and are untouched.
- **No config change**: behavior is keyed off what the engine actually returns.
- **Non-LoRA, `merge=True`, and base pushes are byte-for-byte unaffected**: in
  all three cases the payload contains no `lora_` tensor, so the worker takes
  today's exact path.
- `peft_config` is rebuilt worker-side from `model_config` (both processes are
  constructed from the same config) rather than serialized across the wire;
  vLLM's `PEFTHelper.from_dict` requires only `r` / `lora_alpha` /
  `target_modules`. Field-level equality with the trainer-side config was
  verified explicitly (r=32 / alpha=64 / scaling=2.0 on both sides).

## Validation

Real RL training (Qwen3-4B + LoRA r32, 3 trainer + 1 rollout, fully_async,
NCCL backend) with exactly this diff:

- payload per sync: 8,414 MiB (frozen base, 0 lora_ tensors) → **252 MiB,
  504/504 lora_ tensors**; first sync is already adapter-only
- vLLM `list_loras()`: `[]` → `[<id>]`
- `rollout_corr/kl` returns to the numerical floor after every sync (mean
  8e-4) instead of climbing monotonically (mean 3.9e-3) — i.e. the rollout
  policy now tracks the trainer policy
- end-to-end logprob agreement at the engine floor (~3-6e-4): scaling, module
  targeting and tensor contents verified, not just transfer (60-step + 12-step
  runs of the functionally equivalent patch)
- `param_sync` time 6.25 s → 0.97 s (6.4×)

## Tests

Two mock-based async unit tests (no GPU): the trainer branch runs the
two-phase protocol (`base_sync_done` sequence), and the worker annotates
adapter pushes with `peft_config`/`base_sync_done=True` while forwarding every
tensor in order and leaving base/full pushes untouched.

## Caveats

- Exercised end-to-end on the NCCL backend; other named-tensors backends share
  the same worker code path but were not run.
- SGLang server adapter accepts the same kwargs (cf. #7287) but was not
  exercised here.
````

---

## 3 · 提交时的注意事项

- [ ] issue 先提，拿到编号后 PR 填 `Fixes #<n>`
- [ ] 两个 patch 基于 **verl-project/verl main**（断点已核实未变；`wire_format` 是 main 新加的，patch 已带守卫）
- [x] 数字已回填（2026-08-19）：源码树跑 5 步 / 3 次同步 / exit 0 —— 全部 adapter 推送、基座 0 次、kl 贴地板；三 rank ckpt 504/504 逐位相同（E21 修复同场验证）
- [ ] 与包①同批提交，正文互相引用（同一天发现、同形状：配置意图正确，静默走进错误分支）
- [ ] tag LoRA 相关 codeowner（#7436 显示 HollowMan6 刚被加进 CODEOWNERS）
