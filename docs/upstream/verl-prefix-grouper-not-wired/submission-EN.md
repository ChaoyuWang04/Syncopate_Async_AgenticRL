# 提交件 · verl issue + PR 正文（英文，GitHub 可直接粘贴）

> 状态：**草稿完成，等 Chaoyu 过目后提交**（2026-08-19）。
> 中文分析 → [`analysis.md`](analysis.md)。零 GPU 复现 → [`repro_prefix_grouper_wiring.py`](repro_prefix_grouper_wiring.py)。
> 顺序：先 issue，PR 里填 `Fixes #<n>`。⚠️ 端到端数字待我们的补丁跑通后回填。

---

## 1 · Issue

**Title:**

```
[fsdp, actor] `actor.use_prefix_grouper=True` is a silent no-op — the shared-prefix
forward is never wired in (and the mask it would use drops tool tokens)
```

**Body:**

````markdown
## Summary

`actor.use_prefix_grouper` is accepted by the config, documented in
`actor.yaml` ("Whether to enable PrefixGrouper shared-prefix forward"), read by
three trainers, and even has dedicated group-aware batch balancing. **But the
shared-prefix forward is never actually entered.** Enabling the flag does not
speed anything up and does not error — the only remaining effect is that
`_balance_batch` keeps samples of a GRPO group on the same rank.

This matters beyond a missing feature: a user who enables the flag and measures
wall-clock will observe **no improvement**, and will reasonably conclude that
PrefixGrouper is not worth it for their workload — when in fact it never ran.

Reproduced on verl 0.8.0 with a zero-GPU script (attached below); all three
checks are static/CPU-only.

## Root cause: two disconnected wires

**(1) The attention patch is never applied.**
`apply_monkey_patch()` accepts `use_prefix_grouper` and calls
`apply_prefix_grouper_patch()` when it is true
(`verl/models/transformers/monkey_patch.py:324`). It has exactly **one** call
site in the whole package, and that call site does not forward the flag:

```python
# verl/workers/engine/fsdp/transformer_impl.py:292
apply_monkey_patch(
    model=module,
    use_remove_padding=self.use_remove_padding,
    ulysses_sp_size=self.ulysses_sequence_parallel_size,
    use_fused_kernels=use_fused_kernels,
    fused_kernels_backend=fused_kernels_backend,
)   # <-- no use_prefix_grouper => defaults to False
```

So `ALL_ATTENTION_FUNCTIONS` is never wrapped, and the `prefix_grouper=` kwarg
would be ignored even if something passed it.

**(2) The packed forward has no callers.**
`forward_micro_batch_with_prefix_grouper()` and `build_pg_from_micro_batch()`
(`verl/trainer/ppo/prefix_grouper_utils.py`) are never referenced outside their
own module. The FSDP engine's `forward_step()`
(`verl/workers/engine/fsdp/transformer_impl.py:1253`) always goes through
`prepare_model_inputs()` -> `self.module(**model_inputs)` ->
`prepare_model_outputs()`, i.e. the standard rmpad path.

## Second, latent bug: `suffix_mask` is given the gradient mask

This one does not bite today (because of the above), but it will the moment the
wiring is fixed, so it belongs in the same issue.

`PrefixGrouper` uses `suffix_mask` to decide **which tokens exist**:

```python
# prefix_grouper/__init__.py
suffix_lens = suffix_mask.sum(dim=1)                 # length of each suffix
suffix_mask.nonzero(as_tuple=False)                  # which positions get gathered
```

verl passes `response_mask`:

```python
# verl/trainer/ppo/prefix_grouper_utils.py
PrefixGrouper.from_ungrouped_masks(
    prefix_mask=prefix_ids.ne(pad_token_id),
    suffix_mask=response_mask,          # <-- gradient mask, not existence mask
    ...)
```

In single-turn RLVR the two coincide. In **multi-turn / tool-calling** rollouts
they do not — and verl's own agent loop is explicit about it:

```python
# verl/experimental/agent_loop/tool_agent_loop.py
agent_data.response_mask += [1] * len(response_ids)   # model-generated  -> gradient
agent_data.response_mask += [0] * len(response_ids)   # tool observation -> no gradient
```

Result: every tool-observation token is dropped from the packed model input.
The model would be trained on a conversation with the tool results removed —
silently, since the shapes still line up.

Check C of the repro script shows this directly:

```
packed with existence mask : [[1,2,3,4, 10,11,12,13,14,15, 20,21,22,23,24,25]]
packed with response_mask  : [[1,2,3,4, 10,11,      14,15, 20,21,      24,25]]
tokens dropped from input  : [12,13,22,23]   <-- the tool observations
```

A correct fix needs **two** masks: pack the input with the response *attention*
mask (`attention_mask[:, prompt_length:]`), and keep `response_mask` for the
loss. `pg_forward()` currently reuses the single mask returned by
`split_output()` for both purposes.

## Reproduction

```bash
pip install prefix_grouper          # only needed for check C
python repro_prefix_grouper_wiring.py
```

<details>
<summary>repro_prefix_grouper_wiring.py</summary>

(attached in the PR / gist — static AST scan of the verl package for checks A
and B, plus a 16-token PrefixGrouper round-trip for check C. No GPU, no model.)

</details>

Output on verl 0.8.0:

```
[PASS] A. apply_monkey_patch() is never called with use_prefix_grouper
       verl/workers/engine/fsdp/transformer_impl.py:292
       kwargs=['model','use_remove_padding','ulysses_sp_size','use_fused_kernels','fused_kernels_backend']
[PASS] B. forward_micro_batch_with_prefix_grouper() has zero call sites
[PASS] C. passing response_mask silently drops tool-observation tokens
```

## Suggested fix

1. Forward `use_prefix_grouper` from the engine config into
   `apply_monkey_patch()`.
2. Route `forward_step()` through the packed path when the flag is on (or state
   in the docs that the feature is not available on the FSDP engine yet).
3. Use the response attention mask for packing and keep `response_mask` for the
   loss, so multi-turn rollouts stay correct.
4. Add an assertion or a one-line log when the flag is enabled but the patch did
   not apply, so "enabled but inert" cannot happen silently again.

Happy to send a PR for (1)-(3) if the direction looks right.

## Environment

```
verl 0.8.0 · torch 2.9 · FSDP engine · flash_attention_2
prefix-grouper 0.0.1.post1
```
````

---

## 2 · PR

**Title:**

```
[fsdp, actor] Wire up use_prefix_grouper, and pack with the existence mask
```

**Body:**

````markdown
Fixes #<issue number>

`actor.use_prefix_grouper` was accepted, documented and read, but the
shared-prefix forward was never entered: the attention patch was not applied
(the single `apply_monkey_patch()` call site did not forward the flag) and
`forward_micro_batch_with_prefix_grouper()` had no callers. Enabling the flag
was a no-op that a user would most likely measure as "PrefixGrouper gives no
speedup".

## Changes

- `workers/engine/fsdp/transformer_impl.py`: forward `use_prefix_grouper` into
  `apply_monkey_patch()`; route `forward_step()` through the packed path when
  enabled.
- `trainer/ppo/prefix_grouper_utils.py`: build the packed input from the
  response **attention** mask; keep `response_mask` as the loss mask. This keeps
  multi-turn / tool-calling rollouts correct — previously every tool-observation
  token would have been dropped from the model input.
- Assertion when the flag is on but the attention patch did not apply, so the
  "enabled but inert" state cannot recur silently.

## Validation

- Zero-GPU repro script for the three failure modes (checks A/B/C) — attached in
  the issue.
- **Equivalence**: with the flag on vs off, `log_probs` for the same batch and
  the same weights are 〔bitwise identical / max abs diff = X〕.
  ★ Equivalence is the acceptance criterion here, not speed — the whole point of
  the paper's design is that it is training-equivalent.
- Throughput on a multi-turn agentic workload (group size 8, shared prefix
  〔4196〕 tokens, suffix 〔654〕 tokens): 〔to be filled〕.

## Notes

- No wire-format or config changes; the flag keeps its name and default
  (`false`).
- Single-turn RLVR is unaffected by the mask change (there the two masks are
  identical).
````

---

## 3 · 提交注意事项

```
⚠️ 数字带〔〕的都要用我们自己的补丁跑通后回填 —— 现在**不许**填任何猜测值。
⚠️ issue 与 PR 分开提，PR 里 Fixes #<n>。
⚠️ 与包①② 同批时互相引用一句「same shape: config accepted, wire missing」。
⚠️ C 那条要写清楚「今天不咬人、修好 A/B 之后才咬人」，否则会被当成不存在的问题关掉。
```
