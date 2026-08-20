标题（复制这一行；CI 会检查这个格式）：
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
