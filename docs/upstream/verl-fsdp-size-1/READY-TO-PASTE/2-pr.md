标题（复制这一行；CI 会检查这个格式）：
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
