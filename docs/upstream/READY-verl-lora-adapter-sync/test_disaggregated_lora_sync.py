# Proposed tests for the two-phase disaggregated LoRA sync fix.
# Both are mock-based (no GPU, no Ray) -- they pin the protocol, not the transport.
# Place / adapt per maintainer convention (e.g. tests/checkpoint_engine/).
import asyncio
from types import SimpleNamespace

import torch


# ---------------------------------------------------------------------------
# ① rollout side: CheckpointEngineWorker.update_weights
#    - an adapter push (only `lora_` tensors) must reach the server adapter
#      with peft_config + base_sync_done=True
#    - a base/full push must go through unchanged (no kwargs)
#    - all items must be forwarded, in order (the peek must not eat the first)
# ---------------------------------------------------------------------------

def _make_worker(update_weights_impl, first_names):
    """Build a stub `self` for the unbound update_weights coroutine."""

    async def receive_weights(global_steps=None):
        for name in first_names:
            yield name, torch.zeros(1)

    captured = {}

    class ServerAdapter:
        async def update_weights(self, weights, global_steps=None, **kwargs):
            captured["kwargs"] = kwargs
            captured["names"] = [n async for n, _ in weights]

    worker = SimpleNamespace(
        checkpoint_engine=SimpleNamespace(receive_weights=receive_weights),
        server_adapter=ServerAdapter(),
        model_config=SimpleNamespace(
            lora_rank=32, lora_alpha=64, target_modules=["q_proj", "v_proj"], exclude_modules=None
        ),
        _infer_lora_peft_config=None,   # bound below
        update_weights=None,
    )
    from verl.checkpoint_engine.base import CheckpointEngineWorker

    worker._infer_lora_peft_config = CheckpointEngineWorker._infer_lora_peft_config.__get__(worker)
    worker.update_weights = update_weights_impl.__get__(worker)
    return worker, captured


def test_adapter_push_is_detected_and_annotated():
    from verl.checkpoint_engine.base import CheckpointEngineWorker

    names = [
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight",
        "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight",
    ]
    worker, captured = _make_worker(CheckpointEngineWorker.update_weights.__wrapped__
                                    if hasattr(CheckpointEngineWorker.update_weights, "__wrapped__")
                                    else CheckpointEngineWorker.update_weights, names)
    asyncio.run(worker.update_weights(global_steps=1))
    assert captured["names"] == names, "peek must forward every tensor, in order"
    assert captured["kwargs"].get("base_sync_done") is True
    pc = captured["kwargs"].get("peft_config")
    assert pc and pc["r"] == 32 and pc["lora_alpha"] == 64


def test_base_push_goes_through_unchanged():
    from verl.checkpoint_engine.base import CheckpointEngineWorker

    names = ["model.embed_tokens.weight", "model.layers.0.self_attn.q_proj.base_layer.weight"]
    worker, captured = _make_worker(CheckpointEngineWorker.update_weights.__wrapped__
                                    if hasattr(CheckpointEngineWorker.update_weights, "__wrapped__")
                                    else CheckpointEngineWorker.update_weights, names)
    asyncio.run(worker.update_weights(global_steps=1))
    assert captured["names"] == names
    assert "peft_config" not in captured["kwargs"], "base/full pushes must keep today's behavior"


# ---------------------------------------------------------------------------
# ② trainer side: the disaggregated branch must run the two-phase protocol
#    (with a real load_format, the very first push is already adapter-only)
# ---------------------------------------------------------------------------

def test_trainer_two_phase_protocol():
    from verl.workers.engine_workers import ActorRolloutRefWorker

    calls = []

    def get_per_tensor_param(layered_summon=False, base_sync_done=False):
        calls.append(base_sync_done)
        return iter(()), {"r": 32}          # (params, peft_config)

    async def send_weights(per_tensor_param, global_steps=None):
        return {}

    worker = SimpleNamespace(
        config=SimpleNamespace(rollout=_RolloutCfg()),
        actor=SimpleNamespace(engine=SimpleNamespace(get_per_tensor_param=get_per_tensor_param)),
        checkpoint_engine=SimpleNamespace(send_weights=send_weights),
    )
    impl = ActorRolloutRefWorker.update_weights
    impl = getattr(impl, "__wrapped__", impl)
    bound = impl.__get__(worker)
    asyncio.run(bound(global_steps=1, mode="nccl"))
    asyncio.run(bound(global_steps=2, mode="nccl"))
    # load_format="safetensors" => rollout already has real base weights
    # => both syncs must collect the adapter (base_sync_done=True), never the base
    assert calls == [True, True]


class _RolloutCfg:
    load_format = "safetensors"
    checkpoint_engine = SimpleNamespace(backend="nccl")

    def get(self, key, default=None):
        return getattr(self, key, default)
