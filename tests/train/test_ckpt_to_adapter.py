from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest
import torch

from syncopate.train.ckpt_to_adapter import _write_adapter, export_adapter
from syncopate.train.lora_adapter_check import AdapterValidationError, inspect_adapter


META = {"r": 2, "lora_alpha": 4, "task_type": "CAUSAL_LM"}


def _learned_state() -> dict[str, torch.Tensor]:
    stem = "base_model.model.model.layers.0.self_attn.q_proj"
    return {
        f"{stem}.lora_A.default.weight": torch.tensor([[1.0, 0.0, 2.0], [0.0, 1.0, 3.0]]),
        f"{stem}.lora_B.default.weight": torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0]]
        ),
    }


def test_adapter_gate_checks_pairs_rank_metadata_and_learned_b(tmp_path: Path):
    out = tmp_path / "adapter"
    stats = _write_adapter(_learned_state(), META, out)
    assert stats["pairs"] == 1
    assert stats["nonzero_b_tensors"] == 1
    assert stats["target_modules"] == ["q_proj"]
    assert inspect_adapter(out) == stats


def test_adapter_gate_rejects_initialization_only_zero_b(tmp_path: Path):
    state = _learned_state()
    state[next(key for key in state if ".lora_B." in key)].zero_()
    with pytest.raises(AdapterValidationError, match="all LoRA B tensors are zero"):
        _write_adapter(state, META, tmp_path / "adapter")


def test_writer_refuses_to_silently_drop_base_weights(tmp_path: Path):
    state = {**_learned_state(), "base_model.model.model.embed_tokens.weight": torch.ones(2, 2)}
    with pytest.raises(RuntimeError, match="expected a LoRA-only checkpoint"):
        _write_adapter(state, META, tmp_path / "adapter")


def test_fsdp2_export_uses_verl_shard_reconstruction_but_not_full_model_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    actor = tmp_path / "global_step_2" / "actor"
    actor.mkdir(parents=True)
    (actor / "huggingface").mkdir()
    (actor / "fsdp_config.json").write_text('{"FSDP_version": 2, "world_size": 2}')
    (actor / "lora_train_meta.json").write_text(json.dumps(META))
    for rank in range(2):
        (actor / f"model_world_size_2_rank_{rank}.pt").write_bytes(f"shard-{rank}".encode())

    class FakeConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            self.hf_upload = False

    class FakeFSDPModelMerger:
        cleaned = False

        def __init__(self, config):
            self.config = config

        def merge_and_save(self):
            # Dynamic dispatch must hit the project adapter-only writer.  Calling a
            # generic full-model writer here would make this fake fail immediately.
            self.save_hf_model_and_tokenizer(_learned_state())

        def cleanup(self):
            type(self).cleaned = True

    base_module = types.ModuleType("verl.model_merger.base_model_merger")
    base_module.ModelMergerConfig = FakeConfig
    fsdp_module = types.ModuleType("verl.model_merger.fsdp_model_merger")
    fsdp_module.FSDPModelMerger = FakeFSDPModelMerger
    monkeypatch.setitem(sys.modules, "verl.model_merger.base_model_merger", base_module)
    monkeypatch.setitem(sys.modules, "verl.model_merger.fsdp_model_merger", fsdp_module)

    out = tmp_path / "published" / "lora_adapter"
    result = export_adapter(actor, out)
    assert result["backend"] == "verl-fsdp2"
    assert inspect_adapter(out)["pairs"] == 1
    manifest = json.loads((out / "export_manifest.json").read_text())
    assert manifest["backend"] == "verl-fsdp2"
    assert len(manifest["source_shards"]) == 2
    assert not (out.parent / "model.safetensors").exists()


def test_runbook_uses_project_adapter_exporter_not_generic_full_model_merger():
    source = Path("scripts/v16_pipeline.sh").read_text(encoding="utf-8")
    assert "-m syncopate.train.ckpt_to_adapter" in source
    assert "-m verl.model_merger merge" not in source
