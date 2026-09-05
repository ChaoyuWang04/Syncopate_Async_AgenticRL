from __future__ import annotations

import json

import pytest
import torch
from safetensors.torch import save_file

from syncopate.train.merge_adapter import (
    _assert_merge_landed,
    _converted_base_key_map,
    _resolve_weight_name,
)


class _TinyModel:
    def __init__(self, state: dict[str, torch.Tensor]) -> None:
        self._state = state

    def state_dict(self) -> dict[str, torch.Tensor]:
        return self._state


class _PrefixTransform:
    def rename_source_key(self, source_key: str) -> tuple[str, str | None]:
        prefix = "model.language_model."
        if source_key.startswith(prefix):
            return "model." + source_key[len(prefix):], prefix
        return source_key, None


def test_resolve_weight_name_accepts_unique_wrapper_insertion() -> None:
    names = {
        "model.language_model.layers.0.linear_attn.in_proj_a.weight",
        "model.visual.blocks.0.attn.proj.weight",
    }
    assert _resolve_weight_name(
        "base_model.model.model.layers.0.linear_attn.in_proj_a", names
    ) == "model.language_model.layers.0.linear_attn.in_proj_a.weight"


def test_resolve_weight_name_rejects_ambiguous_suffix() -> None:
    names = {
        "model.language_model.layers.3.self_attn.q_proj.weight",
        "mtp.layers.3.self_attn.q_proj.weight",
    }
    with pytest.raises(SystemExit, match="拒绝猜测"):
        _resolve_weight_name("base_model.model.model.layers.3.self_attn.q_proj", names)


def test_official_conversion_map_excludes_mtp_collision() -> None:
    model_name = "model.layers.0.mlp.shared_expert.down_proj.weight"
    mapping = _converted_base_key_map(
        {
            "model.language_model.layers.0.mlp.shared_expert.down_proj.weight",
            "mtp.layers.0.mlp.shared_expert.down_proj.weight",
        },
        {model_name},
        [_PrefixTransform()],
    )
    assert mapping == {
        model_name: ["model.language_model.layers.0.mlp.shared_expert.down_proj.weight"]
    }


def test_assert_merge_landed_handles_qwen_language_model_wrapper(tmp_path) -> None:
    base = tmp_path / "base"
    adapter = tmp_path / "adapter"
    base.mkdir()
    adapter.mkdir()

    base_name = "model.language_model.layers.0.linear_attn.in_proj_a.weight"
    merged_name = "model.layers.0.linear_attn.in_proj_a.weight"
    base_weight = torch.ones((2, 2), dtype=torch.float32)
    delta = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    save_file({base_name: base_weight}, base / "model.safetensors")
    (adapter / "adapter_config.json").write_text(
        json.dumps({"lora_alpha": 1, "r": 1}), encoding="utf-8"
    )
    stem = "base_model.model.model.layers.0.linear_attn.in_proj_a"
    save_file(
        {
            stem + ".lora_A.weight": torch.tensor([[1.0, 0.0]]),
            stem + ".lora_B.weight": torch.tensor([[1.0], [0.0]]),
        },
        adapter / "adapter_model.safetensors",
    )

    _assert_merge_landed(
        str(base), str(adapter), _TinyModel({merged_name: base_weight + delta}), max_resid=0.01
    )
