"""Export a verl RL checkpoint as a standalone PEFT LoRA adapter.

The current B200 path saves LoRA-only FSDP2 shards.  verl's generic model merger
correctly reconstructs those shards and writes ``lora_adapter/``, but then tries
to validate the result as a *full* Hugging Face model.  A LoRA-only checkpoint
contains no base weights, so that final full-model validation necessarily fails.

This component reuses verl's FSDP2 shard reconstruction and replaces only its
final writer: every reconstructed key must be a LoRA key, then the adapter is
written and checked as PEFT.  Legacy replicated/DDP checkpoints retain their
old exact cross-rank equality check.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from syncopate.train.ckpt_guards import assert_ranks_identical
from syncopate.train.lora_adapter_check import inspect_adapter


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_meta(actor_dir: Path) -> dict[str, Any]:
    path = actor_dir / "lora_train_meta.json"
    if not path.is_file():
        raise RuntimeError(f"missing LoRA training metadata: {path}")
    meta = json.loads(path.read_text(encoding="utf-8"))
    try:
        rank = int(meta["r"])
        alpha = float(meta["lora_alpha"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"invalid LoRA metadata in {path}: {meta!r}") from exc
    task_type = meta.get("task_type")
    if rank <= 0 or alpha <= 0 or task_type != "CAUSAL_LM":
        raise RuntimeError(
            f"invalid LoRA metadata in {path}: r={rank}, alpha={alpha}, task_type={task_type!r}"
        )
    return {"r": rank, "lora_alpha": alpha, "task_type": task_type}


def _write_adapter(state_dict: dict[str, Any], meta: dict[str, Any], out: Path) -> dict[str, Any]:
    """Write only LoRA tensors; refusing base weights makes save_lora_only observable."""

    import torch
    from safetensors.torch import save_file

    non_lora = sorted(key for key in state_dict if "lora_" not in key)
    if non_lora:
        raise RuntimeError(
            "expected a LoRA-only checkpoint, but reconstructed base/non-LoRA weights: "
            f"{non_lora[:5]}"
        )

    tensors: dict[str, torch.Tensor] = {}
    for source_key, value in state_dict.items():
        target_key = source_key.replace(".default.weight", ".weight")
        if target_key in tensors:
            raise RuntimeError(f"LoRA key normalization collision: {target_key}")
        if not isinstance(value, torch.Tensor):
            raise RuntimeError(f"LoRA value is not a tensor: {source_key} -> {type(value).__name__}")
        tensors[target_key] = value.detach().cpu().contiguous()
    if not tensors:
        raise RuntimeError("checkpoint contains no LoRA tensors")

    target_modules = sorted(
        {
            key.removesuffix(".lora_A.weight").rsplit(".", 1)[-1]
            for key in tensors
            if key.endswith(".lora_A.weight")
        }
    )
    config = {
        "task_type": meta["task_type"],
        "peft_type": "LORA",
        "auto_mapping": None,
        "base_model_name_or_path": None,
        "revision": None,
        "inference_mode": False,
        "r": int(meta["r"]),
        "lora_alpha": meta["lora_alpha"],
        "lora_dropout": 0.0,
        "fan_in_fan_out": False,
        "bias": "none",
        "target_modules": target_modules,
        "modules_to_save": None,
        "init_lora_weights": True,
        "layers_to_transform": None,
        "layers_pattern": None,
        "rank_pattern": {},
        "alpha_pattern": {},
    }
    out.mkdir(parents=True, exist_ok=True)
    save_file(tensors, out / "adapter_model.safetensors")
    (out / "adapter_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return inspect_adapter(out)


def _export_fsdp2(actor_dir: Path, staging_dir: Path, meta: dict[str, Any]) -> dict[str, Any]:
    """Let verl merge DTensor shards, while substituting an adapter-only final writer."""

    from verl.model_merger.base_model_merger import ModelMergerConfig
    from verl.model_merger.fsdp_model_merger import FSDPModelMerger

    config = ModelMergerConfig(
        operation="merge",
        backend="fsdp",
        target_dir=str(staging_dir.parent),
        local_dir=str(actor_dir),
        hf_model_config_path=str(actor_dir / "huggingface"),
    )

    class AdapterOnlyFSDPModelMerger(FSDPModelMerger):
        export_stats: dict[str, Any] | None = None

        def save_hf_model_and_tokenizer(self, state_dict):  # type: ignore[no-untyped-def]
            self.export_stats = _write_adapter(state_dict, meta, staging_dir)

    merger = AdapterOnlyFSDPModelMerger(config)
    try:
        merger.merge_and_save()
        if merger.export_stats is None:
            raise RuntimeError("verl merger returned without invoking the adapter writer")
        return merger.export_stats
    finally:
        merger.cleanup()


def _export_replicated(actor_dir: Path, staging_dir: Path, meta: dict[str, Any]) -> dict[str, Any]:
    """Legacy path: each rank is a complete copy, proven equal before rank 0 is read."""

    import torch

    shards = sorted(actor_dir.glob("model_world_size_*_rank_*.pt"))
    if not shards:
        raise RuntimeError(f"no model shards found in {actor_dir}")
    checked = assert_ranks_identical(actor_dir, sample=10**9)
    if len(shards) > 1 and checked == 0:
        raise RuntimeError("multiple replicated shards exist but no tensors were compared")
    state_dict = torch.load(shards[0], map_location="cpu", weights_only=False)
    return _write_adapter(state_dict, meta, staging_dir)


def export_adapter(actor_dir: str | Path, out: str | Path) -> dict[str, Any]:
    actor = Path(actor_dir)
    destination = Path(out)
    if not actor.is_dir():
        raise RuntimeError(f"actor checkpoint directory does not exist: {actor}")
    meta = _read_meta(actor)
    fsdp_config_path = actor / "fsdp_config.json"
    fsdp_config = (
        json.loads(fsdp_config_path.read_text(encoding="utf-8"))
        if fsdp_config_path.is_file()
        else {}
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".adapter-export-", dir=destination.parent) as temp:
        staging = Path(temp) / "adapter"
        if int(fsdp_config.get("FSDP_version", 1)) == 2:
            stats = _export_fsdp2(actor, staging, meta)
            backend = "verl-fsdp2"
        else:
            stats = _export_replicated(actor, staging, meta)
            backend = "replicated-rank0-after-exact-rank-check"

        source_shards = sorted(actor.glob("model_world_size_*_rank_*.pt"))
        manifest = {
            "schema_version": 1,
            "backend": backend,
            "source_actor": str(actor),
            "source_shards": [
                {"name": path.name, "bytes": path.stat().st_size, "sha256": _sha256(path)}
                for path in source_shards
            ],
            "adapter": stats,
            "adapter_config_sha256": _sha256(staging / "adapter_config.json"),
            "adapter_weights_sha256": _sha256(staging / "adapter_model.safetensors"),
        }
        (staging / "export_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        destination.mkdir(parents=True, exist_ok=True)
        for name in ("adapter_config.json", "adapter_model.safetensors", "export_manifest.json"):
            os.replace(staging / name, destination / name)

    final = inspect_adapter(destination)
    print(
        f"[rl-adapter] {actor} -> {destination} · {backend} · "
        f"{final['pairs']} pairs · nonzero B {final['nonzero_b_tensors']} · "
        f"{final['bytes']} bytes"
    )
    return {**final, "backend": backend}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export a verl RL checkpoint as a PEFT LoRA adapter")
    parser.add_argument("actor_dir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    export_adapter(args.actor_dir, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
