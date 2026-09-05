"""Strict PEFT LoRA adapter gate used by the fixed SFT/RL pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


class AdapterValidationError(ValueError):
    """The directory is not a usable, learned PEFT LoRA adapter."""


def inspect_adapter(adapter_dir: str | Path) -> dict[str, Any]:
    """Validate files, metadata, A/B pairing, rank shapes, and learned B weights."""

    directory = Path(adapter_dir)
    config_path = directory / "adapter_config.json"
    weights_path = directory / "adapter_model.safetensors"
    missing = [path.name for path in (config_path, weights_path) if not path.is_file()]
    if missing:
        raise AdapterValidationError(f"missing required file(s): {', '.join(missing)}")
    if weights_path.stat().st_size == 0:
        raise AdapterValidationError("adapter_model.safetensors is empty")

    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AdapterValidationError(f"invalid adapter_config.json: {exc}") from exc
    if not isinstance(config, dict):
        raise AdapterValidationError("adapter_config.json must contain a JSON object")

    try:
        rank = int(config.get("r", 0))
        alpha = float(config.get("lora_alpha", 0))
    except (TypeError, ValueError) as exc:
        raise AdapterValidationError("r and lora_alpha must be numeric") from exc
    if rank <= 0 or alpha <= 0:
        raise AdapterValidationError(f"invalid LoRA metadata: r={rank}, alpha={alpha}")
    if config.get("task_type") != "CAUSAL_LM":
        raise AdapterValidationError(
            f"task_type must be CAUSAL_LM, got {config.get('task_type')!r}"
        )
    if config.get("peft_type") != "LORA":
        raise AdapterValidationError(f"peft_type must be LORA, got {config.get('peft_type')!r}")

    from safetensors import safe_open

    shapes: dict[str, tuple[int, ...]] = {}
    nonzero = 0
    nonzero_b = 0
    with safe_open(str(weights_path), framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        for key in keys:
            tensor = handle.get_tensor(key)
            shapes[key] = tuple(tensor.shape)
            learned = bool(tensor.count_nonzero().item())
            nonzero += int(learned)
            nonzero_b += int(learned and key.endswith(".lora_B.weight"))

    a_keys = {key for key in shapes if key.endswith(".lora_A.weight")}
    b_keys = {key for key in shapes if key.endswith(".lora_B.weight")}
    unexpected = sorted(set(shapes) - a_keys - b_keys)
    a_stems = {key.removesuffix(".lora_A.weight") for key in a_keys}
    b_stems = {key.removesuffix(".lora_B.weight") for key in b_keys}
    if unexpected:
        raise AdapterValidationError(f"unexpected non-LoRA tensors: {unexpected[:3]}")
    if not a_stems:
        raise AdapterValidationError("adapter contains no LoRA A/B pairs")
    if a_stems != b_stems:
        only_a = sorted(a_stems - b_stems)[:3]
        only_b = sorted(b_stems - a_stems)[:3]
        raise AdapterValidationError(f"unpaired LoRA tensors: only_A={only_a}, only_B={only_b}")

    bad_shapes = []
    for stem in sorted(a_stems):
        a_shape = shapes[f"{stem}.lora_A.weight"]
        b_shape = shapes[f"{stem}.lora_B.weight"]
        if len(a_shape) != 2 or len(b_shape) != 2 or a_shape[0] != rank or b_shape[1] != rank:
            bad_shapes.append((stem, a_shape, b_shape))
    if bad_shapes:
        raise AdapterValidationError(
            f"LoRA rank/shape mismatch for r={rank}: {bad_shapes[:3]}"
        )
    if nonzero_b == 0:
        raise AdapterValidationError(
            "all LoRA B tensors are zero; A is nonzero at initialization, so this does not prove training"
        )

    actual_modules = {stem.rsplit(".", 1)[-1] for stem in a_stems}
    configured_modules = config.get("target_modules")
    if not isinstance(configured_modules, list) or set(configured_modules) != actual_modules:
        raise AdapterValidationError(
            "target_modules does not exactly match adapter tensor module names: "
            f"config={sorted(configured_modules or [])}, actual={sorted(actual_modules)}"
        )

    return {
        "r": rank,
        "alpha": alpha,
        "tensors": len(shapes),
        "pairs": len(a_stems),
        "nonzero_tensors": nonzero,
        "nonzero_b_tensors": nonzero_b,
        "bytes": weights_path.stat().st_size,
        "target_modules": sorted(actual_modules),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a learned PEFT LoRA adapter")
    parser.add_argument("adapter_dir", type=Path)
    args = parser.parse_args(argv)
    try:
        stats = inspect_adapter(args.adapter_dir)
    except AdapterValidationError as exc:
        print(f"🔴 {args.adapter_dir} is not a usable learned PEFT adapter: {exc}")
        return 1
    print(
        f"[adapter] {args.adapter_dir}: r={stats['r']} alpha={stats['alpha']:g} · "
        f"{stats['pairs']} A/B pairs · nonzero B {stats['nonzero_b_tensors']} · "
        f"{stats['bytes']} bytes"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
