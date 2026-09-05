"""SFT smoke 的固定健康闸：检查真实更新曲线、位移、显存和本轮 adapter。"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any


def evaluate(log_path: Path, adapter_dir: Path, *, expected_steps: int) -> dict[str, Any]:
    log = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    rows = [
        (int(step), float(loss), float(grad), float(rate))
        for step, loss, grad, rate in re.findall(
            r"\[step (\d+)\] loss=([-+0-9.eE]+) grad_norm=([-+0-9.eE]+).*?sup_tok/s=([-+0-9.eE]+)",
            log,
        )
    ]
    losses = [row[1] for row in rows]
    grads = [row[2] for row in rows]
    rates = [row[3] for row in rows]
    trainable = [float(x) for x in re.findall(r"可训练 ([0-9.]+)M", log)]
    shifts = [float(x) for x in re.findall(r"\|\|ΔW\|\|/\|\|W\|\| = ([0-9.eE+-]+)%", log)]
    peaks = [float(x) for x in re.findall(r"显存峰值 ([0-9.]+) GB", log)]
    window = max(1, min(5, len(losses) // 2))
    trend = bool(losses) and sum(losses[-window:]) / window < sum(losses[:window]) / window
    checks = {
        "log_exists": log_path.is_file() and bool(log.strip()),
        "steps_complete": bool(rows) and max(row[0] for row in rows) >= expected_steps,
        "loss_finite": bool(losses) and all(math.isfinite(x) for x in losses),
        "loss_trending_down": trend,
        "grad_finite_nonzero": bool(grads) and all(math.isfinite(x) for x in grads)
        and any(x > 0 for x in grads),
        "throughput_positive": bool(rates) and all(math.isfinite(x) and x > 0 for x in rates),
        "trainable_expected": bool(trainable) and 29.6 <= trainable[-1] <= 44.4,
        "weights_moved": bool(shifts) and math.isfinite(shifts[-1]) and shifts[-1] > 0,
        "peak_memory_below_b200": bool(peaks) and max(peaks) < 180.0,
        "adapter_files_exist": (adapter_dir / "adapter_config.json").is_file()
        and (adapter_dir / "adapter_model.safetensors").is_file(),
        "no_traceback": "Traceback" not in log,
    }
    quality_names = {"loss_trending_down"}
    health_checks = {name: passed for name, passed in checks.items()
                     if name not in quality_names}
    quality_checks = {name: checks[name] for name in quality_names}
    health_ok = all(health_checks.values())
    quality_ready = all(quality_checks.values())
    status = "fatal" if not health_ok else ("warn" if not quality_ready else "pass")
    return {
        "ok": health_ok and quality_ready,
        "status": status,
        "health_ok": health_ok,
        "quality_ready": quality_ready,
        "log_path": str(log_path),
        "adapter_dir": str(adapter_dir),
        "expected_steps": expected_steps,
        "checks": checks,
        "health_checks": health_checks,
        "quality_checks": quality_checks,
        "metrics": {
            "steps": [row[0] for row in rows],
            "losses": losses,
            "grad_norms": grads,
            "supervised_tokens_per_sec": rates,
            "trainable_millions": trainable,
            "delta_w_percent": shifts,
            "peak_memory_gb": peaks,
        },
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="检查 SFT smoke 是否真的更新且产出可验 adapter")
    ap.add_argument("--log", type=Path, required=True)
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--expected-steps", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    result = evaluate(args.log, args.adapter, expected_steps=args.expected_steps)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    for name, passed in result["checks"].items():
        print(f"  {'✅' if passed else '🔴'} {name}")
    print(f"[sft-run-gate] {result['status'].upper()} -> {args.out}")
    if not result["health_ok"]:
        return 3
    return 0 if result["quality_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
