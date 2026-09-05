"""RL 训练段的固定健康闸与 smoke 质量告警。

只看本轮 ``run_purpose.json``、本轮日志和本轮 checkpoint，避免“进程退出 0”或
旧目录残留被误报成训练成功。健康失败返回 3，任何模式都停止；可继续诊断的
response 截断返回 2，交给 runbook 的 observe/strict 语义决定是否继续。
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

_NUMBER = r"[-+]?(?:nan|inf(?:inity)?|(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"


def metric_values(log: str, key: str) -> list[float]:
    """读取 verl 文本指标；兼容 ``key:1.2`` 与 ``key:np.float64(1.2)``。"""
    pattern = re.compile(
        rf"{re.escape(key)}['\"]?\s*[:=]\s*(?:np\.float(?:16|32|64)\()?\s*({_NUMBER})",
        re.IGNORECASE,
    )
    return [float(match.group(1)) for match in pattern.finditer(log)]


def evaluate(run_dir: Path, log_path: Path, *, expected_profile: str) -> dict[str, Any]:
    log = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    purpose: dict[str, Any] = {}
    purpose_error = ""
    try:
        purpose = json.loads((run_dir / "run_purpose.json").read_text(encoding="utf-8"))
    except Exception as exc:  # 缺文件和坏 JSON 都必须落进完整报告，不能第一项就短路
        purpose_error = repr(exc)

    losses = metric_values(log, "actor/pg_loss")
    grad_norms = metric_values(log, "actor/grad_norm")
    rewards = metric_values(log, "critic/score/mean")
    global_steps = [int(x) for x in metric_values(log, "training/global_step")
                    if math.isfinite(x) and x >= 0]
    sync_seconds = metric_values(log, "timing_s/update_weights")
    response_clip_ratios = metric_values(log, "response_length/clip_ratio")
    checkpoints = sorted(
        (int(match.group(1)), path)
        for path in run_dir.glob("global_step_*")
        if (match := re.fullmatch(r"global_step_(\d+)", path.name)) and path.is_dir()
    )
    requested = purpose.get("steps_requested")
    known_wandb_exit = (
        "Exception ignored in atexit callback" in log and "BrokenPipeError" in log)

    health_checks = {
        "log_exists": log_path.is_file() and bool(log.strip()),
        "purpose_readable": not purpose_error,
        "profile_matches": purpose.get("purpose") == expected_profile
        and purpose.get("profile") == expected_profile,
        "steps_declared": isinstance(requested, int) and requested > 0,
        "loss_finite": bool(losses) and all(math.isfinite(x) for x in losses),
        "grad_finite": bool(grad_norms) and all(math.isfinite(x) for x in grad_norms),
        "reward_nonzero": bool(rewards) and all(math.isfinite(x) for x in rewards)
        and any(x != 0.0 for x in rewards),
        # 量真实 step 指标，不用配置里的 update_weights_bucket_megabytes 冒充接线证据。
        "weight_sync_observed": bool(sync_seconds)
        and all(math.isfinite(x) and x >= 0.0 for x in sync_seconds),
        "reported_steps_complete": isinstance(requested, int) and bool(global_steps)
        and max(global_steps) >= requested,
        "checkpoint_complete": isinstance(requested, int) and bool(checkpoints)
        and checkpoints[-1][0] >= requested,
        "no_unexpected_traceback": "Traceback" not in log or known_wandb_exit,
    }
    clip_metric_valid = bool(response_clip_ratios) and all(
        math.isfinite(x) and 0.0 <= x <= 1.0 for x in response_clip_ratios
    )
    quality_checks = {
        "response_clipping_measured": clip_metric_valid,
        "response_not_clipped": clip_metric_valid
        and all(x == 0.0 for x in response_clip_ratios),
    }
    health_ok = all(health_checks.values())
    quality_ready = all(quality_checks.values())
    status = "fatal" if not health_ok else ("warn" if not quality_ready else "pass")
    return {
        "ok": health_ok and quality_ready,
        "status": status,
        "health_ok": health_ok,
        "quality_ready": quality_ready,
        "expected_profile": expected_profile,
        "run_dir": str(run_dir),
        "log_path": str(log_path),
        "purpose": purpose,
        "purpose_error": purpose_error,
        "checks": {**health_checks, **quality_checks},
        "health_checks": health_checks,
        "quality_checks": quality_checks,
        "metrics": {
            "pg_loss": losses,
            "grad_norm": grad_norms,
            "score_mean": rewards,
            "global_steps": global_steps,
            "update_weights_seconds": sync_seconds,
            "response_clip_ratios": response_clip_ratios,
            "checkpoint_steps": [step for step, _ in checkpoints],
        },
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="检查本轮 RL 是否真实完成训练、同步和存档")
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--log", type=Path, required=True)
    ap.add_argument("--profile", choices=["smoke", "candidate"], required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    result = evaluate(args.run_dir, args.log, expected_profile=args.profile)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    for name, passed in result["checks"].items():
        print(f"  {'✅' if passed else '🔴'} {name}")
    print(f"[rl-run-gate] {result['status'].upper()} -> {args.out}")
    if not result["health_ok"]:
        return 3
    return 0 if result["quality_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
