"""OPD 训练段固定健康闸：真实更新、mask/KL/零泄漏与 final adapter 缺一不可。"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any


def evaluate(log_path: Path, out_dir: Path, *, expected_real_steps: int) -> dict[str, Any]:
    log = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    completion: dict[str, Any] = {}
    completion_error = ""
    try:
        completion = json.loads((out_dir / "completion.json").read_text(encoding="utf-8"))
    except Exception as exc:
        completion_error = repr(exc)
    summary = re.search(
        r"\[opd-summary\] attempted=(\d+) real=(\d+) skipped=(\d+) "
        r"target_real=(\S+) status=(\w+)", log)
    attempted = int(summary.group(1)) if summary else 0
    real = int(summary.group(2)) if summary else 0
    skipped = int(summary.group(3)) if summary else 0
    status = summary.group(5) if summary else "missing"
    masked = [int(x) for x in re.findall(r"\[opd-mask\].*?全局可蒸 token=(\d+)", log)]
    kl_chat = [float(x) for x in re.findall(r"kl_chat/tok=([-+0-9.eE]+)", log)]
    kl_task = [float(x) for x in re.findall(r"kl_task/tok=([-+0-9.eE]+)", log)]
    routes = [(int(a), int(b), int(c), int(d)) for a, b, c, d in re.findall(
        r"\[opd-route\] chat=(\d+) task=(\d+) chat_masked=(\d+) task_masked=(\d+)",
        log,
    )]
    real_routes = [(int(total), int(chat_masked), int(task_masked))
                   for total, chat_masked, task_masked in re.findall(
        r"step \d+ .*?masked=(\d+) \[opd-route\] chat=\d+ task=\d+ "
        r"chat_masked=(\d+) task_masked=(\d+)", log
    )]
    adapter = out_dir / "final"
    prompt = re.search(
        r"\[opd-prompt\] full_menu=(\d+) answer_fields=0 "
        r"history=message_pairs hash=([0-9a-f]{16})", log,
    )
    run_token = re.search(r"\[opd-run\] token=([0-9a-f]{16})\b", log)
    checks = {
        "log_exists": log_path.is_file() and bool(log.strip()),
        "vocab_identity_checked": "[opd-vocab]" in log,
        "sampling_seed_recorded": bool(re.search(
            r"\[opd-seed\] base=-?\d+ rank=\d+ effective=-?\d+", log)),
        "prompt_contract_checked": bool(prompt) and int(prompt.group(1)) > 0,
        "summary_pass": bool(summary) and status == "pass",
        "real_steps_complete": real >= expected_real_steps > 0,
        "nonzero_mask_observed": bool(masked) and any(x > 0 for x in masked),
        "chat_route_observed": any(chat > 0 for chat, _, _, _ in routes),
        "task_route_observed": any(task > 0 for _, task, _, _ in routes),
        "v15_chat_nl_observed": any(chat_masked > 0 for _, _, chat_masked, _ in routes),
        "route_mask_accounted": bool(real_routes)
        and all(total == chat_masked + task_masked
                for total, chat_masked, task_masked in real_routes),
        "kl_finite": bool(kl_chat or kl_task)
        and all(math.isfinite(x) for x in kl_chat + kl_task),
        "zero_mask_control_passed": bool(re.search(r"\[opd-zero\].*对照通过", log)),
        "rank_sync_passed": "[opd-sync]" in log and "权重一致性通过" in log,
        "final_adapter_exists": (adapter / "adapter_config.json").is_file()
        and (adapter / "adapter_model.safetensors").is_file(),
        "completion_marker_matches": not completion_error
        and completion.get("status") == "pass"
        and completion.get("real_steps") == real
        and real >= expected_real_steps
        and bool(run_token)
        and completion.get("run_token") == run_token.group(1)
        and bool(prompt)
        and completion.get("prompt_hash") == prompt.group(2),
        "no_traceback": "Traceback" not in log,
    }
    return {
        "ok": all(checks.values()),
        "log_path": str(log_path),
        "out_dir": str(out_dir),
        "expected_real_steps": expected_real_steps,
        "checks": checks,
        "metrics": {
            "attempted_steps": attempted,
            "real_steps": real,
            "skipped_steps": skipped,
            "masked_tokens": masked,
            "kl_chat_per_token": kl_chat,
            "kl_task_per_token": kl_task,
            "routes": [{"chat": chat, "task": task, "chat_masked": chat_masked,
                        "task_masked": task_masked}
                       for chat, task, chat_masked, task_masked in routes],
            "real_route_masks": [{"total": total, "chat": chat, "task": task}
                                 for total, chat, task in real_routes],
            "summary_status": status,
            "prompt_hash": prompt.group(2) if prompt else None,
            "completion": completion,
            "completion_error": completion_error,
        },
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="检查 OPD 是否完成真实更新并产出有效 final adapter")
    ap.add_argument("--log", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--expected-real-steps", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    result = evaluate(args.log, args.out_dir, expected_real_steps=args.expected_real_steps)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    for name, passed in result["checks"].items():
        print(f"  {'✅' if passed else '🔴'} {name}")
    print(f"[opd-run-gate] {'PASS' if result['ok'] else 'FATAL'} -> {args.out}")
    return 0 if result["ok"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
