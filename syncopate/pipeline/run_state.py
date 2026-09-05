"""固定管线的最小运行账本。

它不判断模型质量，只把每个 stage 的真实退出结果写进同一份 manifest，避免
把“命令打印过”“旧目录还在”误当成本轮通过。
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

STATUSES = {"pass", "warn", "block_next", "fatal"}


def stage_is_resumable(
    manifest: Path,
    *,
    run_id: str,
    profile: str,
    gate_mode: str,
    stage: str,
) -> bool:
    """Return whether an exact-run resume can reuse this completed stage.

    PASS is always reusable.  WARN is reusable only in observe mode: it means
    the diagnostic stage completed and the warning must remain visible, not
    that rerunning unchanged inputs can turn it green.  Fatal/blocking stages
    are never reusable.
    """
    if not manifest.exists():
        return False
    data = json.loads(manifest.read_text(encoding="utf-8"))
    identity = (data.get("run_id"), data.get("profile"), data.get("gate_mode"))
    expected = (run_id, profile, gate_mode)
    if identity != expected:
        raise ValueError(f"run manifest identity mismatch: {identity!r} != {expected!r}")
    status = data.get("stages", {}).get(stage, {}).get("status")
    return status == "pass" or (gate_mode == "observe" and status == "warn")


# Compatibility for callers that imported the original name.  Its semantics
# remain literal PASS-only; the CLI resume path uses stage_is_resumable.
def stage_is_passed(
    manifest: Path,
    *,
    run_id: str,
    profile: str,
    gate_mode: str,
    stage: str,
) -> bool:
    if not manifest.exists():
        return False
    data = json.loads(manifest.read_text(encoding="utf-8"))
    identity = (data.get("run_id"), data.get("profile"), data.get("gate_mode"))
    expected = (run_id, profile, gate_mode)
    if identity != expected:
        raise ValueError(f"run manifest identity mismatch: {identity!r} != {expected!r}")
    return data.get("stages", {}).get(stage, {}).get("status") == "pass"


def record_stage(
    manifest: Path,
    *,
    run_id: str,
    profile: str,
    gate_mode: str,
    stage: str,
    status: str,
    returncode: int,
) -> dict:
    if status not in STATUSES:
        raise ValueError(f"unknown stage status: {status}")
    if manifest.exists():
        data = json.loads(manifest.read_text(encoding="utf-8"))
        identity = (data.get("run_id"), data.get("profile"), data.get("gate_mode"))
        expected = (run_id, profile, gate_mode)
        if identity != expected:
            raise ValueError(f"run manifest identity mismatch: {identity!r} != {expected!r}")
    else:
        data = {
            "schema_version": 1,
            "run_id": run_id,
            "profile": profile,
            "gate_mode": gate_mode,
            "stages": {},
        }
    now = datetime.now(timezone.utc).isoformat()
    attempt = {"status": status, "returncode": int(returncode), "finished_at": now}
    slot = data["stages"].setdefault(stage, {"attempts": []})
    slot["attempts"].append(attempt)
    slot.update(attempt)
    data["updated_at"] = now
    data["pipeline_ok"] = not any(
        item.get("status") in {"block_next", "fatal"}
        for item in data["stages"].values()
    )
    # observe 会让诊断继续跑完，但 WARN 绝不能在总结果里伪装成“全绿”。
    data["all_passed"] = bool(data["stages"]) and all(
        item.get("status") == "pass" for item in data["stages"].values())
    manifest.parent.mkdir(parents=True, exist_ok=True)
    tmp = manifest.with_suffix(manifest.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    tmp.replace(manifest)
    return data


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="记录 v16 固定管线 stage 结果")
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--profile", required=True, choices=["smoke", "candidate"])
    ap.add_argument("--gate-mode", required=True, choices=["observe", "strict"])
    ap.add_argument("--stage", required=True)
    ap.add_argument("--check-passed", action="store_true")
    ap.add_argument("--check-resumable", action="store_true")
    ap.add_argument("--status", choices=sorted(STATUSES))
    ap.add_argument("--returncode", type=int)
    args = ap.parse_args(argv)
    if args.check_passed or args.check_resumable:
        try:
            check = stage_is_resumable if args.check_resumable else stage_is_passed
            passed = check(
                args.manifest,
                run_id=args.run_id,
                profile=args.profile,
                gate_mode=args.gate_mode,
                stage=args.stage,
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"🔴 无法复用运行账本: {exc}")
            return 2
        return 0 if passed else 1
    if args.status is None or args.returncode is None:
        ap.error("记录模式必须同时提供 --status 和 --returncode")
    record_stage(
        args.manifest,
        run_id=args.run_id,
        profile=args.profile,
        gate_mode=args.gate_mode,
        stage=args.stage,
        status=args.status,
        returncode=args.returncode,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
