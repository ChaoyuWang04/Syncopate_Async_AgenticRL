#!/usr/bin/env python
"""Validate one v16 Exam run without importing the retired R5/R6/R7 gate table.

The fixed pipeline has two separate questions:

* did the real Runtime -> model -> judge path finish and leave complete evidence;
* is the model good enough to promote.

Smoke answers the first question and records the second as a diagnostic warning.  A
candidate run is blocked until a candidate policy has been frozen explicitly; an
old v15 table is never used as an implicit policy.

Exit codes are part of the runbook contract:

* 0: evidence is complete and the registered quality policy passes;
* 1: broken/missing evidence (fatal pipeline error);
* 2: evidence is complete, but quality is not ready (WARN in observe, BLOCK in strict).
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
from pathlib import Path
from typing import Any

from syncopate.core.contract import n1_hits


TERMINAL = {"succeeded", "failed", "cancelled", "waiting_for_user"}


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _paths(pattern: str) -> list[Path]:
    return [Path(p) for p in sorted(glob.glob(pattern))]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _served_ids(models_json: Path) -> set[str]:
    try:
        data = json.loads(models_json.read_text(encoding="utf-8"))
        return {str(row["id"]) for row in data.get("data", []) if "id" in row}
    except (OSError, json.JSONDecodeError, TypeError, KeyError):
        return set()


def _candidate_quality(
    policy_path: Path | None,
    *,
    spec_path: Path,
    expected_passes: int,
    level_totals: dict[str, dict[str, int]],
    n1_rate: float,
    judge_failures: int,
) -> tuple[bool, dict[str, bool], list[str]]:
    """Apply only an explicit, identity-bound candidate policy.

    The current project has deliberately not frozen this file yet.  Supporting the
    small schema here makes the eventual hand-off explicit without inventing today's
    thresholds in code.
    """
    checks: dict[str, bool] = {"candidate_policy_registered": bool(policy_path)}
    gaps: list[str] = []
    if policy_path is None:
        return False, checks, ["candidate 质量门槛尚未冻结；禁止沿用旧 R5/R6/R7 表"]
    try:
        policy = json.loads(policy_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        checks["candidate_policy_readable"] = False
        return False, checks, [f"candidate policy 读不到：{exc}"]

    checks["candidate_policy_readable"] = True
    checks["candidate_policy_schema"] = policy.get("schema_version") == 1
    checks["candidate_policy_profile"] = policy.get("profile") == "candidate"
    checks["candidate_policy_spec_identity"] = (
        policy.get("exam_spec_sha256") == _sha256(spec_path)
    )
    checks["candidate_policy_passes"] = policy.get("passes") == expected_passes
    mins = policy.get("min_level_rates")
    checks["candidate_policy_has_levels"] = isinstance(mins, dict) and bool(mins)
    checks["candidate_policy_has_n1"] = isinstance(policy.get("max_n1_rate"), (int, float))
    checks["candidate_policy_has_fail_limit"] = isinstance(
        policy.get("max_judge_failures"), int
    )
    for name, ok in checks.items():
        if not ok:
            gaps.append(name)

    if gaps:
        return False, checks, gaps

    for level, minimum in mins.items():
        total = level_totals.get(level, {})
        n = total.get("n", 0)
        rate = total.get("pass", 0) / n if n else 0.0
        ok = n > 0 and rate >= float(minimum)
        checks[f"level:{level}"] = ok
        if not ok:
            gaps.append(f"{level}={rate:.1%} < {float(minimum):.1%}（或无读数）")
    max_n1 = float(policy["max_n1_rate"])
    checks["n1_rate"] = n1_rate <= max_n1
    if not checks["n1_rate"]:
        gaps.append(f"N1 机器语法命中率 {n1_rate:.1%} > {max_n1:.1%}")
    max_fails = int(policy["max_judge_failures"])
    checks["judge_failures"] = judge_failures <= max_fails
    if not checks["judge_failures"]:
        gaps.append(f"判卷失败 {judge_failures} > {max_fails}")
    return not gaps, checks, gaps


def evaluate(
    *,
    raw_pattern: str,
    judged_pattern: str,
    spec_path: Path,
    models_json: Path,
    served_model: str,
    model_path: str,
    profile: str,
    expected_passes: int,
    limit: int = 0,
    candidate_policy: Path | None = None,
) -> dict[str, Any]:
    raw_paths = _paths(raw_pattern)
    judged_paths = _paths(judged_pattern)
    spec_rows = _jsonl(spec_path) if spec_path.is_file() else []
    expected_rows = spec_rows[:limit] if limit else spec_rows
    expected_ids = {str(row.get("id")) for row in expected_rows}
    expected_items = len(expected_rows)

    raw_runs: list[list[dict[str, Any]]] = []
    judged_runs: list[dict[str, Any]] = []
    read_errors: list[str] = []
    for path in raw_paths:
        try:
            raw_runs.append(_jsonl(path))
        except (OSError, json.JSONDecodeError) as exc:
            read_errors.append(f"{path}: {exc}")
    for path in judged_paths:
        try:
            judged_runs.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError) as exc:
            read_errors.append(f"{path}: {exc}")

    raw_counts = [len(rows) for rows in raw_runs]
    judged_counts = [sum(int(v.get("n", 0)) for v in run.get("levels", {}).values())
                     for run in judged_runs]
    all_raw_rows = [row for rows in raw_runs for row in rows]
    all_turns = [row.get("turns") or [] for row in all_raw_rows]

    checks = {
        "spec_exists": spec_path.is_file(),
        "spec_nonempty": expected_items > 0,
        "raw_file_count": len(raw_paths) == expected_passes,
        "judged_file_count": len(judged_paths) == expected_passes,
        "files_readable": not read_errors
                          and len(raw_runs) == len(raw_paths)
                          and len(judged_runs) == len(judged_paths),
        "raw_item_counts": bool(raw_counts) and all(n == expected_items for n in raw_counts),
        "judged_item_counts": bool(judged_counts)
                             and all(n == expected_items for n in judged_counts),
        "raw_ids_unique": bool(raw_runs) and all(
            len({str(row.get("id")) for row in rows}) == len(rows) for rows in raw_runs
        ),
        "raw_ids_match_spec": bool(raw_runs) and all(
            {str(row.get("id")) for row in rows} == expected_ids for rows in raw_runs
        ),
        "turns_present": bool(all_turns) and all(turns for turns in all_turns),
        "terminal_status_present": bool(all_turns) and all(
            turns and turns[-1].get("status") in TERMINAL for turns in all_turns
        ),
        "behavior_present": bool(all_turns) and all(
            turns and bool(turns[-1].get("behavior")) for turns in all_turns
        ),
        "served_model_identity": served_model in _served_ids(models_json),
        "model_path_recorded": bool(model_path),
    }
    operational_gaps = [name for name, ok in checks.items() if not ok]
    operational_gaps.extend(read_errors)
    operational_ok = not operational_gaps

    level_totals: dict[str, dict[str, int]] = {}
    judge_failures = 0
    for run in judged_runs:
        judge_failures += len(run.get("fails") or [])
        for level, value in (run.get("levels") or {}).items():
            dst = level_totals.setdefault(level, {"pass": 0, "n": 0})
            dst["pass"] += int(value.get("pass", 0))
            dst["n"] += int(value.get("n", 0))
    final_replies = [str(turns[-1].get("reply") or "") for turns in all_turns if turns]
    n1_count = sum(bool(n1_hits(reply)) for reply in final_replies)
    n1_rate = n1_count / len(final_replies) if final_replies else 0.0
    think_count = sum(int((turns[-1].get("think_nonempty") or 0) > 0)
                      for turns in all_turns if turns)
    quality_gaps: list[str] = []
    quality_checks: dict[str, bool]
    if profile == "candidate":
        quality_ready, quality_checks, quality_gaps = _candidate_quality(
            candidate_policy,
            spec_path=spec_path,
            expected_passes=expected_passes,
            level_totals=level_totals,
            n1_rate=n1_rate,
            judge_failures=judge_failures,
        )
    else:
        # Smoke does not promote a model.  These two strict diagnostics make quality
        # debt visible without pretending that a 40-item, one-pass sample is a
        # statistically frozen candidate gate.
        quality_checks = {
            "diagnostic_all_items_pass": judge_failures == 0,
            "diagnostic_n1_clean": n1_count == 0,
        }
        if judge_failures:
            quality_gaps.append(f"40 题诊断中有 {judge_failures} 个判卷失败")
        if n1_count:
            quality_gaps.append(
                f"终答机器语法命中 {n1_count}/{len(final_replies)}（N1 未净化）"
            )
        quality_ready = not quality_gaps

    if not operational_ok:
        status = "fatal"
    elif not quality_ready:
        status = "warn" if profile == "smoke" else "block_next"
    else:
        status = "pass"
    return {
        "schema_version": 1,
        "status": status,
        "operational_ok": operational_ok,
        "quality_ready": quality_ready,
        "profile": profile,
        "model": {"path": model_path, "served_name": served_model},
        "exam": {
            "spec": str(spec_path),
            "spec_sha256": _sha256(spec_path) if spec_path.is_file() else None,
            "limit": limit,
            "expected_passes": expected_passes,
            "expected_items_per_pass": expected_items,
            "raw_files": [str(p) for p in raw_paths],
            "judged_files": [str(p) for p in judged_paths],
        },
        "checks": checks,
        "operational_gaps": operational_gaps,
        "quality_checks": quality_checks,
        "quality_gaps": quality_gaps,
        "metrics": {
            "raw_item_counts": raw_counts,
            "judged_item_counts": judged_counts,
            "judge_failures": judge_failures,
            "n1_hits": n1_count,
            "n1_rate": n1_rate,
            "think_nonempty": think_count,
            "level_totals": level_totals,
        },
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="v16 Exam 本轮完整性与质量分离门禁")
    ap.add_argument("--raw", required=True, help="本轮 raw jsonl glob")
    ap.add_argument("--judged", required=True, help="本轮 judged json glob")
    ap.add_argument("--exam-spec", type=Path, required=True)
    ap.add_argument("--models-json", type=Path, required=True)
    ap.add_argument("--served-model", required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--profile", choices=("smoke", "candidate"), required=True)
    ap.add_argument("--expected-passes", type=int, required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--candidate-policy", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    result = evaluate(
        raw_pattern=args.raw,
        judged_pattern=args.judged,
        spec_path=args.exam_spec,
        models_json=args.models_json,
        served_model=args.served_model,
        model_path=args.model_path,
        profile=args.profile,
        expected_passes=args.expected_passes,
        limit=args.limit,
        candidate_policy=args.candidate_policy,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    print(f"[exam-run-gate] status={result['status']} operational_ok="
          f"{result['operational_ok']} quality_ready={result['quality_ready']}")
    for gap in result["operational_gaps"]:
        print(f"  🔴 证据/链路：{gap}")
    for gap in result["quality_gaps"]:
        print(f"  🟡 质量：{gap}")
    print(f"[exam-run-gate] -> {args.out}")
    if not result["operational_ok"]:
        return 1
    return 0 if result["quality_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
