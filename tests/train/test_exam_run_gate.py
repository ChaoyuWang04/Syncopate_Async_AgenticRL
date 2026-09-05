from __future__ import annotations

import hashlib
import json
from pathlib import Path

from syncopate.evaluation.exam_run import resolved_behavior
from syncopate.evaluation.exam_run_gate import evaluate


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                    encoding="utf-8")


def _fixture(tmp_path: Path, *, dirty_reply: bool = False):
    spec = tmp_path / "context_v4_exam.jsonl"
    _write_jsonl(spec, [{"id": "L1_00"}, {"id": "L1_01"}])
    raw = tmp_path / "run_arm_r1_context_v4.jsonl"
    _write_jsonl(raw, [
        {"id": "L1_00", "turns": [{"status": "succeeded", "behavior": "answer",
                                      "reply": '{"reply": "污染"}' if dirty_reply else "回答一"}]},
        {"id": "L1_01", "turns": [{"status": "waiting_for_user", "behavior": "clarify",
                                      "reply": "请补充信息"}]},
    ])
    judged = tmp_path / "judged_arm_r1_context_v4.jsonl"
    judged.write_text(json.dumps({
        "file": str(raw),
        "levels": {"L1-iv": {"pass": 1, "n": 1}, "L1-oov": {"pass": 1, "n": 1}},
        "fails": [],
    }), encoding="utf-8")
    models = tmp_path / "models.json"
    models.write_text(json.dumps({"data": [{"id": "v16_arm"}]}), encoding="utf-8")
    return spec, raw, judged, models


def _evaluate(tmp_path: Path, **overrides):
    spec, raw, judged, models = _fixture(
        tmp_path, dirty_reply=overrides.pop("dirty_reply", False)
    )
    args = dict(
        raw_pattern=str(raw),
        judged_pattern=str(judged),
        spec_path=spec,
        models_json=models,
        served_model="v16_arm",
        model_path="models/student-sft-v16_run",
        profile="smoke",
        expected_passes=1,
        limit=0,
        candidate_policy=None,
    )
    args.update(overrides)
    return evaluate(**args), (spec, raw, judged, models)


def test_complete_smoke_evidence_passes_without_old_release_table(tmp_path):
    result, _ = _evaluate(tmp_path)
    assert result["status"] == "pass"
    assert result["operational_ok"] is True
    assert result["quality_ready"] is True
    assert all(result["checks"].values())


def test_approval_pause_is_a_tool_call_but_failures_stay_unclassified():
    proposal = {"tool": "memory.write_proposal", "params": {"lane": "semantic"}}
    assert resolved_behavior(
        None, status="waiting_for_user", proposal=proposal
    ) == "tool_call"
    assert resolved_behavior(None, status="failed", proposal=None) is None


def test_smoke_quality_debt_is_warn_not_broken_pipeline(tmp_path):
    result, _ = _evaluate(tmp_path, dirty_reply=True)
    assert result["status"] == "warn"
    assert result["operational_ok"] is True
    assert result["quality_ready"] is False
    assert result["metrics"]["n1_hits"] == 1


def test_missing_or_partial_run_is_fatal_not_observe_warning(tmp_path):
    result, (_, raw, _, _) = _evaluate(tmp_path)
    raw.unlink()
    result = evaluate(
        raw_pattern=str(raw),
        judged_pattern=result["exam"]["judged_files"][0],
        spec_path=Path(result["exam"]["spec"]),
        models_json=tmp_path / "models.json",
        served_model="v16_arm",
        model_path="models/student-sft-v16_run",
        profile="smoke",
        expected_passes=1,
    )
    assert result["status"] == "fatal"
    assert result["operational_ok"] is False
    assert result["checks"]["raw_file_count"] is False


def test_candidate_refuses_to_borrow_historical_thresholds(tmp_path):
    result, _ = _evaluate(tmp_path, profile="candidate")
    assert result["status"] == "block_next"
    assert result["operational_ok"] is True
    assert result["quality_checks"]["candidate_policy_registered"] is False
    assert any("旧 R5/R6/R7" in gap for gap in result["quality_gaps"])


def test_candidate_policy_is_bound_to_exam_identity(tmp_path):
    result, (spec, _, _, _) = _evaluate(tmp_path)
    policy = tmp_path / "candidate_policy.json"
    policy.write_text(json.dumps({
        "schema_version": 1,
        "profile": "candidate",
        "exam_spec_sha256": hashlib.sha256(spec.read_bytes()).hexdigest(),
        "passes": 1,
        "min_level_rates": {"L1-iv": 1.0, "L1-oov": 1.0},
        "max_n1_rate": 0.0,
        "max_judge_failures": 0,
    }), encoding="utf-8")
    result, _ = _evaluate(tmp_path, profile="candidate", candidate_policy=policy)
    assert result["status"] == "pass"
    assert result["quality_ready"] is True
