from __future__ import annotations

import json

import pytest

from syncopate.pipeline.run_state import record_stage, stage_is_passed, stage_is_resumable


def test_manifest_keeps_stage_attempts_and_blocks_pipeline(tmp_path):
    manifest = tmp_path / "manifest.json"
    record_stage(manifest, run_id="r1", profile="smoke", gate_mode="observe",
                 stage="sft-train", status="pass", returncode=0)
    record_stage(manifest, run_id="r1", profile="smoke", gate_mode="observe",
                 stage="exam", status="warn", returncode=10)
    data = json.loads(manifest.read_text(encoding="utf-8"))
    assert data["pipeline_ok"] is True
    assert data["all_passed"] is False
    assert data["stages"]["exam"]["status"] == "warn"

    record_stage(manifest, run_id="r1", profile="smoke", gate_mode="observe",
                 stage="opd-train", status="fatal", returncode=2)
    data = json.loads(manifest.read_text(encoding="utf-8"))
    assert data["pipeline_ok"] is False
    assert data["stages"]["opd-train"]["returncode"] == 2


def test_manifest_refuses_identity_reuse(tmp_path):
    manifest = tmp_path / "manifest.json"
    record_stage(manifest, run_id="r1", profile="smoke", gate_mode="observe",
                 stage="sft-train", status="pass", returncode=0)
    with pytest.raises(ValueError, match="identity mismatch"):
        record_stage(manifest, run_id="r1", profile="candidate", gate_mode="strict",
                     stage="sft-train", status="pass", returncode=0)


def test_manifest_is_all_passed_only_when_every_current_stage_passes(tmp_path):
    manifest = tmp_path / "manifest.json"
    data = record_stage(manifest, run_id="r1", profile="smoke", gate_mode="observe",
                        stage="sft-train", status="pass", returncode=0)
    assert data["pipeline_ok"] is True and data["all_passed"] is True


def test_resume_reuses_only_passed_stage_with_same_identity(tmp_path):
    manifest = tmp_path / "manifest.json"
    record_stage(manifest, run_id="r1", profile="smoke", gate_mode="observe",
                 stage="sft-train", status="pass", returncode=0)
    record_stage(manifest, run_id="r1", profile="smoke", gate_mode="observe",
                 stage="merge", status="fatal", returncode=1)
    assert stage_is_passed(manifest, run_id="r1", profile="smoke", gate_mode="observe",
                           stage="sft-train") is True
    assert stage_is_passed(manifest, run_id="r1", profile="smoke", gate_mode="observe",
                           stage="merge") is False
    with pytest.raises(ValueError, match="identity mismatch"):
        stage_is_passed(manifest, run_id="r1", profile="candidate", gate_mode="strict",
                        stage="sft-train")


def test_observe_resume_reuses_warn_without_turning_it_green(tmp_path):
    manifest = tmp_path / "manifest.json"
    record_stage(manifest, run_id="r1", profile="smoke", gate_mode="observe",
                 stage="exam", status="warn", returncode=10)
    assert stage_is_passed(manifest, run_id="r1", profile="smoke", gate_mode="observe",
                           stage="exam") is False
    assert stage_is_resumable(manifest, run_id="r1", profile="smoke", gate_mode="observe",
                              stage="exam") is True
    data = json.loads(manifest.read_text(encoding="utf-8"))
    assert data["stages"]["exam"]["status"] == "warn"
    assert data["all_passed"] is False


def test_strict_resume_never_reuses_warn(tmp_path):
    manifest = tmp_path / "manifest.json"
    record_stage(manifest, run_id="r1", profile="candidate", gate_mode="strict",
                 stage="exam", status="warn", returncode=10)
    assert stage_is_resumable(manifest, run_id="r1", profile="candidate", gate_mode="strict",
                              stage="exam") is False
