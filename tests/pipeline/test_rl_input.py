from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from syncopate.pipeline.rl_input import bind_input, resolve_input, validate_request

ROOT = Path(__file__).resolve().parents[2]


def cli_env():
    return {**os.environ, "PYTHONPATH": str(ROOT), "PY": sys.executable,
            "SYNCOPATE_CONTRACT": "v15", "SYNCOPATE_THINK": "1"}


def fixture(root: Path):
    from syncopate.core.model_paths import STUDENT_MODEL
    from syncopate.pipeline.split import DATA_VERSION

    manifest = root / "_audit" / DATA_VERSION / "runs" / "source" / "manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"run_id": "source", "profile": "smoke", "all_passed": False,
                                   "stages": {"merge": {"status": "pass"}, "exam": {"status": "warn"}}}))
    model = root / "models" / f"{Path(STUDENT_MODEL).name}-sft-{DATA_VERSION}_smoke_source"
    model.mkdir(parents=True)
    (model / "config.json").write_text("{}")
    (model / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"weight": "part.safetensors"}}))
    (model / "part.safetensors").write_bytes(b"1234")
    return manifest, model


def test_explicit_input_is_not_new_sft_and_quality_warn_does_not_block(tmp_path):
    fixture(tmp_path)
    record = bind_input(tmp_path, "source", "new", hash_weights=True)
    assert record["kind"] == "external_sft_for_rl_diagnostic"
    assert record["content_sha256"]["part.safetensors"]
    assert bind_input(tmp_path, "source", "new") == record
    assert not (tmp_path / "_audit/v16/runs/new/manifest.json").exists()


def test_cpu_hashing_required(tmp_path):
    fixture(tmp_path)
    with pytest.raises(ValueError, match="CPU"):
        bind_input(tmp_path, "source", "new")


@pytest.mark.parametrize("stage,profile", [("all", "smoke"), ("train-all", "smoke"), ("rl-train", "candidate")])
def test_invalid_request_rejected_before_cloud_allocation(stage, profile):
    with pytest.raises(ValueError):
        validate_request(stage, profile, "new", "source")


def test_same_size_corrupt_weight_is_caught_by_cpu_recheck(tmp_path):
    _, model = fixture(tmp_path)
    bind_input(tmp_path, "source", "new", hash_weights=True)
    (model / "part.safetensors").write_bytes(b"5678")
    with pytest.raises(ValueError, match="内容改变"):
        bind_input(tmp_path, "source", "new", hash_weights=True)


@pytest.mark.parametrize("kind", ["wrong_run", "unmerged", "missing_shard", "escape"])
def test_bad_inputs_fail(tmp_path, kind):
    manifest, model = fixture(tmp_path)
    data = json.loads(manifest.read_text())
    if kind == "wrong_run":
        data["run_id"] = "other"
    elif kind == "unmerged":
        data["stages"]["merge"]["status"] = "fatal"
    elif kind == "missing_shard":
        (model / "part.safetensors").unlink()
    else:
        (model / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"w": "../other"}}))
    manifest.write_text(json.dumps(data))
    with pytest.raises((ValueError, FileNotFoundError)):
        resolve_input(tmp_path, "source", "new")


@pytest.mark.parametrize("model_only", [False, True])
def test_fresh_cli_stdout_is_only_machine_readable_result(tmp_path, model_only):
    _, model = fixture(tmp_path)
    command = [sys.executable, "-m", "syncopate.pipeline.rl_input", "--input-run", "source",
               "--run-id", "new", "--hash-weights"]
    if model_only:
        command.append("--model-only")
    result = subprocess.run(command, cwd=tmp_path, env=cli_env(), capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    expected = str(model.relative_to(tmp_path))
    if model_only:
        assert result.stdout == expected + "\n"
    else:
        assert json.loads(result.stdout)["model"] == expected
    assert "[contract]" in result.stderr


@pytest.mark.parametrize("missing", [None, "train", "val"])
def test_real_runbook_resolves_external_input_without_starting_training(tmp_path, missing):
    from syncopate.pipeline.split import DEFAULT_RL_DIR
    fixture(tmp_path)
    bind_input(tmp_path, "source", "new", hash_weights=True)
    runbook = tmp_path / "scripts/v16_pipeline.sh"
    runbook.parent.mkdir()
    runbook.write_text((ROOT / "scripts/v16_pipeline.sh").read_text())
    for name in ("train", "val"):
        if name != missing:
            path = tmp_path / DEFAULT_RL_DIR / f"{name}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"existence fixture; schema is checked separately")
    result = subprocess.run(["bash", str(runbook), "--check-inputs", "--profile", "smoke",
                             "--run-id", "new", "--rl-input-run", "source", "rl-train"],
                            cwd=tmp_path, env=cli_env(), capture_output=True, text=True)
    assert result.returncode == (1 if missing else 0), result.stdout + result.stderr
    assert "launch_rl_v1" not in result.stdout
    assert not (tmp_path / "checkpoints").exists()
    assert not (tmp_path / "_audit/v16/runs/new/manifest.json").exists()
    if missing:
        assert f"{missing}.parquet" in result.stdout
    else:
        assert "[inputs] RL 输入可读；未启动训练" in result.stdout
