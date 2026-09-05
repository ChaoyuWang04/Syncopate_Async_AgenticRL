"""RL 固定健康闸：必须量本轮真实指标，不能被配置文字或旧 checkpoint 骗绿。"""

from __future__ import annotations

import json
from pathlib import Path

from syncopate.train.rl_run_gate import evaluate, main, metric_values


def _healthy(tmp_path: Path) -> tuple[Path, Path]:
    run = tmp_path / "run"
    run.mkdir()
    (run / "run_purpose.json").write_text(json.dumps({
        "purpose": "smoke", "profile": "smoke", "steps_requested": 2,
    }), encoding="utf-8")
    (run / "global_step_2").mkdir()
    log = tmp_path / "rl.log"
    log.write_text(
        "step:1 - actor/pg_loss:np.float64(-0.02) - actor/grad_norm:np.float64(0.4) "
        "- critic/score/mean:0.2 - training/global_step:1 - timing_s/update_weights:0.8 "
        "- response_length/clip_ratio:0.0\n"
        "step:2 - actor/pg_loss:0.01 - actor/grad_norm:0.2 - critic/score/mean:0.1 "
        "- training/global_step:2 - timing_s/update_weights:0.7 "
        "- response_length/clip_ratio:0.0\n",
        encoding="utf-8",
    )
    return run, log


def test_healthy_smoke_passes(tmp_path):
    run, log = _healthy(tmp_path)
    result = evaluate(run, log, expected_profile="smoke")
    assert result["ok"] is True, result
    assert metric_values(log.read_text(), "actor/pg_loss") == [-0.02, 0.01]


def test_config_key_does_not_fake_weight_sync(tmp_path):
    run, log = _healthy(tmp_path)
    text = log.read_text().replace("timing_s/update_weights:0.8", "update_weights_bucket_megabytes:512")
    text = text.replace("timing_s/update_weights:0.7", "update_weights_bucket_megabytes:512")
    log.write_text(text, encoding="utf-8")
    result = evaluate(run, log, expected_profile="smoke")
    assert result["ok"] is False
    assert result["checks"]["weight_sync_observed"] is False


def test_response_clipping_is_quality_warn_not_health_failure(tmp_path):
    run, log = _healthy(tmp_path)
    log.write_text(log.read_text().replace(
        "response_length/clip_ratio:0.0",
        "response_length/clip_ratio:0.25",
        1,
    ), encoding="utf-8")
    result = evaluate(run, log, expected_profile="smoke")
    assert result["status"] == "warn"
    assert result["health_ok"] is True
    assert result["quality_ready"] is False
    assert result["checks"]["response_not_clipped"] is False
    out = tmp_path / "gate.json"
    assert main([
        "--run-dir", str(run), "--log", str(log), "--profile", "smoke",
        "--out", str(out),
    ]) == 2


def test_nan_zero_reward_and_stale_checkpoint_all_reported(tmp_path):
    run, log = _healthy(tmp_path)
    (run / "global_step_2").rmdir()
    (run / "global_step_1").mkdir()
    log.write_text(
        "actor/pg_loss:nan actor/grad_norm:inf critic/score/mean:0 "
        "training/global_step:1 timing_s/update_weights:0.1\n",
        encoding="utf-8",
    )
    result = evaluate(run, log, expected_profile="smoke")
    assert result["ok"] is False
    assert result["checks"]["loss_finite"] is False
    assert result["checks"]["grad_finite"] is False
    assert result["checks"]["reward_nonzero"] is False
    assert result["checks"]["reported_steps_complete"] is False
    assert result["checks"]["checkpoint_complete"] is False
    assert result["status"] == "fatal"
    out = tmp_path / "gate.json"
    assert main([
        "--run-dir", str(run), "--log", str(log), "--profile", "smoke",
        "--out", str(out),
    ]) == 3
