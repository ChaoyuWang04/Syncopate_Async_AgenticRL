"""RL 固定健康闸：必须量本轮真实指标，不能被配置文字或旧 checkpoint 骗绿。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from syncopate.train.rl_run_gate import evaluate, main, metric_values


def _healthy(tmp_path: Path) -> tuple[Path, Path]:
    run = tmp_path / "run"
    run.mkdir()
    (run / "run_purpose.json").write_text(json.dumps({
        "purpose": "smoke", "profile": "smoke", "steps_requested": 2,
    }), encoding="utf-8")
    (run / "launch_config.json").write_text(json.dumps({"overrides": [
        "trainer.v1.trainer_mode=sync", "trainer.total_training_steps=2",
        "data.train_batch_size=1", "actor_rollout_ref.rollout.n=1",
        "data.max_response_length=12",
    ]}), encoding="utf-8")
    for index in range(2):
        artifact = run / "artifacts" / str(index)
        artifact.mkdir(parents=True)
        (artifact / "rollout.json").write_text(json.dumps({
            "token_trace": {
                "schema_version": 3, "response_ids": [5, 6, 7],
                "response_mask": [1, 1, 0], "response_logprobs": [-1., -2., 0.],
                "segments": [{"type": "assistant", "token_count": 2, "mask": 1},
                             {"type": "assistant_template", "token_count": 1, "mask": 0}],
                "generations": [{"response_offset": 0, "retained_model_tokens": 2,
                                 "raw_token_ids": [5, 6], "finish_reason": "stop",
                                 "stop_reason": None, "discarded_model_tokens": 0,
                                 "remaining_response_budget": 12}],
            },
            "metrics": {"placeholder_logprobs": 0, "truncated": False,
                        "truncation_reason": None},
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


def test_padding_clip_metric_cannot_report_actual_token_clipping(tmp_path):
    run, log = _healthy(tmp_path)
    log.write_text(log.read_text().replace(
        "response_length/clip_ratio:0.0",
        "response_length/clip_ratio:0.25",
        1,
    ), encoding="utf-8")
    result = evaluate(run, log, expected_profile="smoke")
    assert result["ok"] is True, result
    assert result["metrics"]["response_clip_ratios"] == [0.25, 0.0]
    assert result["termination"]["token_limited_trajectories"] == 0


@pytest.mark.parametrize("kind", ["length", "dropped", "remaining_budget"])
def test_real_clipping_is_quality_warn_even_when_padding_metric_is_zero(tmp_path, kind):
    run, log = _healthy(tmp_path)
    path = run / "artifacts/0/rollout.json"
    data = json.loads(path.read_text())
    data["metrics"].update(truncated=True, truncation_reason="tokens")
    generation = data["token_trace"]["generations"][0]
    if kind == "length":
        generation["finish_reason"] = "length"
    elif kind == "dropped":
        generation["raw_token_ids"].append(99)
        generation["discarded_model_tokens"] = 1
    path.write_text(json.dumps(data))
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


@pytest.mark.parametrize("reason", ["turns", "observation"])
def test_other_limits_are_not_counted_as_model_token_clipping(tmp_path, reason):
    run, log = _healthy(tmp_path)
    path = run / "artifacts/0/rollout.json"
    data = json.loads(path.read_text())
    data["metrics"].update(truncated=True, truncation_reason=reason)
    path.write_text(json.dumps(data))
    result = evaluate(run, log, expected_profile="smoke")
    assert result["status"] == "warn"
    assert result["health_ok"] is True
    assert result["checks"]["response_not_clipped"] is True
    assert result["termination"]["trajectory_limits"][reason] == 1


@pytest.mark.parametrize("missing", ["artifact", "launch_config", "async_coverage"])
def test_missing_coverage_is_never_a_green_zero_clipping_rate(tmp_path, missing):
    run, log = _healthy(tmp_path)
    if missing == "artifact":
        (run / "artifacts/0/rollout.json").unlink()
    elif missing == "launch_config":
        (run / "launch_config.json").unlink()
    else:
        path = run / "launch_config.json"
        path.write_text(path.read_text().replace("=sync", "=colocate_async"))
    result = evaluate(run, log, expected_profile="smoke")
    assert result["ok"] is False
    assert result["checks"]["response_clipping_measured"] is False
    assert result["termination"]["token_limited_ratio"] is None


@pytest.mark.parametrize("kind", ["mask", "budget", "missing_stop", "bad_json"])
def test_bad_trace_remains_a_hard_failure(tmp_path, kind):
    run, log = _healthy(tmp_path)
    path = run / "artifacts/0/rollout.json"
    data = json.loads(path.read_text())
    if kind == "mask":
        data["token_trace"]["response_mask"][0] = 0
    elif kind == "budget":
        data["token_trace"]["generations"][0]["remaining_response_budget"] = 99
    elif kind == "missing_stop":
        data["token_trace"]["generations"][0]["finish_reason"] = None
    path.write_text("{" if kind == "bad_json" else json.dumps(data))
    result = evaluate(run, log, expected_profile="smoke")
    assert result["health_ok"] is False
    assert result["status"] == "fatal"


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
