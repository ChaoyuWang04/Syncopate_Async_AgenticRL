from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from syncopate.train.sft import resolve_grad_accum

ROOT = Path(__file__).resolve().parents[2]
ENV = {**os.environ, "SYNCOPATE_CONTRACT": "v15", "SYNCOPATE_THINK": "1"}


def _launch(*args: str):
    return subprocess.run(
        [sys.executable, "-m", "syncopate.train.launch_rl_v1", "--dry-run", *args],
        cwd=ROOT, capture_output=True, text=True, env=ENV,
    )


def test_rl_defaults_to_smoke_official_uniform_sampler():
    run = _launch("--logger", "console")
    assert run.returncode == 0, run.stderr
    text = run.stdout + run.stderr
    assert "profile=smoke" in text
    assert "models/Qwen3.6-35B-A3B-sft-v16_smoke" in text
    assert "-m \\\n  verl.trainer.main_ppo" in text
    assert "main_ppo_pool" not in text


def test_dynamic_pool_is_an_explicit_experiment_arm():
    run = _launch("--dynamic-pool", "--logger", "console")
    assert run.returncode == 0, run.stderr
    assert "syncopate.train.main_ppo_pool" in run.stdout


def test_short_candidate_cannot_masquerade_as_candidate():
    run = _launch("--profile", "candidate", "--steps", "2", "--logger", "console")
    assert run.returncode != 0
    assert "candidate 至少" in (run.stdout + run.stderr)


def test_sft_one_vs_two_gpu_keeps_the_same_effective_batch():
    assert resolve_grad_accum(2, 1, 16, None) == 8
    assert resolve_grad_accum(2, 2, 16, None) == 4


def test_sft_rejects_a_silent_batch_change():
    import pytest

    with pytest.raises(ValueError, match="配方漂移"):
        resolve_grad_accum(2, 2, 16, 8)
