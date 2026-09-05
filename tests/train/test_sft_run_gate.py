from __future__ import annotations

import re
from pathlib import Path

from syncopate.train.sft_run_gate import evaluate, main


def _flatten_losses(text: str) -> str:
    return re.sub(r"loss=[-+0-9.eE]+", "loss=0.5", text)


def _healthy(tmp_path: Path) -> tuple[Path, Path]:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"x")
    log = tmp_path / "sft.log"
    lines = ["[模型] x 可训练 37.0M / 总计 35000M"]
    for step in range(1, 31):
        lines.append(f"[step {step}] loss={2.0-step/50:.4f} grad_norm=0.3 lr=1e-4 sup_tok/s=1200")
    lines += ["||ΔW||/||W|| = 0.0123%", "显存峰值 75.0 GB"]
    log.write_text("\n".join(lines), encoding="utf-8")
    return log, adapter


def test_healthy_sft_smoke_passes(tmp_path):
    log, adapter = _healthy(tmp_path)
    result = evaluate(log, adapter, expected_steps=30)
    assert result["ok"] is True, result


def test_flat_loss_and_missing_adapter_both_reported(tmp_path):
    log, adapter = _healthy(tmp_path)
    log.write_text(_flatten_losses(log.read_text()), encoding="utf-8")
    (adapter / "adapter_model.safetensors").unlink()
    result = evaluate(log, adapter, expected_steps=30)
    assert result["ok"] is False
    assert result["checks"]["loss_trending_down"] is False
    assert result["checks"]["adapter_files_exist"] is False
    assert result["status"] == "fatal"
    assert main([
        "--log", str(log), "--adapter", str(adapter), "--expected-steps", "30",
        "--out", str(tmp_path / "fatal.json"),
    ]) == 3


def test_flat_loss_alone_is_quality_warn(tmp_path):
    log, adapter = _healthy(tmp_path)
    log.write_text(_flatten_losses(log.read_text()), encoding="utf-8")
    result = evaluate(log, adapter, expected_steps=30)
    assert result["status"] == "warn"
    assert result["health_ok"] is True
    assert result["quality_ready"] is False
    assert main([
        "--log", str(log), "--adapter", str(adapter), "--expected-steps", "30",
        "--out", str(tmp_path / "warn.json"),
    ]) == 2
