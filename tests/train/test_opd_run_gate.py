from __future__ import annotations

from pathlib import Path

from syncopate.train.opd_run_gate import evaluate


def _healthy(tmp_path: Path) -> tuple[Path, Path]:
    out = tmp_path / "opd"
    (out / "final").mkdir(parents=True)
    (out / "final/adapter_config.json").write_text("{}", encoding="utf-8")
    (out / "final/adapter_model.safetensors").write_bytes(b"x")
    (out / "completion.json").write_text(
        '{"status":"pass","run_token":"0123456789abcdef",'
        '"real_steps":1,"prompt_hash":"fedcba9876543210"}',
        encoding="utf-8",
    )
    log = tmp_path / "opd.log"
    log.write_text(
        "[opd-vocab] student=248320 teacher=248320 anchor=248320\n"
        "[opd-seed] base=100 rank=0 effective=100\n"
        "[opd-prompt] full_menu=34 answer_fields=0 history=message_pairs "
        "hash=fedcba9876543210\n"
        "[opd-run] token=0123456789abcdef base=/base adapter=/adapter out=/out\n"
        "[opd-mask] attempt 1 全局可蒸 token=12\n"
        "step 1 attempt=1 ep0 kl_chat/tok=0.1200 kl_task/tok=0.0000 masked=12 "
        "[opd-route] chat=1 task=1 chat_masked=12 task_masked=0\n"
        "[opd-zero] step 1 对照通过（0/10）\n"
        "[opd-sync] ep0 权重一致性通过（fp=1.0）\n"
        "[opd-summary] attempted=1 real=1 skipped=0 target_real=1 status=pass\n",
        encoding="utf-8",
    )
    return log, out


def test_healthy_opd_passes(tmp_path):
    log, out = _healthy(tmp_path)
    result = evaluate(log, out, expected_real_steps=1)
    assert result["ok"] is True, result


def test_empty_update_cannot_be_green(tmp_path):
    log, out = _healthy(tmp_path)
    log.write_text(
        "[opd-vocab] ok\n[opd-summary] attempted=8 real=0 skipped=8 target_real=1 status=fail\n",
        encoding="utf-8",
    )
    (out / "final/adapter_model.safetensors").unlink()
    result = evaluate(log, out, expected_real_steps=1)
    assert result["ok"] is False
    assert result["checks"]["real_steps_complete"] is False
    assert result["checks"]["nonzero_mask_observed"] is False
    assert result["checks"]["final_adapter_exists"] is False


def test_single_route_smoke_cannot_claim_full_opd_wiring(tmp_path):
    log, out = _healthy(tmp_path)
    log.write_text(log.read_text(encoding="utf-8").replace("chat=1 task=1", "chat=1 task=0"),
                   encoding="utf-8")
    result = evaluate(log, out, expected_real_steps=1)
    assert result["ok"] is False
    assert result["checks"]["chat_route_observed"] is True
    assert result["checks"]["task_route_observed"] is False


def test_legacy_only_mask_cannot_pass_as_v15_chat_distillation(tmp_path):
    log, out = _healthy(tmp_path)
    log.write_text(log.read_text(encoding="utf-8").replace("chat_masked=12", "chat_masked=0"),
                   encoding="utf-8")
    result = evaluate(log, out, expected_real_steps=1)
    assert result["ok"] is False
    assert result["checks"]["v15_chat_nl_observed"] is False


def test_route_token_accounting_mismatch_is_rejected(tmp_path):
    log, out = _healthy(tmp_path)
    log.write_text(log.read_text(encoding="utf-8").replace("task_masked=0", "task_masked=2"),
                   encoding="utf-8")
    result = evaluate(log, out, expected_real_steps=1)
    assert result["ok"] is False
    assert result["checks"]["route_mask_accounted"] is False


def test_stale_completion_marker_is_rejected(tmp_path):
    log, out = _healthy(tmp_path)
    marker = out / "completion.json"
    marker.write_text(marker.read_text(encoding="utf-8").replace(
        "0123456789abcdef", "aaaaaaaaaaaaaaaa"), encoding="utf-8")
    result = evaluate(log, out, expected_real_steps=1)
    assert result["ok"] is False
    assert result["checks"]["completion_marker_matches"] is False
