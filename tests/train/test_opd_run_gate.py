from __future__ import annotations

from pathlib import Path
from functools import lru_cache
import json

import pytest

from syncopate.train.opd_run_gate import evaluate as _evaluate


@lru_cache(maxsize=1)
def frozen_mapper():
    from transformers import AutoTokenizer
    from syncopate.core.model_paths import STUDENT_MODEL
    from syncopate.train.opd_tokens import ByteLevelTokenMapper
    return ByteLevelTokenMapper(AutoTokenizer.from_pretrained(STUDENT_MODEL, local_files_only=True))


def evaluate(*args, **kwargs):
    return _evaluate(*args, mapper=frozen_mapper(), **kwargs)


def _token_evidence(out, *, task_response_ids=(248058, 87, 248059)):
    """Real frozen-tokenizer records; only the small numerical call evidence is synthetic.

    Actual forward behavior remains covered independently in test_opd_objective.
    """
    import hashlib
    from syncopate.train.opd_tokens import capture_generated_sample, TokenAuditWriter
    mapper = frozen_mapper()
    methods = {}
    for name in ("prepare_inputs_for_generation", "_prepare_position_ids_for_generation", "_update_model_kwargs_for_generation"):
        source = f"def {name}():\n    pass\n"
        methods[name] = {"source": source, "sha256": hashlib.sha256(source.encode()).hexdigest()}
    writer = TokenAuditWriter(out, run_token="0123456789abcdef", rank=0,
        runtime={"versions": {"transformers": "5.10.4", "tokenizers": "0.22.2", "torch": "2.11.0"},
            "position_policy": "text_only_cumsum_attention_minus_one_pad_zero", "generation_methods": methods,
            "tokenizer": mapper.evidence()})
    samples = []
    for family, ids in [("chat", (64, 65) * 6), ("task", task_response_ids)]:
        sample = capture_generated_sample(mapper, prompt_ids=(1,), prompt_attention_mask=(1,),
            prompt_position_ids=(0,), generated_ids=(1, *ids), eos_token_ids=(248046, 248044),
            pad_token_id=248044, implicit_think_open=False)
        gen_id = writer.generation(sample, attempt=1, sample_index=len(samples), turn_index=0,
                                   family=family, final_turn=True)
        samples.append((sample, gen_id, family))

    def calls(sample, zero=False):
        active = [int(label == "text" and not zero) for label in sample.alignment.labels]
        forward = int(bool(sample.response_ids) and (any(active) or zero))
        result = {"response_ids": list(sample.response_ids), "response_labels": list(sample.alignment.labels),
            "response_mask": active, "masked_tokens": sum(active), "reply_tokens": len(active),
            "alignment_warnings": list(sample.alignment.warnings), "student_forward": forward, "aux_forward": forward,
            "backward": forward, "loss_sum": 1.44 if any(active) else 0.0}
        if forward:
            for role in ("student", "aux"):
                result.update({f"{role}_{key}": list(getattr(sample, key))
                               for key in ("input_ids", "attention_mask", "position_ids")})
        if zero:
            result.update(gradients=10, zero_gradient_tensors=10, nonfinite_gradient_tensors=0, aux_gradient_tensors=0)
        return result
    for sample, gen_id, family in samples:
        writer.kl(gen_id, attempt=1, family=family, zero_mask=False, audit=calls(sample))
    sample, gen_id, family = samples[-1]
    writer.kl(gen_id, attempt=1, family=family, zero_mask=True, audit=calls(sample, True))
    return [writer.summary()]


def _healthy(tmp_path: Path, **kwargs) -> tuple[Path, Path]:
    from tests.train.test_opd_updates import light_records, write_records

    out = tmp_path / "opd"
    (out / "final").mkdir(parents=True)
    (out / "final/adapter_config.json").write_text("{}", encoding="utf-8")
    (out / "final/adapter_model.safetensors").write_bytes(b"x")
    (out / "completion.json").write_text(json.dumps({"status": "pass", "run_token": "0123456789abcdef",
        "real_steps": 1, "attempted_steps": 1, "skipped_steps": 0, "world_size": 1, "probe_every": 1,
        "prompt_hash": "fedcba9876543210", "objective": "global_masked_token_mean_reverse_kl",
        "parameter_fingerprint": {"sha256": "a" * 64, "tensors": 10, "elements": 20},
        "token_audits": _token_evidence(out, **kwargs),
        "update_audits": [write_records(out, light_records(pairs=((1, 1),)))]}), encoding="utf-8")
    log = tmp_path / "opd.log"
    log.write_text(
        "[opd-vocab] student=248320 teacher=248320 anchor=248320\n"
        "[opd-seed] base=100 rank=0 effective=100\n"
        "[opd-prompt] full_menu=34 answer_fields=0 history=message_pairs "
        "hash=fedcba9876543210\n"
        "[opd-run] token=0123456789abcdef base=/base adapter=/adapter out=/out\n"
        "[opd-mask] attempt 1 全局可蒸 token=12\n"
        "[opd-objective] step 1 kl_per_token=0.12 global_tokens=12\n"
        "step 1 attempt=1 ep0 kl_chat/tok=0.1200 kl_task/tok=0.0000 masked=12 "
        "[opd-route] chat=1 task=1 chat_masked=12 task_masked=0\n"
        "[opd-zero] step 1 对照通过（0/10） student_forward=1 aux_forward=1 backward=1 gradients=10\n"
        f"[opd-sync] ep0 完整可训练参数指纹一致 sha256={'a' * 64} tensors=10\n"
        "[opd-summary] attempted=1 real=1 skipped=0 target_real=1 status=pass\n",
        encoding="utf-8",
    )
    return log, out


def test_healthy_opd_passes(tmp_path):
    log, out = _healthy(tmp_path)
    result = evaluate(log, out, expected_real_steps=1)
    assert result["ok"] is True, result


@pytest.mark.parametrize("fault", ["missing", "zero_grad_decay", "state_counterfake"])
def test_full_gate_requires_raw_signal_updates_not_old_rl_nonzero(tmp_path, fault):
    log, out = _healthy(tmp_path)
    marker = out / "completion.json"
    completion = json.loads(marker.read_text())
    if fault == "missing":
        completion.pop("update_audits")
        marker.write_text(json.dumps(completion))
    else:
        from tests.train.test_opd_updates import fingerprint
        path = Path(completion["update_audits"][0]["path"])
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if fault == "zero_grad_decay":
            rows[1]["gradients"]["weight"] = fingerprint([0., 0.])
        else:
            rows[1]["state_steps_after"]["weight"] = 7
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    result = evaluate(log, out, expected_real_steps=1)
    assert not result["ok"], "completion label and old nonzero adapter do not prove an OPD update"
    assert result["checks"]["raw_update_audit_complete"] is False


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


def test_empty_zero_probe_cannot_pass_just_from_success_text(tmp_path):
    log, out = _healthy(tmp_path)
    log.write_text(log.read_text().replace('[opd-zero] step 1 对照通过（0/10）',
        '[opd-zero] step 1 对照通过（0/10） student_forward=0 aux_forward=0 backward=0 gradients=0'))
    result = evaluate(log, out, expected_real_steps=1)
    assert result['checks']['zero_mask_control_passed'] is False


def test_update_and_skip_counters_must_add_up(tmp_path):
    log, out = _healthy(tmp_path)
    log.write_text(log.read_text().replace('attempted=1 real=1 skipped=0', 'attempted=7 real=1 skipped=0'))
    assert evaluate(log, out, expected_real_steps=1)['ok'] is False


@pytest.mark.parametrize("fault", ["norm_only", "wrong_sha", "wrong_count", "missing_objective", "nan_objective"])
def test_new_objective_and_fingerprint_need_real_matching_records(tmp_path, fault):
    log, out = _healthy(tmp_path)
    if fault == "norm_only":
        log.write_text(log.read_text().replace(
            f"完整可训练参数指纹一致 sha256={'a' * 64} tensors=10", "权重一致性通过（fp=1.0）"))
    elif fault in {"wrong_sha", "wrong_count"}:
        marker = json.loads((out / "completion.json").read_text())
        if fault == "wrong_sha": marker["parameter_fingerprint"]["sha256"] = "b" * 64
        else: marker["parameter_fingerprint"]["tensors"] = 11
        (out / "completion.json").write_text(json.dumps(marker))
    elif fault == "missing_objective":
        log.write_text("\n".join(line for line in log.read_text().splitlines() if "[opd-objective]" not in line))
    else:
        log.write_text(log.read_text().replace("kl_per_token=0.12", "kl_per_token=nan"))
    assert not evaluate(log, out, expected_real_steps=1)["ok"]


@pytest.mark.parametrize("fault", ["missing_file", "missing_manifest", "missing_kl", "prefix", "ids", "attention",
    "position", "labels", "count", "zero_graph", "zero_gradients", "runtime_hash", "response_mask", "extra_kl", "loss"])
def test_raw_generation_and_kl_records_must_reconcile(tmp_path, fault):
    log, out = _healthy(tmp_path)
    marker_path = out / "completion.json"
    marker = json.loads(marker_path.read_text())
    path = Path(marker['token_audits'][0]['path'])
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    kl = next(row for row in rows if row['event'] == 'kl' and not row['zero_mask'] and row['family'] == 'chat')
    generation = next(row for row in rows if row['event'] == 'generation')
    zero = next(row for row in rows if row['event'] == 'kl' and row['zero_mask'])
    if fault == 'missing_file':
        path.unlink()
    elif fault == 'missing_manifest':
        del marker['token_audits']
    elif fault == 'missing_kl':
        rows.remove(kl)
    elif fault == 'extra_kl':
        rows.append(kl)
    elif fault == 'prefix':
        generation['sample']['generated_ids'][0] = 88
    elif fault == 'ids':
        kl['audit']['student_input_ids'][-1] = 88
    elif fault == 'attention':
        kl['audit']['aux_attention_mask'][0] = 0
    elif fault == 'position':
        kl['audit']['student_position_ids'][0] = 7
    elif fault == 'labels':
        generation['sample']['alignment']['labels'][0] = 'tool'
    elif fault == 'count':
        marker['token_audits'][0]['generated_records'] += 1
    elif fault == 'zero_graph':
        zero['audit']['backward'] = 0
    elif fault == 'zero_gradients':
        zero['audit']['zero_gradient_tensors'] = 9
    elif fault == 'runtime_hash':
        rows[0]['runtime']['generation_methods']['_prepare_position_ids_for_generation']['sha256'] = '0' * 64
    elif fault == 'response_mask':
        kl['audit']['response_mask'][0] = 0
    elif fault == 'loss':
        kl['audit']['loss_sum'] *= 2
    if fault != 'missing_file':
        path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    marker_path.write_text(json.dumps(marker))
    assert not evaluate(log, out, expected_real_steps=1)['ok']


def test_conservative_alignment_warning_is_accounted_but_not_a_health_failure(tmp_path):
    # ID 187 really is ff in this frozen tokenizer; it is not a fabricated byte
    # string attached to an unrelated ID.
    log, out = _healthy(tmp_path, task_response_ids=(248058, 87, 248059, 187))
    result = evaluate(log, out, expected_real_steps=1)
    assert result['ok'], result
    assert result['metrics']['token_audit']['warning_counts']['invalid_utf8'] == 1


def test_route_label_alone_cannot_prove_anchor_forward(tmp_path):
    log, out = _healthy(tmp_path)
    marker_path = out / 'completion.json'
    marker = json.loads(marker_path.read_text())
    path = Path(marker['token_audits'][0]['path'])
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    chat = next(row for row in rows if row['event'] == 'kl' and row['family'] == 'chat')
    zero = next(row for row in rows if row['event'] == 'kl' and row['zero_mask'])
    zero.update(generation_id=chat['generation_id'], family='chat')
    zero['audit'].update(chat['audit'])
    zero['audit'].update(response_mask=[0] * 12, masked_tokens=0, loss_sum=0.0)
    marker['token_audits'][0]['route_forwards'] = {'chat': 2, 'task': 0}
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    marker_path.write_text(json.dumps(marker))
    result = evaluate(log, out, expected_real_steps=1)
    assert result['checks']['raw_token_audit_complete'], result
    assert result['checks']['task_route_observed']
    assert not result['checks']['anchor_forward_observed']
    assert not result['ok']


@pytest.mark.parametrize('fault', ['different_bytes', 'special_id', 'unknown_id', 'unknown_prompt',
                                  'fake_special_mask', 'runtime_tokenizer'])
def test_record_cannot_self_certify_its_tokenizer_identity(tmp_path, fault):
    from dataclasses import asdict
    from syncopate.train.opd_tokens import align_token_bytes
    log, out = _healthy(tmp_path)
    marker = json.loads((out / 'completion.json').read_text())
    path = Path(marker['token_audits'][0]['path'])
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    gen = next(row for row in rows if row['event'] == 'generation' and row['family'] == 'chat')
    kl = next(row for row in rows if row['event'] == 'kl' and row['family'] == 'chat')
    if fault == 'different_bytes':
        # Still structurally self-consistent, but real [64,65] means ab, not xy.
        gen['sample']['alignment'] = asdict(align_token_bytes(('78', '79') * 6, (0,) * 12,
                                                            implicit_think_open=False))
    elif fault in {'special_id', 'unknown_id', 'unknown_prompt'}:
        wrong = 248045 if fault == 'special_id' else 999999
        position = 0 if fault == 'unknown_prompt' else 1
        gen['sample']['generated_ids'][position] = wrong
        if fault == 'unknown_prompt':
            gen['sample']['prompt_ids'][0] = wrong
        else:
            gen['sample']['response_ids'][0] = wrong
            kl['audit']['response_ids'][0] = wrong
        for role in ('student', 'aux'):
            kl['audit'][f'{role}_input_ids'][position] = wrong
    elif fault == 'fake_special_mask':
        # A real ordinary token cannot become an arbitrary invisible special.
        # Use the task sample so the rest of its legitimate zero-NL bookkeeping
        # remains unchanged even after falsifying the token's decode.
        gen = next(row for row in rows if row['event'] == 'generation' and row['family'] == 'task')
        alignment = gen['sample']['alignment']
        mask = [0, 1, 0]
        fake = asdict(align_token_bytes(tuple(alignment['token_bytes']), tuple(mask), implicit_think_open=False))
        gen['sample']['alignment'] = fake
        for row in rows:
            if row['event'] == 'kl' and row['generation_id'] == gen['generation_id']:
                row['audit']['response_labels'] = fake['labels']
                row['audit']['alignment_warnings'] = fake['warnings']
    else:
        rows[0]['runtime']['tokenizer']['backend_sha256'] = 'f' * 64
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    result = evaluate(log, out, expected_real_steps=1)
    assert not result['ok'], result


def _insert_valid_history(rows, marker):
    import copy
    final = next(row for row in rows if row['event'] == 'generation' and row['family'] == 'chat')
    history = copy.deepcopy(final)
    history['final_turn'] = False
    for row in rows:
        if 'generation_id' in row:
            row['generation_id'] = {'0:1': '0:2', '0:2': '0:3'}[row['generation_id']]
    final['turn_index'] = 1
    rows.insert(1, history)
    marker['token_audits'][0]['generated_records'] += 1
    return history, final


@pytest.mark.parametrize('fault', ['duplicate_coordinate', 'family_changed', 'after_final', 'turn_gap'])
def test_generation_coordinates_and_turn_lifecycle_are_not_self_certified(tmp_path, fault):
    log, out = _healthy(tmp_path)
    marker_path = out / 'completion.json'
    marker = json.loads(marker_path.read_text())
    path = Path(marker['token_audits'][0]['path'])
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    gens = [row for row in rows if row['event'] == 'generation']
    if fault == 'duplicate_coordinate':
        gens[1]['sample_index'] = 0
    elif fault in {'family_changed', 'turn_gap'}:
        history, final = _insert_valid_history(rows, marker)
        if fault == 'family_changed':
            history['family'] = 'task'
        else:
            final['turn_index'] = 2
    else:
        # A real, unused history generation exercises lifecycle validation without
        # inventing an extra KL loss, changing route counts, or duplicating IDs.
        import copy
        history = copy.deepcopy(gens[0])
        history.update(generation_id='0:3', sample_index=0, turn_index=1, final_turn=False)
        rows.append(history)
        marker['token_audits'][0]['generated_records'] += 1
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    marker_path.write_text(json.dumps(marker))
    assert not evaluate(log, out, expected_real_steps=1)['ok']


def test_normal_contiguous_multiturn_history_still_passes(tmp_path):
    log, out = _healthy(tmp_path)
    marker_path = out / 'completion.json'
    marker = json.loads(marker_path.read_text())
    path = Path(marker['token_audits'][0]['path'])
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    _insert_valid_history(rows, marker)
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    marker_path.write_text(json.dumps(marker))
    result = evaluate(log, out, expected_real_steps=1)
    assert result['ok'], result


def test_python_gate_requires_an_explicit_external_tokenizer(tmp_path):
    log, out = _healthy(tmp_path)
    with pytest.raises(ValueError, match='tokenizer'):
        _evaluate(log, out, expected_real_steps=1)


def test_cli_requires_tokenizer_instead_of_guessing_record_base(tmp_path):
    from syncopate.train.opd_run_gate import main
    log, out = _healthy(tmp_path)
    with pytest.raises(SystemExit) as exc:
        main(['--log', str(log), '--out-dir', str(out), '--expected-real-steps', '1',
              '--out', str(tmp_path / 'result.json')])
    assert exc.value.code == 2


def test_cli_loads_explicit_real_frozen_tokenizer(tmp_path):
    from syncopate.train.opd_run_gate import main
    from syncopate.core.model_paths import STUDENT_MODEL
    log, out = _healthy(tmp_path)
    result_path = tmp_path / 'result.json'
    assert main(['--log', str(log), '--out-dir', str(out), '--expected-real-steps', '1',
                 '--out', str(result_path), '--tokenizer', str(Path(STUDENT_MODEL).resolve())]) == 0
    result = json.loads(result_path.read_text())
    assert result['ok']
    assert result['tokenizer_path'] == str(Path(STUDENT_MODEL).resolve())
    assert result['metrics']['token_audit']['tokenizer'] == frozen_mapper().evidence()


def test_python_gate_rejects_ambiguous_tokenizer_sources(tmp_path):
    from syncopate.core.model_paths import STUDENT_MODEL
    log, out = _healthy(tmp_path)
    with pytest.raises(ValueError, match='exactly one'):
        _evaluate(log, out, expected_real_steps=1, mapper=frozen_mapper(), tokenizer_path=STUDENT_MODEL)
