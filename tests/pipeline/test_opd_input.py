import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from syncopate.pipeline.opd_input import bind_input, validate_request
from syncopate.pipeline.rl_input import bind_input as bind_sft, file_sha256
from tests.pipeline.test_rl_input import fixture as sft_fixture, cli_env, ROOT


def fixture(root):
    _, model = sft_fixture(root)
    bind_sft(root, 'source', 'rl', hash_weights=True)
    audit = root / '_audit/v16/runs/rl'
    (audit / 'manifest.json').write_text(json.dumps({'run_id': 'rl', 'profile': 'smoke',
        'stages': {'rl-train': {'status': 'warn'}, 'rl-adapter': {'status': 'pass'}}}))
    (audit / 'rl_run_gate.json').write_text(json.dumps({'health_ok': True,
        'run_dir': 'checkpoints/grpo/v16_smoke_rl', 'expected_profile': 'smoke',
        'log_path': '_audit/v16/runs/rl/rl_train.log', 'health_checks': {'grad_finite': True},
        'metrics': {'global_steps': [1, 2]}}))
    actor = root / 'checkpoints/grpo/v16_smoke_rl/global_step_2/actor'
    actor.mkdir(parents=True)
    shards = []
    for rank in range(2):
        path = actor / f'model_world_size_2_rank_{rank}.pt'
        path.write_bytes(f'rank{rank}'.encode())
        shards.append({'name': path.name, 'bytes': path.stat().st_size, 'sha256': file_sha256(path)})
    adapter = root / 'models/adapters/rl_v16_smoke_rl/lora_adapter'
    adapter.mkdir(parents=True)
    (adapter / 'adapter_model.safetensors').write_bytes(b'format verified separately on target CPU')
    (adapter / 'adapter_config.json').write_text('{}')
    export = {'source_actor': str(actor.relative_to(root)), 'source_shards': shards,
        'adapter_weights_sha256': file_sha256(adapter / 'adapter_model.safetensors'),
        'adapter_config_sha256': file_sha256(adapter / 'adapter_config.json')}
    (adapter / 'export_manifest.json').write_text(json.dumps(export))
    prompts = root / 'data/u_route/v16_p1_prompts.jsonl'
    prompts.parent.mkdir(parents=True)
    prompts.write_text('{"family":"chat","turns":["fixture"]}\n')
    return audit, model, actor, adapter


def test_input_binds_rl_adapter_and_its_actual_inherited_base_without_new_pass(tmp_path):
    _, model, _, adapter = fixture(tmp_path)
    record = bind_input(tmp_path, 'rl', 'new', hash_weights=True)
    assert record['kind'] == 'external_rl_for_opd_diagnostic'
    assert record['model'] == str(model.relative_to(tmp_path))
    assert record['adapter'] == str(adapter.relative_to(tmp_path))
    assert bind_input(tmp_path, 'rl', 'new') == record
    assert not (tmp_path / '_audit/v16/runs/new/manifest.json').exists()
    assert not (tmp_path / '_audit/v16/runs/new/rl_input.json').exists()


def test_fresh_binding_requires_cpu_content_verification(tmp_path):
    fixture(tmp_path)
    with pytest.raises(ValueError, match='CPU'):
        bind_input(tmp_path, 'rl', 'new')


@pytest.mark.parametrize('stage,profile', [('all','smoke'),('rl-train','smoke'),('opd-train','candidate')])
def test_external_opd_only_allows_explicit_smoke_stages(stage, profile):
    with pytest.raises(ValueError):
        validate_request(stage, profile, 'new', 'rl')


@pytest.mark.parametrize('kind', ['wrong_run', 'bad_rl', 'bad_adapter', 'wrong_actor', 'missing_rank',
                                 'escape', 'corrupt_adapter', 'corrupt_shard'])
def test_bad_upstream_identity_is_rejected_before_gpu(tmp_path, kind):
    audit, _, actor, adapter = fixture(tmp_path)
    path = audit / 'manifest.json'
    manifest = json.loads(path.read_text())
    export_path = adapter / 'export_manifest.json'
    export = json.loads(export_path.read_text())
    if kind == 'wrong_run': manifest['run_id'] = 'other'
    elif kind == 'bad_rl': (audit / 'rl_run_gate.json').write_text('{"health_ok":false}')
    elif kind == 'bad_adapter': manifest['stages']['rl-adapter']['status'] = 'fatal'
    elif kind == 'wrong_actor': export['source_actor'] = 'checkpoints/grpo/v16_smoke_other/global_step_2/actor'
    elif kind == 'missing_rank': export['source_shards'].pop()
    elif kind == 'escape': export['source_shards'][0]['name'] = '../other.pt'
    elif kind == 'corrupt_adapter': (adapter / 'adapter_model.safetensors').write_bytes(b'corrupt')
    else: (actor / 'model_world_size_2_rank_0.pt').write_bytes(b'wrong')
    path.write_text(json.dumps(manifest))
    export_path.write_text(json.dumps(export))
    with pytest.raises((ValueError, OSError)):
        bind_input(tmp_path, 'rl', 'new', hash_weights=True)


def test_cpu_recheck_rejects_same_size_base_corruption(tmp_path):
    _, model, _, _ = fixture(tmp_path)
    bind_input(tmp_path, 'rl', 'new', hash_weights=True)
    (model / 'part.safetensors').write_bytes(b'5678')
    with pytest.raises(ValueError):
        bind_input(tmp_path, 'rl', 'new', hash_weights=True)


@pytest.mark.parametrize('mutation', ['run_dir', 'profile', 'log_path', 'checks', 'empty_checks'])
def test_upstream_gate_must_identify_this_run_and_have_no_failed_health_checks(tmp_path, mutation):
    audit, _, _, _ = fixture(tmp_path)
    path = audit / 'rl_run_gate.json'
    gate = json.loads(path.read_text())
    if mutation == 'run_dir': gate['run_dir'] = 'checkpoints/grpo/v16_smoke_other'
    elif mutation == 'profile': gate['expected_profile'] = 'candidate'
    elif mutation == 'log_path': gate['log_path'] = '_audit/v16/runs/other/rl_train.log'
    elif mutation == 'checks': gate['health_checks']['grad_finite'] = False
    else: gate['health_checks'] = {}
    path.write_text(json.dumps(gate))
    with pytest.raises(ValueError):
        bind_input(tmp_path, 'rl', 'new', hash_weights=True)


@pytest.mark.parametrize('mutation', ['missing', 'extra', 'bad_sha', 'wrong_sha'])
def test_gpu_binding_rejects_incomplete_or_unbound_cpu_hash_record(tmp_path, mutation):
    fixture(tmp_path)
    record = bind_input(tmp_path, 'rl', 'new', hash_weights=True)
    hashes = record['content_sha256']
    key = next(iter(hashes))
    if mutation == 'missing': hashes.pop(key)
    elif mutation == 'extra': hashes['never-hashed.safetensors'] = '0' * 64
    elif mutation == 'bad_sha': hashes[key] = 'not-a-sha256'
    else: hashes[key] = '0' * 64
    (tmp_path / '_audit/v16/runs/new/opd_input.json').write_text(json.dumps(record))
    with pytest.raises(ValueError):
        bind_input(tmp_path, 'rl', 'new')


@pytest.mark.parametrize('name', ['model.safetensors', 'pytorch_model.bin', 'adapter_config.json'])
def test_added_competing_hf_weights_cannot_bypass_source_index_hashes(tmp_path, name):
    _, model, _, _ = fixture(tmp_path)
    (model / name).write_bytes(b'competing loader artifact')
    with pytest.raises(ValueError):
        bind_input(tmp_path, 'rl', 'new', hash_weights=True)


def test_inherited_rl_binding_cannot_drop_hash_then_rebind_changed_base(tmp_path):
    audit, model, _, _ = fixture(tmp_path)
    path = audit / 'rl_input.json'
    binding = json.loads(path.read_text())
    binding['content_sha256'].pop('part.safetensors')
    path.write_text(json.dumps(binding))
    (model / 'part.safetensors').write_bytes(b'5678')
    with pytest.raises(ValueError):
        bind_input(tmp_path, 'rl', 'new', hash_weights=True)


@pytest.mark.parametrize('stages,profile', [({'rl-train': {'status': 'pass'}}, 'smoke'),
    ({'opd-train': {'status': 'pass'}}, 'candidate')])
def test_existing_non_diagnostic_target_manifest_is_rejected(tmp_path, stages, profile):
    fixture(tmp_path)
    target = tmp_path / '_audit/v16/runs/new'
    target.mkdir()
    (target / 'manifest.json').write_text(json.dumps({'run_id': 'new', 'profile': profile, 'stages': stages}))
    with pytest.raises(ValueError):
        bind_input(tmp_path, 'rl', 'new', hash_weights=True)
    assert not (target / 'opd_input.json').exists()


def test_matching_opd_only_manifest_can_continue_without_new_binding(tmp_path):
    fixture(tmp_path)
    record = bind_input(tmp_path, 'rl', 'new', hash_weights=True)
    (tmp_path / '_audit/v16/runs/new/manifest.json').write_text(json.dumps({
        'run_id': 'new', 'profile': 'smoke', 'stages': {'opd-train': {'status': 'warn'}}}))
    assert bind_input(tmp_path, 'rl', 'new') == record


@pytest.mark.parametrize('missing', [False, True])
def test_actual_runbook_checks_bound_opd_inputs_without_training(tmp_path, missing):
    _, _, _, adapter = fixture(tmp_path)
    bind_input(tmp_path, 'rl', 'new', hash_weights=True)
    if missing:
        (adapter / 'adapter_model.safetensors').unlink()
    script = tmp_path / 'scripts/v16_pipeline.sh'
    script.parent.mkdir()
    script.write_text((ROOT / 'scripts/v16_pipeline.sh').read_text())
    result = subprocess.run(['bash', str(script), '--check-inputs', '--profile', 'smoke',
        '--run-id', 'new', '--opd-input-run', 'rl', 'opd-train'], cwd=tmp_path,
        env=cli_env(), capture_output=True, text=True)
    assert (result.returncode != 0) is missing, result.stdout + result.stderr
    assert 'torch.distributed.run' not in result.stdout
    assert not (tmp_path / 'checkpoints/opd').exists()
    assert not (tmp_path / '_audit/v16/runs/new/manifest.json').exists()


def test_dry_run_external_opd_does_not_guess_inherited_base():
    result = subprocess.run(['bash', str(ROOT / 'scripts/v16_pipeline.sh'), '--dry-run',
        '--run-id', 'new', '--opd-input-run', 'rl', 'opd-train'], cwd=ROOT,
        env=cli_env(), capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'CPU_BOUND_OPD_BASE_rl' in result.stdout
    assert '--adapter models/adapters/rl_v16_smoke_rl/lora_adapter' in result.stdout
    assert '--out checkpoints/opd/v16_smoke_new' in result.stdout
    assert 'sft-v16_smoke_rl' not in result.stdout


@pytest.mark.parametrize('flags,stage', [([], 'all'), ([], 'rl-train'),
    (['--profile', 'candidate'], 'opd-train'), (['--rl-input-run', 'sft'], 'opd-train')])
def test_actual_runbook_rejects_invalid_external_opd_scope(flags, stage):
    result = subprocess.run(['bash', str(ROOT / 'scripts/v16_pipeline.sh'), '--dry-run',
        '--run-id', 'new', '--opd-input-run', 'rl', *flags, stage], cwd=ROOT,
        env=cli_env(), capture_output=True, text=True)
    assert result.returncode == 2, result.stdout + result.stderr
    assert '[stage' not in result.stdout


def test_runbook_cannot_silently_ignore_scope_flags_after_stage():
    result = subprocess.run(['bash', str(ROOT / 'scripts/v16_pipeline.sh'), '--dry-run',
        '--run-id', 'new', '--opd-input-run', 'rl', 'opd-train', '--profile', 'candidate'], cwd=ROOT,
        env=cli_env(), capture_output=True, text=True)
    assert result.returncode == 2
    assert '[stage' not in result.stdout
