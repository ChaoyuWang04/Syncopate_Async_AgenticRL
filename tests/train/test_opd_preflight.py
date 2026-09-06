"""OPD CPU 前置不改数据；拒绝坏结构、借用考题或不完整证据。"""
import copy
import json
from pathlib import Path

import pytest

from syncopate.train.opd_preflight import validate_prompt_rows, require_preflight


def rows():
    return [{'id': 'chat1', 'family': 'chat', 'turns': ['普通问句']},
            {'id': 'task1', 'family': 'task', 'turns': ['任务问句', '继续解释']}]


def test_frozen_prompt_validation_does_not_mutate_or_judge_semantics():
    data = rows()
    before = copy.deepcopy(data)
    checked = validate_prompt_rows(data, {'未参与训练的考题'})
    assert checked == {'rows': 2, 'turns': 3, 'families': {'chat': 1, 'task': 1}, 'exact_exam_overlap': 0}
    assert data == before


@pytest.mark.parametrize('mutation', ['empty', 'bad_id', 'duplicate', 'family', 'turns', 'empty_turn', 'exam', 'missing_route'])
def test_bad_structure_or_exact_exam_collision_fails_without_cleaning(mutation):
    data = rows()
    if mutation == 'empty': data = []
    elif mutation == 'bad_id': data[0]['id'] = None
    elif mutation == 'duplicate': data[1]['id'] = data[0]['id']
    elif mutation == 'family': data[0]['family'] = 'unknown'
    elif mutation == 'turns': data[0]['turns'] = 'not a list'
    elif mutation == 'empty_turn': data[0]['turns'] = ['']
    elif mutation == 'exam': data[0]['turns'] = ['考题']
    else: data[0]['family'] = 'task'
    with pytest.raises(ValueError):
        validate_prompt_rows(data, {'考题'})


@pytest.mark.parametrize('mutation', ['none', 'skip', 'check', 'run', 'input', 'missing'])
def test_preflight_labels_cannot_replace_complete_cpu_evidence(mutation):
    record = {'health_ok': True, 'run_id': 'new', 'input': {'run_id': 'new', 'input_run': 'old'},
        'checks': {key: True for key in ('tests', 'inputs', 'runbook_inputs', 'adapter', 'data', 'tokenizers', 'no_cuda')},
        'tests': {'counts': {'tests': 5, 'failures': 0, 'errors': 0, 'skipped': 0}, 'rc': 0}}
    if mutation == 'skip': record['tests']['counts']['skipped'] = 1
    elif mutation == 'check': record['checks']['inputs'] = False
    elif mutation == 'run': record['run_id'] = 'other'
    elif mutation == 'input': record['input']['input_run'] = 'other'
    elif mutation == 'missing': record['checks'] = {}
    with pytest.raises(ValueError):
        require_preflight(record, 'new', 'old')


def test_junit_requires_every_registered_module_and_rejects_selection(tmp_path):
    from syncopate.train.opd_preflight import TESTS, junit_evidence
    xml = tmp_path / 'tests.xml'
    xml.write_text('<testsuite><testcase classname="tests.train.test_opd_tokens" name="only_one"/></testsuite>')
    with pytest.raises(ValueError, match='模块'):
        junit_evidence(xml)
    cases = ''.join(f'<testcase classname="{Path(name).with_suffix("").as_posix().replace("/", ".")}" name="one"/>' for name in TESTS)
    xml.write_text(f'<testsuite>{cases}</testsuite>')
    assert junit_evidence(xml)['counts']['tests'] == len(TESTS)
    xml.write_text(f'<testsuite>{cases}<testcase classname="elsewhere" name="wrong"/></testsuite>')
    with pytest.raises(ValueError, match='模块'):
        junit_evidence(xml)


def test_teacher_metadata_uses_exact_model_loader_files(tmp_path):
    from syncopate.core.model_paths import TEACHER_MODEL
    from syncopate.train.opd_preflight import teacher_metadata
    model = tmp_path / TEACHER_MODEL
    model.mkdir(parents=True)
    (model / 'config.json').write_text('{}')
    (model / 'stray.safetensors').write_bytes(b'wrong name')
    with pytest.raises(ValueError):
        teacher_metadata(tmp_path, hash_weights=True)
    (model / 'stray.safetensors').rename(model / 'model.safetensors')
    record = teacher_metadata(tmp_path, hash_weights=True)
    assert set(record['content_sha256']) == {str(Path(TEACHER_MODEL) / name) for name in ('model.safetensors', 'config.json')}
    assert all(len(value) == 64 for value in record['content_sha256'].values())


def test_all_exam_turns_are_checked_without_claiming_case_provenance(tmp_path, monkeypatch):
    from syncopate.train.opd_preflight import frozen_inputs
    from syncopate.pipeline import multiturn
    from syncopate.pipeline.split import DEFAULT_OPD_PROMPTS
    monkeypatch.setattr(multiturn, 'EXAM_FILES', ['test_exam.jsonl'])
    prompt = tmp_path / DEFAULT_OPD_PROMPTS
    prompt.parent.mkdir(parents=True)
    prompt.write_text('\n'.join(json.dumps(row) for row in rows()))
    exam = tmp_path / 'data/u_route/test_exam.jsonl'
    exam.parent.mkdir(parents=True, exist_ok=True)
    exam.write_text(json.dumps({'turns': ['普通问句', '另外末轮']}))
    with pytest.raises(ValueError, match='逐字重合'):
        frozen_inputs(tmp_path)


@pytest.fixture
def complete_preflight(tmp_path, monkeypatch):
    from syncopate.train import opd_preflight as pre
    from syncopate.core.model_paths import TEACHER_MODEL
    from syncopate.pipeline import multiturn
    from syncopate.pipeline.split import DEFAULT_OPD_PROMPTS
    # Reuse the same fixture as the input-binding tests; no real model content.
    from tests.pipeline.test_opd_input import fixture
    fixture(tmp_path)
    monkeypatch.setenv('SYNCOPATE_LOCAL_SOURCE_SHA', 'a' * 64)
    monkeypatch.setattr(multiturn, 'EXAM_FILES', ['test_exam.jsonl'])
    monkeypatch.setattr(pre, 'TESTS', ('tests/train/test_opd_preflight.py',))
    test = tmp_path / pre.TESTS[0]
    test.parent.mkdir(parents=True)
    test.write_text('# local fixture only')
    (tmp_path / DEFAULT_OPD_PROMPTS).write_text('\n'.join(json.dumps(row) for row in rows()))
    (tmp_path / 'data/u_route/test_exam.jsonl').write_text('{"turns":["独立考题"]}')
    teacher = tmp_path / TEACHER_MODEL
    teacher.mkdir(parents=True)
    (teacher / 'config.json').write_text('{}')
    (teacher / 'model.safetensors').write_bytes(b'teacher fixture')
    bound = pre.bind_input(tmp_path, 'rl', 'new', hash_weights=True)
    directory = tmp_path / '_audit/v16/runs/new/preparation/opd-train'
    directory.mkdir(parents=True)
    xml = directory / 'tests.xml'
    xml.write_text('<testsuite><testcase classname="tests.train.test_opd_preflight" name="one"/></testsuite>')
    data, hashes = pre.frozen_inputs(tmp_path)
    record = {'health_ok': True, 'run_id': 'new', 'source_sha256': 'a' * 64,
        'input': bound, 'checks': {key: True for key in pre.CHECKS},
        'tests': {'rc': 0, **pre.junit_evidence(xml), 'files': {pre.TESTS[0]: pre.file_sha256(test)}},
        'teacher': pre.teacher_metadata(tmp_path, hash_weights=True), 'data': data, 'data_sha256': hashes,
        'adapter': {'pairs': 1, 'tensors': 2, 'nonzero_b_tensors': 1},
        'tokenizers': {'same_vocab': True}, 'runbook_inputs': {'rc': 0}}
    path = directory / 'preflight.json'
    path.write_text(json.dumps(record))
    return tmp_path, path, record


@pytest.mark.parametrize('mutation', ['none', 'input_hash', 'teacher_hash', 'teacher_extra', 'teacher_config',
                                     'data', 'xml', 'test', 'source', 'adapter', 'adapter_config_same_size'])
def test_complete_preflight_checks_bind_actual_cpu_files(complete_preflight, mutation):
    from syncopate.train import opd_preflight as pre
    root, path, record = complete_preflight
    if mutation == 'input_hash': record['input']['content_sha256'] = {}
    elif mutation == 'teacher_hash': record['teacher']['content_sha256'] = {}
    elif mutation == 'teacher_extra':
        (root / record['teacher']['model'] / 'stray.safetensors').write_bytes(b'wrong')
    elif mutation == 'teacher_config':
        (root / record['teacher']['model'] / 'config.json').write_text('{"changed":true}')
    elif mutation == 'data':
        (root / next(iter(record['data_sha256']))).write_text('[]')
    elif mutation == 'xml':
        (path.parent / 'tests.xml').write_text('<testsuite/>')
    elif mutation == 'test':
        (root / pre.TESTS[0]).write_text('# changed')
    elif mutation == 'source': record['source_sha256'] = 'b' * 64
    elif mutation == 'adapter': record['adapter']['nonzero_b_tensors'] = 0
    elif mutation == 'adapter_config_same_size':
        config = root / record['input']['adapter'] / 'adapter_config.json'
        config.write_text('[]')  # same bytes as {}; existing metadata alone cannot detect it
    path.write_text(json.dumps(record))
    if mutation == 'none':
        assert pre.verify_preflight(root, 'new', 'rl') == record
    else:
        with pytest.raises(ValueError):
            pre.verify_preflight(root, 'new', 'rl')
