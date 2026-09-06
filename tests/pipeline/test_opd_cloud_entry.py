"""Execute actual cloud entry bodies locally; failed CPU checks must not allocate GPUs."""
import ast
from contextlib import contextmanager
import json
from pathlib import Path
import pathlib
import re
import shlex
import sys
import time
from types import SimpleNamespace
from uuid import uuid4

import pytest


def function(name, scope):
    tree = ast.parse(Path('modal_app/stack_probe.py').read_text())
    node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    node.decorator_list = []
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<actual cloud entry>', 'exec'), scope)
    return scope[name]


class Remote:
    def __init__(self, name, calls, result):
        self.name, self.calls, self.result = name, calls, result
    def remote(self, *args, **kwargs):
        self.calls.append((self.name, args, kwargs))
        return self.result
    def with_options(self, **options):
        self.calls.append((self.name + '_allocate', (), options))
        return self


@pytest.fixture
def entry(tmp_path):
    calls = []
    scope = {'ALL_STEPS': [], 'time': time, 'uuid4': uuid4, 'json': json,
             'LOCAL_ROOT': tmp_path, 'pathlib': pathlib, 'AUDIT': '/vol/_audit', 'sys': sys}
    scope['p_opd_preflight'] = Remote('cpu', calls, {'ok': True})
    scope['p_prepare_cache'] = Remote('cache', calls, {'ok': True})
    scope['p_pipeline'] = Remote('pipeline', calls, {'ok': False, 'returncode': 0,
        'manifest': {'pipeline_ok': True, 'all_passed': False}})
    return function('main', scope), calls, scope


@pytest.mark.parametrize('case', ['pass', 'fail'])
def test_actual_main_gates_gpu_and_preserves_observe_warning(entry, case):
    main, calls, scope = entry
    scope['p_opd_preflight'].result = {'ok': case == 'pass'}
    with pytest.raises(SystemExit):  # WARN summary remains visibly non-green.
        main(steps='pipeline', pipeline_stage='opd-train', pipeline_run_id='new',
             pipeline_opd_input_run='rl', pipeline_opd_real_steps=2)
    names = [name for name, _, _ in calls]
    assert names == (['cpu', 'cache', 'pipeline_allocate', 'pipeline'] if case == 'pass' else ['cpu'])
    if case == 'pass':
        assert calls[2][2]['gpu'] == 'B200:2'
        assert calls[-1][1][-2:] == ('rl', 2)


@pytest.mark.parametrize('extra', [dict(pipeline_profile='candidate'), dict(pipeline_stage='train-all'),
                                 dict(pipeline_rl_input_run='source'), dict(pipeline_opd_real_steps=-1)])
def test_bad_opd_request_fails_before_any_remote_or_allocation(entry, extra):
    main, calls, _ = entry
    kwargs = dict(steps='pipeline', pipeline_stage='opd-train', pipeline_run_id='new', pipeline_opd_input_run='rl')
    kwargs.update(extra)
    with pytest.raises(ValueError):
        main(**kwargs)
    assert calls == []


def test_eval_verifies_existing_cpu_evidence_before_single_gpu(entry):
    main, calls, _ = entry
    with pytest.raises(SystemExit):
        main(steps='pipeline', pipeline_stage='opd-eval', pipeline_run_id='new', pipeline_opd_input_run='rl')
    assert calls[0] == ('cpu', ('new', 'rl', True), {})
    assert calls[2][2]['gpu'] == 'B200'


def test_stage_consumes_cpu_record_and_passes_fixed_runbook_flags(tmp_path, monkeypatch):
    shells = []
    monkeypatch.setattr('syncopate.pipeline.cloud_execution.bind_source', lambda *args: None)
    monkeypatch.setattr('syncopate.pipeline.cloud_execution.prepared_cache', lambda *args: {})
    @contextmanager
    def claim(*args): yield
    monkeypatch.setattr('syncopate.pipeline.cloud_execution.writer_claims', claim)
    monkeypatch.setattr('syncopate.train.gpu_telemetry.gpu_telemetry', claim)
    audit = tmp_path / '_audit/v16/runs/new'
    audit.mkdir(parents=True)
    (audit / 'manifest.json').write_text('{"pipeline_ok":true,"all_passed":false}')
    scope = {'pathlib': pathlib, 're': re, 'time': time, 'uuid4': uuid4, 'json': json,
        'VOL': str(tmp_path), 'REPO': str(tmp_path), 'PY': '/python', 'SERVICE_ENV': {}, 'writers': {},
        'vol': SimpleNamespace(commit=lambda: None), '_sync_repo': lambda: {'sha': 'same'},
        '_topology': lambda: {}, 'p_gpu': SimpleNamespace(local=lambda count: {'ok': True}),
        '_record': lambda step, ok, details: {'step': step, 'ok': ok, **details},
        '_sh': lambda command, **kwargs: shells.append((command, kwargs)) or {'rc': 0, 'secs': 1}}
    result = function('_pipeline_stage', scope)('opd-train', 'smoke', 'observe', 'new', False,
                                                opd_input_run='rl', opd_real_steps=2)
    assert '--verify-only --input-run rl --run-id new' in shells[0][0]
    assert result['returncode'] == 0 and result['ok'] is False
    command, options = shells[-1]
    assert shlex.split(command)[shlex.split(command).index('--opd-input-run') + 1] == 'rl'
    assert options['env']['OPD_SMOKE_REAL_STEPS'] == '2'
    assert options['stop_grace'] == 105
