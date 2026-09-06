"""Exercise the real cloud function body without importing Modal or allocating hardware."""
import ast
import json
import pathlib
import shlex
import time
from contextlib import contextmanager
from uuid import uuid4
from types import SimpleNamespace

import pytest

from syncopate.pipeline.infra_probe import execute_probe


@pytest.fixture
def entry(tmp_path, monkeypatch):
    source = pathlib.Path('modal_app/stack_probe.py').read_text()
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'p_infra_probe')
    node.decorator_list = []
    calls = []
    records = {}
    state = {'pytest_rc': 0, 'skipped': False, 'telemetry': False, 'module_rc': 0}

    class Registry:
        def put(self, key, value, *, skip_if_exists):
            if key in records:
                return False
            records[key] = value
            return True
        def get(self, key): return records.get(key)
        def pop(self, key): return records.pop(key)

    @contextmanager
    def telemetry(path):
        state['telemetry'] = True
        path.write_text('timestamp,uuid\nnow,gpu0\n')
        try:
            yield
        finally:
            state['telemetry'] = False

    monkeypatch.setattr('syncopate.train.gpu_telemetry.gpu_telemetry', telemetry)

    def shell(command, **kwargs):
        args = shlex.split(command)
        calls.append(command)
        if 'pytest' in args:
            xml_path = pathlib.Path(next(arg.split('=', 1)[1] for arg in args if arg.startswith('--junitxml=')))
            xml = '<testsuites><testsuite><testcase name="pass"/>'
            if state['skipped']:
                xml += '<testcase name="skip"><skipped/></testcase>'
            xml += '</testsuite></testsuites>'
            xml_path.write_text(xml)
            return {'rc': state['pytest_rc'], 'out': f'pytest attempt {len(calls)}'}
        assert args[args.index('-m') + 1] in {'syncopate.infra.hardware_probe', 'syncopate.infra.training_probe'}
        mode = args[args.index('--mode') + 1]
        assert mode == 'cpu' or state['telemetry'], 'GPU must run inside telemetry context'
        output = pathlib.Path(args[args.index('--output') + 1])
        output.mkdir()
        (output / 'result.json').write_text(json.dumps({
            'health_ok': state['module_rc'] == 0, 'interfaces': {'nccl_compiled': True}}))
        return {'rc': state['module_rc'], 'out': 'module ran', 'secs': 1,
                'timed_out': state['module_rc'] != 0, 'process_returncode': state['module_rc'],
                'timeout_reason': 'runtime' if state['module_rc'] else None}

    scope = {'pathlib': pathlib, 'json': json, 'uuid4': uuid4, 'time': time,
             'VOL': str(tmp_path), 'PY': '/python', 'REPO': '.', 'writers': Registry(),
             'vol': SimpleNamespace(commit=lambda: None),
             '_sync_repo': lambda: {'sha': 'fixed'}, '_sh': shell, '_topology': lambda: {},
             '_record': lambda step, ok, details: {'step': step, 'ok': ok, **details}}
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<actual p_infra_probe>', 'exec'), scope)
    return scope['p_infra_probe'], calls, state


def test_failed_cpu_phase_cannot_overwrite_earlier_log(entry, tmp_path):
    function, calls, state = entry
    state['pytest_rc'] = 1
    assert function('b13_hardware', 'test', 'cpu')['ok'] is False
    log = tmp_path / '_audit/infra/B13/test/cpu_tests.log'
    first = log.read_text()
    with pytest.raises((FileExistsError, RuntimeError)):
        function('b13_hardware', 'test', 'cpu')
    assert log.read_text() == first
    assert len(calls) == 1


def test_skipped_cpu_test_blocks_actual_gpu_dispatch(entry):
    function, calls, state = entry
    state['skipped'] = True

    class Remote:
        remote = staticmethod(function)
        def with_options(self, **kwargs):
            pytest.fail('skipped CPU checks must not allocate GPU')

    result = execute_probe(Remote(), 'b13_hardware', 'test', 'gpu')
    assert result['cpu']['ok'] is False
    assert len(calls) == 1


def test_gpu_execution_has_telemetry_and_terminal_process_evidence(entry, tmp_path):
    function, calls, state = entry
    assert function('b13_hardware', 'test', 'cpu')['ok'] is True
    state['module_rc'] = -1
    result = function('b13_hardware', 'test', 'gpu')
    assert result['ok'] is False
    assert result['process']['timed_out'] is True
    assert result['process']['process_returncode'] == -1
    assert pathlib.Path(result['telemetry_path']).is_file()
    assert (tmp_path / '_audit/infra/B13/test/gpu_execution.json').is_file()


def test_direct_gpu_call_rejects_missing_cpu_test_evidence(entry, tmp_path):
    function, _, _ = entry
    root = tmp_path / '_audit/infra/B13/test/cpu'
    root.mkdir(parents=True)
    (root / 'result.json').write_text('{"health_ok":true,"interfaces":{"nccl_compiled":true}}')
    with pytest.raises((OSError, RuntimeError)):
        function('b13_hardware', 'test', 'gpu')


def test_training_entry_runs_its_real_tests_before_its_module(entry, tmp_path):
    function, calls, _ = entry
    cpu = function('b03_training', 'test', 'cpu')
    assert cpu['ok'] is True
    assert 'tests/infra/test_training_probe.py' in shlex.split(calls[0])
    assert '-m syncopate.infra.training_probe' in calls[1]
    assert (tmp_path / '_audit/infra/B03/test/cpu_execution.json').is_file()
