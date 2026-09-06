import pytest

from syncopate.pipeline.infra_probe import probe_spec, probe_command, can_start_gpu


def healthy_cpu():
    return {'ok': True, 'result': {'health_ok': True, 'interfaces': {'nccl_compiled': True}},
            'tests': {'tests': 4, 'failures': 0, 'errors': 0, 'skipped': 0}}


def test_registered_probe_has_bounded_resource_and_exact_module():
    spec = probe_spec('b13_hardware', 'b13_20260906a')
    assert spec['experiment'] == 'B13'
    assert spec['gpus'] == 2 and spec['timeout'] == 1200
    assert spec['module'] == 'syncopate.infra.hardware_probe'
    assert spec['relative_root'] == '_audit/infra/B13/b13_20260906a'


def test_training_probe_uses_own_registered_module_and_cpu_tests():
    spec = probe_spec('b03_training', 'training_test')
    assert spec['experiment'] == 'B03'
    assert spec['module'] == 'syncopate.infra.training_probe'
    assert spec['relative_root'] == '_audit/infra/B03/training_test'
    assert (spec['gpus'], spec['cpu'], spec['memory'], spec['timeout']) == (2, 8, 32768, 1200)
    assert 'tests/infra/test_training_probe.py' in spec['test_files']
    assert 'tests/infra/test_training_probe.py' not in probe_spec('b13_hardware', 'test')['test_files']


@pytest.mark.parametrize('name,run_id', [('all','ok'),('b13_hardware','../x'),('b13_hardware','x;rm')])
def test_unknown_probe_and_unsafe_run_are_rejected(name, run_id):
    with pytest.raises(ValueError):
        probe_spec(name, run_id)


def test_command_uses_module_and_no_training_parameters():
    spec = probe_spec('b13_hardware', 'test')
    argv = probe_command('/env/.venv/bin/python', spec, '/vol/_audit/infra/B13/test/gpu', 'gpu')
    assert argv == ['/env/.venv/bin/python', '-m', 'syncopate.infra.hardware_probe',
                    '--mode', 'gpu', '--output', '/vol/_audit/infra/B13/test/gpu']
    with pytest.raises(ValueError):
        probe_command('python', spec, '/vol/test', 'training')


@pytest.mark.parametrize('record,expected', [
    (healthy_cpu(), True),
    ({'ok': True, 'result': {'health_ok': True}}, False),
    ({'ok': True, 'result': {}}, False),
    ({'ok': False, 'result': {'health_ok': True}}, False),
    ({'ok': True, 'result': {'health_ok': False}}, False),
    ({}, False),
])
def test_missing_or_failed_cpu_evidence_never_allocates_gpu(record, expected):
    assert can_start_gpu(record) is expected


def test_cpu_failure_prevents_real_dispatch_gpu_options():
    from syncopate.pipeline import infra_probe
    assert hasattr(infra_probe, 'execute_probe'), '需要真正使用 CPU 结果的调度路径'

    class Function:
        def remote(self, *args):
            assert args[-1] == 'cpu'
            return {'ok': False, 'result': {'health_ok': False}}

        def with_options(self, **kwargs):
            pytest.fail('失败后不允许分配 GPU')

    result = infra_probe.execute_probe(Function(), 'b13_hardware', 'test', 'gpu')
    assert set(result) == {'cpu'}


def test_cpu_then_exact_gpu_dispatch_with_same_run_identity():
    from syncopate.pipeline import infra_probe
    assert hasattr(infra_probe, 'execute_probe'), '需要受限的 CPU→GPU 调度'
    calls = []

    class Function:
        def remote(self, *args):
            calls.append(args)
            return healthy_cpu()

        def with_options(self, **kwargs):
            assert len(calls) == 1
            assert kwargs == {'gpu': 'B200:2', 'cpu': 8, 'memory': 32768, 'timeout': 1200}
            return self

    result = infra_probe.execute_probe(Function(), 'b13_hardware', 'test', 'gpu')
    assert calls == [('b13_hardware','test','cpu'),('b13_hardware','test','gpu')]
    assert result['gpu']['ok']


@pytest.mark.parametrize('change', ['skipped', 'errors', 'failures', 'zero_tests', 'no_nccl'])
def test_cpu_success_label_cannot_hide_missing_checks(change):
    record = healthy_cpu()
    if change == 'zero_tests':
        record['tests']['tests'] = 0
    elif change == 'no_nccl':
        record['result']['interfaces']['nccl_compiled'] = False
    else:
        record['tests'][change] = 1
    assert not can_start_gpu(record)
