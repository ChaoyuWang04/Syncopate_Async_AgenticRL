"""B 系列微探针的有限入口；不维护第二份训练命令。"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from syncopate.pipeline.cloud_execution import validate_run_id


def probe_spec(name: str, run_id: str) -> dict:
    validate_run_id(run_id)
    registry = {
        'b13_hardware': ('B13', 'syncopate.infra.hardware_probe', ['tests/infra/test_hardware_probe.py']),
        'b03_training': ('B03', 'syncopate.infra.training_probe',
                         ['tests/infra/test_hardware_probe.py', 'tests/infra/test_training_probe.py']),
    }
    if name not in registry:
        raise ValueError(f'未登记的 infra probe: {name}')
    experiment, module, tests = registry[name]
    return {'name': name, 'experiment': experiment, 'module': module,
            'gpus': 2, 'cpu': 8, 'memory': 32768, 'timeout': 1200,
            'relative_root': f'_audit/infra/{experiment}/{run_id}', 'test_files': tests}


def probe_command(python: str, spec: dict, output: str, mode: str) -> list[str]:
    if mode not in {'cpu', 'gpu'}:
        raise ValueError('probe mode 必须为 cpu 或 gpu')
    return [python, '-m', spec['module'], '--mode', mode, '--output', output]


def can_start_gpu(cpu_result: dict) -> bool:
    result = cpu_result.get('result', {})
    return (cpu_result.get('ok') is True and result.get('health_ok') is True
            and result.get('interfaces', {}).get('nccl_compiled') is True
            and tests_passed(cpu_result.get('tests', {})))


def test_counts(path: Path) -> dict:
    """Count real JUnit cases; absent, empty or malformed reports cannot pass."""
    try:
        cases = list(ET.parse(path).getroot().iter('testcase'))
    except (OSError, ET.ParseError) as exc:
        return {'error': str(exc)}
    return {'tests': len(cases), **{
        key: sum(case.find(tag) is not None for case in cases)
        for key, tag in [('failures', 'failure'), ('errors', 'error'), ('skipped', 'skipped')]}}


def tests_passed(counts: dict) -> bool:
    return (type(counts.get('tests')) is int and counts['tests'] > 0
            and all(type(counts.get(key)) is int and counts[key] == 0
                    for key in ('failures', 'errors', 'skipped')))


def execute_probe(function, name: str, run_id: str, mode: str) -> dict:
    spec = probe_spec(name, run_id)
    if mode not in {'cpu', 'gpu'}:
        raise ValueError('probe mode 必须为 cpu 或 gpu')
    results = {'cpu': function.remote(name, run_id, 'cpu')}
    if mode == 'gpu' and can_start_gpu(results['cpu']):
        gpu = 'B200' if spec['gpus'] == 1 else f"B200:{spec['gpus']}"
        results['gpu'] = function.with_options(gpu=gpu, cpu=spec['cpu'],
            memory=spec['memory'], timeout=spec['timeout']).remote(name, run_id, 'gpu')
    return results
