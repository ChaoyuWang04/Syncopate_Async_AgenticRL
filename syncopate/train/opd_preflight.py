"""独立 OPD 的 CPU 前置：只读冻结输入，测试和身份通过后才申请 GPU。"""
from __future__ import annotations

import argparse
from collections import Counter
import importlib.metadata
import json
import os
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET

from syncopate.pipeline.infra_probe import test_counts, tests_passed
from syncopate.pipeline.opd_input import bind_input, validate_request
from syncopate.pipeline.process_guard import run_guarded
from syncopate.pipeline.rl_input import file_sha256
from syncopate.pipeline.model_input import model_files


CHECKS = {'tests', 'inputs', 'runbook_inputs', 'adapter', 'data', 'tokenizers', 'no_cuda'}
TESTS = (
    'tests/pipeline/test_opd_input.py', 'tests/pipeline/test_rl_input.py',
    'tests/pipeline/test_pipeline_defaults.py', 'tests/pipeline/test_cloud_execution.py',
    'tests/train/test_opd_preflight.py', 'tests/train/test_opd_tokens.py',
    'tests/train/test_opd_render_v15.py', 'tests/train/test_opd_objective.py',
    'tests/train/test_opd_completion.py', 'tests/train/test_opd_run_gate.py',
    'tests/train/test_opd_updates.py',
    'tests/train/test_lora_adapter_check.py', 'tests/train/test_ckpt_to_adapter.py',
    'tests/pipeline/test_model_input.py',
    'tests/pipeline/test_opd_cloud_entry.py',
)


def validate_prompt_rows(rows: list[dict], exam_turns: set[str]) -> dict:
    if not rows:
        raise ValueError('冻结 OPD prompts 为空')
    seen, turns, families = set(), [], Counter()
    for row in rows:
        if (not isinstance(row, dict) or not isinstance(row.get('id'), str) or not row['id']
                or row['id'] in seen or row.get('family') not in {'chat', 'task', 'task_neg'}
                or not isinstance(row.get('turns'), list) or not row['turns']
                or any(not isinstance(text, str) or not text.strip() for text in row['turns'])):
            raise ValueError('OPD prompts 身份/结构无效；不静默删行')
        seen.add(row['id'])
        turns.extend(row['turns'])
        families[row['family']] += 1
    overlap = set(turns) & exam_turns
    if overlap or not families['chat'] or not families['task']:
        raise ValueError(f'OPD 与考卷逐字重合 {len(overlap)} 条，或缺少 chat/task 路由')
    return {'rows': len(rows), 'turns': len(turns), 'families': dict(families), 'exact_exam_overlap': 0}


def require_preflight(record: dict, run_id: str, input_run: str) -> None:
    checks = record.get('checks', {})
    if (record.get('health_ok') is not True or record.get('run_id') != run_id
            or record.get('input', {}).get('run_id') != run_id
            or record.get('input', {}).get('input_run') != input_run
            or set(checks) != CHECKS or any(value is not True for value in checks.values())
            or record.get('tests', {}).get('rc') != 0
            or not tests_passed(record.get('tests', {}).get('counts', {}))):
        raise ValueError('本轮 OPD CPU 证据缺失、身份不符或有未通过/跳过检查')
    source_sha = record.get('source_sha256')
    if not _sha(source_sha) or source_sha != os.environ.get('SYNCOPATE_LOCAL_SOURCE_SHA'):
        raise ValueError('CPU 前置与当前源码不一致')
    tests = record['tests']
    if (set(tests.get('files', {})) != set(TESTS) or not _hashes(tests['files'])
            or not _sha(tests.get('xml_sha256')) or set(tests.get('modules', [])) != _test_modules()):
        raise ValueError('CPU 测试文件/模块或 XML 身份不完整')
    for key in ('input', 'teacher'):
        files = record.get(key, {}).get('files', [])
        hashes = record.get(key, {}).get('content_sha256', {})
        if (not files or not _hashes(hashes) or set(hashes) != {item.get('path') for item in files}
                or len(files) != len(hashes)
                or any(type(item.get('bytes')) is not int or item['bytes'] <= 0 for item in files)):
            raise ValueError(f'{key} 文件与完整 CPU 哈希未绑定')
    inputs, adapter, data = record['input'], record.get('adapter', {}), record.get('data', {})
    from syncopate.core.model_paths import TEACHER_MODEL
    if (inputs.get('kind') != 'external_rl_for_opd_diagnostic' or record['teacher'].get('model') != TEACHER_MODEL
            or not _hashes(record.get('data_sha256', {}))
            or type(adapter.get('pairs')) is not int or adapter['pairs'] <= 0
            or type(adapter.get('nonzero_b_tensors')) is not int
            or not 0 < adapter['nonzero_b_tensors'] <= adapter['pairs']
            or adapter.get('tensors') != 2 * adapter['pairs']
            or data.get('exact_exam_overlap') != 0 or not data.get('families', {}).get('chat')
            or not data.get('families', {}).get('task')
            or record.get('tokenizers', {}).get('same_vocab') is not True
            or record.get('runbook_inputs', {}).get('rc') != 0):
        raise ValueError('CPU 记录没有完整的 adapter、数据、词表或实际 runbook 通过证据')


def _sha(value) -> bool:
    return isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) is not None


def _hashes(values) -> bool:
    return isinstance(values, dict) and bool(values) and all(_sha(value) for value in values.values())


def _test_modules() -> set[str]:
    return {Path(name).with_suffix('').as_posix().replace('/', '.') for name in TESTS}


def junit_evidence(path: Path) -> dict:
    counts = test_counts(path)
    if not tests_passed(counts):
        raise ValueError('CPU XML 含失败、跳过或没有测试')
    modules = {case.get('classname') for case in ET.parse(path).getroot().iter('testcase')}
    if modules != _test_modules():
        raise ValueError('CPU XML 没有覆盖每个登记测试模块，或混入未登记模块')
    return {'counts': counts, 'modules': sorted(modules), 'xml_sha256': file_sha256(path)}


def teacher_metadata(root: Path, *, hash_weights: bool = False) -> dict:
    from syncopate.core.model_paths import TEACHER_MODEL
    files = model_files(root / TEACHER_MODEL)
    result = {'model': TEACHER_MODEL, 'files': [
        {'path': str(path.relative_to(root)), 'bytes': path.stat().st_size, 'mtime_ns': path.stat().st_mtime_ns}
        for path in files]}
    if hash_weights:
        result['content_sha256'] = {str(path.relative_to(root)): file_sha256(path) for path in files}
    return result


def verify_preflight(root: Path, run_id: str, input_run: str) -> dict:
    """Re-read actual CPU evidence before launch; large immutable weights are not rehashed on GPU."""
    from syncopate.pipeline.split import DATA_VERSION
    directory = root / '_audit' / DATA_VERSION / 'runs' / run_id / 'preparation/opd-train'
    record = json.loads((directory / 'preflight.json').read_text())
    require_preflight(record, run_id, input_run)
    if bind_input(root, input_run, run_id) != record['input']:
        raise ValueError('OPD 实际上游与 CPU 输入不同')
    if teacher_metadata(root) != {key: value for key, value in record['teacher'].items() if key != 'content_sha256'}:
        raise ValueError('教师实际加载文件与 CPU 绑定不同')
    for side in ('input', 'teacher'):
        for name, sha in record[side]['content_sha256'].items():
            if Path(name).suffix in {'.json', '.jinja'} and file_sha256(root / name) != sha:
                raise ValueError('输入配置/tokenizer 内容已改变')
    data, hashes = frozen_inputs(root)
    if data != record['data'] or hashes != record['data_sha256']:
        raise ValueError('冻结 OPD 输入或考卷内容已改变')
    if {name: file_sha256(root / name) for name in TESTS} != record['tests']['files']:
        raise ValueError('目标测试源码与 CPU 不同')
    if any(record['tests'].get(key) != value for key, value in junit_evidence(directory / 'tests.xml').items()):
        raise ValueError('CPU XML 内容与汇总不同')
    return record


def frozen_inputs(root: Path) -> tuple[dict, dict]:
    # Reuse the project's explicit exam list, not a glob or a new version list.
    from syncopate.pipeline.multiturn import EXAM_FILES
    from syncopate.pipeline.split import DEFAULT_OPD_PROMPTS
    prompts = root / DEFAULT_OPD_PROMPTS
    rows = [json.loads(line) for line in prompts.read_text().splitlines() if line.strip()]
    exam_paths = [root / 'data/u_route' / name for name in EXAM_FILES]
    exam_turns = {turn for path in exam_paths for line in path.read_text().splitlines() if line.strip()
                  for turn in json.loads(line)['turns']}
    return validate_prompt_rows(rows, exam_turns), {
        str(path.relative_to(root)): file_sha256(path) for path in [prompts, *exam_paths]}


def run_check(command: list[str], log: Path, *, timeout: int = 600) -> dict:
    result = run_guarded(command, cwd=Path.cwd(), env=os.environ.copy(), timeout=timeout, stop_grace=105)
    with log.open('x') as stream:
        stream.write(result.pop('out'))
    return {**result, 'log': str(log)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-run', required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args(argv)
    validate_request('opd-train', 'smoke', args.run_id, args.input_run)
    if args.verify_only:
        try:
            verify_preflight(Path.cwd(), args.run_id, args.input_run)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            print(f'OPD CPU evidence failed: {exc}')
            return 3
        print('OPD CPU evidence matches current inputs and source')
        return 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        raise FileExistsError('已有 OPD 前置证据；失败重跑请换 run-id')
    with args.out.with_suffix('.claim.json').open('x') as claim:
        json.dump({'run_id': args.run_id, 'input_run': args.input_run}, claim)
    root = Path.cwd()
    result = {'health_ok': False, 'run_id': args.run_id, 'source_sha256': os.environ.get('SYNCOPATE_LOCAL_SOURCE_SHA'),
              'checks': {key: False for key in CHECKS}}
    try:
        from syncopate.core.model_paths import build_tokenizer_path, TEACHER_MODEL
        os.environ.setdefault('SYNCOPATE_TEST_TOKENIZER', build_tokenizer_path())
        os.environ.pop('PYTEST_ADDOPTS', None)
        xml = args.out.parent / 'tests.xml'
        result['tests'] = run_check([sys.executable, '-m', 'pytest', *TESTS, '-q', '-rs',
                                    f'--junitxml={xml}'], args.out.parent / 'tests.log')
        result['tests']['counts'] = test_counts(xml)
        result['checks']['tests'] = result['tests']['rc'] == 0 and tests_passed(result['tests']['counts'])
        if not result['checks']['tests']:
            raise ValueError('CPU 测试未全过；不分配 GPU')
        result['tests'].update(junit_evidence(xml))
        result['tests']['files'] = {name: file_sha256(root / name) for name in TESTS}
        record = bind_input(root, args.input_run, args.run_id, hash_weights=True)
        result['input'] = record
        result['checks']['inputs'] = True
        command = ['bash', 'scripts/v16_pipeline.sh', '--check-inputs', '--profile', 'smoke',
                   '--gate-mode', 'observe', '--run-id', args.run_id, '--opd-input-run', args.input_run, 'opd-train']
        result['runbook_inputs'] = run_check(command, args.out.parent / 'runbook_inputs.log', timeout=60)
        result['checks']['runbook_inputs'] = result['runbook_inputs']['rc'] == 0
        if not result['checks']['runbook_inputs']:
            raise ValueError('固定 runbook 输入检查未通过')
        from syncopate.train.lora_adapter_check import inspect_adapter
        result['adapter'] = inspect_adapter(root / record['adapter'])
        result['checks']['adapter'] = True
        result['data'], result['data_sha256'] = frozen_inputs(root)
        result['checks']['data'] = True
        from transformers import AutoTokenizer
        student = AutoTokenizer.from_pretrained(root / record['model'], local_files_only=True)
        teacher = AutoTokenizer.from_pretrained(root / TEACHER_MODEL, local_files_only=True)
        same_vocab = student.get_vocab() == teacher.get_vocab()
        if not same_vocab:
            raise ValueError('学生/教师词表不同，逐 token KL 不合法')
        result['teacher'] = teacher_metadata(root, hash_weights=True)
        result['tokenizers'] = {'same_vocab': same_vocab, 'vocab_size': len(student.get_vocab())}
        result['checks']['tokenizers'] = True
        import torch
        result['checks']['no_cuda'] = not torch.cuda.is_initialized()
        result['versions'] = {name: importlib.metadata.version(name) for name in ('torch', 'transformers', 'peft')}
        result['health_ok'] = all(result['checks'].values())
        require_preflight(result, args.run_id, args.input_run)
    except Exception as exc:
        result['health_ok'] = False
        result['error'] = f'{type(exc).__name__}: {exc}'
    with args.out.open('x') as output:
        json.dump(result, output, ensure_ascii=False, indent=2)
    print(json.dumps({'health_ok': result['health_ok'], 'checks': result['checks'], 'error': result.get('error')}, ensure_ascii=False))
    return 0 if result['health_ok'] else 3


if __name__ == '__main__':
    raise SystemExit(main())
