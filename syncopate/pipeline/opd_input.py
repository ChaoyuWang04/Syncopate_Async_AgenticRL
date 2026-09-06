"""Bind an existing RL adapter and its exact SFT base for a separate OPD diagnostic."""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import json
from pathlib import Path
import re
import shlex
import sys

from syncopate.pipeline.cloud_execution import validate_run_id
from syncopate.pipeline.rl_input import bind_input as bind_sft, file_sha256, resolve_input as resolve_sft


def validate_request(stage: str, profile: str, run_id: str, input_run: str) -> None:
    for value in (run_id, input_run):
        validate_run_id(value)
    if run_id == input_run or profile != 'smoke' or stage not in {'opd-train', 'opd-eval'}:
        raise ValueError('外部 RL 只用于独立 smoke OPD 对照，不能冒充全链或 candidate')


def validate_cloud_request(stage: str, profile: str, run_id: str, input_run: str,
                           rl_input_run: str, real_steps: int) -> None:
    if input_run:
        validate_request(stage, profile, run_id, input_run)
    if input_run and rl_input_run:
        raise ValueError('外部 SFT 和外部 RL 输入不能同时指定')
    if type(real_steps) is not int or not 0 <= real_steps <= 16:
        raise ValueError('OPD 诊断更新数须为 0（使用默认）或 1～16')
    if real_steps and (not input_run or stage != 'opd-train' or profile != 'smoke'):
        raise ValueError('更新数覆盖只用于显式上游的 smoke opd-train')


def _describe(root: Path, input_run: str, run_id: str) -> tuple[dict, dict]:
    from syncopate.pipeline.split import DATA_VERSION
    validate_request('opd-train', 'smoke', run_id, input_run)
    audit = root / '_audit' / DATA_VERSION / 'runs' / input_run
    manifest_path = audit / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    stages = manifest.get('stages', {})
    if (manifest.get('run_id') != input_run or manifest.get('profile') != 'smoke'
            or stages.get('rl-train', {}).get('status') not in {'pass', 'warn'}
            or stages.get('rl-adapter', {}).get('status') != 'pass'):
        raise ValueError('上游不是指定的已完成 RL 和 adapter 的 smoke run')
    gate_path = audit / 'rl_run_gate.json'
    gate = json.loads(gate_path.read_text())
    checkpoint_root = root / f'checkpoints/grpo/{DATA_VERSION}_smoke_{input_run}'
    health_checks = gate.get('health_checks')
    if (gate.get('health_ok') is not True or gate.get('expected_profile') != 'smoke'
            or not isinstance(health_checks, dict) or not health_checks
            or any(value is not True for value in health_checks.values())
            or (root / gate.get('run_dir', '')).resolve() != checkpoint_root.resolve()
            or (root / gate.get('log_path', '')).resolve() != (audit / 'rl_train.log').resolve()):
        raise ValueError('上游 RL 健康闸未通过或没有对应本轮输入')

    base_binding = audit / 'rl_input.json'
    if base_binding.is_file():
        old = json.loads(base_binding.read_text())
        if old.get('kind') != 'external_sft_for_rl_diagnostic' or old.get('run_id') != input_run:
            raise ValueError('上游 RL 底座绑定身份错误')
        base = bind_sft(root, old['input_run'], input_run)
    else:
        base = resolve_sft(root, input_run, run_id)

    adapter = root / f'models/adapters/rl_{DATA_VERSION}_smoke_{input_run}/lora_adapter'
    export_path = adapter / 'export_manifest.json'
    exported = json.loads(export_path.read_text())
    declared_actor = Path(exported['source_actor'])
    actor = (root / declared_actor).resolve()
    try:
        suffix = actor.relative_to(checkpoint_root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError('adapter 来自另一个 RL run 的 actor') from exc
    match = re.fullmatch(r'global_step_([1-9][0-9]*)/actor', suffix)
    if match is None or int(match.group(1)) not in gate.get('metrics', {}).get('global_steps', []):
        raise ValueError('adapter actor 步数没有本轮真实更新证据')
    shards = exported.get('source_shards', [])
    matches = [re.fullmatch(r'model_world_size_([1-9][0-9]*)_rank_([0-9]+)\.pt', s['name']) for s in shards]
    if not matches or any(m is None for m in matches):
        raise ValueError('actor 分片清单无效或越界')
    world = int(matches[0].group(1))
    if (len(matches) != world or {int(m.group(1)) for m in matches} != {world}
            or {int(m.group(2)) for m in matches} != set(range(world))):
        raise ValueError('actor 分片缺 rank 或重复')

    actor_relative = checkpoint_root.relative_to(root) / suffix
    expected = {str(Path(base['model']) / name): sha for name, sha in base.get('content_sha256', {}).items()}
    expected.update({str(actor_relative / entry['name']): entry['sha256'] for entry in shards})
    expected.update({str(adapter.relative_to(root) / name): exported[key] for name, key in (
        ('adapter_config.json', 'adapter_config_sha256'), ('adapter_model.safetensors', 'adapter_weights_sha256'))})
    files = [Path(base['model']) / item['name'] for item in base['files']]
    files += [Path(name) for name in expected if not name.startswith(base['model'] + '/')]
    descriptions = []
    for path in sorted(set(files)):
        stat = (root / path).stat()
        if not (root / path).is_file() or stat.st_size <= 0:
            raise ValueError(f'输入文件为空或缺失: {path}')
        descriptions.append({'path': path.as_posix(), 'bytes': stat.st_size})
    for entry in shards:
        if (actor / entry['name']).stat().st_size != entry['bytes']:
            raise ValueError('actor 分片大小与 export manifest 不同')
    metadata = {
        'schema_version': 1, 'kind': 'external_rl_for_opd_diagnostic',
        'run_id': run_id, 'input_run': input_run, 'base_input_run': base['input_run'],
        'model': base['model'], 'adapter': adapter.relative_to(root).as_posix(),
        'actor': actor_relative.as_posix(), 'files': descriptions,
        'source_manifest_sha256': file_sha256(manifest_path), 'rl_gate_sha256': file_sha256(gate_path),
        'export_manifest_sha256': file_sha256(export_path),
        'rl_base_binding_sha256': file_sha256(base_binding) if base_binding.is_file() else None,
    }
    return metadata, expected


def bind_input(root: Path, input_run: str, run_id: str, *, hash_weights: bool = False) -> dict:
    from syncopate.pipeline.split import DATA_VERSION
    metadata, expected = _describe(root, input_run, run_id)
    target = root / '_audit' / DATA_VERSION / 'runs' / run_id / 'opd_input.json'
    existing = json.loads(target.read_text()) if target.exists() else None
    manifest_path = target.parent / 'manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if (existing is None or manifest.get('run_id') != run_id or manifest.get('profile') != 'smoke'
                or not isinstance(manifest.get('stages'), dict)
                or not set(manifest['stages']) <= {'opd-train', 'opd-eval'}):
            raise ValueError('输出 run 已有非本次独立 OPD 的账本，必须换 run-id')
    if existing is not None:
        hashes = existing.get('content_sha256')
        if ({key: existing.get(key) for key in metadata} != metadata
                or not isinstance(hashes, dict)
                or set(hashes) != {entry['path'] for entry in metadata['files']}
                or any(not isinstance(sha, str) or re.fullmatch(r'[0-9a-f]{64}', sha) is None for sha in hashes.values())
                or any(hashes.get(name) != sha for name, sha in expected.items())):
            raise ValueError('本轮 OPD 输入身份变化或没有 CPU 内容校验，必须换 run-id')
        if not hash_weights:
            return existing
    elif not hash_weights:
        raise ValueError('先在 CPU 绑定 OPD 输入内容，不能用 GPU 做准备')
    hashes = {entry['path']: file_sha256(root / entry['path']) for entry in metadata['files']}
    if any(hashes.get(name) != sha for name, sha in expected.items()):
        raise ValueError('OPD 输入内容与源 RL 底座/actor/adapter 身份不同')
    if existing is not None:
        if hashes != existing['content_sha256']:
            raise ValueError('OPD 输入内容改变，必须换 run-id')
        return existing
    record = {**metadata, 'content_sha256': hashes}
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('x') as output:
        json.dump(record, output, ensure_ascii=False, indent=2)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-run', required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--hash-weights', action='store_true')
    parser.add_argument('--shell-vars', action='store_true')
    args = parser.parse_args()
    with redirect_stdout(sys.stderr):
        record = bind_input(Path.cwd(), args.input_run, args.run_id, hash_weights=args.hash_weights)
    if args.shell_vars:
        print('MERGED=' + shlex.quote(record['model']))
        print('RL_ADAPTER=' + shlex.quote(str(Path(record['adapter']).parent)))
    else:
        print(json.dumps(record, ensure_ascii=False))


if __name__ == '__main__':
    main()
