"""Exact local safetensors selection used by CPU model-identity checks."""
from __future__ import annotations

import json
from pathlib import Path


def model_files(directory: Path, *, indexed_only: bool = False) -> list[Path]:
    config = json.loads((directory / 'config.json').read_text())
    if not isinstance(config, dict) or config.get('transformers_weights') is not None:
        raise ValueError('模型含未登记的显式 transformers_weights，不能保证加载已校验文件')
    index = directory / 'model.safetensors.index.json'
    if index.is_file():
        mapping = json.loads(index.read_text()).get('weight_map')
        if (not isinstance(mapping, dict) or not mapping
                or any(not isinstance(name, str) or Path(name).name != name
                       or not name.endswith('.safetensors') for name in mapping.values())):
            raise ValueError('模型 shard 清单为空或包含越界路径')
        shards = sorted(set(mapping.values()))
    elif indexed_only:
        raise ValueError('本轮上游必须有 model.safetensors.index.json')
    else:
        shards = ['model.safetensors']
    payloads = {path.name for pattern in ('*.safetensors', '*.bin') for path in directory.glob(pattern)}
    if payloads != set(shards) or (directory / 'adapter_config.json').exists():
        raise ValueError('模型含未登记的竞争权重或 PEFT 配置，不能保证加载已校验分片')
    files = [directory / name for name in shards]
    files += sorted(directory.glob('*.json')) + sorted(directory.glob('*.jinja'))
    if any(not path.is_file() or path.stat().st_size <= 0 for path in files):
        raise ValueError('模型文件为空或缺失')
    return files
