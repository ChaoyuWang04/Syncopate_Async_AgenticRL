"""Resolve exactly the HF safetensors payload, with no competing loader paths."""
import json

import pytest

from syncopate.pipeline.model_input import model_files


def fixture(root, *, single=False):
    (root / 'config.json').write_text('{}')
    if single:
        (root / 'model.safetensors').write_bytes(b'weight')
    else:
        (root / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {'w': 'part.safetensors'}}))
        (root / 'part.safetensors').write_bytes(b'weight')


@pytest.mark.parametrize('single', [False, True])
def test_exact_canonical_payload_is_selected(tmp_path, single):
    fixture(tmp_path, single=single)
    names = [path.name for path in model_files(tmp_path)]
    assert names == (['model.safetensors', 'config.json'] if single else
                     ['part.safetensors', 'config.json', 'model.safetensors.index.json'])


@pytest.mark.parametrize('kind', ['stray', 'competing', 'missing', 'peft', 'bin', 'escape', 'explicit', 'empty', 'no_config'])
def test_ambiguous_or_incomplete_loader_selection_is_rejected(tmp_path, kind):
    fixture(tmp_path)
    if kind == 'stray':
        (tmp_path / 'model.safetensors.index.json').unlink()
    elif kind == 'competing': (tmp_path / 'model.safetensors').write_bytes(b'wrong')
    elif kind == 'missing': (tmp_path / 'part.safetensors').unlink()
    elif kind == 'peft': (tmp_path / 'adapter_config.json').write_text('{}')
    elif kind == 'bin': (tmp_path / 'pytorch_model.bin').write_bytes(b'wrong')
    elif kind == 'escape':
        (tmp_path / 'model.safetensors.index.json').write_text('{"weight_map":{"w":"../elsewhere"}}')
    elif kind == 'explicit': (tmp_path / 'config.json').write_text('{"transformers_weights":"other.safetensors"}')
    elif kind == 'empty': (tmp_path / 'part.safetensors').write_bytes(b'')
    else: (tmp_path / 'config.json').unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        model_files(tmp_path)
