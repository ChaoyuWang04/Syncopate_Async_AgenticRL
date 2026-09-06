"""The shared adapter gate must reject non-finite payloads, not count them as learned."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from syncopate.train.lora_adapter_check import AdapterValidationError, inspect_adapter


@pytest.fixture
def adapter(tmp_path, monkeypatch):
    config = {'r': 2, 'lora_alpha': 4, 'task_type': 'CAUSAL_LM', 'peft_type': 'LORA',
              'target_modules': ['q_proj']}
    (tmp_path / 'adapter_config.json').write_text(json.dumps(config))
    (tmp_path / 'adapter_model.safetensors').write_bytes(b'fake payload for boundary test')
    values = {'model.q_proj.lora_A.weight': np.ones((2, 4), dtype=np.float32),
              'model.q_proj.lora_B.weight': np.ones((4, 2), dtype=np.float32)}

    class Handle:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def keys(self): return values.keys()
        def get_tensor(self, key):
            value = values[key]
            return SimpleNamespace(shape=value.shape, count_nonzero=lambda: np.int64(np.count_nonzero(value)),
                isfinite=lambda: np.isfinite(value), is_floating_point=lambda: value.dtype.kind == 'f')

    monkeypatch.setattr('safetensors.safe_open', lambda *args, **kwargs: Handle())
    return tmp_path, config, values


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf')])
@pytest.mark.parametrize('matrix', ['A', 'B'])
def test_nonfinite_adapter_payload_never_counts_as_training(adapter, value, matrix):
    directory, _, tensors = adapter
    tensors[f'model.q_proj.lora_{matrix}.weight'][0, 0] = value
    with pytest.raises(AdapterValidationError, match='finite'):
        inspect_adapter(directory)


@pytest.mark.parametrize('alpha', [float('nan'), float('inf'), -float('inf')])
def test_nonfinite_lora_scaling_is_rejected(adapter, alpha):
    directory, config, _ = adapter
    config['lora_alpha'] = alpha
    (directory / 'adapter_config.json').write_text(json.dumps(config))
    with pytest.raises(AdapterValidationError, match='metadata'):
        inspect_adapter(directory)


def test_integer_weights_are_not_a_valid_trainable_adapter(adapter):
    directory, _, tensors = adapter
    tensors['model.q_proj.lora_B.weight'] = np.ones((4, 2), dtype=np.int64)
    with pytest.raises(AdapterValidationError, match='floating'):
        inspect_adapter(directory)


def test_finite_nonzero_adapter_keeps_existing_statistics(adapter):
    directory, _, _ = adapter
    result = inspect_adapter(directory)
    assert result['pairs'] == 1 and result['nonzero_b_tensors'] == 1


@pytest.mark.parametrize('value', [1.0, float('nan'), float('inf')])
def test_real_safetensors_payload_is_checked_on_target_cpu(tmp_path, value):
    torch = pytest.importorskip('torch')
    from safetensors.torch import save_file
    (tmp_path / 'adapter_config.json').write_text(json.dumps({
        'r': 2, 'lora_alpha': 4, 'task_type': 'CAUSAL_LM', 'peft_type': 'LORA',
        'target_modules': ['q_proj']}))
    save_file({'model.q_proj.lora_A.weight': torch.ones(2, 4),
               'model.q_proj.lora_B.weight': torch.full((4, 2), value)},
              str(tmp_path / 'adapter_model.safetensors'))
    if value == 1.0:
        assert inspect_adapter(tmp_path)['pairs'] == 1
    else:
        with pytest.raises(AdapterValidationError, match='finite'):
            inspect_adapter(tmp_path)
