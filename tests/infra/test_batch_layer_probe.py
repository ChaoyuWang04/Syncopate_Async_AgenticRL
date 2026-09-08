import hashlib
import json
import pytest
from syncopate.infra.batch_layer_probe import validate_input, first_divergence, validate_result


def payload():
    rows = [[1, 2], [2, 3]]
    return {'tokens': rows, 'tokens_sha256': hashlib.sha256(json.dumps(rows).encode()).hexdigest(),
            'positions': [0, 1], 'attention_mask': [1, 1],
            'config': {'text_config': {'vocab_size': 4}}}


def test_input_rejects_identity_and_shape_drift():
    validate_input(payload())
    for key, value in [('tokens_sha256', 'bad'), ('positions', [1, 2]),
                       ('attention_mask', [1, 0]), ('tokens', [[1, 2], [3]])]:
        data = payload(); data[key] = value
        with pytest.raises(ValueError): validate_input(data)


def test_first_difference_respects_execution_order():
    rows = [{'name': 'z', 'max_abs': 0}, {'name': 'a', 'max_abs': .01}]
    assert first_divergence(rows) == 'a'
    assert first_divergence(rows[:1]) is None


def test_result_fails_closed_on_missing_observer_or_repeats():
    result = {'records': [{'batch_size': b, 'repeat': r, 'observer_equal': True,
               'layers': [{'name': 'layer', 'max_abs': 0}], 'logprobs': [0.]}
              for b in (1, 2) for r in range(3)]}
    validate_result(result)
    result['records'][0]['observer_equal'] = False
    with pytest.raises(ValueError): validate_result(result)
    result['records'][0]['observer_equal'] = True
    result['records'].pop()
    with pytest.raises(ValueError): validate_result(result)
