import copy

import pytest

torch = pytest.importorskip("torch")

from syncopate.train.policy_evidence import validate_batch_trace


def example():
    trace = {"prompt_ids": [1, 2], "response_ids": [3, 4, 5], "response_mask": [1, 0, 1],
             "response_logprobs": [-0.123456789, 0., -1.23456789]}
    tensors = {name: [torch.tensor(values)] for name, values in {
        "input_ids": [1, 2, 3, 4, 5], "prompts": [1, 2], "responses": [3, 4, 5],
        "response_mask": [1, 0, 1], "rollout_log_probs": trace["response_logprobs"],
        "old_log_probs": [-.13, float("nan"), -1.24], "repeat_log_probs": [-.13, 0., -1.24],
    }.items()}
    return tensors, trace


def test_saved_batch_matches_raw_generation_after_same_dtype_cast():
    tensors, trace = example()
    assert validate_batch_trace(tensors, 0, trace) == 2


@pytest.mark.parametrize("field", ["prompts", "responses", "input_ids", "response_mask", "rollout_log_probs"])
def test_wrong_training_input_is_not_probability_noise(field):
    tensors, trace = example()
    tensors[field][0][0] += 1
    with pytest.raises(ValueError):
        validate_batch_trace(tensors, 0, trace)


@pytest.mark.parametrize("field", ["old_log_probs", "repeat_log_probs"])
def test_nonfinite_active_probability_is_rejected(field):
    tensors, trace = example()
    tensors[field][0][0] = float("nan")
    with pytest.raises(ValueError, match="有限性"):
        validate_batch_trace(tensors, 0, trace)


def test_wrong_trace_cannot_be_matched_to_same_length_sample():
    tensors, trace = example()
    other = copy.deepcopy(trace)
    other["response_ids"][1] += 1
    with pytest.raises(ValueError, match="rollout"):
        validate_batch_trace(tensors, 0, other)
