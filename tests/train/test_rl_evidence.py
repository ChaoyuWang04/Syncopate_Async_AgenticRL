import copy

import pytest

from syncopate.train.rl_evidence import validate_trace


def example():
    return {"schema_version": 3, "response_ids": [5, 6, 7, 8], "response_mask": [1, 1, 0, 0],
            "response_logprobs": [-1., -2., 0., 0.], "segments": [
                {"type": "assistant", "token_count": 2, "mask": 1},
                {"type": "assistant_template", "token_count": 1, "mask": 0},
                {"type": "tool", "token_count": 1, "mask": 0}],
            "generations": [{"response_offset": 0, "retained_model_tokens": 2,
                             "raw_token_ids": [5, 6], "finish_reason": "stop"}]}


def test_quality_warnings_are_not_tensor_failures():
    counts = validate_trace(example(), {"placeholder_logprobs": 0, "truncated": True,
                                        "unclosed_think_turns": 1, "tool_errors": 9})
    assert counts["model_tokens"] == 2 and counts["template_tokens"] == 1


@pytest.mark.parametrize("kind", ["mask", "missing_probability", "nan", "token", "segment", "missing_stop"])
def test_wrong_training_signal_fails(kind):
    trace, metrics = copy.deepcopy(example()), {"placeholder_logprobs": 0}
    if kind == "mask": trace["response_mask"][2] = 1
    elif kind == "missing_probability": metrics["placeholder_logprobs"] = 1
    elif kind == "nan": trace["response_logprobs"][0] = float("nan")
    elif kind == "token": trace["response_ids"][0] = 999
    elif kind == "missing_stop": trace["generations"][0].pop("finish_reason")
    else: trace["segments"].pop()
    with pytest.raises(ValueError):
        validate_trace(trace, metrics)
