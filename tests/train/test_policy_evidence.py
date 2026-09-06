import copy

import pytest

from syncopate.train.policy_evidence import validate_records


def records():
    tensor = {"sha256": "a", "shape": [2], "local_shape": [2], "dtype": "float32", "bytes": 8,
              "placements": None, "finite": True, "nonzero": 2}
    weights = {"x.lora_A.weight": tensor}
    updated = {"x.lora_A.weight": {**tensor, "sha256": "b"}}
    result = []
    for rank in (0, 1):
        result.append({"kind": "optimizer", "rank": rank, "attempt": 1, "completed_optimizer_calls": 1,
            "error": None, "parameters_before": weights, "parameters_after": updated,
            "changed_parameters": 1, "gradients_before_clip": weights, "reported_grad_norm": 1.,
            "state_steps_after": {"1.0": 1}})
        for version in (0, 1):
            sent = weights if version == 0 else updated
            result.append({"kind": "weight_source", "rank": rank, "global_steps": version,
                           "stream_complete": True, "tensors": sent})
            result.append({"kind": "weight_receiver", "replica_rank": rank, "global_steps": version,
                           "error": None, "tensors": sent, "add_lora_calls": [
                               {"accepted": True, "slot_present": True, "tensors": sent}]})
    values = {"tokens": 2, "mean_signed": 0., "mean_abs": .1, "max_abs": .1, "p99_abs": .1}
    result.append({"kind": "logprob", "step": 1, "keys": ["a", "b"],
        "tags": [{"min_global_steps": 0, "max_global_steps": 0}] * 2,
        "trainer_rollout": values, "trainer_repeat": values})
    return copy.deepcopy(result)


def test_complete_observations_still_do_not_claim_full_b03():
    result = validate_records(records(), steps=1, ranks=2)
    assert result["health_ok"] and result["not_proven"]


@pytest.mark.parametrize("fault", ["missing", "duplicate", "skip", "unchanged", "bad_grad", "bad_state",
                                    "bad_version", "wrong_receiver", "not_accepted", "stale", "nan", "fake_changed"])
def test_known_identity_failures_are_not_quality_warnings(fault):
    events = records()
    update = next(r for r in events if r["kind"] == "optimizer")
    receiver = next(r for r in events if r["kind"] == "weight_receiver")
    probability = events[-1]
    if fault == "missing": events.pop(0)
    if fault == "duplicate": events.append(copy.deepcopy(update))
    if fault == "skip": update["completed_optimizer_calls"] = 0
    if fault == "unchanged": update["changed_parameters"] = 0
    if fault == "fake_changed": update["parameters_after"] = copy.deepcopy(update["parameters_before"])
    if fault == "bad_grad": update["reported_grad_norm"] = "nan"
    if fault == "bad_state": update["state_steps_after"] = {"0.0": 1}
    if fault == "bad_version": receiver["global_steps"] = 8
    if fault == "wrong_receiver": receiver["tensors"] = {"wrong": {}}
    if fault == "not_accepted": receiver["add_lora_calls"][0]["accepted"] = False
    if fault == "stale": probability["tags"] = [{"min_global_steps": -1, "max_global_steps": -1}] * 2
    if fault == "nan": probability["trainer_rollout"]["mean_abs"] = float("nan")
    result = validate_records(events, steps=1, ranks=2)
    assert not result["health_ok"] and result["errors"]
