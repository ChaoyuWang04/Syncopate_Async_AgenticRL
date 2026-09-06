import json
from pathlib import Path

import pytest

from syncopate.train.generation_probe import parse_arms, sampling_for, validate_provenance


def test_only_allocate_requested_independent_arms():
    assert parse_arms('full') == ('full',)
    assert parse_arms('full,model_defaults') == ('full', 'model_defaults')
    for bad in ('', 'full,full', 'full,', 'other', 'full, other'):
        with pytest.raises(ValueError):
            parse_arms(bad)


def test_model_identity_requires_exact_base_and_adapter_path():
    from syncopate.core.model_paths import STUDENT_MODEL
    run = Path("/vol/checkpoints/grpo/v16_smoke_example")
    valid = dict(base=STUDENT_MODEL, adapter="checkpoints/sft/v16_smoke_example/SELECTED")
    validate_provenance(valid, run)
    for bad in ({**valid, "base": "models/wrong"},
                {**valid, "adapter": "checkpoints/sft/v16_smoke_example_old/SELECTED"},
                {**valid, "adapter": "checkpoints/sft/v16_smoke_example/not-selected"}):
        with pytest.raises(ValueError, match="身份不符"):
            validate_provenance(bad, run)


def test_diagnostic_sampling_does_not_mutate_shared_contract(tmp_path):
    from syncopate.train import rollout_budget as budget
    before = (budget.SAMPLING_TEMPERATURE, budget.SAMPLING_TOP_P, budget.SAMPLING_TOP_K)
    (tmp_path / "generation_config.json").write_text(json.dumps(dict(temperature=1, top_p=.95, top_k=20)))
    assert sampling_for("model_defaults", tmp_path)["top_p"] == .95
    assert tuple(sampling_for("full", tmp_path).values()) == before
    assert (budget.SAMPLING_TEMPERATURE, budget.SAMPLING_TOP_P, budget.SAMPLING_TOP_K) == before
    with pytest.raises(ValueError):
        sampling_for("unknown", tmp_path)


def test_eval_engine_consumes_per_request_sampling_on_cpu():
    pytest.importorskip("torch")
    pytest.importorskip("vllm")
    import asyncio
    import itertools
    from types import SimpleNamespace
    from syncopate.train.eval_local import VLLMEngine

    seen = []
    class Engine:
        async def generate(self, prompt, sampling, request_id, **kwargs):
            seen.append(sampling)
            yield SimpleNamespace(outputs=[SimpleNamespace(token_ids=[1], finish_reason="stop", stop_reason=2)])

    engine = object.__new__(VLLMEngine)
    engine.engine, engine._request_counter, engine.lora = Engine(), itertools.count(), None
    engine.sampling_defaults = dict(max_tokens=20, top_p=1.0, top_k=-1, temperature=1.0)
    result = asyncio.run(engine([3], {"max_tokens": 10, "top_p": .95, "top_k": 20, "seed": 1234}))
    assert (seen[0].max_tokens, seen[0].top_p, seen[0].top_k, seen[0].seed) == (10, .95, 20, 1234)
    assert engine.sampling_defaults["top_p"] == 1
    assert result.sampling_params["seed"] == 1234 and result.finish_reason == "stop"
