import pytest
from syncopate.infra.probability_probe import compare_logprobs, validate_tokens


def test_comparison_reports_alignment_and_nonfinite_errors():
    assert compare_logprobs([1., 2.], [1., 2.])['max_abs'] == 0
    assert compare_logprobs([1., 2.], [1.1, 2.])['max_abs'] == pytest.approx(.1)
    for a,b in [([1.], [1.,2.]), ([float('nan')], [1.]), ([],[])]:
        with pytest.raises(ValueError):
            compare_logprobs(a,b)


def test_tokens_must_be_original_valid_nonempty_sequences():
    validate_tokens([[3,4], [1,2,3]], 10)
    for rows in [[[]], [[-1,2]], [[1,10]], [[1.0,2]], [[True,2]]]:
        with pytest.raises(ValueError):
            validate_tokens(rows, 10)


def test_weight_index_requires_complete_exact_shard_set():
    from syncopate.infra.probability_probe import validate_shards
    validate_shards({'weight_map': {'x':'model-1.safetensors', 'y':'model-2.safetensors'}},
                    ['model-1.safetensors','model-2.safetensors'])
    for actual in [['model-1.safetensors'], ['model-1.safetensors','extra.safetensors']]:
        with pytest.raises(ValueError):
            validate_shards({'weight_map': {'x':'model-1.safetensors','y':'model-2.safetensors'}},actual)


def test_text_mrope_requires_all_three_axes_equal():
    from syncopate.infra.probability_probe import matches_prefill
    rows=[[10,11],[20,21]];flat=[10,11,20,21];pos=[0,1,0,1]
    assert matches_prefill({'tokens':flat,'positions':pos},rows)
    assert matches_prefill({'tokens':flat,'positions':[pos,pos,pos]},rows)
    assert not matches_prefill({'tokens':flat,'positions':[pos,pos,[0,2,0,1]]},rows)
    assert not matches_prefill({'tokens':flat,'positions':[]},rows)
    assert not matches_prefill({'tokens':flat[:-1],'positions':pos},rows)
