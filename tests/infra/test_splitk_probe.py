import pytest
from syncopate.infra.splitk_probe import validate_disabled, tensor_digest_bytes


def test_requires_algorithm_and_trace_proof():
    validate_disabled({'split_k':1,'reduction':0}, [{'name':'actual_gemm'}])
    for attributes,kernels in [({'split_k':4,'reduction':0},[{'name':'gemm'}]),
                                ({'split_k':1,'reduction':1},[{'name':'gemm'}]),
                                ({'split_k':1,'reduction':0},[{'name':'splitKreduce_kernel'}]),
                                ({'split_k':1,'reduction':0},[])]:
        with pytest.raises(ValueError): validate_disabled(attributes,kernels)


def test_hash_compares_bytes_not_statistics():
    assert tensor_digest_bytes(b'\x00\x01') != tensor_digest_bytes(b'\x01\x00')


def test_causal_gate_rejects_reselected_or_changed_algorithm():
    from syncopate.infra.splitk_probe import causal_status
    original = dict(id=4, tile=2, split_k=4, reduction=1, stages=3, swizzle=0, custom=0)
    base = dict(arm='unrestricted', supported=True, equals_torch_M64=True, algorithm=original,
                original_heuristic_algorithm=original)
    edited = dict(original, split_k=1, reduction=0)
    intervention = dict(arm='no_splitk_same_algorithm', supported=True,
                        algorithm=edited, original_heuristic_algorithm=dict(original))
    assert causal_status([base, intervention])['torch_attribution_ok']
    intervention['original_heuristic_algorithm']['id'] = 5
    with pytest.raises(ValueError, match='heuristic'): causal_status([base, intervention])
    intervention['original_heuristic_algorithm'] = dict(original)
    intervention['algorithm'] = dict(edited, tile=9)
    with pytest.raises(ValueError, match='nonintervened'): causal_status([base, intervention])


def test_diagnostic_success_does_not_imply_torch_attribution():
    from syncopate.infra.splitk_probe import causal_status
    rows = [dict(arm='unrestricted', supported=True, equals_torch_M64=False),
            dict(arm='no_splitk_preference', supported=True)]
    result = causal_status(rows)
    assert result['diagnostic_ok'] and not result['torch_attribution_ok']
