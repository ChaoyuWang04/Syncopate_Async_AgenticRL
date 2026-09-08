import pytest
from syncopate.infra.gemm_cause_probe import kernel_index, validate_operator


def test_cuda_gate_uses_kernel_events_not_host_ranges():
    trace = {'traceEvents': [{'cat': 'cpu_op', 'name': 'aten::linear'},
                             {'cat': 'kernel', 'name': 'actual_gemm', 'args': {'grid': [1,2,3], 'block': [128,1,1]}}]}
    assert kernel_index(trace)[0]['name'] == 'actual_gemm'
    with pytest.raises(ValueError, match='CUDA kernel'):
        kernel_index({'traceEvents': trace['traceEvents'][:1]})


def test_incomplete_or_drifting_replay_cannot_pass():
    with pytest.raises(ValueError):
        validate_operator({'records': [], 'capture_replay_equal': True})
    records = [{'mode': mode, 'batch': b, 'neighbors': n, 'target_equal': True,
                'repeat_equal': True, 'profile_equal': True, 'kernels': [{'name': 'gemm'}]}
               for mode in ('bf16_on', 'bf16_off', 'fp32') for b in (1,2,4,8)
               for n in ('duplicate', 'real', 'zero')]
    validate_operator({'records': records, 'capture_replay_equal': True})
    records[0]['profile_equal'] = False
    with pytest.raises(ValueError):
        validate_operator({'records': records, 'capture_replay_equal': True})
