"""Required target-CPU acceptance; dependency absence fails rather than skips."""
from syncopate.infra.opd_contract_probe import run_cpu


def test_real_upstream_opd_contract(tmp_path):
    result = run_cpu(tmp_path)
    assert result['ok']
    assert result['wrong_local_denominator_detected']
    assert result['max_gradient_error'] <= 2e-6
    assert result['masked_gradient_nonzero'] == 0
    assert result['zero_mask_gradient_nonzero'] == 0
    assert (tmp_path / 'opd-cpu.json').is_file()
    assert result['rejection_helper_all_rejected']
    assert [c['global_tokens'] for c in result['empty_dispatcher_cases']] == [0, 4]
    assert all(c['exception'] == 'RuntimeError' for c in result['empty_dispatcher_cases'])
