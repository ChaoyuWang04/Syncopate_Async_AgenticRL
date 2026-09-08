import pytest
from syncopate.infra.gdn_boundary_probe import compare_top1, validate_result


def test_top1_reports_changes_and_margins_without_confusing_ties():
    a = {'top1_ids': [4, 5], 'top2_margin': [0., .5]}
    b = {'top1_ids': [7, 5], 'top2_margin': [0., .25]}
    result = compare_top1(a, b)
    assert result['changed_positions'] == [0]
    assert result['changed_tokens'] == 1
    assert result['reference_margin_at_changes'] == [0.]
    with pytest.raises(ValueError): compare_top1(a, {'top1_ids': [4], 'top2_margin': [0.]})
    with pytest.raises(ValueError): compare_top1(a, {'top1_ids': [4, 5], 'top2_margin': [0., float('nan')]})


def test_result_rejects_missing_replays_and_perturbing_observer():
    result = {'records': [{'batch_size': b, 'repeat': r, 'observer_equal': True}
                         for b in (1, 2) for r in range(3)],
              'replay_records': [{'repeat': r, 'identical_target': True,
                                  'single_matches_capture': True, 'batch_matches_capture': True}
                                 for r in range(3)]}
    validate_result(result)
    result['records'][0]['observer_equal'] = False
    with pytest.raises(ValueError): validate_result(result)
    result['records'][0]['observer_equal'] = True
    result['replay_records'][0]['batch_matches_capture'] = False
    with pytest.raises(ValueError): validate_result(result)
