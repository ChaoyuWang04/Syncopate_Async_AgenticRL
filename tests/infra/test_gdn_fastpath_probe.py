import pytest
from syncopate.infra.gdn_fastpath_probe import validate_records


def record(arm):
    return dict(arm=arm, hits={'conv': int(arm == 'fast'), 'core': int(arm != 'torch')},
                comparisons=[dict(relative_l2=0.001, max_abs=0.001)], ms=[1., 2., 3.])


def test_reject_missing_real_fast_call():
    rows = [record(a) for a in ('torch', 'native', 'fast')]
    rows[2]['hits']['conv'] = 0
    with pytest.raises(ValueError, match='hit'):
        validate_records(rows)


def test_reject_inaccurate_gradients_and_empty_measurement():
    rows = [record(a) for a in ('torch', 'native', 'fast')]
    rows[2]['comparisons'][0]['relative_l2'] = .5
    with pytest.raises(ValueError, match='numerics'):
        validate_records(rows)
    rows[2]['comparisons'] = []
    with pytest.raises(ValueError):
        validate_records(rows)


def test_cpu_requires_both_imports_exact_source_and_versions():
    from copy import deepcopy
    from syncopate.infra.gdn_fastpath_probe import validate_dependencies, PINNED_VERSIONS, B03_SOURCE_SHA
    good={'cuda_initialized':False,'direct_imports':{p:{'ok':True} for p in ('causal_conv1d','fla.ops.gated_delta_rule')},
          'source_sha256':{'modeling':B03_SOURCE_SHA},'versions':dict(PINNED_VERSIONS)}
    validate_dependencies(good,cpu=True)
    for package in good['direct_imports']:
        bad=deepcopy(good);bad['direct_imports'][package]['ok']=False
        with pytest.raises(ValueError): validate_dependencies(bad,cpu=True)
    for key in PINNED_VERSIONS:
        bad=deepcopy(good);bad['versions'][key]='wrong'
        with pytest.raises(ValueError): validate_dependencies(bad,cpu=True)
    bad=deepcopy(good);bad['source_sha256']['modeling']='wrong'
    with pytest.raises(ValueError): validate_dependencies(bad,cpu=True)
    bad=deepcopy(good);bad['cuda_initialized']=True
    with pytest.raises(ValueError): validate_dependencies(bad,cpu=True)
