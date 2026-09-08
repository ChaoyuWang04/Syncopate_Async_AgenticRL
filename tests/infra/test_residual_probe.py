import pytest
torch = pytest.importorskip("torch")
from syncopate.infra.residual_probe import observe, compare, substitute, target


def test_observer_and_causal_positive_negative_controls():
    class Sensitive(torch.nn.Module):
        def forward(self, x):
            y=x.clone()
            if len(x)==2: y[0,0,0]+=1
            return y
    model=Sensitive()
    x=torch.zeros(2,3,4)
    a,ca=observe(model, x[:1], names=[''])
    b,cb=observe(model, x, names=[''])
    assert compare(ca['']['output'],cb['']['output'])['changed']==1
    sham, _=observe(model,x,names=[''],replacement=('',cb['']['output']))
    fixed,_=observe(model,x,names=[''],replacement=('',ca['']['output']))
    assert torch.equal(sham,b)
    assert torch.equal(fixed[:1],a)
    assert torch.equal(fixed[1:],b[1:])
    assert compare(ca['']['args'],cb['']['args'])['equal']


def test_tuple_router_target_and_neighbor():
    x=(torch.randn(6,8),torch.randn(6,2),torch.ones(6,2,dtype=torch.long))
    reference=tuple(t[:3]*0 for t in x)
    edited=substitute(x,reference)
    assert all(torch.equal(t[:3],r) and torch.equal(t[3:],o[3:]) for t,r,o in zip(edited,reference,x))
    assert target(x,3)[0].shape==(3,8)


def test_bad_substitution_rejected():
    with pytest.raises(ValueError):substitute(torch.zeros(2,3),torch.zeros(1,4))


def test_joint_intervention_needed():
    from syncopate.infra.residual_probe import joint_fixture
    result=joint_fixture()
    assert result['single_changed']==1 and result['joint_changed']==0


def test_exact_dot_reference_distinguishes_nearer_arm():
    from syncopate.infra.residual_probe import exact_coordinates
    x=torch.tensor([[1.,2.]],dtype=torch.bfloat16)
    w=torch.tensor([[2.,3.]],dtype=torch.bfloat16)
    result=exact_coordinates(x,w,torch.tensor([[8.]]),torch.tensor([[9.]]))
    assert result['coordinates']==1
    assert result['counts']=={'single_nearer':1,'paired_nearer':0,'equal_distance':0}
    assert result['rows'][0]['exact_numerator']=='8'


def test_native_loader_contract_is_shared():
    from syncopate.infra.residual_probe import load_official_model
    from syncopate.infra.probability_probe import MODEL
    calls=[]
    class Fake:
        def eval(self):self.evaluated=True;return self
    fake=Fake()
    def loader(path,**kwargs):calls.append((path,kwargs));return fake
    assert load_official_model(loader) is fake and fake.evaluated
    assert calls==[(MODEL,{'local_files_only':True,'dtype':torch.bfloat16,'attn_implementation':'sdpa','device_map':'cuda'})]
    # No manual layer reconstruction or whole-layer dtype coercion is allowed.
    import inspect
    from syncopate.infra.residual_probe import followup,run
    assert 'load_official_model()' in inspect.getsource(followup)
    assert 'load_official_model()' in inspect.getsource(run)
    assert 'to_empty' not in inspect.getsource(followup)
