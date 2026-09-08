import pytest
from syncopate.infra.lora_handoff_probe import assess_cycle, select_target


def test_select_only_last_full_attention():
    config={'layer_types':['linear_attention','full_attention','linear_attention','full_attention']}
    mapping={'model.language_model.layers.3.self_attn.o_proj.weight':'shard.safetensors'}
    assert select_target(config,mapping)=='model.language_model.layers.3.self_attn.o_proj.weight'
    with pytest.raises(ValueError):select_target(config,{})


def test_cycle_rejects_stale_version_and_accepts_recovery():
    rows={'base':[[-1.,-2.]]*2,'A':[[-1.1,-2.1]]*2,'B':[[-1.2,-2.2]]*2,
          'A_reload':[[-1.1,-2.1]]*2,'base_restored':[[-1.,-2.]]*2}
    assert assess_cycle(rows)['ok']
    rows['B']=rows['A']
    assert not assess_cycle(rows)['ok']


def test_cycle_rejects_noise_recovery_failure_and_nan():
    rows={'base':[[-1.,-2.]]*2,'A':[[-1.1,-2.1]]*2,'B':[[-1.2,-2.2]]*2,
          'A_reload':[[-1.3,-2.1]]*2,'base_restored':[[-1.,-2.]]*2}
    assert not assess_cycle(rows)['ok']
    rows['B']=[[float('nan'),-2.]]*2
    with pytest.raises(ValueError):assess_cycle(rows)
