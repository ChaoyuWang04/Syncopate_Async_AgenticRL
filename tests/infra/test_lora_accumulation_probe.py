from syncopate.infra.lora_accumulation_probe import windows


def test_tail_window_uses_its_own_denominator_and_target_mask():
    batches=windows()
    assert [len(b) for b in batches]==[4,2]
    assert [[sum(r['loss_mask']) for r in b] for b in batches]==[[3,5,4,7],[2,6]]
    for batch in batches:
        assert len({r['row_id'] for r in batch})==len(batch)
        assert all(r['loss_mask'][0]==0 and r['loss_mask'][-1]==1 for r in batch)


def test_optimizer_projection_keeps_valid_frozen_group_members_visible():
    import pytest
    from syncopate.infra.lora_accumulation_probe import trainable_optimizer
    original={'states':{},'param_groups':[{'params':['base','lora'],'lr':.001}]}
    projected=trainable_optimizer(original,{'lora'},{'base','lora'})
    assert projected['param_groups'][0]['params']==['lora']
    assert original['param_groups'][0]['params']==['base','lora']
    for invalid in [
        {'states':{'base':{'step':1}},'param_groups':[{'params':['base','lora']}]},
        {'states':{},'param_groups':[{'params':['base']}]},
        {'states':{},'param_groups':[{'params':['lora','lora']}]},
    ]:
        with pytest.raises(ValueError): trainable_optimizer(invalid,{'lora'},{'base','lora'})
