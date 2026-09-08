import copy

import pytest

from syncopate.infra.fp8_probe import quantization_kwargs, validate_inventory


def test_online_moe_default_activation_is_selected_inside_method():
    from types import SimpleNamespace
    from syncopate.infra.fp8_probe import validate_moe_spec
    expected = SimpleNamespace(weight='block-fp8-key', activation=None)
    validate_moe_spec(SimpleNamespace(weight='block-fp8-key', activation=None), expected)
    for bad in [SimpleNamespace(weight=None, activation=None),
                SimpleNamespace(weight='other-key', activation=None),
                SimpleNamespace(weight='block-fp8-key', activation='explicit-override')]:
        with pytest.raises(ValueError):
            validate_moe_spec(bad, expected)


def inventory():
    return [
        {'name': 'model.layers.0.mlp.experts', 'is_moe': True, 'is_linear': False,
         'quant_method': 'Fp8PerBlockOnlineMoEMethod',
         'tensors': {'w13_weight': {'dtype': 'torch.float8_e4m3fn', 'finite': True},
                     'w2_weight': {'dtype': 'torch.float8_e4m3fn', 'finite': True},
                     'w13_weight_scale_inv': {'dtype': 'torch.float32', 'finite': True}}},
        {'name': 'model.layers.0.linear_attn.in_proj_qkvz', 'is_moe': False,
         'is_linear': True, 'quant_method': 'UnquantizedLinearMethod',
         'tensors': {'weight': {'dtype': 'torch.bfloat16', 'finite': True}}},
    ]


def test_kwargs_have_no_mutable_shared_state():
    a = quantization_kwargs()
    a['quantization_config']['linear']['weight'] = 'bad'
    assert quantization_kwargs()['quantization_config']['linear'] == {'weight': None, 'activation': None}
    assert quantization_kwargs()['kv_cache_dtype'] == 'auto'


def test_inventory_requires_real_weight_dtype_and_gdn_witness():
    assert validate_inventory(inventory())['moe_fp8_modules'] == 1
    for rows in [[], inventory()[:1], inventory()[1:]]:
        with pytest.raises(ValueError):
            validate_inventory(rows)
    rows = inventory()
    rows[0]['tensors']['w13_weight']['dtype'] = 'torch.bfloat16'
    with pytest.raises(ValueError):
        validate_inventory(rows)


@pytest.mark.parametrize('change', ['gdn_fp8', 'gdn_method', 'nonfinite', 'missing_scale'])
def test_rejects_bad_actual_mechanism(change):
    rows = copy.deepcopy(inventory())
    if change == 'gdn_fp8': rows[1]['tensors']['weight']['dtype'] = 'torch.float8_e4m3fn'
    if change == 'gdn_method': rows[1]['quant_method'] = 'Fp8PerBlockOnlineLinearMethod'
    if change == 'nonfinite': rows[0]['tensors']['w13_weight_scale_inv']['finite'] = False
    if change == 'missing_scale': del rows[0]['tensors']['w13_weight_scale_inv']
    with pytest.raises(ValueError):
        validate_inventory(rows)
