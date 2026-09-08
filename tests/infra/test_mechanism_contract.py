import pytest
from syncopate.infra.mechanism_contract import validate_preflight

def test_preflight_requires_exact_completed_phase_and_identity():
    current={'sources':{'probe':'sha'},'lock_sha256':'lock','overlay_sha256':'overlay','image_id':'image'}
    good={**current,'ok':True,'phase':'gemm_cpu'}
    validate_preflight(good,current,'gemm_cpu')
    for key,value in [('ok',False),('phase','lora_cpu'),('image_id','other'),('overlay_sha256','other'),('sources',{}),('lock_sha256','other')]:
        with pytest.raises(ValueError):validate_preflight({**good,key:value},current,'gemm_cpu')
