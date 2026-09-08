"""Exact CPU-to-GPU identity gate shared by mechanism probe dispatchers."""

def validate_preflight(prior, current, phase):
    if prior.get('ok') is not True or prior.get('phase') != phase:
        raise ValueError('CPU preflight is incomplete or belongs to another phase')
    for key in ('sources', 'lock_sha256', 'overlay_sha256', 'image_id'):
        if key not in prior or key not in current or prior[key] != current[key]:
            raise ValueError('CPU/GPU preflight identity mismatch: ' + key)
