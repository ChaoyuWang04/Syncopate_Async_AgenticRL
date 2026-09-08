"""B04 online MoE-only FP8 witnesses; no CUDA imports at module import time."""
from __future__ import annotations

import hashlib
import importlib
from importlib.metadata import version
from pathlib import Path


def quantization_kwargs():
    """Official v0.28 online schema; keep ordinary Linear and KV at BF16."""
    return {'quantization': 'fp8_per_block',
            'quantization_config': {'linear': {'weight': None, 'activation': None}},
            'kv_cache_dtype': 'auto'}


def validate_moe_spec(resolved, expected):
    # v0.28 OnlineQuantizationConfig._dispatch rejects explicit activation
    # overrides: the selected FP8 method chooses dynamic activation internally.
    if resolved != expected or resolved.weight is None or resolved.activation is not None:
        raise ValueError(f'Unexpected MoE spec: resolved={resolved!r}, expected={expected!r}')


def cpu_check():
    """Resolve EngineArgs only, without building an engine or loading weights.

    This is API/config evidence, not a GPU backend compatibility claim.
    """
    import torch
    if torch.cuda.is_initialized():
        raise RuntimeError('CUDA was already initialized before CPU config probe')
    from vllm.engine.arg_utils import EngineArgs
    from vllm.config.quantization import resolve_quantization_config
    args = EngineArgs(model='/vol/models/Qwen3.6-35B-A3B', dtype='bfloat16',
                      **quantization_kwargs())
    resolved = args.quantization_config
    expected = resolve_quantization_config('fp8_per_block', None)
    if resolved.linear.weight is not None or resolved.linear.activation is not None:
        raise ValueError('EngineArgs did not preserve ordinary Linear exclusions')
    validate_moe_spec(resolved.moe, expected.moe)
    if args.dtype != 'bfloat16' or args.kv_cache_dtype != 'auto':
        raise ValueError('BF16 model/KV contract changed')
    sources = {}
    for name in ['vllm.engine.arg_utils', 'vllm.config.quantization',
                 'vllm.model_executor.layers.quantization.online.fp8']:
        path = Path(importlib.import_module(name).__file__)
        sources[name] = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    if torch.cuda.is_initialized():
        raise RuntimeError('CPU config probe unexpectedly initialized CUDA')
    return {'ok': True, 'scope': 'EngineArgs resolution only; no engine or CUDA execution',
            'kwargs': quantization_kwargs(),
            'resolved': {'linear': {'weight': None, 'activation': None},
                         'moe': {'weight': str(resolved.moe.weight), 'activation': resolved.moe.activation,
                                 'activation_selection': 'selected internally by Fp8PerBlockOnlineMoEMethod'}},
            'versions': {name: version(name) for name in ['vllm', 'torch', 'transformers', 'verl']},
            'installed_source': sources, 'training_gdn_dependencies': gdn_dependencies()}


def validate_inventory(rows):
    """Fail closed on absent mechanism, changed precision, or invalid scales."""
    moe_count = gdn_count = 0
    for row in rows:
        tensors = row['tensors']
        if any(t['finite'] is not True for t in tensors.values()):
            raise ValueError('nonfinite tensor: ' + row['name'])
        if row['is_moe']:
            if 'Fp8PerBlockOnlineMoEMethod' not in row['quant_method']:
                raise ValueError('unexpected MoE quant method: ' + row['name'])
            for name in ['w13_weight', 'w2_weight']:
                if tensors.get(name, {}).get('dtype') != 'torch.float8_e4m3fn':
                    raise ValueError('MoE FP8 weight missing: ' + row['name'] + '.' + name)
            if not any('scale' in name for name in tensors):
                raise ValueError('MoE scale witness missing: ' + row['name'])
            moe_count += 1
        if row['is_linear'] and '.linear_attn.' in row['name']:
            if not row['quant_method'].endswith('UnquantizedLinearMethod'):
                raise ValueError('GDN Linear quantized: ' + row['name'])
            if tensors.get('weight', {}).get('dtype') != 'torch.bfloat16':
                raise ValueError('GDN weight is not BF16: ' + row['name'])
            gdn_count += 1
    if not moe_count or not gdn_count:
        raise ValueError('missing actual FP8 MoE or BF16 GDN Linear witnesses')
    return {'moe_fp8_modules': moe_count, 'gdn_bf16_linear_modules': gdn_count}


def inspect_model(model):
    """Run through LLM.apply_model; full scale/weight finiteness costs GPU time.

    Inspection is diagnostic overhead and must not enter performance timing.
    Shared expert fusion is recorded, not inferred from a name's absence.
    """
    import torch
    from vllm.model_executor.layers.linear import LinearBase

    def finite(tensor):
        # Bound scratch space even for an entire experts tensor or a packed,
        # non-contiguous weight; do not materialize a whole FP32 checkpoint.
        if tensor.numel() > 1_048_576:
            axis = next(i for i, size in enumerate(tensor.shape) if size > 1)
            half = tensor.shape[axis] // 2
            return finite(tensor.narrow(axis, 0, half)) and finite(
                tensor.narrow(axis, half, tensor.shape[axis] - half))
        return bool(torch.isfinite(tensor.detach().float()).all().item())

    rows = []
    shared = []
    for name, module in model.named_modules():
        if 'shared_expert' in name:
            shared.append({'name': name, 'class': type(module).__name__})
        is_linear = isinstance(module, LinearBase)
        # Loaded fused experts expose both named weight tensors. This remains
        # observable when the upstream FusedMoE factory chooses a subclass.
        is_moe = hasattr(module, 'w13_weight') and hasattr(module, 'w2_weight')
        method = getattr(module, 'quant_method', None)
        if not is_linear and not is_moe and method is None:
            continue
        tensors = dict(module.named_parameters(recurse=False))
        tensors.update(dict(module.named_buffers(recurse=False)))
        # Quantization scales can be plain Tensor attributes after repacking.
        tensors.update({key: val for key, val in vars(module).items()
                        if isinstance(val, torch.Tensor) and ('scale' in key or 'weight' in key)})
        entries = {}
        for key, val in tensors.items():
            if 'weight' not in key and 'scale' not in key:
                continue
            entries[key] = {'dtype': str(val.dtype), 'shape': list(val.shape),
                            'finite': finite(val)}
        rows.append({'name': name, 'class': type(module).__name__,
                     'is_linear': is_linear, 'is_moe': is_moe,
                     'quant_method': type(method).__name__ if method is not None else '',
                     'fp8_backend': str(getattr(method, 'fp8_backend', None)),
                     'experts_class': str(getattr(method, 'experts_cls', None)),
                     'tensors': entries,
                     'shared_expert_attributes': {key: val for key, val in vars(module).items()
                         if 'shared_expert' in key and isinstance(val, (int, float, bool, str, type(None)))}})
    counts = validate_inventory(rows)
    return {'ok': True, **counts, 'modules': rows, 'shared_expert_modules': shared,
            'shared_expert_note': 'Absent standalone modules do not prove absence: shared experts may be fused into FP8 MoE.'}



def gdn_dependencies():
    import importlib.util
    from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as module
    return {'causal_conv1d_installed':importlib.util.find_spec('causal_conv1d') is not None,
            'fla_installed':importlib.util.find_spec('fla') is not None,
            'fast_path_available':getattr(module,'is_fast_path_available',None),
            'functions':{n: callable(getattr(module,n,None)) for n in
                         ['causal_conv1d_fn','causal_conv1d_update','chunk_gated_delta_rule','fused_recurrent_gated_delta_rule']}}

if __name__ == '__main__':
    import argparse
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', type=Path, required=True, metavar='OUT')
    output = parser.parse_args().cpu
    output.mkdir(parents=True, exist_ok=True)
    result = cpu_check()
    (output / 'fp8-cpu.json').write_text(json.dumps(result, indent=2) + '\n')
