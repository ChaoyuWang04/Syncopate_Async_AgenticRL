"""B07: exact captured GDN Linear, shape/precision kernels and causal substitution."""
from __future__ import annotations
import argparse
import inspect
import json
from pathlib import Path

from syncopate.infra.gdn_boundary_probe import (
    PROJECTION, B03_SOURCE_SHA, tensor_metric, logits_summary, compare_top1)
from syncopate.infra.batch_layer_probe import validate_input
from syncopate.infra.probability_probe import MODEL, sha_file, verify_model, compare_logprobs

CAPTURE = Path('/vol/_audit/infra-probes/B05/batch3-gdn-01/attempt-266daa98c57143ab9b6e022642fa48f5/gdn-captures.pt')
CAPTURE_SHA = '24c0face5106870ac0001256d7773f694eb63243b4c478251b40b0e88bffbe23'
TOKEN_SHA = '3cb2f099114e7a95e9cd485d05998abc40ebac1e7dced96e6a69a2aa23c14d40'
REVISION = '995ad96eacd98c81ed38be0c5b274b04031597b0'
BATCHES = (1, 2, 4, 8)
MODES = ('bf16_on', 'bf16_off', 'fp32')
NEIGHBORS = ('duplicate', 'real', 'zero')


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def kernel_index(trace):
    kernels = [{'name': e['name'], 'grid': e.get('args', {}).get('grid'),
                'block': e.get('args', {}).get('block'), 'duration_us': e.get('dur')}
               for e in trace.get('traceEvents', []) if e.get('cat') == 'kernel']
    if not kernels:
        raise ValueError('profiler recorded no actual CUDA kernel events')
    return kernels


def validate_operator(result):
    expected = {(m, b, n) for m in MODES for b in BATCHES for n in NEIGHBORS}
    rows = result['records']
    if len(rows) != len(expected) or {(r['mode'], r['batch'], r['neighbors']) for r in rows} != expected:
        raise ValueError('incomplete operator matrix')
    if not result['capture_replay_equal'] or not all(
            r['target_equal'] and r['repeat_equal'] and r['profile_equal'] and r['kernels'] for r in rows):
        raise ValueError('operator identity/repeat/profiler gate failed')


def detailed_metric(reference, value):
    """FP64 comparison, exact difference count, fixed absolute-error bins."""
    import torch
    a, b = reference.detach().cpu().double(), value.detach().cpu().double()
    result = tensor_metric(a, b)
    delta = (a-b).abs()
    edges = [0., 2**-24, 2**-20, 2**-16, 2**-12, 2**-8, 1.]
    counts = [int(((delta > lo) & (delta <= hi)).sum()) for lo, hi in zip(edges, edges[1:])]
    result.update(elements=a.numel(), changed=int((delta != 0).sum()),
                  histogram={'zero': int((delta == 0).sum()), 'right_closed_edges': edges,
                             'counts': counts, 'above_last': int((delta > edges[-1]).sum())})
    return result


def batch_input(target, neighbor, size, kind):
    import torch
    other = {'duplicate': target, 'real': neighbor, 'zero': torch.zeros_like(target)}[kind]
    return torch.cat([target] + [other]*(size-1), dim=0).contiguous()


def profile_linear(linear, x, path):
    import torch
    from torch.profiler import profile, ProfilerActivity
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        value = linear(x)
        torch.cuda.synchronize()
    prof.export_chrome_trace(str(path))
    trace = json.loads(path.read_text())
    kernels = kernel_index(trace)
    ops = sorted({e['name'] for e in trace['traceEvents'] if e.get('cat') == 'cpu_op'})
    return value, {'path': str(path), 'sha256': sha_file(path), 'bytes': path.stat().st_size,
                   'kernels': kernels, 'aten_ops': ops}


def operator(linear, captures, output, *, cpu=False):
    import torch
    device = linear.weight.device
    target = captures['batch1_input'].to(device)
    neighbor = captures['batch2_input'][1:2].to(device)
    if not torch.equal(target[0], captures['batch2_input'][0].to(device)):
        raise ValueError('B05 identical input witness changed')
    backend = torch.backends.cuda.matmul
    old_reduction, old_precision = backend.allow_bf16_reduced_precision_reduction, backend.fp32_precision
    result = {'ok': False, 'records': [], 'reference': 'FP64 same BF16 inputs/weights; no original unquantized weight claim'}
    try:
        backend.fp32_precision = 'ieee'
        # Only one target needs FP64: rows do not interact mathematically.
        reference = torch.nn.functional.linear(target.double(), linear.weight.double(),
                      None if linear.bias is None else linear.bias.double()).cpu()
        base_weight = linear.weight.detach().clone()
        base_bias = None if linear.bias is None else linear.bias.detach().clone()
        backend.allow_bf16_reduced_precision_reduction = True
        with torch.inference_mode():
            one = linear(target).cpu()
            two = linear(captures['batch2_input'].to(device)).cpu()
        result['capture_replay_equal'] = torch.equal(one, captures['batch1_output']) and torch.equal(two, captures['batch2_output'])
        if not result['capture_replay_equal']:
            raise RuntimeError('actual Linear does not reproduce B05 captures')
        outputs = {}
        for mode in MODES:
            backend.allow_bf16_reduced_precision_reduction = mode == 'bf16_on'
            dtype = torch.float32 if mode == 'fp32' else base_weight.dtype
            linear.to(dtype=dtype)
            with torch.no_grad():
                linear.weight.copy_(base_weight.to(dtype))
                if base_bias is not None: linear.bias.copy_(base_bias.to(dtype))
            for size in BATCHES:
                for kind in NEIGHBORS:
                    x = batch_input(target, neighbor, size, kind).to(dtype)
                    with torch.inference_mode():
                        values = [linear(x).detach().cpu() for _ in range(3)]
                        flattened = linear(x.reshape(-1, x.shape[-1])).reshape_as(values[0]).detach().cpu()
                        if cpu:
                            profiled, trace = values[0], {'kernels': [{'name': 'CPU fixture; no CUDA claim'}]}
                        else:
                            profiled, trace = profile_linear(linear, x, output/f'gemm-{mode}-{size}-{kind}.trace.json')
                    target_out = values[0][:1]
                    outputs[mode, size, kind] = target_out
                    baseline = outputs[mode, 1, 'duplicate']
                    record = {'mode': mode, 'batch': size, 'neighbors': kind,
                              'M': size*target.shape[1], 'N': linear.out_features, 'K': linear.in_features,
                              'target_equal': torch.equal(x[:1], target.to(dtype)),
                              'repeat_equal': all(torch.equal(v, values[0]) for v in values),
                              'profile_equal': torch.equal(profiled.cpu(), values[0]),
                              'shape': list(x.shape), 'stride': list(x.stride()),
                              'flatten_2d_vs_3d': detailed_metric(values[0], flattened),
                              'fp32_precision': backend.fp32_precision,
                              'allow_bf16_reduced_precision_reduction': backend.allow_bf16_reduced_precision_reduction,
                              'vs_batch1': detailed_metric(baseline, target_out),
                              'vs_fp64': detailed_metric(reference, target_out),
                              'vs_fp64_rounded_bf16': detailed_metric(reference.to(base_weight.dtype), target_out), **trace}
                    result['records'].append(record)
                    write(output/'gemm-partial.json', result)
        # Same M for all targets, changing only irrelevant neighboring rows.
        result['fixed_M_neighbor_controls'] = [
            {'mode': mode, 'batch': size, 'neighbor': kind,
             'difference': detailed_metric(outputs[mode, size, 'duplicate'], outputs[mode, size, kind])}
            for mode in MODES for size in BATCHES for kind in ('real', 'zero')]
        validate_operator(result)
        result['ok'] = True
        write(output/'gemm-operator.json', result)
        return result
    finally:
        backend.allow_bf16_reduced_precision_reduction = old_reduction
        backend.fp32_precision = old_precision


def propagation(model, inputs, captures, output, *, device='cuda'):
    """Replace first projection only; same-shape sham and repeated interventions."""
    import torch
    modules = dict(model.named_modules())
    projection = modules[PROJECTION]
    layers = {n: m for n, m in modules.items() if type(m).__name__ == 'Qwen3_5MoeDecoderLayer'}
    if not layers: raise RuntimeError('no exact decoder layer class')
    result = {'ok': False, 'records': [], 'scope': 'teacher-forced same HF model; first projection intervention, no cross-engine claim'}
    baseline_layers, baseline_summary, baseline_projection = {}, {}, {}
    for batch in (1, 2):
        ids = torch.tensor(inputs['tokens'][:batch], device=device)
        kwargs = dict(input_ids=ids, position_ids=torch.tensor(inputs['positions'], device=device).unsqueeze(0).expand_as(ids),
                      attention_mask=torch.ones_like(ids), use_cache=False)
        with torch.inference_mode():
            off = logits_summary(model(**kwargs).logits, inputs['tokens'][0])
        for arm in ('observe', 'sham', 'substitute'):
            for repeat in range(2):
                seen, layer_values = {}, {}
                def replace(module, args, value):
                    if 'projection' in seen: raise RuntimeError('projection invoked twice')
                    seen['projection'] = True
                    original = value.detach().cpu().clone()
                    expected = captures[f'batch{batch}_output']
                    if not torch.equal(original, expected):
                        raise RuntimeError('full-model projection differs from B05 capture')
                    if arm == 'observe':
                        baseline_projection[batch] = original
                        return None
                    replacement = baseline_projection[batch][:1] if arm == 'sham' else baseline_projection[1][:1]
                    edited = value.clone()
                    edited[:1] = replacement.to(value.device)
                    if not torch.equal(edited[:1].cpu(), replacement): raise RuntimeError('substitution missed target')
                    if not torch.equal(edited[1:], value[1:]): raise RuntimeError('substitution changed neighbor')
                    seen['substitution_verified'] = True
                    return edited
                def hook(name):
                    def observe(module, args, value):
                        if name in layer_values: raise RuntimeError('decoder invoked twice')
                        tensor = value[0] if isinstance(value, tuple) else value
                        if tensor.ndim != 3 or tensor.shape[:2] != ids.shape:
                            raise RuntimeError('decoder output shape changed')
                        layer_values[name] = tensor[:1].detach().cpu().clone()
                    return observe
                handles = [projection.register_forward_hook(replace)] + [m.register_forward_hook(hook(n)) for n, m in layers.items()]
                try:
                    with torch.inference_mode(): summary = logits_summary(model(**kwargs).logits, inputs['tokens'][0])
                finally:
                    for handle in handles: handle.remove()
                if set(layer_values) != set(layers) or not seen.get('projection'):
                    raise RuntimeError('missing causal observer')
                if arm in ('observe', 'sham') or batch == 1:
                    if summary['full_logits_sha256'] != off['full_logits_sha256']:
                        raise RuntimeError('observer/sham/batch1 substitution altered logits')
                if arm == 'observe' and repeat == 0:
                    baseline_layers[batch] = layer_values
                    baseline_summary[batch] = summary
                metrics = [{'name': n, 'vs_single': detailed_metric(baseline_layers[1][n], t),
                            'vs_same_shape': detailed_metric(baseline_layers[batch][n], t)} for n, t in layer_values.items()]
                result['records'].append({'batch': batch, 'arm': arm, 'repeat': repeat, 'summary': summary,
                    'observer_or_sham_equal': summary['full_logits_sha256'] == off['full_logits_sha256'],
                    'substitution_verified': seen.get('substitution_verified', False), 'layers': metrics,
                    'chosen_logp_vs_single': compare_logprobs(baseline_summary[1]['logprobs'], summary['logprobs']),
                    'top1_vs_single': compare_top1(baseline_summary[1], summary)})
                write(output/'gemm-propagation-partial.json', result)
    result['ok'] = True
    write(output/'gemm-propagation.json', result)
    return result


def run(input_path, output, stage):
    import torch
    from importlib.metadata import version
    inputs = json.loads(input_path.read_text())
    validate_input(inputs)
    if inputs['tokens_sha256'] != TOKEN_SHA or inputs['revision'] != REVISION:
        raise ValueError('B05 input identity changed')
    versions = {p: version(p) for p in ('torch', 'transformers')}
    if versions != {'torch': '2.13.0', 'transformers': '5.10.4'}:
        raise RuntimeError('frozen B05 versions changed')
    verify_model(inputs)
    if sha_file(CAPTURE) != CAPTURE_SHA: raise ValueError('B05 captures changed')
    captures = torch.load(CAPTURE, map_location='cpu', weights_only=True)
    metadata = {'capture_sha256': CAPTURE_SHA, 'versions': versions, 'torch_git_version': torch.version.git_version,
                'cuda': torch.version.cuda, 'device': torch.cuda.get_device_name(), 'stage': stage,
                'tokens_sha256': TOKEN_SHA, 'revision': REVISION}
    write(output/'gemm-identity.json', metadata)
    result = {'ok': False, **metadata}
    if stage in ('operator', 'all'):
        from safetensors import safe_open
        index = json.loads((MODEL/'model.safetensors.index.json').read_text())
        key = PROJECTION + '.weight'
        if key not in index['weight_map']: raise ValueError('exact projection weight key missing')
        with safe_open(str(MODEL/index['weight_map'][key]), framework='pt', device='cpu') as file:
            weight = file.get_tensor(key)
        if tuple(weight.shape) != (2048, 4096) or weight.dtype != torch.bfloat16:
            raise ValueError('projection weight contract changed')
        if PROJECTION + '.bias' in index['weight_map']: raise ValueError('unexpected projection bias')
        linear = torch.nn.Linear(4096, 2048, bias=False, dtype=torch.bfloat16, device='cuda').eval()
        with torch.no_grad(): linear.weight.copy_(weight)
        result['operator'] = operator(linear, captures, output)['ok']
        del linear, weight
        torch.cuda.empty_cache()
    if stage in ('propagation', 'all'):
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(MODEL, local_files_only=True, dtype=torch.bfloat16,
                                                          attn_implementation='sdpa', device_map='cuda').eval()
        source = Path(inspect.getfile(type(dict(model.named_modules())[PROJECTION.rsplit('.', 1)[0]])))
        if sha_file(source) != B03_SOURCE_SHA: raise RuntimeError('model source changed from B05')
        result['propagation'] = propagation(model, inputs, captures, output)['ok']
    result['ok'] = True
    write(output/'gemm-result.json', result)
    return result


def cpu_check(output):
    import torch
    torch.manual_seed(137)
    linear = torch.nn.Linear(8, 4, bias=False, dtype=torch.bfloat16).eval()
    x = torch.randn(2, 3, 8).bfloat16()
    with torch.inference_mode():
        captures = {f'batch{b}_{key}': value for b in (1,2)
                    for key, value in [('input', x[:b].clone()), ('output', linear(x[:b]).clone())]}
    result = operator(linear, captures, output, cpu=True)
    assert detailed_metric(x, x + .25)['changed'] > 0
    assert not torch.cuda.is_initialized()
    class BatchSensitiveLinear(torch.nn.Linear):
        def forward(self, value):
            result = super().forward(value)
            if value.shape[0] == 2:
                result = result.clone()
                result[..., 0] += .25
            return result
    class Qwen3_5MoeDecoderLayer(torch.nn.Module):
        def __init__(self, sensitive=False):
            super().__init__()
            self.linear_attn = torch.nn.Module()
            self.linear_attn.out_proj = (BatchSensitiveLinear if sensitive else torch.nn.Linear)(8, 8, bias=False, dtype=torch.bfloat16)
            with torch.no_grad(): self.linear_attn.out_proj.weight.copy_(torch.eye(8, dtype=torch.bfloat16))
        def forward(self, value):
            return self.linear_attn.out_proj(value)
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.language_model = torch.nn.Module()
            self.model.language_model.layers = torch.nn.ModuleList([Qwen3_5MoeDecoderLayer(sensitive=True), Qwen3_5MoeDecoderLayer()])
            self.embedding = torch.nn.Embedding(8, 8, dtype=torch.bfloat16)
        def forward(self, input_ids, **kwargs):
            from types import SimpleNamespace
            value = self.embedding(input_ids)
            for layer in self.model.language_model.layers: value = layer(value)
            return SimpleNamespace(logits=value)
    tiny = Tiny().eval()
    inputs = {'tokens': [[1,2,3], [3,4,5]], 'positions': [0,1,2]}
    projection = dict(tiny.named_modules())[PROJECTION]
    with torch.inference_mode():
        captured = {f'batch{b}_output': projection(tiny.embedding(torch.tensor(inputs['tokens'][:b]))).clone() for b in (1,2)}
    causal = propagation(tiny, inputs, captured, output, device='cpu')
    assert len(causal['records']) == 12 and causal['ok']
    natural = [r for r in causal['records'] if r['batch'] == 2 and r['arm'] == 'observe']
    sham = [r for r in causal['records'] if r['batch'] == 2 and r['arm'] == 'sham']
    replaced = [r for r in causal['records'] if r['batch'] == 2 and r['arm'] == 'substitute']
    assert all(all(layer['vs_single']['changed'] > 0 for layer in r['layers']) for r in natural)
    assert all(r['observer_or_sham_equal'] for r in sham)
    assert all(r['substitution_verified'] and all(layer['vs_single']['equal'] for layer in r['layers'])
               and r['chosen_logp_vs_single']['max_abs'] == 0 for r in replaced)
    assert all(r['chosen_logp_vs_single']['max_abs'] > 0 for r in natural)
    result = {'ok': result['ok'] and causal['ok'], 'cuda_initialized': False,
              'causal_positive_control': 'batch2 firstprojection coordinate +0.25; natural differs, sham unchanged, replacement restores all layers and chosen-logp',
              'scope': 'tiny CPU shared operator matrix and causal intervention; actual CUDA profiler/model pending'}
    write(output/'gemm-cpu.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--input', type=Path)
    parser.add_argument('--stage', choices=('all', 'operator', 'propagation'), default='all')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if not args.cpu and args.input is None: parser.error('--input required')
    args.out.mkdir(parents=True, exist_ok=True)
    if any(args.out.glob('gemm-*')): raise RuntimeError('use fresh output directory')
    try:
        if args.cpu: cpu_check(args.out)
        else: run(args.input, args.out, args.stage)
    except Exception as exc:
        write(args.out/'gemm-failure.json', {'ok': False, 'type': type(exc).__name__, 'error': str(exc)})
        raise


if __name__ == '__main__': main()
