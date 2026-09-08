"""Q14 BF16 batch mechanism reference, explicitly not a trainer.

Hooks observe decoder boundaries and their leaves in execution order. Functional
GDN kernels and expert GEMMs are not individually visible. No backend patching.
"""
from __future__ import annotations
import argparse
import hashlib
import inspect
import json
import math
from pathlib import Path

from syncopate.infra.probability_probe import MODEL, validate_tokens, verify_model, sha_file


def validate_input(inputs):
    rows = inputs['tokens']
    validate_tokens(rows, inputs['config']['text_config']['vocab_size'])
    if len(rows) != 2 or len(rows[0]) != len(rows[1]) or rows[0] == rows[1]:
        raise ValueError('requires two distinct equal-length raw sequences')
    length = len(rows[0])
    if length > 256 or inputs['positions'] != list(range(length)) or inputs['attention_mask'] != [1] * length:
        raise ValueError('requires bounded unpadded contiguous positions')
    if hashlib.sha256(json.dumps(rows).encode()).hexdigest() != inputs['tokens_sha256']:
        raise ValueError('token identity changed')


def first_divergence(rows):
    return next((row['name'] for row in rows if row['max_abs'] > 0), None)


def validate_result(result):
    records = result['records']
    if sorted((r['batch_size'], r['repeat']) for r in records) != [(b, r) for b in (1, 2) for r in range(3)]:
        raise ValueError('missing or duplicate shape/repeat')
    for record in records:
        if not record['observer_equal'] or not record['layers']:
            raise ValueError('observer changed logits or observed no layers')
        if not record['logprobs'] or not all(math.isfinite(x) for x in record['logprobs']):
            raise ValueError('invalid final probabilities')


def run(inputs, output):
    validate_input(inputs)
    verify_model(inputs)
    import torch
    from importlib.metadata import version
    from transformers import AutoModelForImageTextToText
    torch.manual_seed(137)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL, local_files_only=True, dtype=torch.bfloat16,
        attn_implementation='sdpa', device_map='cuda').eval()
    modules = dict(model.named_modules())
    decoders = [n for n, m in modules.items() if type(m).__name__ == 'Qwen3_5MoeDecoderLayer']
    if not decoders:
        raise RuntimeError('no supported Qwen3_5MoeDecoderLayer found')
    experts = [n for n, m in modules.items() if type(m).__name__ == 'Qwen3_5MoeExperts']
    selected = {n: m for n, m in modules.items()
                if any(n == d or n.startswith(d + '.') for d in decoders)
                and not any(n.startswith(e + '.') for e in experts)
                and (not list(m.children()) or type(m).__name__ in {
                    'Qwen3_5MoeDecoderLayer', 'Qwen3_5MoeGatedDeltaNet',
                    'Qwen3_5MoeAttention', 'Qwen3_5MoeSparseMoeBlock', 'Qwen3_5MoeExperts'})}
    source = Path(inspect.getfile(type(modules[decoders[0]])))
    result = {'ok': False, 'engine': 'hf_mechanism_reference_not_trainer',
              'dtype': 'bfloat16', 'attention': 'sdpa', 'use_cache': False,
              'input_sha256': inputs['tokens_sha256'], 'revision': inputs['revision'],
              'versions': {p: version(p) for p in ('torch', 'transformers')},
              'modeling_source': {'path': str(source), 'sha256': sha_file(source)},
              'model_class': type(model).__name__, 'records': [],
              'limitations': ['functional kernels and internal expert GEMMs are not module boundaries',
                              'first observed divergence is not proof of root cause',
                              'three repeats share one GPU allocation; no performance conclusion'],
              'module_inventory': {n: type(m).__name__ for n, m in selected.items()}}
    length = len(inputs['tokens'][0])
    baseline = {}
    shape_baseline = {}
    baseline_bytes = 0

    def metric(a, b):
        if a.shape != b.shape or a.dtype != b.dtype:
            raise RuntimeError('activation alignment changed')
        x, y = a.double(), b.double()
        if not torch.isfinite(x).all() or not torch.isfinite(y).all():
            raise RuntimeError('nonfinite activation')
        delta = y - x
        norm = torch.linalg.vector_norm(x).item()
        distance = torch.linalg.vector_norm(delta).item()
        return {'max_abs': delta.abs().max().item(),
                'relative_l2': distance / norm if norm else (0.0 if not distance else None),
                'reference_l2': norm, 'difference_l2': distance,
                'equal': torch.equal(a, b)}

    def logits_hash(tensor):
        if not torch.isfinite(tensor).all():
            raise RuntimeError('nonfinite logits')
        return hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()

    def forward(batch):
        ids = torch.tensor(batch, device='cuda')
        positions = torch.tensor(inputs['positions'], device='cuda').unsqueeze(0).expand_as(ids)
        mask = torch.tensor(inputs['attention_mask'], device='cuda').unsqueeze(0).expand_as(ids)
        logits = model(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=False).logits
        return logits.detach().cpu()

    with torch.inference_mode():
        for batch_size in (1, 2):
            batch = inputs['tokens'][:batch_size]
            # OFF repetitions bracket no instrumentation; all full batch logits retained only as hashes.
            off = [logits_hash(forward(batch)) for _ in range(3)]
            shape_baseline = {}
            for repeat in range(3):
                layers, skipped, seen = [], [], set()

                def observe(name, module, args, value):
                    nonlocal baseline_bytes
                    if name in seen:
                        raise RuntimeError('module invoked more than once: ' + name)
                    seen.add(name)
                    router_ids = None
                    if type(module).__name__ == 'Qwen3_5MoeTopKRouter':
                        if not isinstance(value, tuple) or len(value) != 3:
                            raise RuntimeError('router output contract changed')
                        router_ids = value[2][:length].detach().cpu().clone()
                    tensor = value[0] if isinstance(value, tuple) else value
                    if not isinstance(tensor, torch.Tensor):
                        skipped.append({'name': name, 'reason': 'non-tensor output'}); return
                    shape = list(tensor.shape)
                    if tensor.ndim >= 3 and tensor.shape[0] == batch_size and tensor.shape[1] == length:
                        target = tensor[0]
                    elif tensor.ndim == 2 and tensor.shape[0] == batch_size * length:
                        target = tensor[:length]
                    else:
                        skipped.append({'name': name, 'shape': shape, 'reason': 'no verified token axis'}); return
                    if target.numel() > 2_000_000:
                        raise RuntimeError('activation exceeds registered bound')
                    target = target.detach().cpu().clone()
                    if batch_size == 1 and repeat == 0:
                        baseline_bytes += target.numel() * target.element_size()
                        if baseline_bytes > 512 * 1024**2:
                            raise RuntimeError('activation baseline exceeds 512 MiB')
                        baseline[name] = (target, router_ids)
                    if name not in baseline:
                        raise RuntimeError('observed module missing from single baseline')
                    if repeat == 0:
                        shape_baseline[name] = (target, router_ids)
                    reference, ref_ids = baseline[name]
                    row = {'name': name, 'class': type(module).__name__, 'shape': shape,
                           'target_shape': list(target.shape), **metric(reference, target),
                           'same_shape_repeat': metric(shape_baseline[name][0], target)}
                    if router_ids is not None:
                        row['router_ids'] = router_ids.tolist()
                        row['router_ids_equal_single'] = torch.equal(ref_ids, router_ids)
                        row['router_changed_tokens'] = (ref_ids != router_ids).any(-1).sum().item()
                    layers.append(row)

                handles = [m.register_forward_hook(
                    lambda m, args, value, name=n: observe(name, m, args, value)) for n, m in selected.items()]
                try:
                    logits = forward(batch)
                finally:
                    for handle in handles: handle.remove()
                if set(row['name'] for row in layers) != set(baseline):
                    raise RuntimeError('observer coverage differs across forwards')
                lp = logits[0, :-1].float().log_softmax(-1).gather(
                    -1, torch.tensor(batch[0][1:]).unsqueeze(-1)).squeeze(-1)
                digest = logits_hash(logits)
                record = {'batch_size': batch_size, 'repeat': repeat,
                          'off_logits_sha256': off[repeat], 'on_logits_sha256': digest,
                          'observer_equal': digest == off[repeat], 'off_repeat_equal': len(set(off)) == 1,
                          'logprobs': lp.tolist(), 'layers': layers, 'skipped': skipped,
                          'first_observed_divergence': first_divergence(layers),
                          'first_decoder_divergence': first_divergence([
                              row for row in layers if row['name'] in decoders])}
                result['records'].append(record)
                (output / 'layers-partial.json').write_text(json.dumps(result, indent=2) + '\n')
                if not record['observer_equal']:
                    raise RuntimeError('observer OFF/ON logits differ; localization invalid')
    validate_result(result)
    result['ok'] = True
    result['baseline_bytes'] = baseline_bytes
    result['peak_allocated'] = torch.cuda.max_memory_allocated()
    (output / 'layers-result.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if any(args.out.glob('layers-*.json')):
        raise RuntimeError('output already contains layer artifacts; use a fresh run directory')
    run(json.loads(args.input.read_text()), args.out)


if __name__ == '__main__':
    main()
