"""Q18: observe GDN projection input/output, then replay the actual Linear.

Ordinary HF BF16 mechanism reference, not a trainer or generation benchmark.
"""
from __future__ import annotations
import argparse
import hashlib
import inspect
import json
import math
from pathlib import Path

from syncopate.infra.batch_layer_probe import validate_input
from syncopate.infra.probability_probe import MODEL, sha_file, verify_model

PROJECTION = 'model.language_model.layers.0.linear_attn.out_proj'
B03_SOURCE_SHA = '40da264c51fcfadd7b87271c5485f029ee9efd05e6e54b34728c8ee252db4c9c'


def compare_top1(a, b):
    n = len(a['top1_ids'])
    if not n or any(len(v[key]) != n for v in (a, b) for key in ('top1_ids', 'top2_margin')):
        raise ValueError('unaligned top1 summaries')
    if not all(math.isfinite(x) and x >= 0 for v in (a, b) for x in v['top2_margin']):
        raise ValueError('invalid top2 margins')
    changes = [i for i, (x, y) in enumerate(zip(a['top1_ids'], b['top1_ids'])) if x != y]
    return {'tokens': n, 'changed_tokens': len(changes), 'changed_positions': changes,
            'reference_margin_at_changes': [a['top2_margin'][i] for i in changes],
            'batch_margin_at_changes': [b['top2_margin'][i] for i in changes]}


def validate_result(result):
    if sorted((r['batch_size'], r['repeat']) for r in result['records']) != [(b, r) for b in (1, 2) for r in range(3)]:
        raise ValueError('incomplete full-model observations')
    if not all(r['observer_equal'] for r in result['records']):
        raise ValueError('observer changed full logits')
    replay = result['replay_records']
    if sorted(r['repeat'] for r in replay) != [0, 1, 2]:
        raise ValueError('incomplete projection replay')
    if not all(r[k] for r in replay for k in ('identical_target', 'single_matches_capture', 'batch_matches_capture')):
        raise ValueError('projection replay identity failed')


def tensor_metric(a, b):
    import torch
    if a.shape != b.shape or a.dtype != b.dtype or not a.numel():
        raise ValueError('unaligned activation tensors')
    a, b = a.detach().cpu(), b.detach().cpu()
    x, y = a.double(), b.double()
    if not torch.isfinite(x).all() or not torch.isfinite(y).all():
        raise ValueError('nonfinite activation tensor')
    distance = torch.linalg.vector_norm(y - x).item()
    norm = torch.linalg.vector_norm(x).item()
    return {'equal': torch.equal(a, b), 'max_abs': (y-x).abs().max().item(),
            'relative_l2': distance/norm if norm else (0. if distance == 0 else None),
            'difference_l2': distance, 'reference_l2': norm}


def logits_summary(logits, tokens):
    import torch
    if not torch.isfinite(logits).all():
        raise ValueError('nonfinite full logits')
    logits = logits.detach().cpu().contiguous()
    target = logits[0, :-1].float()
    best = target.topk(2, dim=-1).values
    lp = target.log_softmax(-1).gather(-1, torch.tensor(tokens[1:]).unsqueeze(-1)).squeeze(-1)
    return {'full_logits_sha256': hashlib.sha256(logits.view(torch.uint8).numpy().tobytes()).hexdigest(),
            'logprobs': lp.tolist(), 'top1_ids': target.argmax(-1).tolist(),
            'top2_margin': (best[:, 0]-best[:, 1]).tolist(),
            'margin_units': 'raw logit difference; top1 uses argmax first-index tie rule'}


def measure(model, projection, inputs, device, persist=lambda result: None):
    """Same observer/replay code used by target CPU fixture and real CUDA model."""
    import torch
    length = len(inputs['tokens'][0])
    result = {'ok': False, 'records': [], 'replay_records': [],
              'scope': 'teacher-forced first sequence; next-token positions 0..length-2',
              'replay_scope': 'same actual projection and weights; target input exactly identical across replay shapes'}
    captures, off_baselines = {}, {}
    with torch.inference_mode():
        for size in (1, 2):
            rows = inputs['tokens'][:size]
            ids = torch.tensor(rows, device=device)
            kwargs = {'input_ids': ids,
                      'position_ids': torch.tensor(inputs['positions'], device=device).unsqueeze(0).expand_as(ids),
                      'attention_mask': torch.tensor(inputs['attention_mask'], device=device).unsqueeze(0).expand_as(ids),
                      'use_cache': False}
            off = [logits_summary(model(**kwargs).logits, rows[0]) for _ in range(3)]
            off_baselines[size] = off[0]
            for repeat in range(3):
                capture = {}
                def keep(key, tensor):
                    if key in capture or tensor.ndim != 3 or tuple(tensor.shape[:2]) != (size, length):
                        raise RuntimeError('projection boundary shape/invocation mismatch')
                    if not tensor.is_contiguous() or tensor.numel() * tensor.element_size() > 16 * 1024**2:
                        raise RuntimeError('projection activation layout/bound changed')
                    capture[key] = tensor.detach().cpu().clone()
                def pre(module, args):
                    if len(args) != 1:
                        raise RuntimeError('projection must consume one positional tensor')
                    keep('input', args[0])
                def post(module, args, value):
                    keep('output', value)
                handles = [projection.register_forward_pre_hook(pre), projection.register_forward_hook(post)]
                try:
                    on = logits_summary(model(**kwargs).logits, rows[0])
                finally:
                    for handle in handles: handle.remove()
                if set(capture) != {'input', 'output'}:
                    raise RuntimeError('projection observer did not fire')
                if repeat == 0: captures[size] = capture
                reference = captures[1]
                record = {'batch_size': size, 'repeat': repeat,
                          'observer_equal': off[repeat]['full_logits_sha256'] == on['full_logits_sha256'],
                          'off': off[repeat], 'on_logits_sha256': on['full_logits_sha256'],
                          'off_same_shape_repeat_equal': off[repeat]['full_logits_sha256'] == off[0]['full_logits_sha256'],
                          'input_shape': list(capture['input'].shape),
                          'output_shape': list(capture['output'].shape),
                          'target_input_difference': tensor_metric(reference['input'][0], capture['input'][0]),
                          'target_output_difference': tensor_metric(reference['output'][0], capture['output'][0]),
                          'same_shape_input_repeat': tensor_metric(captures[size]['input'], capture['input']),
                          'same_shape_output_repeat': tensor_metric(captures[size]['output'], capture['output'])}
                result['records'].append(record); persist(result)
                if not record['observer_equal']:
                    raise RuntimeError('observer changed full logits; stop localization')
        # Replay only after removing all observers. Copy exact original target,
        # replacing only the target of the natural two-row capture.
        single = captures[1]['input'].to(device)
        natural = captures[2]['input'].to(device)
        hybrid = torch.cat([single, natural[1:]], dim=0)
        if not torch.equal(single[0], hybrid[0]) or not torch.equal(natural[1], hybrid[1]):
            raise RuntimeError('replay tensor identity failed')
        replay_baseline = None
        for repeat in range(3):
            a = projection(single).detach().cpu()
            b = projection(hybrid).detach().cpu()
            c = projection(natural).detach().cpu()
            if replay_baseline is None: replay_baseline = (a, b, c)
            row = {'repeat': repeat, 'identical_target': torch.equal(single[0], hybrid[0]),
                   'single_matches_capture': torch.equal(a, captures[1]['output']),
                   'batch_matches_capture': torch.equal(c, captures[2]['output']),
                   'same_input_batch_difference': tensor_metric(a[0], b[0]),
                   'natural_input_vs_identical_input_in_batch': tensor_metric(b[0], c[0]),
                   'repeat_single': tensor_metric(replay_baseline[0], a),
                   'repeat_hybrid': tensor_metric(replay_baseline[1], b),
                   'repeat_natural': tensor_metric(replay_baseline[2], c)}
            result['replay_records'].append(row); persist(result)
        result['top1_batch_difference'] = compare_top1(off_baselines[1], off_baselines[2])
    validate_result(result)
    result['ok'] = True
    return result, captures


def cpu_check(output):
    import torch
    from types import SimpleNamespace
    if torch.cuda.is_initialized(): raise RuntimeError('CPU check inherited CUDA')
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(8, 4)
            self.projection = torch.nn.Linear(4, 8, bias=False)
        def forward(self, input_ids, **kwargs):
            hidden = self.embedding(input_ids)
            # Known positive upstream drift, while replay keeps target identical.
            if input_ids.shape[0] == 2: hidden = hidden + 0.25
            return SimpleNamespace(logits=self.projection(hidden))
    torch.manual_seed(137)
    model = Tiny().eval()
    inputs = {'tokens': [[1, 2, 3], [4, 5, 6]], 'positions': [0, 1, 2], 'attention_mask': [1, 1, 1]}
    result, _ = measure(model, model.projection, inputs, 'cpu')
    assert result['records'][3]['target_input_difference']['max_abs'] > 0
    assert all(r['observer_equal'] for r in result['records'])
    assert not torch.cuda.is_initialized()
    result.update(scope='CPU tiny fixture only; shared observer and replay contract', cuda_initialized=False)
    (output/'gdn-cpu.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


def run(input_path, output):
    inputs = json.loads(input_path.read_text())
    validate_input(inputs); verify_model(inputs)
    import torch
    from importlib.metadata import version
    from transformers import AutoModelForImageTextToText
    torch.manual_seed(137)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL, local_files_only=True, dtype=torch.bfloat16,
        attn_implementation='sdpa', device_map='cuda').eval()
    modules = dict(model.named_modules())
    projection = modules[PROJECTION]
    if type(projection) is not torch.nn.Linear or projection.weight.dtype != torch.bfloat16:
        raise RuntimeError('expected actual BF16 torch Linear')
    source = Path(inspect.getfile(type(modules[PROJECTION.rsplit('.', 1)[0]])))
    if sha_file(source) != B03_SOURCE_SHA:
        raise RuntimeError('installed GDN modeling source differs from B03; re-register comparison')
    metadata = {'model': inputs['model'], 'revision': inputs['revision'],
                'input_sha256': sha_file(input_path), 'tokens_sha256': inputs['tokens_sha256'],
                'projection': PROJECTION, 'dtype': 'bfloat16', 'attention': 'sdpa',
                'modeling_source_sha256': sha_file(source),
                'versions': {p: version(p) for p in ('torch', 'transformers')}}
    def persist(result):
        (output/'gdn-partial.json').write_text(json.dumps({**metadata, **result}, indent=2)+'\n')
    result, captures = measure(model, projection, inputs, 'cuda', persist)
    tensor_path = output/'gdn-captures.pt'
    torch.save({f'batch{b}_{k}': t for b, data in captures.items() for k, t in data.items()}, tensor_path)
    result.update(metadata)
    result['captures'] = {'path': str(tensor_path), 'bytes': tensor_path.stat().st_size,
                          'sha256': sha_file(tensor_path), 'scope': 'first ON input/output, both shapes; no full-vocab logits'}
    result['peak_allocated'] = torch.cuda.max_memory_allocated()
    (output/'gdn-result.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--input', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if not args.cpu and args.input is None: parser.error('--input required for GPU probe')
    args.out.mkdir(parents=True, exist_ok=True)
    if any(args.out.glob('gdn-*')): raise RuntimeError('use a fresh artifact directory')
    if args.cpu: cpu_check(args.out)
    else: run(args.input, args.out)


if __name__ == '__main__': main()
