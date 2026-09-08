"""Actual verl FSDP OPD loss primitives, independent CPU reference; no trainer claim."""
from __future__ import annotations
import argparse
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace


def run_cpu(out: Path) -> dict:
    import torch
    import verl
    from verl.trainer.distillation.fsdp import losses as upstream
    from verl.trainer.ppo.core_algos import agg_loss
    from verl.trainer.distillation import losses as dispatcher

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.manual_seed(610)
    base = torch.randn(5, 4, 7)
    teacher = torch.randn(5, 4, 7).log_softmax(-1).detach()
    mask = torch.arange(4)[None, :] < torch.tensor([4, 1, 3, 2, 1])[:, None]
    mask[0, 1] = False  # Excluded channel inside an otherwise valid response.
    total = int(mask.sum())
    errors, grad_errors, cases = [], [], []

    def tokens(x, t, k):
        vals, ids = t.topk(k, dim=-1)
        # Formal FSDP logits processor receives unpadded tokens with a leading 1.
        vals_n = torch.nested.as_nested_tensor(list(vals), layout=torch.jagged)
        ids_n = torch.nested.as_nested_tensor(list(ids), layout=torch.jagged)
        cfg = SimpleNamespace(distillation_loss=SimpleNamespace(
            use_chunked_topk=False, log_prob_min_clamp=None))
        result = upstream.compute_forward_kl_topk(
            x.reshape(1, -1, 7), vals_n, ids_n, cfg, 'thd')
        return result['distillation_losses'].reshape(x.shape[:2])

    for k in (7, 3):
        xref = base.double().clone().requires_grad_()
        probs = teacher.double().exp()
        ids = teacher.topk(k, dim=-1).indices
        # Explicit sum over teacher-selected vocabulary; no KL helper reused.
        terms = probs * (teacher.double() - xref.log_softmax(-1))
        reference = (terms.gather(-1, ids).sum(-1) * mask).sum() / total
        gref, = torch.autograd.grad(reference, xref)
        for chunks in ((5,), (2, 2, 1), (1, 1, 1, 1, 1)):
            x = base.clone().requires_grad_()
            loss = x.sum() * 0
            offset = 0
            for count in chunks:
                sl = slice(offset, offset + count)
                loss = loss + agg_loss(tokens(x[sl], teacher[sl], k), mask[sl],
                                       'token-mean', batch_num_tokens=total)
                offset += count
            grad, = torch.autograd.grad(loss, x)
            le = abs(float(loss.detach()) - float(reference.detach()))
            ge = float((grad.double() - gref).abs().max())
            errors.append(le); grad_errors.append(ge)
            assert le <= 2e-6 and ge <= 2e-6, (k, chunks, le, ge)
            assert torch.count_nonzero(grad[~mask]) == 0
            cases.append({'topk': k, 'chunks': chunks, 'loss': float(loss.detach()),
                          'loss_error': le, 'gradient_error': ge})
    # Exercise actual direct-distillation dispatcher, including response slicing,
    # fresh empty global_batch_info, and its normalization wiring (#7200/#7225).
    dispatcher_cases = []
    for chunks in ((5,), (2, 2, 1), (1, 1, 1, 1, 1)):
        x = base.clone().requires_grad_()
        value = x.sum() * 0
        offset = 0
        for count in chunks:
            sl = slice(offset, offset + count)
            kl = tokens(x[sl], teacher[sl], 7)
            # One prompt token and four response tokens. Logit position t predicts t+1.
            packed = torch.cat([kl, torch.zeros(count, 1)], dim=1).flatten()
            mass = torch.ones_like(packed)
            data = {'prompts': torch.ones(count, 1, dtype=torch.long),
                    'responses': torch.ones(count, 4, dtype=torch.long),
                    'attention_mask': torch.ones(count, 5, dtype=torch.long),
                    'response_mask': mask[sl], 'dp_size': 1,
                    'batch_num_tokens': total, 'global_batch_size': 5}
            cfg = SimpleNamespace(loss_agg_mode='token-mean', global_batch_info={}, loss_scale_factor=None)
            dc = SimpleNamespace(distillation_loss=SimpleNamespace(loss_mode='forward_kl_topk',
                use_policy_gradient=False, loss_max_clamp=None, topk=7))
            part, _ = dispatcher.distillation_loss(cfg, dc,
                {'distillation_losses': packed, 'student_mass': mass, 'teacher_mass': mass}, data)
            assert cfg.global_batch_info['batch_num_tokens'] == total
            value = value + part
            offset += count
        actual_grad, = torch.autograd.grad(value, x)
        xr = base.double().clone().requires_grad_()
        expected = ((teacher.double().exp() * (teacher.double()-xr.log_softmax(-1))).sum(-1)*mask).sum()/total
        expected_grad, = torch.autograd.grad(expected, xr)
        error = float((actual_grad.double()-expected_grad).abs().max())
        assert error <= 2e-6
        assert abs(float(value.detach()-expected.detach())) <= 2e-6
        dispatcher_cases.append({'chunks': chunks, 'gradient_error': error, 'loss': float(value.detach())})
    x = base.clone().requires_grad_()
    raw = tokens(x, teacher, 7)
    whole = agg_loss(raw, mask, 'token-mean', batch_num_tokens=total)
    wrong = sum(agg_loss(raw[i:i+1], mask[i:i+1], 'token-mean') for i in range(5))
    wrong_delta = abs(float((whole-wrong).detach()))
    assert wrong_delta > 1e-5
    gradient, = torch.autograd.grad(whole, x, retain_graph=True)
    assert torch.count_nonzero(gradient) > 0
    reverse = (x.log_softmax(-1).exp() * (x.log_softmax(-1)-teacher)).sum(-1)
    direction_gap = abs(float(((reverse * mask).sum()/total - whole).detach()))
    assert direction_gap > 1e-5
    zero = agg_loss(raw, torch.zeros_like(mask), 'token-mean', batch_num_tokens=total)
    zg, = torch.autograd.grad(zero, x)
    assert zero.item() == 0 and torch.count_nonzero(zg) == 0
    same = base.clone().requires_grad_()
    same_loss = tokens(same, base.log_softmax(-1).detach(), 7).sum()
    same_grad, = torch.autograd.grad(same_loss, same)
    assert abs(same_loss.item()) <= 2e-6 and same_grad.abs().max().item() <= 2e-6
    p = torch.nn.Parameter(torch.ones(3))
    opt = torch.optim.AdamW([p], lr=.01, weight_decay=.1)
    before = p.detach().clone()
    (p.sum()*0).backward()
    assert torch.count_nonzero(p.grad) == 0
    opt.step()
    assert not torch.equal(p, before)
    # Record, do not silently fix, the primitive's undefined global empty denominator.
    empty = agg_loss(torch.ones_like(mask, dtype=torch.float32), torch.zeros_like(mask),
                     'token-mean', batch_num_tokens=0)
    # Reachable producer -> consumer counterexample, not a hand-invented zero mask.
    from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_rejection_mask
    import traceback
    initial_mask = torch.ones(1, 4)
    _, rejected, _ = compute_rollout_correction_and_rejection_mask(
        old_log_prob=torch.full((1, 4), -1.), rollout_log_prob=torch.full((1, 4), -3.),
        response_mask=initial_mask, rollout_rs='token_k2', rollout_rs_threshold=.1)
    _, kept, _ = compute_rollout_correction_and_rejection_mask(
        old_log_prob=torch.full((1, 4), -1.), rollout_log_prob=torch.full((1, 4), -1.),
        response_mask=initial_mask, rollout_rs='token_k2', rollout_rs_threshold=.1)
    assert rejected.sum().item() == 0 and torch.equal(kept, initial_mask)
    empty_dispatcher_cases = []
    for global_tokens in (0, 4):
        ex = base[:1].clone().requires_grad_()
        kl = tokens(ex, teacher[:1], 7)
        packed = torch.cat([kl, torch.zeros(1, 1)], dim=1).flatten()
        cfg = SimpleNamespace(loss_agg_mode='token-mean', global_batch_info={}, loss_scale_factor=None)
        dc = SimpleNamespace(distillation_loss=SimpleNamespace(loss_mode='forward_kl_topk',
            use_policy_gradient=False, loss_max_clamp=None, topk=7))
        data = {'prompts': torch.ones(1, 1, dtype=torch.long),
                'responses': torch.ones(1, 4, dtype=torch.long),
                'attention_mask': torch.ones(1, 5, dtype=torch.long),
                'response_mask': rejected, 'dp_size': 1,
                'batch_num_tokens': global_tokens, 'global_batch_size': 1}
        try:
            dispatcher.distillation_loss(cfg, dc, {'distillation_losses': packed,
                'student_mass': torch.ones_like(packed), 'teacher_mass': torch.ones_like(packed)}, data)
        except RuntimeError as exc:
            frames = traceback.extract_tb(exc.__traceback__)
            assert 'numel() == 0' in str(exc), str(exc)
            assert any(f.name == 'compute_forward_kl_topk' and 'student_mass.min()' in (f.line or '') for f in frames)
            trace = traceback.format_exc()
            (out / f'empty-dispatcher-global-{global_tokens}.txt').write_text(trace)
            empty_dispatcher_cases.append({'global_tokens': global_tokens, 'exception': type(exc).__name__,
                'message': str(exc), 'frames': [{'file': f.filename, 'line': f.lineno,
                'function': f.name, 'source': f.line} for f in frames], 'backward_reached': False})
        else:
            raise AssertionError('Expected upstream empty-mask error no longer reproduces; investigate version boundary')
    sources = {}
    for name, obj in [('fsdp_losses', upstream), ('agg_loss', agg_loss), ('dispatcher', dispatcher)]:
        source = inspect.getsource(obj)
        (out / (name + '.py')).write_text(source)
        sources[name] = {'path': inspect.getfile(obj), 'sha256': hashlib.sha256(source.encode()).hexdigest()}
    dispatch_source = inspect.getsource(dispatcher.distillation_loss)
    result = {'ok': True, 'scope': 'actual FSDP loss primitives; no trainer, GPU, DP or Megatron schedule',
              'torch': torch.__version__, 'verl': verl.__version__, 'sources': sources,
              'cases': cases, 'dispatcher_cases': dispatcher_cases,
              'empty_dispatcher_cases': empty_dispatcher_cases, 'rejection_helper_all_rejected': True, 'max_loss_error': max(errors), 'max_gradient_error': max(grad_errors),
              'wrong_local_denominator_detected': True, 'wrong_denominator_delta': wrong_delta,
              'forward_reverse_gap': direction_gap, 'masked_gradient_nonzero': 0,
              'zero_mask_gradient_nonzero': int(torch.count_nonzero(zg)),
              'same_distribution_gradient_max': same_grad.abs().max().item(),
              'global_empty_denominator_finite': bool(torch.isfinite(empty)),
              'weight_decay_only_has_target_signal': False,
              'dispatcher_populates_global_tokens': 'config.global_batch_info["batch_num_tokens"] = data["batch_num_tokens"]' in dispatch_source}
    (out / 'opd-cpu.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', action='store_true', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_cpu(args.out), allow_nan=False))


if __name__ == '__main__':
    main()
