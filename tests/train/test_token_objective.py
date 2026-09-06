"""SFT 均值目标的独立整批参考、尾窗口与 CPU 双 rank 对照。"""
from datetime import timedelta

import pytest

torch = pytest.importorskip("torch")
import torch.distributed as dist
import torch.multiprocessing as mp

from syncopate.train.token_objective import finish_token_window, assert_replicated_parameters


def test_real_sft_sparse_forward_has_same_gradient_across_microbatch_splits(record_property):
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from syncopate.train.sft import collate, token_losses

    torch.manual_seed(7)
    model = Qwen3ForCausalLM(Qwen3Config(
        vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        tie_word_embeddings=True, attention_dropout=0.0,
    )).eval()
    rows = [
        {"input_ids": [5, 6, 7, 8, 9, 10], "loss_mask": [0, 0, 1, 1, 0, 0]},
        {"input_ids": [11, 12, 13, 14], "loss_mask": [0, 1, 1, 1]},
    ]
    whole, _ = token_losses(model, collate(rows, 0))
    whole.mean().backward()
    reference = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])
    model.zero_grad(set_to_none=True)
    total, count = 0.0, 0
    for row in rows:
        losses, _ = token_losses(model, collate([row], 0))
        losses.sum().backward()
        total += float(losses.detach().sum())
        count += losses.numel()
    result = finish_token_window(model.parameters(), local_tokens=count, local_loss_sum=total, device="cpu")
    actual = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])
    relative = float((actual - reference).norm() / reference.norm())
    record_property("relative_gradient_l2", relative)
    assert relative < 1e-4
    assert result["supervised_tokens"] == 5
    assert result["loss"] == pytest.approx(float(whole.detach().mean()), rel=1e-6)


def test_two_updates_including_short_tail_match_whole_window_reference():
    actual = torch.nn.Parameter(torch.tensor([0.25, -0.75], dtype=torch.float64))
    reference = torch.nn.Parameter(actual.detach().clone())
    actual_opt = torch.optim.SGD([actual], lr=0.1)
    reference_opt = torch.optim.SGD([reference], lr=0.1)
    # micro-batch 监督量为 2、1、3；最后一组只有一个 micro-batch。
    groups = [torch.tensor(x, dtype=torch.float64) for x in
              ([[1, 2], [3, 4]], [[5, 6]], [[7, 8], [9, 10], [11, 12]])]
    for window in (groups[:2], groups[2:]):
        reference_opt.zero_grad()
        (torch.cat(window) @ reference).square().mean().backward()
        reference_opt.step()
        actual_opt.zero_grad()
        total = 0.0
        for group in window:
            loss = (group @ actual).square().sum()
            loss.backward()
            total += float(loss.detach())
        result = finish_token_window([actual], local_tokens=sum(len(x) for x in window),
                                     local_loss_sum=total, device="cpu")
        actual_opt.step()
        torch.testing.assert_close(actual, reference, atol=1e-12, rtol=1e-12)
        assert result["supervised_tokens"] == 3


@pytest.mark.parametrize("tokens,loss", [(0, 0.0), (1, float("nan")), (1, float("inf"))])
def test_empty_or_nonfinite_window_cannot_reach_optimizer(tokens, loss):
    parameter = torch.nn.Parameter(torch.ones(2))
    with pytest.raises(ValueError):
        finish_token_window([parameter], local_tokens=tokens, local_loss_sum=loss, device="cpu")


def test_same_norm_is_not_same_parameter_identity():
    left = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    right = torch.nn.Parameter(torch.tensor([2.0, 1.0]))
    assert left.norm() == right.norm()
    assert assert_replicated_parameters([("p", left)]) != assert_replicated_parameters([("p", right)])


def _distributed_worker(rank, rendezvous, empty_rank):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=45))
    try:
        parameter = torch.nn.Parameter(torch.tensor([0.5, -0.25], dtype=torch.float64))
        rows = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.float64)
        mine = rows if empty_rank and rank == 0 else rows[:0] if empty_rank else rows[:1] if rank == 0 else rows[1:]
        local_sum = 0.0
        if len(mine):
            loss = (mine @ parameter).square().sum()
            loss.backward()
            local_sum = float(loss.detach())
        result = finish_token_window([parameter], local_tokens=len(mine), local_loss_sum=local_sum, device="cpu")
        reference = parameter.detach().clone().requires_grad_()
        expected_loss = (rows @ reference).square().mean()
        expected_loss.backward()
        torch.testing.assert_close(parameter.grad, reference.grad, atol=1e-12, rtol=1e-12)
        assert result["supervised_tokens"] == 3
        assert result["loss"] == pytest.approx(float(expected_loss.detach()))
        torch.optim.SGD([parameter], lr=0.1).step()
        assert_replicated_parameters([("p", parameter)])
        # 刻意改变 rank 1，不改变范数；所有 rank 都应收到同一个错误，不能死锁。
        if rank == 1:
            with torch.no_grad():
                parameter.copy_(parameter.flip(0))
        with pytest.raises(ValueError, match="指纹不同"):
            assert_replicated_parameters([("p", parameter)])
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("empty_rank", [False, True])
def test_cpu_two_rank_gradient_and_parameter_identity(tmp_path, empty_rank):
    mp.spawn(_distributed_worker, args=((tmp_path / "rendezvous").as_uri(), empty_rank),
             nprocs=2, join=True)
