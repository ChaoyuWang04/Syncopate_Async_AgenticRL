"""稀疏投影必须和朴素路径**逐位等价** —— 这是它唯一的正确性门槛。

★ 为什么这个测试非有不可
`token_losses` 绕开了 HF 的 `model(**batch).loss`，自己接管了「投影 + CE」这一段。
一旦错位（shift 差一位、mask 取反、按样本平均而不是按 token 平均），
loss 依然是个能降的数，训练照样"收敛"——但学的东西是错的。
这个项目已经栽过一次一模一样的跟头：SFT 标签 bug 让 val_loss 降到 0.0000，
而监督目标全是错的。**loss 会降不代表 loss 是对的。**

用一个极小的随机 Qwen3 结构跑，不下载权重、CPU 秒级。
"""

from __future__ import annotations

import torch

from syncopate.train.sft import collate, token_losses


def _tiny_model():
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(0)
    config = Qwen3Config(
        vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8, tie_word_embeddings=True,
    )
    return Qwen3ForCausalLM(config).eval()


def _batch(pad_id: int = 0):
    # 两条长度不同、监督位置也不同的样本 —— batch>1 且 mask 不齐是最容易错的情况
    rows = [
        {"input_ids": [5, 6, 7, 8, 9, 10], "loss_mask": [0, 0, 1, 1, 0, 0]},
        {"input_ids": [11, 12, 13, 14], "loss_mask": [0, 1, 1, 1]},
    ]
    return collate(rows, pad_token_id=pad_id)


def test_sparse_loss_matches_hf_loss():
    """稀疏投影的 mean 必须等于 HF 全量 logits 路径的 loss。"""
    model = _tiny_model()
    batch = _batch()

    with torch.no_grad():
        reference = model(**batch).loss
        losses, _ = token_losses(model, batch)

    assert losses.numel() == 5          # 两条样本的 loss_mask 里一共 5 个 1
    torch.testing.assert_close(losses.mean(), reference, rtol=1e-5, atol=1e-6)


def test_row_index_attributes_each_loss_to_its_sample():
    """rows 必须能把每个 loss 归到正确的样本 —— 分组报 val loss 全靠它。"""
    model = _tiny_model()
    batch = _batch()

    with torch.no_grad():
        losses, rows = token_losses(model, batch)
        # 逐条单独跑，和批量跑的结果对齐
        for i in range(2):
            single = {k: v[i: i + 1] for k, v in batch.items()}
            solo, _ = token_losses(model, single)
            torch.testing.assert_close(losses[rows == i], solo, rtol=1e-4, atol=1e-5)


def test_gradients_flow_to_all_supervised_positions():
    """被选中的位置要有梯度，没被选中的位置梯度必须是 0（而不是"没算"）。"""
    model = _tiny_model()
    batch = _batch()

    trunk, _ = model.model, model.lm_head
    captured = {}

    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output.last_hidden_state
        hidden.retain_grad()
        captured["hidden"] = hidden

    handle = trunk.register_forward_hook(hook)
    losses, _ = token_losses(model, batch)
    losses.mean().backward()
    handle.remove()

    grad = captured["hidden"].grad
    shift_labels = batch["labels"][:, 1:]
    supervised = (shift_labels != -100)
    # 监督位置（错开一位后对应 hidden 的第 i 列）梯度非零
    assert grad[:, :-1][supervised].abs().sum() > 0
    # 非监督位置只能通过注意力间接受影响，但**最后一层的直接梯度**为 0：
    # 它们根本没进 lm_head
    assert torch.count_nonzero(grad[:, :-1][~supervised]) == 0


def test_empty_supervision_returns_empty_without_crashing():
    """一个监督 token 都没有的 batch 不能炸 —— 训练循环靠 numel()==0 跳过。"""
    model = _tiny_model()
    batch = collate([{"input_ids": [3, 4, 5], "loss_mask": [0, 0, 0]}], pad_token_id=0)

    losses, rows = token_losses(model, batch)

    assert losses.numel() == 0 and rows.numel() == 0
