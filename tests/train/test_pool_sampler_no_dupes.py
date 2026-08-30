"""GRPO 组内重复题：排除窗口必须**跨 epoch / 跨续训**都成立（`25 §6⑧`⒝）。

⛔ 这条红在 08-29 就登记过（e31s12_cand_unified 1/400 步、smoke 3/400 步），
   当时的判据只说"新跑零重复"——**要等一次 400 步长跑才知道修没修好**。
   ⇒ 现在把它降成一条秒级的结构测试：直接消费采样器的输出流，
     按**任意**训练批大小切开，检查每个训练批里有没有重复题。

★ 为什么重复题是致命的：GRPO 的优势是**组内**比较算出来的。同一道题在一个训练批里
  出现两次 ⇒ 两组的 baseline 互相污染，梯度方向就不是这道题真实的相对优劣了。
"""
from __future__ import annotations

import pytest

from syncopate.train.main_ppo_pool import DynamicPoolSampler


class _DS:
    """最小数据集：只提供 extra_info.case_id 和长度（采样器只用这两样）。"""

    def __init__(self, n: int) -> None:
        self._n = n

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, i: int) -> dict:
        return {"extra_info": {"case_id": f"C{i:04d}"}}


def _stream(sampler, epochs: int) -> list[int]:
    out: list[int] = []
    for _ in range(epochs):
        out.extend(iter(sampler))
    return out


@pytest.mark.parametrize("train_batch", [8, 16, 24])
def test_no_duplicate_case_within_any_training_batch(train_batch: int) -> None:
    """★ 关键：训练批的边界**不保证**和采样批对齐，所以要按各种切法都验一遍。"""
    sampler = DynamicPoolSampler(_DS(200), batch_size=8, seed=7)
    idx = _stream(sampler, epochs=3)          # 跨 epoch 边界
    for start in range(0, len(idx) - train_batch, train_batch):
        window = idx[start:start + train_batch]
        assert len(set(window)) == len(window), (
            f"训练批（大小 {train_batch}，起点 {start}）里出现重复题：{window}")


def test_exclusion_window_survives_resume() -> None:
    """续训也要恢复排除窗口 —— 否则"续训第一批"和"中断前最后一批"可以撞题。"""
    a = DynamicPoolSampler(_DS(200), batch_size=8, seed=7)
    tail = _stream(a, epochs=1)[-8:]
    b = DynamicPoolSampler(_DS(200), batch_size=8, seed=7)
    b.load_state_dict(a.state_dict())
    head = next(iter([list(iter(b))]))[:8]
    assert not (set(tail) & set(head)), (
        f"续训第一批与中断前最后一批撞题：{sorted(set(tail) & set(head))}")
