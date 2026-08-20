"""SFT 数据集的长度契约。

守住的命题：**超过长度预算的样本必须硬报错，不许静默截断。**

2026-08-19 抓到的隐患：sft.py 曾是 `--max-length` 默认 4096 + 静默 `[:max_length]`
切片，而 v13 数据 92.6% 超过 4096 —— 实跑靠手传 6656 侥幸躲过。静默截断砍掉的是
轨迹**结尾**（最终结论那段），loss 恰恰集中在那里，且没有任何计数报警。
与 RL 侧 prompt 截断（defer 崩塌归因翻案）同族。
"""

from __future__ import annotations

import pandas as pd
import pytest

from syncopate.train.rollout_budget import MAX_PROMPT_LENGTH, MAX_RESPONSE_LENGTH
from syncopate.train.sft import SFT_MAX_LENGTH, PretokenizedDataset


def _write_parquet(path, rows):
    pd.DataFrame(rows).to_parquet(path)


def test_max_length_is_derived_from_rollout_budget():
    """守则⑨：长度上限是契约的推论，不是独立参数。"""
    assert SFT_MAX_LENGTH == MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH


def test_overlong_sample_refuses_to_start(tmp_path):
    """超长样本 ⇒ SystemExit，且报错里带条数和 case_id —— 不是切掉继续跑。"""
    p = tmp_path / "train.parquet"
    _write_parquet(p, [
        {"case_id": "ok", "input_ids": [1] * 10, "loss_mask": [0] * 5 + [1] * 5},
        {"case_id": "too_long", "input_ids": [1] * 20, "loss_mask": [1] * 20},
    ])
    with pytest.raises(SystemExit) as exc:
        PretokenizedDataset(p, max_length=16)
    assert "too_long" in str(exc.value)
    assert "1/2" in str(exc.value)


def test_within_budget_sample_is_not_touched(tmp_path):
    """预算之内的样本必须**原样**进训练 —— 一个 token 都不动。"""
    p = tmp_path / "train.parquet"
    ids, mask = list(range(12)), [0] * 6 + [1] * 6
    _write_parquet(p, [{"case_id": "ok", "input_ids": ids, "loss_mask": mask}])
    ds = PretokenizedDataset(p, max_length=16)
    assert ds[0]["input_ids"] == ids
    assert ds[0]["loss_mask"] == mask
