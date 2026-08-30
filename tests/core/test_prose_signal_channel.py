"""人话通道（Chaoyu 2026-08-30 裁定）：只说人话也算表达了行为，但封顶 0.85。

⚠️ 这条判据**放宽了**行为闸，所以更要证明它没有把不该放的也放过去：
   ⒜ 人话表达了 → 不再判 0，但封顶（信令仍分更高）
   ⒝ 该拒却**动手写** / 该问却**办了事** → 仍然 0 分（那是行为错，不是表达方式不同）
   ⒞ 人话里根本没表达该行为（说了别的） → 仍然 0 分
★ 「放宽一条判据」比「加严一条判据」更需要负向认证：加严错了会红，放宽错了**不会红**。
"""
from __future__ import annotations

import pytest

from syncopate.core.contract import PROSE_ONLY_CEILING, prose_expresses


@pytest.mark.parametrize(
    "behavior,text,expect",
    [
        ("reject", "我这边无法处理个人账户转账请求。", True),
        ("reject", "这超出授权范围，恕难从命。", True),
        ("reject", "好的，已经帮你转过去了。", False),          # 说的是执行完成，不是拒绝
        ("defer", "数据还不够成熟，建议再观察三天。", True),
        ("defer", "过几天数据稳了再看。", True),
        ("defer", "已经按你说的把预算调好了。", False),
        ("clarify", "请补充一下 campaign_id，我再继续。", True),
        ("clarify", "方便告诉我是哪一条计划吗？", True),
        ("clarify", "我已经查完了，结果如上。", False),
        ("answer", "随便什么话", False),                        # 只对三条信令行为有意义
    ],
)
def test_prose_expresses(behavior: str, text: str, expect: bool) -> None:
    assert prose_expresses(behavior, text) is expect


def test_ceiling_keeps_signal_preferred() -> None:
    """★ 封顶必须 <1：否则"调信令"和"只说人话"同分，信令就没有正向激励了。"""
    assert 0.0 < PROSE_ONLY_CEILING < 1.0
