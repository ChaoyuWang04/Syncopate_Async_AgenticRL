"""runtime 侧 v15 契约（`25 §R4` 门槛③：训练=评测=部署同一份解析）。

★ 这组测试守的是 N5「一份契约」：runtime **不许**另抄一份解析逻辑。
   前科（decider.py 抬头）：vLLM 的 hermes parser ≠ 我们的 parse_step，
   两边不一致时「训练时的最优策略在线上就不是最优」。
"""

from __future__ import annotations

import importlib
import os

import pytest


@pytest.fixture
def v15(monkeypatch):
    """在 v15 契约下重新加载受影响的模块（进程内切换要显式 reload）。"""
    monkeypatch.setenv("SYNCOPATE_CONTRACT", "v15")
    import syncopate.core.contract as C
    importlib.reload(C)
    import syncopate.runtime.decider as D
    importlib.reload(D)
    yield D
    monkeypatch.delenv("SYNCOPATE_CONTRACT", raising=False)
    importlib.reload(C)
    importlib.reload(D)


TC = '<tool_call>\n{{"name": "{n}", "arguments": {a}}}\n</tool_call>'


def test_signal_becomes_terminal_proposal(v15):
    p = v15.VllmDecider._to_proposal(
        TC.format(n="session.defer", a='{"reason": "太新", "recheck_after_days": 3}'))
    assert p.kind == "final"
    assert p.final_answer["signal"] == "defer"
    assert p.final_answer["arguments"]["recheck_after_days"] == 3


@pytest.mark.parametrize("sig", ["defer", "clarify", "reject"])
def test_each_signal_reaches_runtime(v15, sig):
    args = {"defer": '{"reason": "x", "recheck_after_days": 1}',
            "clarify": '{"question": "哪条?", "missing_fields": ["campaign_id"]}',
            "reject": '{"reason_code": "policy", "explanation": "不行"}'}[sig]
    p = v15.VllmDecider._to_proposal(TC.format(n=f"session.{sig}", a=args))
    assert p.kind == "final" and p.final_answer["signal"] == sig


def test_plain_text_final_has_no_behavior_label(v15):
    """★ 纯文本终答的行为是**轨迹级**推导的，单步这里必须留空 —— 不许在 runtime 猜。"""
    p = v15.VllmDecider._to_proposal("ROAS 就是投入产出比。")
    assert p.kind == "final" and p.final_answer["behavior"] is None
    assert "投入产出比" in p.final_answer["text"]


def test_business_tool_still_flows(v15):
    p = v15.VllmDecider._to_proposal(
        TC.format(n="campaign.get_metrics", a='{"campaign_id": "CMP_1"}'))
    assert p.kind == "tool_call" and p.tool == "campaign.get_metrics"


def test_shell_residue_is_parse_error_not_silently_accepted(v15):
    """★ 契约回潮必须在 runtime 也被抓住 —— 静默接受等于线上悄悄退回 v14。"""
    p = v15.VllmDecider._to_proposal('```json\n{"behavior": "defer", "answer": {}}\n```')
    assert p.kind == "tool_call" and p.tool is None
    assert "shell_residue" in p.rationale


def test_thinking_is_surfaced(v15):
    p = v15.VllmDecider._to_proposal("<think>先想想</think>好的。")
    assert p.thinking == "先想想"


def test_multi_tool_still_blocked(v15):
    p = v15.VllmDecider._to_proposal(
        TC.format(n="a.b", a="{}") + TC.format(n="c.d", a="{}"))
    assert p.kind == "tool_call" and p.tool is None and "一个 tool call" in p.rationale


def test_runtime_uses_the_same_parser_as_training(v15):
    """★ N5 的硬判据：runtime 与训练**同一个函数**，不是两份长得像的实现。"""
    import inspect

    from syncopate.core import parsing_v15
    src = inspect.getsource(v15.VllmDecider._to_proposal_v15)
    assert "parse_step_v15" in src
    assert parsing_v15.parse_step_v15 is not None


def test_v14_default_unchanged(monkeypatch):
    """不设环境变量时，runtime 行为与 v14 逐字节相同。"""
    monkeypatch.delenv("SYNCOPATE_CONTRACT", raising=False)
    import syncopate.core.contract as C
    importlib.reload(C)
    import syncopate.runtime.decider as D
    importlib.reload(D)
    p = D.VllmDecider._to_proposal('```json\n{"behavior": "defer", "answer": {"a": 1}}\n```')
    assert p.kind == "final" and p.final_answer["behavior"] == "defer"
