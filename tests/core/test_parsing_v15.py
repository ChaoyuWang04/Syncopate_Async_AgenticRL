"""v15 解析器与行为推导器的单测（`25 §R1` 门槛①：≥30 用例，三通道 × 五类畸形）。

★ 覆盖矩阵（每一格都要有用例，缺格比用例少更危险）：
    通道      正常 · 截断 · 嵌套 · 空文本 · 多调用
    think 段  有/无/空/多块
    信令      三种各一 · 混合 · 重复 · 壳残留
    推导      轨迹级三条规则
"""

from __future__ import annotations

import pytest

from syncopate.core.parsing_v15 import (
    ParsedStepV15, derive_behavior, parse_step_v15, parse_tool_calls,
    render_report, render_signal, strip_thinking,
)

TC = '<tool_call>\n{{"name": "{n}", "arguments": {a}}}\n</tool_call>'
DEFER = TC.format(n="session.defer", a='{"reason": "太新", "recheck_after_days": 3}')
CLARIFY = TC.format(n="session.clarify", a='{"question": "哪条?", "missing_fields": ["campaign_id"]}')
REJECT = TC.format(n="session.reject", a='{"reason_code": "unauthorized", "explanation": "越权"}')
BIZ = TC.format(n="campaign.get_metrics", a='{"campaign_id": "CMP_1"}')
REPORT = TC.format(n="session.report", a='{"decision": "hold"}')


# ── 通道一：think 段 ────────────────────────────────────────────────────
def test_think_stripped_and_recorded():
    body, had, inner = strip_thinking("<think>我在想</think>正文")
    assert body == "正文" and had is True and inner == "我在想"


def test_think_absent():
    body, had, inner = strip_thinking("正文")
    assert body == "正文" and had is False and inner == ""


def test_empty_think_counts_as_present_but_empty():
    """★ N3 的读数依据：空 think 块 ≠ 没有 think 块，两者必须分得开。"""
    _, had, inner = strip_thinking("<think>\n\n</think>正文")
    assert had is True and inner == ""


def test_multiple_think_blocks_concatenated():
    _, had, inner = strip_thinking("<think>A</think>x<think>B</think>y")
    assert had is True and "A" in inner and "B" in inner


def test_think_does_not_swallow_tool_call():
    p = parse_step_v15("<think>先查一下</think>" + BIZ)
    assert p.kind == "tool_calls" and p.had_thinking is True


# ── 通道二：业务工具 ────────────────────────────────────────────────────
def test_single_business_tool():
    p = parse_step_v15(BIZ)
    assert p.kind == "tool_calls" and p.tool_calls[0]["name"] == "campaign.get_metrics"


def test_multiple_business_tools_in_one_step():
    p = parse_step_v15(BIZ + BIZ)
    assert p.kind == "tool_calls" and len(p.tool_calls) == 2


def test_report_tool_is_not_terminal():
    """session.report 是非终止性的 —— 调它之后轨迹还要继续。"""
    p = parse_step_v15(REPORT)
    assert p.kind == "tool_calls" and p.signal is None


def test_arguments_as_nested_json_string():
    p = parse_step_v15('<tool_call>\n{"name": "x.y", "arguments": "{\\"a\\": 1}"}\n</tool_call>')
    assert p.kind == "tool_calls" and p.tool_calls[0]["arguments"] == {"a": 1}


def test_trailing_comma_is_repaired():
    p = parse_step_v15('<tool_call>\n{"name": "x.y", "arguments": {"a": 1,}}\n</tool_call>')
    assert p.kind == "tool_calls"


def test_single_quotes_are_not_repaired():
    p = parse_step_v15("<tool_call>\n{'name': 'x.y'}\n</tool_call>")
    assert p.kind == "error" and "malformed" in p.error


def test_truncated_tool_call_block():
    """截断：闭合标签没了 ⇒ 正则匹配不到 ⇒ 退化成纯文本，而不是崩。"""
    p = parse_step_v15('<tool_call>\n{"name": "x.y", "arguments": {}}')
    assert p.kind == "final_text"


def test_malformed_is_counted_not_silently_dropped():
    """★ 与 v14 的关键差别：畸形块必须被数出来判错，不能静默丢弃。"""
    calls, malformed = parse_tool_calls("<tool_call>\nnot json\n</tool_call>")
    assert calls == [] and malformed == 1


def test_tool_call_payload_not_object():
    p = parse_step_v15('<tool_call>\n"just a string"\n</tool_call>')
    assert p.kind == "error"


def test_tool_call_missing_name():
    p = parse_step_v15('<tool_call>\n{"arguments": {}}\n</tool_call>')
    assert p.kind == "error"


# ── 通道三：信令 ────────────────────────────────────────────────────────
@pytest.mark.parametrize("text,want", [(DEFER, "defer"), (CLARIFY, "clarify"), (REJECT, "reject")])
def test_each_terminal_signal(text, want):
    p = parse_step_v15(text)
    assert p.kind == "signal" and p.signal == want


def test_signal_args_are_kept():
    p = parse_step_v15(DEFER)
    assert p.signal_args == {"reason": "太新", "recheck_after_days": 3}


def test_signal_with_trailing_text_is_allowed():
    """信令 + 一句收尾话是合法形态（`25 §3.1` 终答段）。"""
    p = parse_step_v15(DEFER + "\n那我三天后再看。")
    assert p.kind == "signal" and "三天后" in p.text


def test_two_terminal_signals_is_error():
    p = parse_step_v15(DEFER + CLARIFY)
    assert p.kind == "error" and p.error == "multi_terminal_signal"


def test_same_signal_twice_is_error():
    p = parse_step_v15(DEFER + DEFER)
    assert p.kind == "error" and p.error == "multi_terminal_signal"


def test_signal_mixed_with_business_tool_is_error():
    """混合形态（`25 §6③`）：一轮里既发信令又办事，语义未定义 ⇒ 判错并计数。"""
    p = parse_step_v15(BIZ + DEFER)
    assert p.kind == "error" and p.error == "mixed_signal_and_tool"


def test_signal_mixed_with_report_is_error():
    p = parse_step_v15(REPORT + REJECT)
    assert p.kind == "error" and p.error == "mixed_signal_and_tool"


# ── 通道四：纯文本终答 ─────────────────────────────────────────────────
def test_plain_text_final():
    p = parse_step_v15("ROAS 是广告花的钱换回多少收入。")
    assert p.kind == "final_text" and p.text.startswith("ROAS")


def test_empty_final_text_is_error():
    """★ 空终答 = 空头支票老病（`25 §1③`），必须判错不能放过。"""
    assert parse_step_v15("   \n  ").kind == "error"


def test_think_only_output_is_empty_final():
    assert parse_step_v15("<think>想了半天</think>").error == "empty_final_text"


def test_shell_residue_is_error():
    """★ 壳残留 = 契约回潮，判错（不能照着壳解析，那等于契约没换）。"""
    p = parse_step_v15('```json\n{"behavior": "defer", "answer": {}}\n```')
    assert p.kind == "error" and p.error == "shell_residue"


def test_shell_residue_detected_even_with_signal():
    p = parse_step_v15(DEFER + '```json\n{"behavior": "defer", "answer": {}}\n```')
    assert p.kind == "error" and p.error == "shell_residue"


def test_plain_code_block_without_behavior_is_not_residue():
    """普通代码块（比如给用户看的示例）不该被误判成壳 —— 判据只认 behavior 字段。"""
    p = parse_step_v15("参考：\n```json\n{\"cpi\": 2.1}\n```")
    assert p.kind == "final_text"


# ── 轨迹级行为推导 ─────────────────────────────────────────────────────
@pytest.mark.parametrize("text,want", [(DEFER, "defer"), (CLARIFY, "clarify"), (REJECT, "reject")])
def test_derive_signal_behaviors(text, want):
    assert derive_behavior(terminal=parse_step_v15(text), business_tools_used=False) == want


def test_derive_tool_call_when_business_tools_used():
    """★ 这条是轨迹级的：最后一步只是纯文本，光看它推不出 tool_call。"""
    t = parse_step_v15("查完了，CPI 是 2.10。")
    assert derive_behavior(terminal=t, business_tools_used=True) == "tool_call"


def test_derive_answer_when_no_tools_used():
    t = parse_step_v15("ROAS 就是投入产出比。")
    assert derive_behavior(terminal=t, business_tools_used=False) == "answer"


def test_signal_behavior_ignores_business_flag():
    """信令是显式动作，不受"之前调过业务工具"影响。"""
    assert derive_behavior(terminal=parse_step_v15(DEFER), business_tools_used=True) == "defer"


def test_derive_refuses_non_terminal_step():
    with pytest.raises(ValueError):
        derive_behavior(terminal=parse_step_v15(BIZ), business_tools_used=False)


# ── 反向渲染（造 gold 用）────────────────────────────────────────────────
def test_render_signal_roundtrips():
    text = render_signal("session.defer", {"reason": "x", "recheck_after_days": 2})
    p = parse_step_v15(text)
    assert p.kind == "signal" and p.signal == "defer" and p.signal_args["recheck_after_days"] == 2


def test_render_report_roundtrips():
    p = parse_step_v15(render_report({"decision": "hold", "cpi": 2.1}))
    assert p.kind == "tool_calls" and p.tool_calls[0]["name"] == "session.report"


def test_render_rejects_unknown_signal():
    with pytest.raises(AssertionError):
        render_signal("session.whatever", {})


def test_v14_parser_untouched_by_v15_module():
    """★ 并存判据：v15 模块存在不改变 v14 解析器的任何行为。"""
    from syncopate.core.parsing import parse_step
    p = parse_step('```json\n{"behavior": "defer", "answer": {"a": 1}}\n```')
    assert p.kind == "final" and p.behavior == "defer" and p.answer == {"a": 1}
