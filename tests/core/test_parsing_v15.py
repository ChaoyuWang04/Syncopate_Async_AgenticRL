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


# ── 门槛③：往返逐字节一致（渲染 → 解析 → 重渲染）─────────────────────────
@pytest.mark.parametrize("name,args", [
    ("session.defer", {"reason": "数据太新", "recheck_after_days": 3}),
    ("session.clarify", {"question": "哪条计划?", "missing_fields": ["campaign_id", "region"]}),
    ("session.reject", {"reason_code": "unauthorized", "explanation": "越权，无法执行。"}),
    ("session.report", {"decision": "hold", "approved_budget": 1200, "cpi": 2.1}),
    ("session.report", {"note": "含中文与「引号」和 \\ 反斜杠"}),
    ("session.defer", {"reason": "", "recheck_after_days": 0}),
])
def test_render_parse_render_is_byte_identical(name, args):
    """★ 往返判据：渲染出的文本 → 解析 → 再渲染，必须**逐字节**相等。

    这条守的是"渲染与解析是同一份契约的两面"。任何一边偷偷规范化了什么
    （改引号、丢字段、重排 key），往返就会不等 —— 而那种偏差在训练里是静默的。
    """
    text1 = render_signal(name, args)
    p = parse_step_v15(text1)
    got_args = p.signal_args if p.kind == "signal" else p.tool_calls[0]["arguments"]
    text2 = render_signal(name, got_args)
    assert text1 == text2, f"往返不等\n第一次: {text1!r}\n第二次: {text2!r}"


def test_roundtrip_preserves_argument_types():
    """int 不许在往返里变成 str（recheck_after_days 是 int，判据②a 会查类型）。"""
    p = parse_step_v15(render_signal("session.defer",
                                     {"reason": "x", "recheck_after_days": 7}))
    assert isinstance(p.signal_args["recheck_after_days"], int)


def test_roundtrip_with_think_prefix():
    """带 think 前缀时，剥掉 think 后的往返仍要逐字节相等。"""
    text1 = render_signal("session.defer", {"reason": "x", "recheck_after_days": 1})
    p = parse_step_v15("<think>先想一下</think>" + text1)
    assert render_signal("session.defer", p.signal_args) == text1


# ── 通道四：Qwen3.5+ XML 线格式（2026-09-03 裁定⑫，学生换 Qwen3.6-35B-A3B）──────────────
# 模板原生格式：<tool_call>\n<function=NAME>\n<parameter=K>\nV\n</parameter>\n</function>\n</tool_call>
# 解析器两种线格式都认；渲染默认 xml（SYNCOPATE_TOOLCALL_FORMAT=json 回到 Qwen3 世代）。
XML = ('<tool_call>\n<function={n}>\n{params}</function>\n</tool_call>')


def _xml(name, **kw):
    params = "".join(f"<parameter={k}>\n{v}\n</parameter>\n" for k, v in kw.items())
    return XML.format(n=name, params=params)


def test_xml_business_call_parses():
    p = parse_step_v15(_xml("campaign.get_metrics", campaign_id="CMP_1"))
    assert p.kind == "tool_calls" and p.tool_calls == [{"name": "campaign.get_metrics", "arguments": {"campaign_id": "CMP_1"}}]


def test_xml_signal_parses_and_coerces_by_schema():
    """信令参数按注册表 schema 收型：recheck_after_days 是 integer ⇒ 3 不是 "3"。"""
    p = parse_step_v15(_xml("session.defer", reason="太新", recheck_after_days="3"))
    assert p.kind == "signal" and p.signal == "defer"
    assert p.signal_args == {"reason": "太新", "recheck_after_days": 3}


def test_xml_nested_json_parameter():
    p = parse_step_v15(_xml("x.y", opts='{"a": 1, "b": [1, 2]}'))
    assert p.kind == "tool_calls" and p.tool_calls[0]["arguments"] == {"opts": {"a": 1, "b": [1, 2]}}


def test_xml_schemaless_param_infers_json_scalars():
    """无 schema 的参数（session.report 自由字段等）：XML 线格式里数字与数字形字串不可区分（模板一律 str），
    取舍=能按 JSON 标量解的就解（这是 report 字段 2.1 不变成 "2.1" 的代价，parsing_v15._coerce 注释有账）。"""
    p = parse_step_v15(_xml("no.such_tool", n="7", f="2.1", b="true", z="null", s="abc"))
    assert p.tool_calls[0]["arguments"] == {"n": 7, "f": 2.1, "b": True, "z": None, "s": "abc"}


def test_xml_typed_string_param_keeps_numeric_looking_string():
    """有 schema 且 type=string 的参数不受推断影响：campaign_id="2024" 仍是字串。"""
    p = parse_step_v15(_xml("campaign.get_metrics", campaign_id="2024"))
    assert p.tool_calls[0]["arguments"] == {"campaign_id": "2024"}


def test_xml_missing_function_close_is_malformed():
    p = parse_step_v15("<tool_call>\n<function=x.y>\n<parameter=a>\n1\n</parameter>\n</tool_call>")
    assert p.kind == "error" and p.error.startswith("malformed_tool_call")


def test_xml_residue_between_parameters_is_malformed():
    p = parse_step_v15("<tool_call>\n<function=x.y>\n<parameter=a>\n1\n</parameter>\ngarbage\n</function>\n</tool_call>")
    assert p.kind == "error"


def test_xml_and_json_both_accepted_in_one_step():
    p = parse_step_v15(_xml("x.y", a="1") + "\n" + TC.format(n="x.z", a='{"b": 2}'))
    assert p.kind == "tool_calls" and [c["name"] for c in p.tool_calls] == ["x.y", "x.z"]


def test_render_tool_call_default_is_xml_and_roundtrips(monkeypatch):
    from syncopate.core import parsing_v15 as m
    monkeypatch.setattr(m, "TOOLCALL_FORMAT", "xml")
    text = m.render_tool_call("session.defer", {"reason": "太新", "recheck_after_days": 3})
    assert text.startswith("<tool_call>\n<function=session.defer>\n<parameter=reason>\n太新\n</parameter>")
    p = parse_step_v15(text)
    assert p.signal == "defer" and p.signal_args == {"reason": "太新", "recheck_after_days": 3}


def test_render_tool_call_json_mode_roundtrips(monkeypatch):
    from syncopate.core import parsing_v15 as m
    monkeypatch.setattr(m, "TOOLCALL_FORMAT", "json")
    text = m.render_tool_call("x.y", {"a": 1, "b": {"c": True}})
    assert text.startswith("<tool_call>\n{") and parse_step_v15(text).tool_calls[0]["arguments"] == {"a": 1, "b": {"c": True}}


def test_xml_render_matches_qwen35_chat_template_exactly():
    """★ 守则⑮：我们渲染出的 assistant 文本必须和 Qwen3.5 模板对同一 tool_calls 结构渲染的**逐字节相同**，
    否则训练目标与模型在 system 里被教的格式不同形。分词器不在时跳过（CI 无权重）。"""
    from pathlib import Path
    from syncopate.core import parsing_v15 as m
    from syncopate.core.model_paths import TEST_TOKENIZER
    if not Path(TEST_TOKENIZER, "chat_template.jinja").exists():
        pytest.skip("no Qwen3.5 tokenizer locally")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TEST_TOKENIZER)
    args = {"campaign_id": "CMP_1", "days": 7, "opts": {"a": [1, 2]}}
    msgs = [{"role": "user", "content": "q"},
            {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": "campaign.get_metrics", "arguments": args}}]}]
    rendered = tok.apply_chat_template(msgs, tokenize=False)
    body = rendered.split("<|im_start|>assistant\n", 1)[1].split("<|im_end|>", 1)[0]
    # 模板在最后一轮 assistant 前放一个空 think 段；剥掉后应与我们的渲染逐字节相同
    body = body.split("</think>\n\n", 1)[-1] if "</think>" in body else body
    ours = m.render_tool_call("campaign.get_metrics", args) if m.TOOLCALL_FORMAT == "xml" else None
    assert ours is not None and body == ours, f"模板:\n{body!r}\n我们:\n{ours!r}"
