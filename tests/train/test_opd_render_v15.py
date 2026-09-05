from __future__ import annotations

import pytest

from syncopate.core.contract import IS_V15
from syncopate.prompts import load_system_prompt
from syncopate.train.opd_render import render_prompt_text, segment_text, v15_char_labels


def _kept(text: str) -> str:
    labels = v15_char_labels(text)
    return "".join(ch for ch, label in zip(text, labels) if label == "text").strip()


@pytest.mark.parametrize(("wire", "expected"), [
    ("这是纯自然语言。", "这是纯自然语言。"),
    ("先想\n</think>\n\n这是终答。", "这是终答。"),
    ("<think>先想</think>这是终答。", "这是终答。"),
    ("<think>先想</think>", ""),
    ('<tool_call>{"name":"x.y","arguments":{}}</tool_call>', ""),
    ("<tool_call>\n<function=x.y>\n</function>\n</tool_call>", ""),
    ("<think>想</think><tool_call>动作</tool_call>", ""),
    ("<think>想</think><tool_call>动作</tool_call>最后说明。", "最后说明。"),
    ("先说明。<tool_call>动作</tool_call>", "先说明。"),
    ("<tool_call>A</tool_call><tool_call>B</tool_call>收尾。", "收尾。"),
    ('{"behavior":"answer","answer":{"reply":"旧壳"}}', ""),
    ('{"reply":"旧壳"}', ""),
    ('```json\n{"summary":"旧壳"}\n```', ""),
    ("<think>没有闭合", ""),
    ("<tool_call>没有闭合", ""),
    ("隐式思考</think>{\"reply\":\"旧壳\"}", ""),
    ("<tool_call><function=session.defer></function></tool_call>", ""),
    ("结果集合是 {1, 2}。", "结果集合是 {1, 2}。"),
    ("这里解释 summary 这个英文词。", "这里解释 summary 这个英文词。"),
    ("", ""),
    ("   \n", ""),
])
def test_v15_mask_human_audit_cases(wire, expected):
    assert _kept(wire) == expected


class _CharTokenizer:
    def __init__(self):
        self.messages = None
        self.tools = None

    def __call__(self, text, **kwargs):
        return {"input_ids": list(range(len(text))),
                "offset_mapping": [(i, i + 1) for i in range(len(text))]}

    def encode(self, text, add_special_tokens=False):
        return list(range(len(text)))

    def decode(self, ids):
        return "x" * len(ids)

    def apply_chat_template(self, messages, *, tools, **kwargs):
        self.messages = messages
        self.tools = tools
        return "rendered"


def test_token_labels_follow_character_offsets():
    tok = _CharTokenizer()
    text = "thinking</think>人话"
    ids, labels = segment_text(tok, text)
    assert len(ids) == len(labels) == len(text)
    assert "text" not in labels[:text.index("</think>") + len("</think>")]
    assert labels[-2:] == ["text", "text"]


def test_think_on_generation_without_close_has_no_distillable_text():
    tok = _CharTokenizer()
    text = "还在思考，没有形成终答"
    ids, labels = segment_text(tok, text, implicit_think_open=True)
    assert len(ids) == len(labels) == len(text)
    assert set(labels) == {"think"}


@pytest.mark.skipif(not IS_V15, reason="v15 多轮消息契约专有")
def test_prompt_uses_real_message_pairs_and_passed_full_menu():
    tok = _CharTokenizer()
    tools = [{"type": "function", "function": {"name": "x.y"}}]
    rendered = render_prompt_text(
        tok,
        "本轮问题",
        tools,
        reference_now="2026-09-05",
        prior=[{"user_message": "上一轮问题", "result": {"text": "上一轮回答"}}],
    )
    assert rendered == "rendered"
    assert [m["role"] for m in tok.messages] == ["system", "user", "assistant", "user"]
    assert tok.messages[0]["content"] == load_system_prompt()
    assert tok.messages[1]["content"] == "上一轮问题"
    assert tok.messages[2]["content"] == "上一轮回答"
    assert "本轮问题" in tok.messages[3]["content"]
    assert tok.tools is tools
