"""守则⑮ 的判据形状：训练样例与线上请求**逐项同形**——「两个东西应当相同」，不需要阈值。

对比对象：训练侧 rollout_loop.build_messages(bundle, prior=…) vs 生产侧 VllmDecider._messages(…, prior=…)。
逐项：历史位置=消息对（system 之后、本轮 user 之前）· 助手历史=人话无 think 无 JSON · 无字段清单
（v15 多轮行）· 当前时间=纯日期 · 本轮 user 是最后一条。
负向认证：把历史折回题面文本 ⇒ 断言必须红（空门槛不算过）。
"""
from __future__ import annotations

import re

import pytest

from syncopate.authoring.seed_cases import SEED_BUILDERS
from syncopate.core.contract import IS_V15
from syncopate.core.prior_turns import render_prior_messages
from syncopate.train.rollout_loop import build_messages

pytestmark = pytest.mark.skipif(not IS_V15, reason="v15 契约专有")


class _Tok:
    def encode(self, t, add_special_tokens=False): return list(range(len(t)))
    def decode(self, ids): return "x" * len(ids)


PRIOR = [{"user_message": "CMP_1 最近消耗多少", "result": {"text": "CMP_1 近 7 天消耗 31500。"}},
         {"user_message": "能扩量吗", "result": {"text": "", "signal": "defer",
                                                 "arguments": {"reason": "数据还没收敛，再等几天。"}}}]


def _bundle():
    b = SEED_BUILDERS["FRESH_0001"]() if "FRESH_0001" in SEED_BUILDERS else next(iter(SEED_BUILDERS.values()))()
    return b


def _prod_messages(prior, user_message, context, answer_fields):
    from syncopate.runtime.decider import VllmDecider
    d = VllmDecider.__new__(VllmDecider)
    d.tokenizer = _Tok(); d.context = context; d.answer_fields = answer_fields
    return VllmDecider._messages(d, user_message, [], prior)


def _shape(msgs):
    """一条渲染结果的「形状」：角色序列 + 历史助手内容 + 本轮 user 里的结构特征。"""
    roles = [m["role"] for m in msgs]
    last = msgs[-1]["content"]
    return {
        "roles": roles,
        "history_assistant": [m["content"] for m in msgs[1:-1] if m["role"] == "assistant"],
        "date_only": bool(re.search(r"当前时间：\d{4}-\d{2}-\d{2}\n", last)),
        "has_field_list": "本次结论需要给出的字段" in last,
        "history_in_user_text": "[上一轮]" in last,
    }


def assert_same_shape(train_msgs, prod_msgs):
    a, b = _shape(train_msgs), _shape(prod_msgs)
    assert a["roles"] == b["roles"], f"历史位置不同形：{a['roles']} vs {b['roles']}"
    assert a["history_assistant"] == b["history_assistant"], "历史助手内容不同形"
    for m in a["history_assistant"]:
        assert "<think>" not in m and "{" not in m, f"历史里带 think/JSON：{m[:40]}"
    assert a["date_only"] and b["date_only"], "当前时间不是纯日期"
    assert a["has_field_list"] == b["has_field_list"] is False, "多轮行不该有字段清单"
    assert not a["history_in_user_text"] and not b["history_in_user_text"]
    assert a["roles"][-1] == "user" and a["roles"][0] == "system"


def test_training_multiturn_row_matches_production_request():
    b = _bundle()
    b.prior = PRIOR
    b.verifier.required_answer_fields = [f for f in b.verifier.required_answer_fields if f.key in ("reply", "summary")]
    train = build_messages(b, None, tokenizer=_Tok())
    prod = _prod_messages(PRIOR, b.case.user_message, b.case.context, b.verifier.required_answer_fields)
    assert_same_shape(train, prod)
    assert train[1:-1] == render_prior_messages(PRIOR, _Tok()), "训练侧历史必须就是共用渲染函数的产物"
    assert train[1:-1] == prod[1:-1], "两侧历史消息对必须逐字相同（同一个函数）"


def test_signal_ended_turn_renders_as_its_own_words_on_both_sides():
    msgs = render_prior_messages(PRIOR, _Tok())
    assert msgs[-1] == {"role": "assistant", "content": "数据还没收敛，再等几天。"}


def test_same_shape_assertion_fails_on_folded_history():
    """负向认证：把历史折成题面文本（08-31 之前的训练形状）⇒ 判据必须红。"""
    b = _bundle()
    b.case.user_message = "[上一轮] 用户：CMP_1 最近消耗多少\n[上一轮] 助手：已给出结论\n\n" + b.case.user_message
    b.verifier.required_answer_fields = []
    train = build_messages(b, None, tokenizer=_Tok())
    prod = _prod_messages(PRIOR, b.case.user_message, b.case.context, [])
    with pytest.raises(AssertionError):
        assert_same_shape(train, prod)


def test_same_shape_assertion_fails_on_iso_time(monkeypatch):
    b = _bundle(); b.prior = PRIOR; b.verifier.required_answer_fields = []
    train = build_messages(b, None, tokenizer=_Tok())
    bad = [dict(m) for m in train]
    bad[-1]["content"] = bad[-1]["content"].replace("当前时间：2026-08-01", "当前时间：2026-08-01T00:00:00+00:00", 1)
    prod = _prod_messages(PRIOR, b.case.user_message, b.case.context, [])
    with pytest.raises(AssertionError):
        assert_same_shape(bad, prod)
