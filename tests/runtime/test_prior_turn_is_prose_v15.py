"""v15 下多轮历史里的上一轮回答必须是**人话**，不是 JSON 壳（`25 §7㉝`）。

⛔ 考场炸出来的：`_prior_turn_messages` 把上一轮结果 `json.dumps` 塞回历史 ⇒
  模型在多轮题里看到 `{"text": "...", "behavior": null}` —— 既不是训过的形状，
  也正是 v15 明令不再输出的东西。context_v3 整份考卷都是多轮题。
"""
from __future__ import annotations

import json

import pytest

from syncopate.core.contract import IS_V15

pytestmark = pytest.mark.skipif(not IS_V15, reason="v15 契约专有")


class _Tok:
    def encode(self, t, add_special_tokens=False):
        return list(range(len(t)))

    def decode(self, ids):
        return "x" * len(ids)


def _render(prior):
    from syncopate.runtime.decider import VllmDecider

    d = VllmDecider.__new__(VllmDecider)
    d.tokenizer = _Tok()
    return VllmDecider._prior_turn_messages(d, prior)


def test_prior_answer_is_the_prose() -> None:
    out = _render([{"user_message": "CMP_1 的 CPI 是多少",
                    "result": {"text": "CMP_1 的 CPI 是 5.8。", "behavior": None}}])
    assert out[-1]["role"] == "assistant"
    assert out[-1]["content"] == "CMP_1 的 CPI 是 5.8。"
    assert "{" not in out[-1]["content"], "上一轮又变回 JSON 壳了"


def test_signal_turn_falls_back_to_its_text_not_json() -> None:
    """信令收场没有终答文本 ⇒ 用信令自己的话，**不能**退回 json.dumps。"""
    out = _render([{"user_message": "能扩量吗",
                    "result": {"text": "", "signal": "defer", "behavior": "defer",
                               "arguments": {"reason": "数据还在动", "recheck_after_days": 5}}}])
    assert out[-1]["content"] == "数据还在动"
    assert "recheck_after_days" not in out[-1]["content"]
