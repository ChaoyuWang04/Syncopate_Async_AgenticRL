"""B-4 · `VllmDecider` 的解析映射（纯函数部分，不碰模型/网络/tokenizer）。

★ 解析器本身（parse_step）在训练侧有整族测试；这里钉的是**映射**：
  parse_step 的三种结果 → agent_loop 的 Proposal，一步多调用按 P0-2 拦在发生点。
"""

from __future__ import annotations

import pytest
from syncopate.core.contract import IS_V15
from syncopate.runtime.decider import DEFAULT_MENU, INTENT_MENUS, VllmDecider


def test_single_tool_call_maps_to_proposal():
    p = VllmDecider._to_proposal(
        '<tool_call>\n{"name": "campaign.get_metrics", '
        '"arguments": {"campaign_id": "CMP_1"}}\n</tool_call>')
    assert p.kind == "tool_call" and p.tool == "campaign.get_metrics"
    assert p.arguments == {"campaign_id": "CMP_1"}
    assert p.param_source == "model"


# ★ 下面两条**只对 v14 契约成立** —— 它们断言的正是被 v15 换掉的那两件事：
#   ① 壳 JSON 是终答（v15 里壳是残留，终答是纯文本 + session.report）
#   ② 一段没有结构的自然语言 = 解析错误（v15 里它就是**合法终答**）
# v15 的对应行为由 tests/runtime/test_decider_v15.py 覆盖（含 v14 默认不变那条）。
# ⇒ 显式按契约跳过，而不是让它在 v15 下红着 —— 但**跳过必须写清楚谁接了这个班**，
#   否则就是"删掉一条判据"（守则⑦：空着的门槛应读作"无法判定"）。
_V14_ONLY = pytest.mark.skipif(
    IS_V15, reason="v14 契约专属断言；v15 的对应行为见 tests/runtime/test_decider_v15.py")


@_V14_ONLY
def test_final_answer_maps_to_final():
    p = VllmDecider._to_proposal(
        '```json\n{"behavior": "defer", "answer": {"summary": "数据未成熟，D7 再判"}}\n```')
    assert p.kind == "final"
    assert p.final_answer == {"behavior": "defer",
                              "answer": {"summary": "数据未成熟，D7 再判"}}


def test_multi_tool_call_is_intercepted_at_source():
    """P0-2 同法：一步多调用 ⇒ 不执行任何一个，纠正文本回灌。"""
    text = ('<tool_call>\n{"name": "a.b", "arguments": {}}\n</tool_call>\n'
            '<tool_call>\n{"name": "c.d", "arguments": {}}\n</tool_call>')
    p = VllmDecider._to_proposal(text)
    assert p.kind == "tool_call" and p.tool is None
    assert "一个 tool call" in p.rationale


@_V14_ONLY
def test_garbage_output_becomes_correction_not_guess():
    p = VllmDecider._to_proposal("嗯让我想想，大概应该先看看数据吧")
    assert p.kind == "tool_call" and p.tool is None
    assert p.rationale.startswith("parse_error")


class _FakeTok:
    """只做长度/文本的最小 tokenizer（渲染逻辑不需要真分词器）。"""

    def encode(self, text, add_special_tokens=False):        # noqa: ANN001
        return list(text)

    def decode(self, ids):                                   # noqa: ANN001
        return "".join(ids)


def _decider_shell() -> VllmDecider:
    """不走 __init__（那会拉真 tokenizer）——只测纯渲染方法。"""
    d = object.__new__(VllmDecider)
    d.tokenizer = _FakeTok()
    return d


def test_prior_turns_render_as_user_assistant_pairs():
    """★ 多轮壳层：历史渲染成 user/assistant 对，本轮永远是最后一条 user。"""
    turns = [{"user_message": "我是王超宇",
              "result": {"behavior": "answer", "answer": {"summary": "你好，王超宇"}}}]
    msgs = _decider_shell()._prior_turn_messages(turns)
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert msgs[0]["content"] == "我是王超宇"
    assert "王超宇" in msgs[1]["content"]


def test_prior_turn_result_accepts_json_string():
    """库里 JSONB 读回来可能是 str —— 不能因此把整轮历史丢掉。"""
    turns = [{"user_message": "查 CMP_1", "result": '{"answer": {"spend": 31500}}'}]
    msgs = _decider_shell()._prior_turn_messages(turns)
    assert "31500" in msgs[1]["content"]


def test_long_prior_answer_is_truncated_and_marked():
    """★ 截断必须**可见**（budget-truncation-family：静默砍是禁的）。"""
    turns = [{"user_message": "q", "result": {"answer": {"x": "长" * 2000}}}]
    msgs = _decider_shell()._prior_turn_messages(turns)
    assert msgs[1]["content"].endswith("…（已截断）")


def test_intent_menus_stay_training_shaped_as_fallback():
    """★ 默认已改全量 30（模型自选，探针 probe_full_menu 全绿）；
    但 INTENT_MENUS 作为**打回路径**保留（SYNCOPATE_TOOL_MENU=intent），
    它必须一直保持训练形状（12–16 个），否则打回时打回到一个没验过的分布。"""
    for intent, menu in INTENT_MENUS.items():
        assert 10 <= len(menu) <= 18, f"{intent} 菜单 {len(menu)} 个，偏离训练分布"
        assert len(set(menu)) == len(menu)
    assert DEFAULT_MENU == INTENT_MENUS["I01"]
