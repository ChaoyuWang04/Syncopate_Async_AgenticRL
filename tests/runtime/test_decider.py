"""B-4 · `VllmDecider` 的解析映射（纯函数部分，不碰模型/网络/tokenizer）。

★ 解析器本身（parse_step）在训练侧有整族测试；这里钉的是**映射**：
  parse_step 的三种结果 → agent_loop 的 Proposal，一步多调用按 P0-2 拦在发生点。
"""

from __future__ import annotations

from syncopate.runtime.decider import DEFAULT_MENU, INTENT_MENUS, VllmDecider


def test_single_tool_call_maps_to_proposal():
    p = VllmDecider._to_proposal(
        '<tool_call>\n{"name": "campaign.get_metrics", '
        '"arguments": {"campaign_id": "CMP_1"}}\n</tool_call>')
    assert p.kind == "tool_call" and p.tool == "campaign.get_metrics"
    assert p.arguments == {"campaign_id": "CMP_1"}
    assert p.param_source == "model"


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


def test_garbage_output_becomes_correction_not_guess():
    p = VllmDecider._to_proposal("嗯让我想想，大概应该先看看数据吧")
    assert p.kind == "tool_call" and p.tool is None
    assert p.rationale.startswith("parse_error")


def test_intent_menus_are_training_shaped():
    """菜单是训练 case 的众数（12–16 个工具），绝不该膨胀回全量 30。"""
    for intent, menu in INTENT_MENUS.items():
        assert 10 <= len(menu) <= 18, f"{intent} 菜单 {len(menu)} 个，偏离训练分布"
        assert len(set(menu)) == len(menu)
    assert DEFAULT_MENU == INTENT_MENUS["I01"]
