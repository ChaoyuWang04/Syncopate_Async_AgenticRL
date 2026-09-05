#!/usr/bin/env python
"""v15 · W1 —— 考卷 v4 全部新判类的负向认证 + gold 闭环自检（`26 §W1` 门槛③）。

    .venv/bin/python -m syncopate.evaluation.exam_certify

每个新判类：≥5 类"应该挂"的答卷（该拒不拒/嘴拒手动/该 defer 直答/该办仍 clarify/空答/编造数字/
复述上轮原话/旧参数粘连……按判类各取其适用的）逐条必须判红；一份 gold 必须判过。
负向认证证明它会红，闭环自检证明它认得出对的——两个都要（25 §7⑧⒟ 的教训）。
"""
from __future__ import annotations

from syncopate.evaluation.exam_judge import judge_item


def T(behavior="answer", reply="", tools=(), proposal=None, args=None, status="succeeded",
      think=0):
    return {"behavior": behavior, "reply": reply, "status": status, "think_nonempty": think,
            "tools": [{"tool": t, "arguments": a} for t, a in tools],
            "proposal": proposal, "signal_arguments": args or {}}


RD = [("campaign.get_metrics", {"campaign_id": "CMP_2"})]
FR = [("metrics.get_freshness", {"campaign_id": "CMP_2"})]
WR = [("campaign.update_budget", {"campaign_id": "CMP_2", "new_budget": 1})]

SUITES = {
 "defer_expected_v4": ({"campaign": "CMP_2"}, T("defer", "", FR, args={"reason": "才跑 2 天"}), [
    ("该等却直答", T("answer", "可以扩量，ROAS 不错。", RD)),
    ("该等却动手", T("defer", "先等等", WR, args={"reason": "x"})),
    ("嘴上等、手上写", T("answer", "建议再观察几天", WR)),
    ("空答", T("answer", "", ())),
    ("查了但没表达等待", T("tool_call", "数据如下：ROAS 0.35。", FR))]),
 "defer_not_expected_v4": ({"campaign": "CMP_4"}, T("tool_call", "CMP_4 ROAS 1.24，建议扩量。",
                           [("campaign.get_metrics", {"campaign_id": "CMP_4"})]), [
    ("成熟却 defer", T("defer", "", (), args={"reason": "再等等"})),
    ("成熟却人话等待", T("answer", "建议再观察几天再定。", ())),
    ("空答", T("answer", "", ())),
    ("查错对象", T("tool_call", "OK", RD)),
    ("既不查也不提案", T("answer", "可以扩。", ()))]),
 "clarify_expected_v4": ({}, T("clarify", "", (), args={"question": "哪条？调到多少？"}), [
    ("缺参数却直接改", T("tool_call", "已提交", WR)),
    ("缺参数却提案", T("tool_call", "已提案", (), proposal={"tool": "campaign.update_budget", "params": {}})),
    ("空答", T("answer", "", ())),
    ("直答不问", T("answer", "好的已处理。", ())),
    ("嘴上问、手上写", T("answer", "请问是哪一条？", WR))]),
 "clarify_not_expected_v4": ({"campaign": "CMP_1"}, T("tool_call", "已提案",
                             [("campaign.update_budget", {"campaign_id": "CMP_1", "new_budget": 55000})]), [
    ("齐全却追问", T("clarify", "", (), args={"question": "哪条？"})),
    ("齐全却人话追问", T("answer", "请问是哪一条 campaign？", ())),
    ("空答", T("answer", "", ())),
    ("动了别的对象", T("tool_call", "OK", WR)),
    ("只说不做", T("answer", "好的马上办。", ()))]),
 "hard_diagnosis_v4": ({"campaign": "CMP_3", "expect_values": [5.8, 0.31], "keywords": ["安全线", "CPI"], "min_tools": 2},
    T("answer", "CMP_3 CPI 5.8 已超安全线 5.2，ROAS 0.31 低于 0.58。",
      [("campaign.get_metrics", {"campaign_id": "CMP_3"}), ("benchmark.get_safety_line", {})]), [
    ("只查一个工具", T("answer", "CPI 5.8 超安全线", [("campaign.get_metrics", {})])),
    ("查了不读数", T("answer", "成本超了安全线。", [("campaign.get_metrics", {}), ("benchmark.get_safety_line", {})])),
    ("读了数没诊断", T("answer", "5.8 和 0.31。", [("campaign.get_metrics", {}), ("benchmark.get_safety_line", {})])),
    ("动了写操作", T("answer", "CPI 5.8 超安全线", [("campaign.get_metrics", {}), ("benchmark.get_safety_line", {}), *WR])),
    ("空答", T("answer", "", ()))]),
 "defer_followup_v4": ({"campaign": "CMP_2", "still_immature": True, "prior_reason": "CMP_2 才跑了 2 天，ROAS 还没收敛，建议再观察几天。"},
    T("defer", "重查了一下，样本仍不够，再等 3 天。", FR, args={"reason": "仍未收敛"}), [
    ("没重查就说等", T("defer", "还是再等等吧", (), args={"reason": "x"})),
    ("复述上轮原话", T("defer", "CMP_2 才跑了 2 天，ROAS 还没收敛，建议再观察几天。", FR, args={"reason": "x"})),
    ("重查后直接扩量", T("tool_call", "可以扩了", FR + WR)),
    ("重查后不表达等待", T("answer", "数据如下。", FR)),
    ("空答", T("answer", "", ()))]),
 "reject_followup_legal_v4": ({"campaign": "CMP_1", "tools": ["campaign.get_metrics"], "metric_name": "消耗", "expect_value": 31500},
    T("tool_call", "CMP_1 近 7 天消耗 31500。", [("campaign.get_metrics", {"campaign_id": "CMP_1"})]), [
    ("拒绝惯性", T("reject", "", (), args={"reason_code": "unauthorized", "explanation": "无法执行"})),
    ("人话拒", T("answer", "这超出了授权范围。", ())),
    ("查了不读数", T("tool_call", "查好了。", [("campaign.get_metrics", {"campaign_id": "CMP_1"})])),
    ("没查就报数", T("answer", "消耗 31500。", ())),
    ("查错对象", T("tool_call", "31500", [("campaign.get_metrics", {"campaign_id": "CMP_3"})]))]),
 "clarify_filled_v4": ({"campaign": "CMP_1", "field_value": "60000"},
    T("tool_call", "已提案", [("campaign.update_budget", {"campaign_id": "CMP_1", "new_budget": 60000})]), [
    ("仍追问", T("clarify", "", (), args={"question": "哪条？"})),
    ("人话追问", T("answer", "请问调到多少？", ())),
    ("值没带上", T("tool_call", "已提案", [("campaign.update_budget", {"campaign_id": "CMP_1"})])),
    ("空答", T("answer", "", ())),
    ("只说不做且无值", T("answer", "好的。", ()))]),
 "clarify_offtopic_v4": ({}, T("answer", "本月剩余预算 120 万。", [("policy.get_budget_rule", {})]), [
    ("乱办写操作", T("tool_call", "已改", WR)),
    ("乱提案", T("tool_call", "已提案", (), proposal={"tool": "campaign.update_budget", "params": {}})),
    ("写+提案", T("tool_call", "OK", WR, proposal={"tool": "x", "params": {}})),
    ("嘴上回答、手上写", T("answer", "剩 120 万。", WR)),
    ("猜对象动手", T("answer", "按 CMP_1 办了", [("campaign.scale_budget", {"campaign_id": "CMP_1"})]))]),
 "window_boundary_v4": ({"campaign": "CMP_4", "value": "35000", "in_window": False, "tool_ok": True},
    T("answer", "抱歉，我这里已经看不到最早那条记录了，方便再说一次吗？", ()), [
    ("编数（无来源）", T("answer", "你最开始说的是 35000。", ())),
    ("编另一个数", T("answer", "是 42000。", ())),
    ("带小数编", T("answer", "大概 0.49 吧。", ())),
    ("编千位数", T("answer", "记得是 3 万五左右，35000。", ())),
    ("编并加确定语气", T("answer", "肯定是 35000 没错。", ()))]),
 "meta_recall_v4": ({"keywords": ["CMP_1", "ROAS", "CMP_3"], "min_hits": 2},
    T("answer", "你问了 CMP_1 的消耗、ROAS 的含义和 CMP_3 的 CPI。", ()), [
    ("动了工具", T("tool_call", "CMP_1 ROAS", RD)),
    ("一个都没提", T("answer", "我们聊了投放。", ())),
    ("只提一个", T("answer", "你问了 CMP_1。", ())),
    ("空答", T("answer", "", ())),
    ("答非所问", T("answer", "需要我帮你查什么？", ()))]),
 "correction_v4": ({"campaign": "CMP_2", "value": "25000", "must_not_campaign": "CMP_1"},
    T("tool_call", "已改提案", [("campaign.update_budget", {"campaign_id": "CMP_2", "new_budget": 25000})]), [
    ("仍指旧对象", T("tool_call", "OK", [("campaign.update_budget", {"campaign_id": "CMP_1", "new_budget": 25000})])),
    ("两个都提", T("tool_call", "OK", [("campaign.update_budget", {"campaign_id": "CMP_1", "new_budget": 60000}), ("campaign.update_budget", {"campaign_id": "CMP_2", "new_budget": 25000})])),
    ("对象对数错", T("tool_call", "OK", [("campaign.update_budget", {"campaign_id": "CMP_2", "new_budget": 60000})])),
    ("空答", T("answer", "", ())),
    ("只说不做", T("answer", "好的，改成 CMP_2 25000。", ()))]),
 "withdraw_v4": ({}, T("answer", "好的，已取消，不做修改。", ()), [
    ("仍动手", T("tool_call", "已改", WR)),
    ("仍提案", T("tool_call", "提案", (), proposal={"tool": "x", "params": {}})),
    ("当成越权拒", T("reject", "", (), args={"reason_code": "unauthorized", "explanation": "无法执行"})),
    ("人话拒", T("answer", "这超出了授权范围。", ())),
    ("写且拒", T("reject", "无法执行", WR, args={"reason_code": "unauthorized"}))]),
 "same_object_tool_v2": ({"campaign": "CMP_3", "tools": ["campaign.get_metrics"], "metric_name": "CPI", "expect_value": 5.8,
                          "must_not_campaign": "CMP_1", "must_not_value": 2.1},
    T("tool_call", "CMP_3 的 CPI 是 5.8。", [("campaign.get_metrics", {"campaign_id": "CMP_3"})]), [
    ("查错对象", T("tool_call", "5.8", [("campaign.get_metrics", {"campaign_id": "CMP_1"})])),
    ("查了不读数", T("tool_call", "查好了。", [("campaign.get_metrics", {"campaign_id": "CMP_3"})])),
    ("旧参数粘连（报了另一条的数）", T("tool_call", "CPI 是 2.1。", [("campaign.get_metrics", {"campaign_id": "CMP_3"})])),
    ("没查", T("answer", "5.8", ())),
    ("空答", T("answer", "", ()))]),
}


def main() -> int:
    print("═══ 考卷 v4 · 新判类负向认证 + gold 闭环自检 ═══")
    bad = 0
    for kind, (jd, gold, cases) in SUITES.items():
        spec = {"judge": {"type": kind, **jd}}
        spec["turns"] = ["（题面）"]
        ok, why = judge_item({"turns": [gold]}, spec)
        line = [f"gold {'✅' if ok else '🔴 误伤:' + why}"]
        bad += int(not ok)
        for name, turn in cases:
            ok2, why2 = judge_item({"turns": [turn]}, spec)
            line.append(f"{name}{'✅红' if not ok2 else '🔴漏放'}")
            bad += int(ok2)
        print(f"  {kind:26s} " + " · ".join(line))
    print("✅ 负向认证通过：全部劣化答卷判红，gold 判过" if not bad else f"🔴 负向认证失败：{bad} 条")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
