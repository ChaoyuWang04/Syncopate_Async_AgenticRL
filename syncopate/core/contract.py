"""契约版本开关 —— **唯一真相来源**（v14 壳 ↔ v15 信令工具）。

为什么单开一个模块：切契约同时动**解析、渲染、判分取数、prompt 模板**四处。
按「这个值应该和那边一致 ⇒ 这里根本不该有第二份」（`00-START 守则⑨`），
四处全部从这里取，不许各存一份。

    SYNCOPATE_CONTRACT=v15   切到信令工具契约
    不设 / =v14              保持 v14 行为（**逐字节不变**，历史审计重放靠它）

⚠️ 判据行：v15 必须**显式可见**。静默切契约 = 下一个「机制在但没接上」。
"""

from __future__ import annotations

import os

CONTRACT = os.environ.get("SYNCOPATE_CONTRACT", "v14")
if CONTRACT not in ("v14", "v15"):
    raise SystemExit(f"🔴 SYNCOPATE_CONTRACT 只能是 v14 / v15，收到 {CONTRACT!r}")

IS_V15 = CONTRACT == "v15"

if IS_V15:
    print("[contract] SYNCOPATE_CONTRACT=v15 ⇒ 行为走 session.* 信令工具、终答纯自然语言",
          flush=True)


# ── session 信令工具族（spec 的唯一真相来源；ToolRegistry 与数据构建都从这里取）──
#
# ⚠️ 2026-08-29：本 spec 此前在 scripts/v15_r0_build.py 与 scripts/v15_probes.py
# 各存了一份副本。三份"当时都是对的"，正是下一次漂移的来源（守则⑨的原案）。
# ⇒ 收口到这里，副本改为 import。
#
# ⛔⛔ 收口当场就抓到一次漂移（`25 §7`）：作者在写探针副本时顺手给 clarify.missing_fields
# 和 reject.explanation **各加了一个 description**，而 R0 双臂数据是用**没有这两个字段**
# 的版本冻结的。若照"改进版"去评测，arm B 的训练 prompt 与评测 prompt 就不一致，
# R0 结论直接作废。⇒ **本文件保存的是与 data/v15_r0/ 冻结数据逐字一致的那一版。**
# 想改 spec ⇒ 必须重建 R0 数据，不能只改这里（判据：v15_r0_build.py 的 spec 一致性断言）。
SESSION_TOOL_SPECS: list[dict] = [
    {"type": "function", "function": {
        "name": "session.defer",
        "description": "数据尚不成熟、需等待后复查时调用（终止本轮任务并可挂起复查）",
        "parameters": {"type": "object", "properties": {
            "reason": {"type": "string", "description": "为什么现在不能下结论"},
            "recheck_after_days": {"type": "integer", "description": "建议几天后复查"}},
            "required": ["reason", "recheck_after_days"]}}},
    {"type": "function", "function": {
        "name": "session.clarify",
        "description": "信息不足需要用户补充时调用（终止本轮，等待用户回答）",
        "parameters": {"type": "object", "properties": {
            "question": {"type": "string", "description": "向用户提出的具体问题"},
            "missing_fields": {"type": "array", "items": {"type": "string"}}},
            "required": ["question", "missing_fields"]}}},
    {"type": "function", "function": {
        "name": "session.reject",
        "description": "请求越权、离题或违反政策时调用（终止本轮并说明）",
        "parameters": {"type": "object", "properties": {
            "reason_code": {"type": "string",
                            "enum": ["out_of_scope", "unauthorized", "policy"]},
            "explanation": {"type": "string"}},
            "required": ["reason_code", "explanation"]}}},
]

# 信令工具名 → 行为标签。**终止性**：调用即结束本轮轨迹。
TERMINAL_SIGNALS: dict[str, str] = {
    "session.defer": "defer",
    "session.clarify": "clarify",
    "session.reject": "reject",
}
# 非终止性：机器可核字段的结构化通道（承接 v14 的 answer_fields）
REPORT_TOOL = "session.report"
SESSION_TOOL_NAMES = frozenset(TERMINAL_SIGNALS) | {REPORT_TOOL}


# ── ★ 人话也算表达了行为（Chaoyu 2026-08-30 裁定）────────────────────────────
#
# 原话：「用人话拒绝我觉得是可以接受的，包括 defer 这些，掉了相应的 session 也可以，
#        人话说出来也可以。」
#
# ⚠️ 但**信令不是装饰**：它承载两个真实需求（`25 §1`）——判分可验证 + runtime 可编排。
#   如果人话算数、而 `trajectory.behavior` 仍按形态推成 `answer`，
#   那 verifier 的行为闸会给 0 分 ⇒ **RL 会用错误的信号去训**（这是最贵的那种错）。
# ⇒ 改成**两段式**（形状取自 `25 §1.1③` 的 Abstain-R1，本就是 R0 ②b 的备选）：
#     调了信令      → 正常判分（结构 + 语义都对）
#     只说了人话    → 也算表达了该行为，但**封顶 0.85**（信令保留正向激励）
#     两样都没有    → 仍然 0 分（行为闸不变松）
# ★ 规则判分，不引入 LLM judge：便宜、精确、不可刷、可负向认证。
import re as _re

PROSE_ONLY_CEILING = 0.85

_PROSE_SIGNAL = {
    "defer": _re.compile(
        r"再(观察|等|看)|等(几天|一等|数据)|数据(还)?(不|没)(够|足|成熟|稳)|"
        r"暂(时)?(不|别)|先(不|别)|不宜(现在|立即)|过几天|尚(未|不)成熟|观察期"),
    "clarify": _re.compile(
        r"请(问|补充|提供|告知|确认)|需要(你|您)?(补充|提供|确认|指定)|"
        r"能否(告诉|提供|说明)|想(先)?确认|是哪(个|条|种)|缺少.*信息|方便(告诉|提供)"),
    "reject": _re.compile(
        r"无法(执行|处理|完成|帮|操作)|不能(执行|处理|帮|操作)|超出.*(授权|范围|职责|权限)|"
        r"越权|不(予|能)(执行|受理)|没有(权限|授权)|恕难|不在.*范围内"),
}


def prose_expresses(behavior: str, text: str) -> bool:
    """终答的人话里有没有把这个行为**说出来**（只对三条信令行为有意义）。"""
    pat = _PROSE_SIGNAL.get(behavior)
    return bool(pat and text and pat.search(text))
