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
