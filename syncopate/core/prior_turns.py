"""多轮历史 → 消息对的**唯一**渲染函数（训练侧 build_messages 与 runtime decider 共用）。

★ 09-02（`26 §W2①`，守则⑮）：此前只有 decider 会把上一轮渲染成 user/assistant 消息对，
  训练数据把历史折成「[上一轮] …」文本塞进题面 —— 两边不同形，模型学到的是错的那一种。
  现在两侧 import 同一个函数：**同形靠只有一条代码路径保证**（副本会漂，25 §7⑥）。

一条历史轮 = {"user_message": str, "result": dict}，result 形状与线上 agent_runs.result 相同：
  · 普通终答  {"text": "…"}
  · 信令收场  {"text": "", "signal": "defer|clarify|reject", "arguments": {...}}
  · v14 旧壳  {"answer": {...}}（按原路径 json.dumps，逐字节不变）
助手侧只带**结论人话**（信令收场用信令自己的话），不带工具步骤、不带 think；
超过 PRIOR_ANSWER_BUDGET 个 token 截断并标注（静默截断是记录在案的失效家族）。
"""
from __future__ import annotations

import json
from typing import Any

from syncopate.core.contract import IS_V15

# 每轮结论最多回灌多少 token（F-5：最近 6 轮 · 每轮 400 tok 封顶，超了标注）
PRIOR_ANSWER_BUDGET = 400
PRIOR_TURNS_LIMIT = 6


def prior_answer_text(result: Any) -> str:
    """上一轮结果 → 助手那句人话。"""
    if isinstance(result, str):
        try:
            result = json.loads(result)
        except ValueError:
            result = {"summary": result}
    if IS_V15 and isinstance(result, dict) and ("text" in result or "signal" in result):
        text = str(result.get("text") or "").strip()
        if not text:                      # 信令收场（defer/clarify/reject）没有终答文本
            a = result.get("arguments") or {}
            text = str(a.get("question") or a.get("explanation")
                       or a.get("reason") or "（上一轮以信令收场）")
        return text
    answer = (result or {}).get("answer", result or {}) if isinstance(result, dict) else (result or {})
    return json.dumps(answer, ensure_ascii=False)


def render_prior_messages(turns: list[dict], tokenizer: Any,
                          budget: int = PRIOR_ANSWER_BUDGET) -> list[dict[str, str]]:
    """把之前几轮渲染成 user/assistant 对（插在 system 之后、本轮 user 之前）。"""
    # ★ 09-02（Chaoyu 在画廊里抓到的）：窗口语义必须在**这里**裁——线上 prior_turns 只取最近 6 轮，
    #   训练行若把 8 轮全渲染进去、gold 却说"最早那条看不到了"，就是在教模型撒谎。
    turns = list(turns or [])[-PRIOR_TURNS_LIMIT:]
    out: list[dict[str, str]] = []
    for t in turns:
        out.append({"role": "user", "content": t.get("user_message") or ""})
        text = prior_answer_text(t.get("result"))
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) > budget:
            text = tokenizer.decode(ids[:budget]) + "…（已截断）"
        out.append({"role": "assistant", "content": text})
    return out
