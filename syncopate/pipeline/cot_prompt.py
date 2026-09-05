"""v15 · W3 CoT 触发显性化（`26 §W3②`）：难例行的题面必须**看得出该想**。

现状：难例 = 隐藏的模板族标签（BUD/DIA/FAIL/RAG/SCALE 里被 triage 判"卡死/死格"的 case），
题面和同族简单题长得一样 ⇒ 模型在题面上看不出「该想」，永远学不会触发（W3② 探针证实）。
修法：CoT 行的 user_message 加一句**多步诊断类问法**（前缀或后缀，按 case_id 确定性选，不改语义、
不改 gold），让"该想"成为题面上可学的特征。⚠️ 只加在 CoT 行；简单题不加（N3 的另一半）。
同一份函数给 gen_cot_v15 与 W3② 探针共用。
"""
from __future__ import annotations

import hashlib

HARD_PREFIX = [
    "这个问题要把该查的都查一遍再下结论：",
    "先别急着给答案，逐项排查后再说：",
    "这条需要多步诊断，把依据一条条列清楚：",
    "帮我系统地分析一下，中间每一步的判断都说明理由：",
]
HARD_SUFFIX = [
    "——请逐步核对相关数据、政策和风控后再给结论。",
    "，注意把可能的原因逐个排除，说明依据。",
    "。这类判断容易出错，先把安全线、成熟度、账户状态都查一遍。",
    "，要有推导过程，不要只给结果。",
]


def explicit_hard_prompt(user_message: str, case_id: str) -> str:
    h = int(hashlib.md5(case_id.encode()).hexdigest()[:8], 16)
    if h % 2 == 0:
        return HARD_PREFIX[h // 2 % len(HARD_PREFIX)] + user_message
    return user_message.rstrip("。？！?!") + HARD_SUFFIX[h // 2 % len(HARD_SUFFIX)]
