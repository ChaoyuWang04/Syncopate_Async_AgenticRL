#!/usr/bin/env python
"""O-2a · 先试最便宜的那条路：**改 `answer_fields` 契约，能不能不训练就说人话**。

    python scripts/v16/probe_answer_fields.py

★ 为什么先做这个（守则⑤ 先测量后动手）：
  O-1 定性里那条 —— 「谢谢，辛苦了」→ `{"summary": "无操作"}` —— 是**两件事叠加**：
    ① 表达退化（要 OPD）
    ② `answer_fields` 只给了 `summary`「本次任务的结论」这一个字段，
       **逼着模型把任何回应都塞进"任务结论"的形状里**
  ⇒ ② 是 prompt 层的，改一行就能试。**先把它剥掉，剩下的才是 OPD 真正要修的**。
  ⇒ 若改契约就够 ⇒ OPD 范围大幅缩小（甚至只剩少量补强）。

★ 判据（不需要人判的那部分）：终答里**自然语言段的 token 数**
  —— 机器标签（`no_change`）1–3 个 token，人话一句 15+。
  ⚠️ 但"更长"不等于"更好"：同时打印原文供人判，且**必须看任务 prompt 有没有被带跑**
    （变话痨、不调工具就是把领域能力换走了）。
"""

from __future__ import annotations

import asyncio
import json
import statistics

import httpx

from syncopate.prompts import load_prompt, render_prompt
from syncopate.train.opd_render import segment_tokens
from syncopate.train.rollout_budget import (
    SAMPLING_TEMPERATURE, SAMPLING_TOP_K, SAMPLING_TOP_P)
from syncopate.train.rollout_loop import ASSISTANT_TURN_END, CHAT_TEMPLATE_KWARGS

CAND_URL, CAND_MODEL = "http://127.0.0.1:8100", "candidate"

# 三个契约变体。★ A 是现状（对照组），B/C 是候选改法。
VARIANTS = {
    "A·现状": [{"key": "summary", "description": "本次任务的结论"}],
    "B·明确要人话": [
        {"key": "reply",
         "description": "用完整的自然语言回复用户，像人一样把结论和理由说清楚"
                        "（**不是**机器标签，不要写 no_change / executed 这类词）"},
    ],
    # ★ C 保留机器可校验的字段**并列**一个人话字段 —— 这是最可能被采纳的形状：
    #   评分器要的结构不丢，人要读的话也有。
    "C·结构+人话并列": [
        {"key": "summary", "description": "结论的机器可校验形式（简短标签或数值）"},
        {"key": "reply",
         "description": "给用户读的完整回复：一到三句自然语言，说清结论和依据"},
    ],
}

CHAT = ["你好，你是做什么的？", "谢谢，辛苦了", "你能帮我改预算吗？",
        "刚才那个结论你是怎么得出来的？"]
TASK = ["查一下 CMP_1 昨天的花费和转化",
        "CMP_4 表现不错，帮我评估并执行扩量，预算提高 20%"]


def render(tokenizer, msg: str, tools, answer_fields) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": load_prompt("system.txt")},
         {"role": "user", "content": render_prompt("step_user.txt", {
             "reference_now": "2026-08-20",
             "context": {"account_id": "ACC_DEMO"},
             "user_message": msg,
             "answer_fields": answer_fields})}],
        tools=tools, add_generation_prompt=True, tokenize=False, **CHAT_TEMPLATE_KWARGS)


async def gen(c: httpx.AsyncClient, prompt: str) -> tuple[str, list[str]]:
    r = await c.post("/v1/completions", json={
        "model": CAND_MODEL, "prompt": prompt, "max_tokens": 320,
        "temperature": SAMPLING_TEMPERATURE, "top_p": SAMPLING_TOP_P,
        "top_k": SAMPLING_TOP_K, "stop": [ASSISTANT_TURN_END], "logprobs": 0},
        timeout=180)
    r.raise_for_status()
    ch = r.json()["choices"][0]
    return ch["text"], (ch.get("logprobs") or {}).get("tokens", [])


async def main() -> int:
    from transformers import AutoTokenizer
    import syncopate.domains.adcampaign  # noqa: F401
    from syncopate.core.tool_registry import REGISTRY

    tokenizer = AutoTokenizer.from_pretrained("models/Qwen3-4B-sft-v13r2-e1")
    tools = REGISTRY.menu(None)
    summary: dict[str, dict[str, float]] = {}

    async with httpx.AsyncClient(base_url=CAND_URL) as c:
        for vname, fields in VARIANTS.items():
            print("\n" + "=" * 78)
            print(f"【{vname}】answer_fields = "
                  f"{[f['key'] for f in fields]}")
            print("=" * 78)
            text_lens, tool_first = [], 0
            for msg in CHAT:
                text, toks = await gen(c, render(tokenizer, msg, tools, fields))
                n_text = sum(1 for lab in segment_tokens(toks) if lab == "text")
                text_lens.append(n_text)
                print(f"\n  [闲聊] {msg}")
                print(f"    自然语言 token 数 = {n_text}")
                print(f"    {text.strip()[:260]}")
            for msg in TASK:
                text, toks = await gen(c, render(tokenizer, msg, tools, fields))
                is_tool = "<tool_call>" in text
                tool_first += is_tool
                print(f"\n  [任务] {msg}")
                print(f"    首步调工具 = {'✅' if is_tool else '🔴 没调（领域能力被带跑了）'}")
                print(f"    {text.strip()[:200]}")
            summary[vname] = {
                "闲聊·自然语言 token 中位": statistics.median(text_lens),
                "任务·首步调工具": f"{tool_first}/{len(TASK)}",
            }

    print("\n" + "=" * 78)
    print("汇总（判据：闲聊要长出人话，任务**首步仍必须调工具**）")
    print("=" * 78)
    for vname, m in summary.items():
        print(f"  {vname:<16} 闲聊自然语言 token 中位 = "
              f"{m['闲聊·自然语言 token 中位']:>5}   任务首步调工具 = "
              f"{m['任务·首步调工具']}")
    print("\n⇒ 若某个变体**闲聊显著变长且任务不退**：那部分退化是 prompt 层的，"
          "不必用训练解决 ⇒ OPD 范围缩小到剩下的部分。")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
