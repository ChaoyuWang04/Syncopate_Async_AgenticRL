#!/usr/bin/env python
"""O-1 · OPD 退化探针：候选 vs 裸底座，**分歧到底在哪些 token 上**。

    python scripts/probe_opd_divergence.py            # 全部
    python scripts/probe_opd_divergence.py --qual     # 只看定性对照（快）

前置：候选端点 :8100（sft-base + candidate LoRA）· 裸底座端点 :8101（base）。
起法见 `docs/syncopate/09 §0` 与 `logs/runtime/start_vllm_base.sh`。

★★★ 这个探针要回答的**唯一问题**（`22 §J-5`）：

    「说不出人话」的退化，是**集中在自然语言段**，还是**弥散在所有位置**？

    集中  ⇒ OPD 对症：只在自然语言段用底座当老师，格式/工具段不动
    弥散  ⇒ 退化更深（连格式和决策一起偏），方案要重想

⇒ **先测量后动手**（守则⑤）：不拿一个"看起来合理"的方案直接上训练。

★ 量法：让**候选**生成（on-policy —— OPD 就是在学生自己的轨迹上蒸），
  再把**同一串 token** 拿去让底座算 logprob，逐 token 比。
  ⚠️ 必须是同一串 token：各自生成再比文本，量的是"两个模型说了不同的话"，
    而不是"同一句话上两个模型的分歧" —— 后者才是蒸馏损失的形状。

★ token 分段（这是判据的核心，分错了整份报告就没意义）：
    工具段    `<tool_call> … </tool_call>` 之间
    格式段    终答的 ```json 围栏、键名、标点等结构 token
    自然语言段 终答 answer 里**字符串值**的内容
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics
import sys
from typing import Any

import httpx

sys.path.insert(0, ".")
from syncopate.prompts import load_prompt, render_prompt  # noqa: E402
from syncopate.train.rollout_budget import (  # noqa: E402
    SAMPLING_TEMPERATURE, SAMPLING_TOP_K, SAMPLING_TOP_P)
from syncopate.train.rollout_loop import ASSISTANT_TURN_END, CHAT_TEMPLATE_KWARGS  # noqa: E402

CAND_URL, BASE_URL = "http://127.0.0.1:8100", "http://127.0.0.1:8101"
CAND_MODEL, BASE_MODEL = "candidate", "base"

# 两类 prompt 必须都在：只测闲聊会得出"全面退化"的错误结论，
# 因为任务 prompt 上的分歧**本来就该大**（那是我们训出来的领域能力）。
PROMPTS = {
    "闲聊/能力询问（应当会说人话）": [
        "你好，你是做什么的？",
        "你能帮我改预算吗？",
        "帮我优化一下",
        "刚才那个结论你是怎么得出来的？",
        "谢谢，辛苦了",
    ],
    "任务（领域能力，分歧大是正常的）": [
        "查一下 CMP_1 昨天的花费和转化",
        "CMP_4 表现不错，帮我评估并执行扩量，预算提高 20%",
        "CMP_3 最近成本很高，帮我分析原因",
    ],
}


from syncopate.train.opd_render import _char_labels, render_prompt_text, segment_text, segment_tokens  # noqa: E402,F401  09-05：搬进正规模块

async def generate(client: httpx.AsyncClient, prompt: str, *, max_tokens: int = 512):
    r = await client.post("/v1/completions", json={
        "model": CAND_MODEL, "prompt": prompt, "max_tokens": max_tokens,
        "temperature": SAMPLING_TEMPERATURE, "top_p": SAMPLING_TOP_P,
        "top_k": SAMPLING_TOP_K, "stop": [ASSISTANT_TURN_END], "logprobs": 0,
    }, timeout=180)
    r.raise_for_status()
    ch = r.json()["choices"][0]
    lp = ch.get("logprobs") or {}
    return ch["text"], lp.get("tokens", []), lp.get("token_logprobs", [])


async def score_under_base(client: httpx.AsyncClient, prompt: str, completion: str,
                           n_completion_tokens: int):
    """把 prompt+completion 整段送给底座，取 **completion 部分**的 prompt_logprobs。

    ★ 这是"同一串 token 上两个模型的分歧"的正确量法（见模块 docstring）。
    """
    r = await client.post("/v1/completions", json={
        "model": BASE_MODEL, "prompt": prompt + completion, "max_tokens": 1,
        "temperature": 0, "prompt_logprobs": 0,
    }, timeout=180)
    r.raise_for_status()
    pl = r.json()["choices"][0].get("prompt_logprobs") or []
    out: list[float | None] = []
    for entry in pl[-n_completion_tokens:] if n_completion_tokens else []:
        if not entry:
            out.append(None)
            continue
        # {token_id: {"logprob": x, ...}} —— 取被实际采用的那个（唯一一项）
        out.append(next(iter(entry.values())).get("logprob"))
    return out


async def qualitative(cand: httpx.AsyncClient, base: httpx.AsyncClient,
                      tokenizer, tools) -> None:
    print("\n" + "=" * 78)
    print("定性对照：同一个问题，候选 vs 裸底座各说了什么")
    print("=" * 78)
    for group, prompts in PROMPTS.items():
        print(f"\n──── {group} ────")
        for msg in prompts:
            p = render_prompt_text(tokenizer, msg, tools)
            ctext, _, _ = await generate(cand, p, max_tokens=256)
            r = await base.post("/v1/completions", json={
                "model": BASE_MODEL, "prompt": p, "max_tokens": 256,
                "temperature": SAMPLING_TEMPERATURE, "top_p": SAMPLING_TOP_P,
                "stop": [ASSISTANT_TURN_END]}, timeout=180)
            btext = r.json()["choices"][0]["text"]
            print(f"\n  用户：{msg}")
            print(f"    候选：{ctext.strip()[:230]}")
            print(f"    底座：{btext.strip()[:230]}")


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qual", action="store_true", help="只跑定性对照")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    import syncopate.domains.adcampaign  # noqa: F401
    from syncopate.core.tool_registry import REGISTRY

    tokenizer = AutoTokenizer.from_pretrained("models/Qwen3-4B-sft-v13r2-e1")
    tools = REGISTRY.menu(None)

    async with httpx.AsyncClient(base_url=CAND_URL) as cand, \
               httpx.AsyncClient(base_url=BASE_URL) as base:
        for url, c, name in ((CAND_URL, cand, "候选"), (BASE_URL, base, "底座")):
            try:
                (await c.get("/v1/models", timeout=5)).raise_for_status()
            except Exception as e:                       # noqa: BLE001
                print(f"🔴 {name}端点 {url} 不可用：{e}")
                return 1

        if args.qual:
            await qualitative(cand, base, tokenizer, tools)
            return 0

        rows: list[dict[str, Any]] = []
        for group, prompts in PROMPTS.items():
            for msg in prompts:
                p = render_prompt_text(tokenizer, msg, tools)
                text, toks, cand_lp = await generate(cand, p)
                if not toks:
                    print(f"⚠️ 跳过（端点没返回 logprobs）：{msg}")
                    continue
                base_lp = await score_under_base(base, p, text, len(toks))
                labels = segment_tokens(toks)
                n = min(len(toks), len(cand_lp), len(base_lp), len(labels))
                for i in range(n):
                    if cand_lp[i] is None or base_lp[i] is None:
                        continue
                    rows.append({"group": group, "msg": msg, "tok": toks[i],
                                 "seg": labels[i],
                                 "d": cand_lp[i] - base_lp[i]})
                print(f"  ✓ {msg[:26]:<28} {n:>4} token")

        if not rows:
            print("🔴 一条都没量到")
            return 1

        print("\n" + "=" * 78)
        print("逐 token 分歧（候选 logprob − 底座 logprob；正=候选更笃定）")
        print("=" * 78)
        print(f"{'分组':<26}{'段':<8}{'token 数':>8}{'均值 Δ':>10}{'中位 Δ':>10}"
              f"{'|Δ|>3 占比':>12}")
        for group in PROMPTS:
            for seg, seg_cn in (("text", "自然语言"), ("format", "格式"), ("tool", "工具")):
                xs = [r["d"] for r in rows if r["group"] == group and r["seg"] == seg]
                if not xs:
                    continue
                big = sum(1 for x in xs if abs(x) > 3) / len(xs)
                print(f"{group[:24]:<26}{seg_cn:<8}{len(xs):>8}"
                      f"{statistics.mean(xs):>10.2f}{statistics.median(xs):>10.2f}"
                      f"{big:>11.0%}")

        chat_text = [r["d"] for r in rows
                     if r["seg"] == "text" and r["group"].startswith("闲聊")]
        task_text = [r["d"] for r in rows
                     if r["seg"] == "text" and r["group"].startswith("任务")]
        fmt = [r["d"] for r in rows if r["seg"] == "format"]
        print("\n判定参照（`22 §J-5`）：")
        if chat_text and fmt:
            ratio = abs(statistics.mean(chat_text)) / max(1e-6, abs(statistics.mean(fmt)))
            print(f"  闲聊·自然语言段 均值 Δ = {statistics.mean(chat_text):+.2f}"
                  f"（n={len(chat_text)}）")
            print(f"  格式段          均值 Δ = {statistics.mean(fmt):+.2f}（n={len(fmt)}）")
            if task_text:
                print(f"  任务·自然语言段 均值 Δ = {statistics.mean(task_text):+.2f}"
                      f"（n={len(task_text)}）")
            print(f"  比值 |闲聊文本| / |格式| = {ratio:.2f}")
            print("  ⇒ 比值大（分歧集中在自然语言段）= OPD 对症；"
                  "接近 1（弥散）= 退化更深，方案要重想")
        await qualitative(cand, base, tokenizer, tools)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
