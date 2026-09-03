#!/usr/bin/env python
"""v15 · W3③ —— 行为/信令类思考探针（`26 §W3` 门槛④ ≥70% 才入库）。训练机跑（需 8B 教师 @:8211）。

    SYNCOPATE_CONTRACT=v15 SYNCOPATE_THINK=1 .venv/bin/python scripts/v15_w3_behavior_think_probe.py [--n 20]

P0-5 用裸 8B 探 reject 类思考只有 2/5：教师没见过我们的信令契约。这里把**契约上下文**（v15 system prompt
+ 全量 34 工具 schema + gold 前缀）喂给教师，从 <think>\\n 续写，命中判据与 gen_cot_v15 相同（教师自己也选中
gold 的下一步动作 = 对 reject/defer/clarify 就是对应的 session.* 调用），且过 W3① 画像闸。
按行为分别报命中率；≥70% 的行为类才允许进 CoT 池，不过线的**显式不带 think 并登记欠账**。
读数落盘 _audit/v15_w3/behavior_think_probe.json。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import httpx
from syncopate.core.model_paths import TEST_TOKENIZER, STUDENT_MODEL, TEACHER_MODEL
from syncopate.pipeline.split import DEFAULT_BATCH_DIR, DEFAULT_SPLIT_DIR, DEFAULT_SFT_DIR, DEFAULT_RL_DIR

sys.path.insert(0, "scripts")


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20, help="每种行为抽几条 case")
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--teacher", default="http://127.0.0.1:8211/v1")
    args = ap.parse_args()
    from transformers import AutoTokenizer
    from syncopate.domains.adcampaign import build_domain
    from syncopate.pipeline.build_dataset import build_sft_row
    from syncopate.pipeline.split import load_bundles
    import u_build_v14_5 as B

    tok = AutoTokenizer.from_pretrained(STUDENT_MODEL)
    reg = build_domain().registry; reg.latency_scale = 0.0
    bundles = load_bundles(Path(DEFAULT_BATCH_DIR))
    by = defaultdict(list)
    for c, b in bundles.items():
        if b.gold and b.verifier.expected_behavior in ("reject", "defer", "clarify"):
            by[b.verifier.expected_behavior].append(b)
    rng = B.rng
    res = {}
    async with httpx.AsyncClient(timeout=180) as client:
        async def one_think(ctx):
            B._SEED[0] += 1
            r = await client.post(f"{args.teacher}/completions", json={
                "model": "t", "prompt": ctx + "<think>\n", "max_tokens": B.THINK_MAX_TOKENS,
                "seed": B._SEED[0], "temperature": 0.7, "top_p": 0.95})
            r.raise_for_status(); return r.json()["choices"][0]["text"]
        for beh, lst in by.items():
            rng.shuffle(lst)
            hit = tried = 0; kept = []
            for b in lst[: args.n]:
                base = await build_sft_row(b, tokenizer=tok, registry=reg, index=0, split="train", config=None)
                full = tok.decode(list(base["input_ids"])[:base["total_length"]])
                segs = full.split(B.ASST)
                # 目标步 = 该 case 里 session.<beh> 那一步（gold 最后一个 assistant 轮）
                want = f"session.{beh}"
                for k in range(1, len(segs)):
                    step = segs[k]
                    # 线格式无关：JSON `"name": "session.x"` 或 Qwen3.5 XML `<function=session.x>`
                    if f'"name": "{want}"' not in step and f"<function={want}>" not in step:
                        continue
                    ctx = B.ASST.join(segs[:k]) + B.ASST + "\n"
                    tried += 1
                    gens = await asyncio.gather(*[one_think(ctx) for _ in range(args.samples)], return_exceptions=True)
                    ok = None
                    for g in gens:
                        if isinstance(g, Exception) or "</think>" not in g:
                            continue
                        think, post = g.split("</think>", 1); think = think.strip()
                        n_seg = len([p for p in re.split(r"\n\s*\n|\n", think) if p.strip()])
                        cjk = len(re.findall(r"[一-鿿]", think)) / max(1, len(think))
                        from syncopate.core.parsing_v15 import parse_tool_calls
                        calls, _ = parse_tool_calls(post) if "<tool_call>" in post else ([], 0)
                        name = calls[0]["name"] if calls else None
                        if think and len(think) <= B.THINK_MAX_CHARS and n_seg <= B.THINK_MAX_SEGS and cjk >= 0.5 and name == want:
                            ok = think; break
                    if ok:
                        hit += 1; kept.append({"case_id": b.case_id, "think": ok})
                    break
            rate = hit / max(1, tried)
            res[beh] = {"tried": tried, "hit": hit, "rate": round(rate, 3), "pass": rate >= 0.70, "examples": kept[:3]}
            print(f"[behavior-think] {beh:8s} {hit}/{tried} = {rate:.0%}  {'✅ 入库' if rate >= 0.7 else '🔴 不带 think，登记欠账'}")
    Path("_audit/v15_w3").mkdir(parents=True, exist_ok=True)
    json.dump(res, open("_audit/v15_w3/behavior_think_probe.json", "w"), ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
