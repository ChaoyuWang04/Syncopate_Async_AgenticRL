#!/usr/bin/env python
"""U 路 P0-5 · CoT 教师选型探针（`24 §4-P0`）：Qwen3-8B think-on 在 10 条 gold 轨迹
（完整工具+观测上下文）上生成思考+终答，判「末答与 gold 一致率 ≥70%」。

    CUDA_VISIBLE_DEVICES=2 .venv/bin/python scripts/u_teacher_probe.py [--model models/Qwen3-8B]

做法：SFT v13 的 answer 行 = (上下文 tokens, gold 终答 tokens)。把上下文解码后
**强制补开 `<think>\n`** 让教师先思考再答；判定 = 教师 </think> 后的输出里，
gold answer JSON 的全部原始值（数字/标签）都出现（宽松包含，不比格式——
P2 造数据时终答仍用 gold，教师只贡献 think 段，所以这里量的是"教师的推理
能否在证据齐全时落到正确结论"）。
"""

from __future__ import annotations

import argparse
import json
import re
import sys

import pandas as pd
import torch

sys.path.insert(0, ".")


def gold_values(ans_text: str) -> list[str]:
    """从 gold 终答 JSON 里抽 answer 字段的原始值（数字与短标签）。"""
    m = re.search(r"\{.*\}", ans_text, re.S)
    if not m:
        return []
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    vals: list[str] = []

    def walk(x):
        if isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
        elif x is not None:
            s = str(x).strip()
            if s and s.lower() not in ("answered",):
                vals.append(s)
    walk(d.get("answer", {}))
    return vals


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Qwen3-8B")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--max-think", type=int, default=1024)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    df = pd.read_parquet("data/sft/v13/train.parquet")
    # ⚠️ answer 行全是 CHAT（gold=自由句，逐字包含判据不适用）——首版 0/10 的探针病。
    #   改抽 defer/reject：gold=机器标签+短字段（可判），且"该不该做"正是 CoT 主场。
    rows = pd.concat([df[df.behavior == "defer"].head(args.n // 2),
                      df[df.behavior == "reject"].head(args.n - args.n // 2)])
    print(f"取 {len(rows)} 条 defer/reject 行；教师 = {args.model}")

    tok = AutoTokenizer.from_pretrained(args.model)
    stok = AutoTokenizer.from_pretrained("models/Qwen3-4B-sft-v13r2-e1")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map={"": 0}).eval()

    ASST = "<|im_start|>assistant"
    EMPTY_THINK = "<think>\n\n</think>\n\n"
    hits, results = 0, []
    for _, r in rows.iterrows():
        ids = list(r.input_ids)
        full = stok.decode(ids[: r.total_length])
        # 监督段=整条多步轨迹 ⇒ gold 终答=最后一个 assistant 段；上下文=其之前全部
        cut = full.rfind(ASST)
        head_end = full.find("\n", cut) + 1
        ctx, gold = full[:head_end], full[head_end:]
        vals = gold_values(gold)
        # 强制真思考：剥掉 think-off 空对，换成敞开的 <think>\n
        prompt = ctx + EMPTY_THINK
        prompt = ctx + "<think>\n" if gold.startswith(EMPTY_THINK) or True else prompt
        if gold.startswith(EMPTY_THINK):
            gold = gold[len(EMPTY_THINK):]
        enc = tok(prompt, return_tensors="pt").to(0)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=args.max_think + 512,
                                 do_sample=True, temperature=0.6, top_p=0.95)
        gen = tok.decode(out[0][enc.input_ids.shape[1]:], skip_special_tokens=False)
        post = gen.split("</think>")[-1] if "</think>" in gen else gen
        ok = bool(vals) and all(v in post for v in vals)
        hits += ok
        results.append({"case_id": r.case_id, "gold_vals": vals, "ok": ok,
                        "think_len": len(gen.split("</think>")[0]) if "</think>" in gen else -1,
                        "post_head": post[:160]})
        print(f"  {r.case_id}: {'✅' if ok else '✗'} gold={vals[:3]} "
              f"think_chars={results[-1]['think_len']}")
    rate = hits / max(len(rows), 1)
    json.dump({"model": args.model, "rate": rate, "results": results},
              open("logs/u_route/teacher_probe.json", "w"), ensure_ascii=False, indent=2)
    print(f"末答一致率 = {hits}/{len(rows)} = {rate:.0%}  （门槛 ≥70%）")
    print("TEACHER-" + ("PASS" if rate >= 0.7 else "FAIL"))
    return 0 if rate >= 0.7 else 1


if __name__ == "__main__":
    raise SystemExit(main())
