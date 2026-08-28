#!/usr/bin/env python
"""U 路 P2 · CoT 冷启轨迹生成（`24 §4-P2`，P0-5 管线放大）。

    CUDA_VISIBLE_DEVICES=2 .venv/bin/python scripts/u_gen_cot.py

范围 v1 = **单步判断案**（defer/reject/answer 行，mask 结构简单精确）：
8B think-on 在完整上下文上生成思考，**末答含全部 gold 值才留**（R1-Distill 同法）；
每案采样 2 次（不同思考路径=多样性），p95 think ≤4096 字符闸。
产物 data/u_route/cot_traces.jsonl：{case_id, sample_idx, think, gold_tail}
"""

from __future__ import annotations

import json
import sys

import pandas as pd
import torch

sys.path.insert(0, "."); sys.path.insert(0, "scripts")
from u_teacher_probe import gold_values  # noqa: E402  复用值抽取

ASST = "<|im_start|>assistant"
EMPTY_THINK = "<think>\n\n</think>\n\n"


def main() -> int:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    df = pd.read_parquet("data/sft/v13/train.parquet")
    dfv = pd.read_parquet("data/sft/v13/val.parquet")
    allr = pd.concat([df, dfv])
    rows = allr[allr.behavior.isin(("defer", "reject", "answer"))]
    print(f"单步判断案 {len(rows)} 行（defer/reject/answer）")

    tok = AutoTokenizer.from_pretrained("models/Qwen3-8B")
    stok = AutoTokenizer.from_pretrained("models/Qwen3-4B-sft-v13r2-e1")
    model = AutoModelForCausalLM.from_pretrained(
        "models/Qwen3-8B", torch_dtype=torch.bfloat16, device_map={"": 0}).eval()

    out, kept, tried = [], 0, 0
    for _, r in rows.iterrows():
        ids = list(r.input_ids)
        full = stok.decode(ids[: r.total_length])
        cut = full.rfind(ASST)
        head_end = full.find("\n", cut) + 1
        ctx, gold = full[:head_end], full[head_end:]
        if gold.startswith(EMPTY_THINK):
            gold_tail = gold[len(EMPTY_THINK):]
        else:
            gold_tail = gold
        vals = gold_values(gold_tail)
        # CHAT answer 行 gold=自由句 ⇒ 不做值判据，改判「非空思考+格式收口」
        judge_vals = vals if r.behavior != "answer" else []
        for si in range(2):
            tried += 1
            enc = tok(ctx + "<think>\n", return_tensors="pt").to(0)
            with torch.no_grad():
                o = model.generate(**enc, max_new_tokens=1400, do_sample=True,
                                   temperature=0.7, top_p=0.95)
            gen = tok.decode(o[0][enc.input_ids.shape[1]:], skip_special_tokens=False)
            if "</think>" not in gen:
                continue
            think, post = gen.split("</think>", 1)
            think = think.strip()
            ok = (bool(think) and len(think) <= 4096
                  and (not judge_vals or all(v in post for v in judge_vals)))
            if ok:
                out.append({"case_id": r.case_id, "sample_idx": si,
                            "behavior": r.behavior, "think": think,
                            "gold_tail": gold_tail})
                kept += 1
        print(f"  {r.case_id}[{r.behavior}] 留 {sum(1 for x in out if x['case_id']==r.case_id)}/2",
              flush=True)
    with open("data/u_route/cot_traces.jsonl", "w") as f:
        for x in out:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    import numpy as np
    lens = [len(x["think"]) for x in out]
    print(f"✅ 保留 {kept}/{tried}（{kept/max(tried,1):.0%}）· think p95={int(np.percentile(lens,95)) if lens else 0} 字符")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
