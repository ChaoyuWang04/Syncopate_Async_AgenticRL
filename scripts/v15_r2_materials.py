#!/usr/bin/env python
"""v15 · R2 教师物料抽取 —— 把 v14.5 缓存里**与契约无关**的文本捞出来复用。

    .venv/bin/python scripts/v15_r2_materials.py

为什么这么做：v14.5 的缓存存的是**已 token 化的行**（契约绑定，v15 不能直接用），
但里面的**教师文本**（L2/L1 的 reply、CoT 的 think、chat 的 reply）是自然语言，
和契约无关 —— 重放一遍就能换契约，没必要再花几小时重跑教师。

⚠️ 但**不复用 summary**：v15 已废除该字段（`25 §3.1`），而它正是 08-29 真人实测
   发现③「summary 通道被『X 释义』模板污染」的病灶。捞出来会把污染带进 v15。

产物 data/u_route/v15_materials.json
    l2_replies  {case_id: reply}      L2 数据追问的教师回复（含读数）
    l1_replies  {case_id: reply}      L1 概念追问的教师回复
    cot_think   {case_id: {step: think}}  难例逐步思考（v14.5 只有终答步，v15 要补）
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from transformers import AutoTokenizer

SHELL = re.compile(r"```json\s*(\{.*?\})\s*```", re.S)
THINK = re.compile(r"<think>(.*?)</think>", re.S)
ASST = "<|im_start|>assistant"


def main() -> int:
    tok = AutoTokenizer.from_pretrained("models/Qwen3-4B")
    out: dict[str, dict] = {"l2_replies": {}, "l1_replies": {}, "cot_think": {}}

    c = json.load(open("data/u_route/v145_l2l1_rows.json"))
    for key, rows in (("l2_replies", c["l2"]), ("l1_replies", c["l1"])):
        for r in rows:
            sup = tok.decode([t for t, m in zip(r["input_ids"], r["loss_mask"]) if m == 1])
            m = SHELL.search(sup)
            if not m:
                continue
            try:
                payload = json.loads(m.group(1))
            except json.JSONDecodeError:
                continue
            reply = (payload.get("answer") or {}).get("reply")
            if reply:                       # ★ 只捞 reply，不捞 summary（v15 已废除）
                out[key][r["case_id"]] = reply

    for r in json.load(open("data/u_route/v145_cot_rows.json")):
        full = tok.decode(r["input_ids"])
        segs = full.split(ASST)
        per_step = {}
        for i, seg in enumerate(segs[1:], start=1):
            blocks = THINK.findall(seg)
            for b in blocks:
                if b.strip():
                    per_step[str(i)] = b.strip()
                    break
        if per_step:
            out["cot_think"][r["case_id"]] = per_step

    p = Path("data/u_route/v15_materials.json")
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"L2 reply {len(out['l2_replies'])} · L1 reply {len(out['l1_replies'])} · "
          f"CoT think {len(out['cot_think'])} 行（合计 "
          f"{sum(len(v) for v in out['cot_think'].values())} 步）")
    print(f"产物 → {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
