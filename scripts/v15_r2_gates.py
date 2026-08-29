#!/usr/bin/env python
"""v15 · R2 数据层门槛检查（`25 §R2`）。

    SYNCOPATE_CONTRACT=v15 SYNCOPATE_THINK=1 .venv/bin/python scripts/v15_r2_gates.py \
        [--parquet data/sft/v15/train.parquet] [--dry-run 12]

两种用法：
  --dry-run N   不建库，直接回放每种行为各 N 条 gold，验**构建路径**（不吃 GPU，秒级）
  --parquet P   查已建好的数据集（正式门槛）

门槛（`25 §R2`）：③壳残留=0 · ④信令语法=100% · ⑤⒜think出现率=100% · ⑤⒝非空占比 20–35%
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
from pathlib import Path

REQ = {"session.defer": {"reason", "recheck_after_days"},
       "session.clarify": {"question", "missing_fields"},
       "session.reject": {"reason_code", "explanation"}}
_SHELL = re.compile(r"```json.*?\"behavior\"", re.S)
_THINK = re.compile(r"<think>(.*?)</think>", re.S)
_TC = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)


def audit(supervised_texts: list[str]) -> dict:
    n = len(supervised_texts)
    shell = summary = rows_think = blocks = nonempty = sig = sig_ok = 0
    for sup in supervised_texts:
        if _SHELL.search(sup):
            shell += 1
        if re.search(r'"summary"\s*:', sup):
            summary += 1
        bl = _THINK.findall(sup)
        if bl:
            rows_think += 1
        blocks += len(bl)
        nonempty += sum(1 for x in bl if x.strip())
        for blk in _TC.findall(sup):
            try:
                p = json.loads(blk)
            except json.JSONDecodeError:
                continue
            if p.get("name") in REQ:
                sig += 1
                if REQ[p["name"]].issubset(p.get("arguments") or {}):
                    sig_ok += 1
    return {"rows": n, "shell_residue_rows": shell, "summary_rows": summary,
            "rows_with_think": rows_think, "think_blocks": blocks,
            "think_nonempty": nonempty, "signal_calls": sig, "signal_schema_ok": sig_ok}


def report(a: dict) -> int:
    n = max(1, a["rows"])
    blocks = max(1, a["think_blocks"])
    ratio = a["think_nonempty"] / blocks
    bad = 0
    print("════ R2 数据层门槛 ════")
    print(f"样本 {a['rows']} 行 · think 块 {a['think_blocks']} 个 · 信令调用 {a['signal_calls']} 次")
    for label, val, ok in [
        (f"③ 壳残留 ```json+behavior : {a['shell_residue_rows']} 行   门槛 =0",
         a["shell_residue_rows"], a["shell_residue_rows"] == 0),
        (f"③ summary 字段出现        : {a['summary_rows']} 行   门槛 =0",
         a["summary_rows"], a["summary_rows"] == 0),
        (f"④ 信令 schema 合法率      : {a['signal_schema_ok']}/{a['signal_calls']} = "
         f"{a['signal_schema_ok']/max(1,a['signal_calls']):.1%}   门槛 =100%",
         0, a["signal_calls"] > 0 and a["signal_schema_ok"] == a["signal_calls"]),
        (f"⑤⒜ 监督段含 think 的行    : {a['rows_with_think']}/{a['rows']} = "
         f"{a['rows_with_think']/n:.1%}   门槛 =100%",
         0, a["rows_with_think"] == a["rows"]),
        (f"⑤⒝ 非空 think 块占比      : {a['think_nonempty']}/{a['think_blocks']} = "
         f"{ratio:.1%}   门槛 20–35%", 0, 0.20 <= ratio <= 0.35),
    ]:
        print(f"  {label}   {'✅' if ok else '🔴'}")
        bad += int(not ok)
    return bad


def dry_run(per: int) -> list[str]:
    from transformers import AutoTokenizer

    from syncopate.domains.adcampaign import build_domain
    from syncopate.pipeline.sft_replay import build_sft_sample
    from syncopate.pipeline.split import load_bundles

    tok = AutoTokenizer.from_pretrained("models/Qwen3-4B")
    reg = build_domain().registry
    reg.latency_scale = 0.0
    by: dict[str, list] = {}
    for b in load_bundles(Path("data/batches/v13")).values():
        if b.gold:
            by.setdefault(b.verifier.expected_behavior, []).append(b)
    texts = []
    for beh in sorted(by):
        for b in by[beh][:per]:
            s = asyncio.run(build_sft_sample(b, tokenizer=tok, registry=reg))
            texts.append(tok.decode([t for t, m in zip(s.input_ids, s.loss_mask) if m == 1],
                                    skip_special_tokens=False))
    return texts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet")
    ap.add_argument("--dry-run", type=int, default=0)
    args = ap.parse_args()
    if args.parquet:
        import pandas as pd
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("models/Qwen3-4B")
        d = pd.read_parquet(args.parquet)
        texts = [tok.decode([t for t, m in zip(list(r["input_ids"]), list(r["loss_mask"]))
                             if m == 1], skip_special_tokens=False)
                 for _, r in d.iterrows()]
    elif args.dry_run:
        texts = dry_run(args.dry_run)
        print(f"（dry-run：五种行为各 {args.dry_run} 条，只验构建路径，不是正式门槛）")
    else:
        ap.error("给 --parquet 或 --dry-run N")
    return report(audit(texts))


if __name__ == "__main__":
    raise SystemExit(main())
