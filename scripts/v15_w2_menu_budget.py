#!/usr/bin/env python
"""v15 · W2⑤ —— 全量菜单（34 个工具）下的 prompt 重量实测（`26 §W2⑤`，Chaoyu 08-31 裁定②）。

    SYNCOPATE_CONTRACT=v15 .venv/bin/python scripts/v15_w2_menu_budget.py --batch data/batches/v13

对每条 case 渲染两种 prompt：按 case 裁剪菜单（训练现状）vs 全量菜单（线上形状），
用 models/Qwen3-4B tokenizer 数 token，与 rollout_budget.MAX_PROMPT_LENGTH 比：
  放得下 ⇒ 直接改全量；放不下 ⇒ 精简 tool_registry 的工具描述（两侧共用一份，精简后仍同形）。
读数落盘 _audit/v15_w2/menu_budget_<batch>.json（守则：建库/量重的输出一律入库）。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", default="data/batches/v13")
    ap.add_argument("--model", default="models/Qwen3-4B")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    from transformers import AutoTokenizer
    from syncopate.domains.adcampaign import build_domain
    from syncopate.pipeline.split import load_bundles
    from syncopate.train.rollout_budget import MAX_PROMPT_LENGTH
    from syncopate.train.rollout_loop import CHAT_TEMPLATE_KWARGS, build_messages

    tok = AutoTokenizer.from_pretrained(args.model)
    reg = build_domain().registry
    full_tools = reg.menu(None)
    bundles = list(load_bundles(Path(args.batch)).values())
    if args.limit:
        bundles = bundles[: args.limit]
    rows = []
    for b in bundles:
        msgs = build_messages(b, b.case.tool_menu)
        n_case = len(tok.apply_chat_template(msgs, tools=reg.menu(b.case.tool_menu),
                                             add_generation_prompt=True, tokenize=True,
                                             **CHAT_TEMPLATE_KWARGS))
        n_full = len(tok.apply_chat_template(msgs, tools=full_tools,
                                             add_generation_prompt=True, tokenize=True,
                                             **CHAT_TEMPLATE_KWARGS))
        rows.append({"case_id": b.case_id, "menu_n": len(b.case.tool_menu or full_tools),
                     "tok_case_menu": n_case, "tok_full_menu": n_full})
    mx_c = max(r["tok_case_menu"] for r in rows); mx_f = max(r["tok_full_menu"] for r in rows)
    over = sum(r["tok_full_menu"] > MAX_PROMPT_LENGTH for r in rows)
    tools_only = len(tok.apply_chat_template([{"role": "system", "content": ""}, {"role": "user", "content": ""}],
                                             tools=full_tools, add_generation_prompt=True, tokenize=True,
                                             **CHAT_TEMPLATE_KWARGS))
    out = {"batch": args.batch, "n_cases": len(rows), "full_menu_tools": len(full_tools),
           "tools_block_tokens": tools_only, "max_prompt_case_menu": mx_c, "max_prompt_full_menu": mx_f,
           "MAX_PROMPT_LENGTH": MAX_PROMPT_LENGTH, "over_budget_full_menu": over,
           "headroom_full_menu": MAX_PROMPT_LENGTH - mx_f, "rows": rows}
    Path("_audit/v15_w2").mkdir(parents=True, exist_ok=True)
    p = Path(f"_audit/v15_w2/menu_budget_{Path(args.batch).name}.json")
    json.dump(out, open(p, "w"), ensure_ascii=False, indent=1)
    print(f"[menu-budget] {args.batch}: {len(rows)} 条 · 全量菜单 {len(full_tools)} 工具（工具块 {tools_only} tok）")
    print(f"  prompt max：按题裁剪 {mx_c} → 全量 {mx_f}（上限 {MAX_PROMPT_LENGTH}，余量 {MAX_PROMPT_LENGTH - mx_f}，超限 {over} 条）")
    print(f"  → {p}")
    return 0 if over == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
