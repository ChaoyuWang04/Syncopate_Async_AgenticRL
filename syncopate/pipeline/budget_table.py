#!/usr/bin/env python
"""v15 · W3③ —— CoT 可达性预算表：**先算后训**（`26 §4.4`、门槛③预注册；W4/W5 实测回填）。

    SYNCOPATE_CONTRACT=v15 .venv/bin/python -m syncopate.pipeline.budget_table

从现役 CoT 池（data/u_route/v16_cot_rows.json）实测：think 长度/段数/token 画像、行重；
按 W3① 约束（≤350 字 · ≤2 段）推算做轻后的行重；按裁定④带宽 30% 推算可装行数与
全库非空块占比；预注册 HARD 档触发率预测带 20–50%。读数落盘 _audit/v15_w3/budget_table.json。
⚠️ 非 CoT 桶总 token 本机没有 parquet ⇒ 用 v15 manifest 的份额反推（cot 18.13% ↔ 19 行）；W4 实测回填。
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from syncopate.core.model_paths import TEST_TOKENIZER, STUDENT_MODEL, TEACHER_MODEL
from syncopate.pipeline.split import DEFAULT_BATCH_DIR, DEFAULT_SPLIT_DIR, DEFAULT_SFT_DIR, DEFAULT_RL_DIR


def main() -> int:
    from transformers import AutoTokenizer
    from syncopate.core.model_paths import build_tokenizer_path
    tok_path = build_tokenizer_path()
    tok = AutoTokenizer.from_pretrained(tok_path)
    rows = json.load(open("data/u_route/v16_cot_rows.json"))   # 裁定⑭：v16 缓存名
    TH = re.compile(r"<think>(.*?)</think>", re.S)
    chars, segs, tk = [], [], []
    per_row = []
    for r in rows:
        txt = tok.decode(r["input_ids"][r["prompt_length"]:r["total_length"]])
        ths = [t.strip() for t in TH.findall(txt) if t.strip()]
        t_tok = 0
        for t in ths:
            chars.append(len(t)); segs.append(len([p for p in re.split(r"\n\s*\n|\n", t) if p.strip()]))
            n = len(tok.encode(t)); tk.append(n); t_tok += n
        per_row.append({"case_id": r["case_id"], "sup_tok": r["supervised_tokens"], "think_tok": t_tok,
                        "nonempty": len(ths), "blocks": r.get("_blocks", 0)})
    chars.sort(); segs.sort(); tk.sort()
    q = lambda a, p: a[min(len(a) - 1, int(len(a) * p))]
    tok_per_char = sum(tk) / max(1, sum(chars))
    now = {"rows": len(rows), "think_blocks_nonempty": len(chars),
           "think_chars_p50": q(chars, .5), "think_chars_p95": q(chars, .95),
           "think_segs_p50": q(segs, .5), "think_segs_p95": q(segs, .95),
           "think_tok_p50": q(tk, .5), "tok_per_char": round(tok_per_char, 3),
           "row_sup_tok_p50": sorted(r["sup_tok"] for r in per_row)[len(per_row) // 2],
           "row_think_tok_p50": sorted(r["think_tok"] for r in per_row)[len(per_row) // 2]}
    # ⛔ 09-02 Chaoyu 裁定：不缩短 CoT ⇒ 行重按现行画像原样算（THINK_CAP 只用于对照列）
    new_block_tok = q(tk, .5)
    row_new = [r["sup_tok"] - r["think_tok"] + r["nonempty"] * new_block_tok for r in per_row]
    row_new_p50 = sorted(row_new)[len(row_new) // 2]
    # 预算：v15 manifest 份额反推（cot 18.13% ↔ 19 行 × 现行重）
    man = json.load(open(f"{DEFAULT_SFT_DIR}/manifest.json"))
    cot_share_prev, cot_rows_prev = man["sup_tok_share"]["cot"], man["sources"]["cot_hard"]
    cot_tok_prev = cot_rows_prev * now["row_sup_tok_p50"]
    non_cot = cot_tok_prev * (1 - cot_share_prev) / cot_share_prev
    band_hi = 0.30
    budget = non_cot * band_hi / (1 - band_hi)
    fit_rows = int(budget // row_new_p50)
    blocks_total_est = 4049 + 0    # ⚠️ 欠账（09-05 登记）：v15 949 行版全库 think 块数（_audit/v15_r2/gates.json）；v16 首次 strict 建库后按 manifest 重填
    nonempty_est = fit_rows * (sum(r["nonempty"] for r in per_row) / len(per_row))
    out = {"now": now, "projection": {
        "think_block_tok_after": new_block_tok, "row_sup_tok_after_p50": row_new_p50,
        "non_cot_tok_est": int(non_cot), "cot_budget_at_30pct": int(budget), "cot_rows_fit": fit_rows,
        "cot_rows_before": cot_rows_prev,
        "global_nonempty_share_est": round(nonempty_est / (blocks_total_est + fit_rows * now["think_segs_p50"] * 0), 3),
        "hard_tier_trigger_rate_preregistered": [0.20, 0.50],
        "assumptions": ["非 CoT token 按 v15 manifest 份额反推，W4 实测回填",
                        "CoT 不缩短（Chaoyu 09-02 裁定）：行重按现池原样；空 think 块已 mask，不计监督 token",
                        "全库块数按 949 行版 4049 估，fam 行未计"]}}
    Path("_audit/v15_w3").mkdir(parents=True, exist_ok=True)
    json.dump(out, open("_audit/v15_w3/budget_table.json", "w"), ensure_ascii=False, indent=1)
    print(f"[budget] 现池 {now['rows']} 行：think p50 {now['think_chars_p50']} 字/{now['think_segs_p50']} 段/{now['think_tok_p50']} tok · 行重 p50 {now['row_sup_tok_p50']}")
    print(f"[budget] 不缩短：块 p50 {new_block_tok} tok · 行重 p50 {row_new_p50} · 30% 带宽预算 ≈{int(budget)} ⇒ 可装 ≈{fit_rows} 行（此前 {cot_rows_prev}）")
    print(f"[budget] 全库非空块占比估 ≈{out['projection']['global_nonempty_share_est']:.1%} · HARD 档触发率预注册带 20–50%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
