#!/usr/bin/env python
"""v15 · 训练数据画廊：把最终喂进训练的每种数据**逐条解码成可读文本**（Chaoyu 09-02：要亲眼看终态）。

    SYNCOPATE_CONTRACT=v15 .venv/bin/python scripts/v15_data_gallery.py --parquet data/sft/v15/train.parquet [--per-bucket 3]
    SYNCOPATE_CONTRACT=v15 .venv/bin/python scripts/v15_data_gallery.py --parquet _audit/v15_w2/dry_rows.parquet   # 本机 DRY 演练产物

存储形态：parquet 每行 = 一条训练样本，列 input_ids（整段对话的 token）· loss_mask（哪些 token 算 loss）·
prompt_length · total_length · supervised_tokens · bucket · sub_axis · behavior · case_id …
文本不直接存，用同一个 tokenizer 解码即可还原。本脚本把每行渲染成：
  ① 元信息（桶 / 轴 / 行为 / token 计数 / think 块：非空几个 · 空块几个且已 mask）
  ② prompt 段：system（折叠，只给长度和尾段 40 字）· 历史消息对 · 本轮 user（原文）
  ③ response 段：逐 assistant 轮，think / tool_call / observation / 终答；**被监督的 token 用 ⟦ ⟧ 包起来**，
     没监督的（工具返回、空 think 块）原样显示 —— 一眼看出模型到底在学哪些字
产物：_audit/<out>.md（按桶分节，每桶 N 条）+ 汇总表（每桶行数 / 监督 token 份额 / think 统计 / 同形检查）。
"""
from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from syncopate.core.model_paths import TEST_TOKENIZER, STUDENT_MODEL, TEACHER_MODEL
from syncopate.pipeline.split import DEFAULT_BATCH_DIR, DEFAULT_SPLIT_DIR, DEFAULT_SFT_DIR, DEFAULT_RL_DIR


def spans(ids, mask, tok):
    """把 token 序列按 mask 连续段切开并解码，监督段包 ⟦⟧。"""
    out, i = [], 0
    while i < len(ids):
        j = i
        while j < len(ids) and mask[j] == mask[i]:
            j += 1
        txt = tok.decode(ids[i:j])
        out.append(f"⟦{txt}⟧" if mask[i] else txt)
        i = j
    return "".join(out)


def render_row(r, tok) -> tuple[str, dict]:
    ids, mask = list(r["input_ids"]), list(r["loss_mask"])
    pl = int(r["prompt_length"])
    prompt = tok.decode(ids[:pl])
    resp_txt = spans(ids[pl:], mask[pl:], tok)
    # prompt 拆段
    sys_m = re.search(r"<\|im_start\|>system\n(.*?)<\|im_end\|>", prompt, re.S)
    system = sys_m.group(1) if sys_m else ""
    tools_n = system.count('"type": "function"')
    body = prompt[sys_m.end():] if sys_m else prompt
    turns = re.findall(r"<\|im_start\|>(user|assistant)\n(.*?)<\|im_end\|>", body, re.S)
    hist = turns[:-1] if turns and turns[-1][0] == "user" else turns
    cur = turns[-1][1] if turns and turns[-1][0] == "user" else ""
    think_all = re.findall(r"<think>(.*?)</think>", tok.decode(ids[pl:]), re.S)
    nonempty = sum(1 for t in think_all if t.strip())
    empty = len(think_all) - nonempty
    # 空块是否已 mask：找 EMPTY_THINK 段落在 mask=1 里
    from syncopate.pipeline.sft_replay import EMPTY_THINK
    pat = tok.encode(EMPTY_THINK, add_special_tokens=False)
    resp_ids, resp_mask = ids[pl:], mask[pl:]
    leaked = sum(1 for i in range(len(resp_ids) - len(pat) + 1)
                 if resp_ids[i:i + len(pat)] == pat and any(resp_mask[i:i + len(pat)]))
    meta = {"case_id": r["case_id"], "bucket": r.get("bucket"), "sub_axis": r.get("sub_axis"),
            "behavior": r.get("behavior"), "prompt_tok": pl, "total_tok": int(r["total_length"]),
            "sup_tok": int(r["supervised_tokens"]), "think_nonempty": nonempty, "think_empty": empty,
            "empty_think_supervised": leaked, "tools_in_menu": tools_n, "history_turns": len(hist) // 2,
            "date_only": bool(re.search(r"当前时间：\d{4}-\d{2}-\d{2}\n", cur)),
            "field_list": "本次结论需要给出的字段" in cur, "folded_history": "[上一轮]" in cur}
    md = [f"### {meta['case_id']}  ·  桶 {meta['bucket']}  ·  轴 {meta['sub_axis']}  ·  行为 {meta['behavior']}",
          f"- prompt {pl} tok · 总 {meta['total_tok']} · 监督 {meta['sup_tok']} · 菜单 {tools_n} 工具 · 历史 {meta['history_turns']} 轮 · "
          f"think 非空 {nonempty} / 空 {empty}（空块有梯度：{leaked}）· 纯日期 {meta['date_only']} · 字段清单 {meta['field_list']}",
          f"- system：{len(system)} 字，尾段「…{system[-60:].strip()}」"]
    if hist:
        md.append(f"- 历史消息对（system 之后、本轮 user 之前的 **独立消息**，不在 system 里；线上同形，最近 {len(hist)//2} 轮）：")
    for k, (role, txt) in enumerate(hist, 1):
        md.append(f"  - 消息 {k} `{role}`：{txt.strip()[:300]}")
    md.append("- 本轮 user（最后一条 user 消息）：")
    md.append("```\n" + cur.strip() + "\n```")
    md.append("- response（⟦…⟧ = 被监督的 token）：")
    md.append("```\n" + resp_txt.strip()[:6000] + "\n```")
    return "\n".join(md), meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default=f"{DEFAULT_SFT_DIR}/train.parquet")
    ap.add_argument("--per-bucket", type=int, default=3)
    ap.add_argument("--out", default=None)
    ap.add_argument("--model", default=None)
    args = ap.parse_args()
    from transformers import AutoTokenizer
    tok_path = args.model or (STUDENT_MODEL if Path("models/Qwen3-4B/tokenizer.json").exists() else TEST_TOKENIZER)
    tok = AutoTokenizer.from_pretrained(tok_path)
    df = pd.read_parquet(args.parquet)
    if "bucket" not in df.columns:
        df["bucket"] = "v13_ballast"
    df["bucket"] = df["bucket"].fillna("v13_ballast")
    out = Path(args.out or f"_audit/gallery_{Path(args.parquet).stem}.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    sections, metas = defaultdict(list), []
    for bucket, g in df.groupby("bucket", sort=True):
        for _, r in g.head(args.per_bucket).iterrows():
            md, meta = render_row(r, tok)
            sections[bucket].append(md); metas.append(meta)
    # 全量统计（不只画廊里的）
    tot = int(df["supervised_tokens"].sum())
    stats = []
    for bucket, g in df.groupby("bucket", sort=True):
        stats.append((bucket, len(g), int(g["supervised_tokens"].sum()) / max(1, tot), int(g["prompt_length"].max()), int(g["total_length"].max())))
    lines = [f"# 训练数据画廊 · {args.parquet}（{len(df)} 行 · tokenizer {tok_path}）", "",
             "| 桶 | 行数 | 监督 token 份额 | prompt 最长 | 总长最长 |", "|---|---|---|---|---|"]
    lines += [f"| {b} | {n} | {s:.1%} | {pm} | {tm} |" for b, n, s, pm, tm in stats]
    bad = [m for m in metas if m["empty_think_supervised"] or m["folded_history"] or m["field_list"] or not m["date_only"]]
    lines += ["", f"画廊抽样 {len(metas)} 条：空 think 有梯度 {sum(m['empty_think_supervised'] for m in metas)} · 折叠历史 {sum(m['folded_history'] for m in metas)} · "
              f"字段清单 {sum(m['field_list'] for m in metas)} · 非纯日期 {sum(not m['date_only'] for m in metas)}"
              + ("  ✅ 同形" if not bad else "  🔴 有不同形，见各条元信息"), ""]
    for bucket in sections:
        lines.append(f"## 桶 {bucket}"); lines += sections[bucket]; lines.append("")
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"[gallery] {len(df)} 行 · {len(sections)} 桶 · 抽样 {len(metas)} 条 → {out}")
    for b, n, s, pm, tm in stats:
        print(f"  {b:16s} {n:5d} 行  份额 {s:6.1%}  prompt≤{pm}  总≤{tm}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
