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


# ⚠️ 难例桶 = **教师思考桶**（我们主动教了思考的那些行），不是"题目难的桶"。
#   ⛔ 2026-08-30：初版把 v13 的 `reasoning`（推理类 case 家族，69 行，从来没教过思考）
#     也算进分母 ⇒ 实测 16.2% 假红；只算 cot_hard 是 84/140 = 60.0%，与选择环节一致。
#   ★ 判据的**分母定义**和判据本身一样重要——分母错了，量的就是另一件事。
HARD_BUCKETS = {"cot_hard", "cot_distill"}


def audit(supervised_texts: list[str], buckets: list[str] | None = None) -> dict:
    """★ 分层统计（Chaoyu 08-30 裁定）：难例桶 / 非难例桶分开报。

    全库口径分不清「思考撑在难例上」和「均匀撒在全库」——后者同样能达标，
    却把"简单题也思考"训进权重，直接撞 R5⑤ 的「简单集 ≤10%」。
    ⇒ 判据写在它想要的那件事上（守则①）。
    """
    n = len(supervised_texts)
    buckets = buckets or [""] * n
    layer = {"hard": [0, 0], "easy": [0, 0]}          # [非空块, 总块]
    shell = summary = rows_think = blocks = nonempty = sig = sig_ok = 0
    for sup, bk in zip(supervised_texts, buckets):
        if _SHELL.search(sup):
            shell += 1
        if re.search(r'"summary"\s*:', sup):
            summary += 1
        bl = _THINK.findall(sup)
        if bl:
            rows_think += 1
        blocks += len(bl)
        ne = sum(1 for x in bl if x.strip())
        nonempty += ne
        key = "hard" if bk in HARD_BUCKETS else "easy"
        layer[key][0] += ne
        layer[key][1] += len(bl)
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
            "think_nonempty": nonempty, "signal_calls": sig, "signal_schema_ok": sig_ok,
            "hard_nonempty": layer["hard"][0], "hard_blocks": layer["hard"][1],
            "easy_nonempty": layer["easy"][0], "easy_blocks": layer["easy"][1]}


def _r(a: dict, layer: str) -> float:
    return a.get(f"{layer}_nonempty", 0) / max(1, a.get(f"{layer}_blocks", 0))


def gate_status(a: dict) -> list[tuple[str, bool]]:
    """★ 门槛判定的**唯一实现**——report 与 --certify 共用一份。

    保护性/判定性逻辑必须提成一份函数、所有路径共用（守则②：当时另外两个读 ckpt 的
    脚本没有那句断言）。两份实现 = 负向认证证明的是另一个判据会红。
    """
    n = max(1, a["rows"])
    blocks = max(1, a["think_blocks"])
    ratio = a["think_nonempty"] / blocks
    return [
        (f"③ 壳残留 ```json+behavior : {a['shell_residue_rows']} 行   门槛 =0",
         a["shell_residue_rows"] == 0),
        (f"③ summary 字段出现        : {a['summary_rows']} 行   门槛 =0",
         a["summary_rows"] == 0),
        (f"④ 信令 schema 合法率      : {a['signal_schema_ok']}/{a['signal_calls']} = "
         f"{a['signal_schema_ok']/max(1,a['signal_calls']):.1%}   门槛 =100%",
         a["signal_calls"] > 0 and a["signal_schema_ok"] == a["signal_calls"]),
        (f"⑤⒜ 监督段含 think 的行    : {a['rows_with_think']}/{a['rows']} = "
         f"{a['rows_with_think']/n:.1%}   门槛 =100%",
         a["rows_with_think"] == a["rows"]),
        (f"⑤⒝-难 难例桶非空 think    : {a.get('hard_nonempty', 0)}/{a.get('hard_blocks', 0)} = "
         f"{_r(a, 'hard'):.1%}   门槛 ≥60%",
         a.get("hard_blocks", 0) > 0 and _r(a, "hard") >= 0.60),
        (f"⑤⒝-易 非难例桶非空 think  : {a.get('easy_nonempty', 0)}/{a.get('easy_blocks', 0)} = "
         f"{_r(a, 'easy'):.1%}   门槛 ≤10%", _r(a, "easy") <= 0.10),
        (f"   （报告项）全库非空占比   : {a['think_nonempty']}/{a['think_blocks']} = "
         f"{ratio:.1%}   由上两条推出，不作门槛", True),
    ]


def report(a: dict) -> int:
    print("════ R2 数据层门槛 ════")
    print(f"样本 {a['rows']} 行 · think 块 {a['think_blocks']} 个 · 信令调用 {a['signal_calls']} 次")
    bad = 0
    for label, ok in gate_status(a):
        print(f"  {label}   {'✅' if ok else '🔴'}")
        bad += int(not ok)
    return bad


# ── 负向认证：每条门槛都要被证明「会红」（守则③⑬；R2 S0）────────────────────
_GOOD_SIGNAL = ('<think>\n想一下\n</think>\n\n<tool_call>{"name": "session.defer", '
                '"arguments": {"reason": "数据未成熟", "recheck_after_days": 3}}</tool_call>')
_GOOD_PLAIN = "<think>\n\n</think>\n\n这条计划的 ROAS 是 1.8，建议先观察三天。"

_BAD_CASES = [
    ("③ 壳残留", 0, [_GOOD_SIGNAL, '<think>\n\n</think>\n\n```json\n{"behavior": "defer"}\n```'], None),
    ("③ summary", 1, [_GOOD_SIGNAL, '<think>\n\n</think>\n\n{"summary": "CMP_1 释义"}'], None),
    ("④ 信令 schema", 2, ['<think>\n\n</think>\n\n<tool_call>{"name": "session.defer", '
                        '"arguments": {"reason": "缺了 recheck_after_days"}}</tool_call>'], None),
    ("⑤⒜ think 出现率", 3, [_GOOD_SIGNAL, "这一行压根没有 think 段。"], None),
    # ⑤⒝-难：难例桶全是空块 ⇒ 必须红（idx 4）
    ("⑤⒝-难 难例桶不思考", 4, [_GOOD_PLAIN] * 5, ["cot_hard"] * 5),
    # ⑤⒝-易：非难例桶到处思考 ⇒ 必须红（idx 5）——这条正是"均匀撒 think"的反例
    ("⑤⒝-易 简单题乱思考", 5, ["<think>\n每一行都在想\n</think>\n\n好的。"] * 5, ["multiturn"] * 5),
]


def certify() -> int:
    """喂手工构造的坏样本，**每条门槛都必须红**；再喂一条好样本，必须全绿。"""
    print("═══ R2 门槛负向认证：以下每条都必须判红 ═══")
    bad = 0
    for name, idx, texts, bks in _BAD_CASES:
        ok = gate_status(audit(texts, bks))[idx][1]
        print(f"  {name:22s} → {'🔴 会红 ✅' if not ok else '✅ 没红 ← 判据失效'}")
        bad += int(ok)
    # 正样本：难例桶全思考（≥60%）+ 非难例桶不思考（≤10%）
    good = [_GOOD_SIGNAL] * 2 + [_GOOD_PLAIN] * 3
    good_bk = ["cot_hard"] * 2 + ["multiturn"] * 3
    st = gate_status(audit(good, good_bk))
    for idx in (0, 1, 2, 3, 4, 5):
        if not st[idx][1]:
            print(f"  🔴 正样本被误判：{st[idx][0]}")
            bad += 1
    print("✅ 负向认证通过：五条门槛全部会红，且正样本全绿" if not bad
          else f"🔴 负向认证失败：{bad} 条")
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
    ap.add_argument("--certify", action="store_true")
    ap.add_argument("--out", help="把审计数与逐条判定落盘（留证）")
    args = ap.parse_args()
    if args.certify:
        return certify()
    buckets = None
    if args.parquet:
        import pandas as pd
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("models/Qwen3-4B")
        d = pd.read_parquet(args.parquet)
        texts = [tok.decode([t for t, m in zip(list(r["input_ids"]), list(r["loss_mask"]))
                             if m == 1], skip_special_tokens=False)
                 for _, r in d.iterrows()]
        buckets = list(d["bucket"]) if "bucket" in d.columns else None
    elif args.dry_run:
        texts, buckets = dry_run(args.dry_run), None
        print(f"（dry-run：五种行为各 {args.dry_run} 条，只验构建路径，不是正式门槛）")
    else:
        ap.error("给 --parquet 或 --dry-run N")
    a = audit(texts, buckets)
    bad = report(a)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(
            {"audit": a, "gates": [{"label": l, "pass": ok} for l, ok in gate_status(a)],
             "failed": bad}, ensure_ascii=False, indent=2))
        print(f"产物 → {args.out}")
    return bad


if __name__ == "__main__":
    raise SystemExit(main())
