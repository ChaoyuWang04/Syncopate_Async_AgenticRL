#!/usr/bin/env python
"""v14.5 · S3 revision 正则库（`24 §4-P2` S3，模式 B：外部数据一个字不入库，只抽模式）。

    .venv/bin/python -m syncopate.pipeline.materials.mine_revision_patterns

从 Congliu/Chinese-DeepSeek-R1-Distill-data-110k 流式采样 think 段，统计"自我修正"
标记的真实出现频率，产出 ≥8 条正则 → syncopate/train/revision_patterns.py，
并在我们现有 59 条 8B think（data/u_route/cot_traces.jsonl）上试跑报命中率。
门槛：正则 ≥8 条（每条在外部样本中出现 ≥30 次才收）；本地 59 条试跑正常执行。
"""

from __future__ import annotations

import itertools
import json
import re
from pathlib import Path

SAMPLE_N = 30000

# 候选修正标记（人工先验起点，按外部真实频率筛选留存）
CANDIDATES = {
    "wait_stop": r"等等[，,、]",
    "not_right": r"不对[，,。]",
    "hold_on": r"等一下",
    "redo": r"重新(想|算|看|考虑|梳理|检查)",
    "switch": r"换(个|一个|种)(思路|角度|方法)",
    "oh_no": r"哦[，,]?\s*不",
    "actually": r"其实(不|并不|应该)",
    "recheck": r"再(检查|确认|核对|验证)一?下?",
    "went_wrong": r"(算|想|理解)错了",
    "backtrack": r"回(过头|头)来?(看|想)",
    "let_me_again": r"让我再",
    "correct_it": r"修正一?下?",
    "maybe_wrong": r"(可能|似乎|好像)(不对|有问题|错了)",
    "double_check": r"验算",
    "reconsider": r"重新考虑",
    "however_think": r"但(是)?再想",
}


def iter_thinks():
    from datasets import load_dataset
    ds = load_dataset("Congliu/Chinese-DeepSeek-R1-Distill-data-110k",
                      split="train", streaming=True)
    for row in itertools.islice(ds, SAMPLE_N):
        t = row.get("reasoning_content") or ""
        if not t:
            out = row.get("output") or row.get("content") or ""
            m = re.search(r"<think>(.*?)</think>", out, re.S)
            t = m.group(1) if m else ""
        if t:
            yield t


def main() -> int:
    counts = {k: 0 for k in CANDIDATES}
    n = 0
    for think in iter_thinks():
        n += 1
        for k, pat in CANDIDATES.items():
            if re.search(pat, think):
                counts[k] += 1
        if n % 5000 == 0:
            print(f"  …{n} 条", flush=True)
    print(f"外部样本 {n} 条 think；标记出现条数：")
    kept = {}
    for k, c in sorted(counts.items(), key=lambda x: -x[1]):
        mark = "✅" if c >= 30 else "✗"
        print(f"  {mark} {k:<14} {c}")
        if c >= 30:
            kept[k] = CANDIDATES[k]
    assert len(kept) >= 8, f"🔴 S3 门槛未过：仅 {len(kept)} 条正则（要 ≥8）"

    # 本地 59 条 8B think 试跑
    local = [json.loads(x)["think"] for x in open("data/u_route/cot_traces.jsonl")]
    hits = sum(1 for t in local if any(re.search(p, t) for p in kept.values()))
    print(f"本地 59 条 think 命中率：{hits}/{len(local)} = {hits/len(local):.0%}")

    out = Path("syncopate/train/revision_patterns.py")
    body = '"""revision（自我修正）标记正则库 —— v14.5 S3 产出（24 §4-P2）。\n\n'
    body += f"来源：Congliu R1 中文 110k 流式采样 {n} 条 think 的频率筛选（≥30 次才收）；\n"
    body += "用途：CoT 承诺闸的修正样本优先标记 + P3 aha 观测器（first_success_events）。\n"
    body += f"本地 59 条 8B think 命中率 {hits}/{len(local)}。外部数据零入库（模式 B）。\n"
    body += '"""\n\nREVISION_PATTERNS = {\n'
    for k, p in kept.items():
        body += f"    {k!r}: r{p!r},\n"
    body += "}\n\n\ndef has_revision(text: str) -> bool:\n"
    body += "    import re\n"
    body += "    return any(re.search(p, text) for p in REVISION_PATTERNS.values())\n"
    out.write_text(body, encoding="utf-8")
    print(f"✅ S3 完成 → {out}（{len(kept)} 条正则）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
