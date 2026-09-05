#!/usr/bin/env python
"""v14.5 · S2 省略句式模式库（`24 §4-P2` S2，模式 B：只抽分布形状，外部文本零入库）。

    .venv/bin/python -m syncopate.pipeline.materials.mine_ellipsis_patterns

从本地 WildChat-1M（中文段·无毒）挖真实的省略式追问：多轮里第二个 user 轮 ≤15 字
且含 {呢|那|它|这|又|还}。骨架化 = 把与上文重叠的内容词替换为 {X}，按骨架频次聚类。
产出 data/u_route/ellipsis_patterns.json：{template, freq, split(train/exam)}。
门槛：模板 ≥30（每条外部频次 ≥3）· 与手写 REF_FORMS 重叠 ≤4 · 70/30 切分随机种子固定。
人核样本落 logs/u_route/s2_pattern_audit.txt（30 条 + 各 3 个脱敏实例）供抽查。
"""

from __future__ import annotations

import glob
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

MARK = re.compile(r"[呢那它这又还]")
CJK = re.compile(r"[一-鿿]{2,}")
HAND_REF_FORMS = ["它的{m}呢？", "这条计划的{m}呢", "那{m}怎么样", "顺便看下它的{m}"]


def skeletonize(follow: str, context: str) -> str:
    s = re.sub(r"[A-Za-z0-9_.\-]+", "{X}", follow)
    # 与上文重叠的内容词（≥2 字 CJK 且出现在上文）→ {X}，取最长优先
    words = sorted(set(CJK.findall(s)), key=len, reverse=True)
    for w in words:
        if w in context and len(w) >= 2:
            s = s.replace(w, "{X}")
    s = re.sub(r"(\{X\})+", "{X}", s)
    return s.strip()


def main() -> int:
    files = sorted(glob.glob(
        "/workspace/hf/hub/datasets--allenai--WildChat-1M/snapshots/*/data/*.parquet"))
    assert files, "WildChat parquet 未找到"
    raw = []          # (skeleton, follow_example)
    scanned = 0
    for f in files:
        t = pq.read_table(f, columns=["conversation", "language", "toxic"])
        for conv, lang, toxic in zip(t["conversation"].to_pylist(),
                                     t["language"].to_pylist(),
                                     t["toxic"].to_pylist()):
            if lang != "Chinese" or toxic:
                continue
            scanned += 1
            users = [(i, m["content"]) for i, m in enumerate(conv)
                     if m.get("role") == "user" and m.get("content")]
            for k in range(1, len(users)):
                fu = users[k][1].strip()
                if not (2 <= len(fu) <= 15) or not MARK.search(fu):
                    continue
                if re.search(r"[a-zA-Z]{20,}|http", fu):
                    continue
                ctx = users[k - 1][1][:500]
                sk = skeletonize(fu, ctx)
                # 必须真的抽象出了指代/省略（含 {X} 或以承接词开头）
                if "{X}" in sk or re.match(r"^(那|它|这|还有|又)", sk):
                    raw.append((sk, fu))
        if len(raw) >= 20000:
            break
    print(f"扫描中文无毒对话 {scanned} 段，候选追问 {len(raw)} 条")
    assert len(raw) >= 2000, f"🔴 S2 门槛未过：原始候选 {len(raw)} < 2000"

    freq = Counter(sk for sk, _ in raw)
    examples = defaultdict(list)
    for sk, fu in raw:
        if len(examples[sk]) < 3:
            examples[sk].append(fu)
    # 收模板：频次 ≥3、长度合理、不是纯 {X} 噪声
    kept = [(sk, c) for sk, c in freq.most_common(200)
            if c >= 3 and 2 <= len(sk.replace("{X}", "")) and len(sk) <= 20
            and sk.count("{X}") <= 2 and sk not in ("{X}", "{X}呢")]
    kept = kept[:60]
    assert len(kept) >= 30, f"🔴 S2 门槛未过：模板 {len(kept)} < 30"
    overlap = sum(1 for sk, _ in kept
                  for h in HAND_REF_FORMS
                  if sk.replace("{X}", "{m}") == h.replace("？", "").replace("?", ""))
    assert overlap <= 4, f"🔴 与手写 REF_FORMS 重叠 {overlap} > 4"

    rng = random.Random(1405)
    idx = list(range(len(kept)))
    rng.shuffle(idx)
    n_exam = max(1, round(len(kept) * 0.3))
    exam_set = set(idx[:n_exam])
    out = [{"template": sk, "freq": c,
            "split": "exam" if i in exam_set else "train"}
           for i, (sk, c) in enumerate(kept)]
    Path("data/u_route").mkdir(parents=True, exist_ok=True)
    with open("data/u_route/ellipsis_patterns.json", "w") as f:
        json.dump({"source": "WildChat-1M zh (mode-B pattern mining, no raw text)",
                   "scanned_dialogs": scanned, "raw_candidates": len(raw),
                   "templates": out, "seed": 1405,
                   "split_ratio": {"train": len(kept) - n_exam, "exam": n_exam}},
                  f, ensure_ascii=False, indent=1)
    with open("logs/u_route/s2_pattern_audit.txt", "w") as f:
        for i, (sk, c) in enumerate(kept):
            sp = "exam" if i in exam_set else "train"
            f.write(f"[{sp}] {sk}   (freq={c})   例: {examples[sk]}\n")
    print(f"✅ S2 完成：模板 {len(kept)}（train {len(kept)-n_exam} / exam {n_exam}）"
          f"· 手写重叠 {overlap} · 审计样本 → logs/u_route/s2_pattern_audit.txt")
    for sk, c in kept[:15]:
        print(f"   {sk}  ×{c}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
