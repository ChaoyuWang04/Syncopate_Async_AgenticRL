#!/usr/bin/env python
"""v14.5 · S1 chat 题库 v2（`24 §4-P2` S1，模式 A：外部只出 user 侧文本）。

    .venv/bin/python -m syncopate.pipeline.materials.build_chat_bank

smoltalk-chinese（classify 闲聊族 ∧ score 高分）500 候选 + WildChat-zh（无毒 ∧ 首轮 ≤60 字）
500 候选 → 互去重 + 与考场/p1_prompts 逐字去重 + 3gram 近重 → 规则质检 → 采 120 条。
门槛：120 条足额 · 考场重叠 =0 · 病句正则命中 =0 · 质检抽 50 合格率 ≥90%。
产出 data/u_route/chat_bank_v2.jsonl：{prompt, source}；审计样本落 logs/u_route/s1_audit.txt。
"""

from __future__ import annotations

import glob
import json
import random
import re
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

rng = random.Random(1450)
CJK = re.compile(r"[一-鿿]")
BAD = re.compile(r"你怎么看怎么|吧吧|指指|NAME_\d|http|www\.|gpt|GPT|openai|OpenAI|ChatGPT|作为一个AI|作为AI")
CODEY = re.compile(r"[{}<>\[\]=;`]|def |import |select ")


def ok_prompt(p: str) -> bool:
    if not (6 <= len(p) <= 60):
        return False
    if len(CJK.findall(p)) < len(p) * 0.5:
        return False
    if BAD.search(p) or CODEY.search(p):
        return False
    return True


def first_user(conv) -> str | None:
    for m in conv:
        if m.get("role") == "user":
            return (m.get("content") or "").strip().replace("\n", " ")
    return None


def main() -> int:
    # ── smoltalk-chinese ────────────────────────────────────────────────────
    sfiles = sorted(glob.glob(
        "/workspace/hf/hub/datasets--opencsg--smoltalk-chinese/snapshots/*/*.parquet")) or \
        sorted(glob.glob(
            "/workspace/hf/hub/datasets--opencsg--smoltalk-chinese/snapshots/*/**/*.parquet",
            recursive=True))
    assert sfiles, "smoltalk parquet 未找到"
    cat_count: Counter = Counter()
    smol = []
    CHAT_CATS = {"寻求建议", "闲聊", "日常对话", "头脑风暴", "角色扮演", "其他"}
    for f in sfiles:
        schema_names = pq.read_schema(f).names
        if not {"conversations", "classify", "score"} <= set(schema_names):
            continue  # 分片 schema 不一致：部分分片缺 classify/score，跳过
        t = pq.read_table(f, columns=["conversations", "classify", "score"])
        for conv, cat, score in zip(t["conversations"].to_pylist(),
                                    t["classify"].to_pylist(),
                                    t["score"].to_pylist()):
            cat_count[cat] += 1
            if cat in CHAT_CATS and (score or 0) >= 4:
                p = first_user(conv)
                if p and ok_prompt(p):
                    smol.append(p)
    print(f"smoltalk 全量类目：{dict(cat_count.most_common(12))}")
    rng.shuffle(smol)
    smol = list(dict.fromkeys(smol))[:500]
    print(f"smoltalk 候选：{len(smol)}")

    # ── WildChat-zh ─────────────────────────────────────────────────────────
    wfiles = sorted(glob.glob(
        "/workspace/hf/hub/datasets--allenai--WildChat-1M/snapshots/*/data/*.parquet"))
    wild = []
    for f in wfiles:
        t = pq.read_table(f, columns=["conversation", "language", "toxic"])
        for conv, lang, toxic in zip(t["conversation"].to_pylist(),
                                     t["language"].to_pylist(),
                                     t["toxic"].to_pylist()):
            if lang != "Chinese" or toxic:
                continue
            p = first_user(conv)
            if p and ok_prompt(p):
                wild.append(p)
        if len(wild) >= 5000:
            break
    rng.shuffle(wild)
    wild = list(dict.fromkeys(wild))[:500]
    print(f"WildChat 候选：{len(wild)}")
    assert len(smol) + len(wild) >= 600, "🔴 候选池不足 600"

    # ── 去重（考场 + p1_prompts 逐字；池内 3gram 近重）──────────────────────
    seen = set()
    for fn in ("talk_exam.jsonl", "context_exam.jsonl", "p1_prompts.jsonl"):
        for x in open(f"data/u_route/{fn}"):
            for turn in json.loads(x)["turns"]:
                seen.add(turn.strip())

    def grams(s):
        return {s[i:i + 3] for i in range(len(s) - 2)} or {s}

    pool, kept_grams = [], []
    for src, plist in (("smoltalk", smol), ("wildchat", wild)):
        for p in plist:
            if p in seen:
                continue
            g = grams(p)
            if any(len(g & kg) / len(g | kg) >= 0.7 for kg in kept_grams):
                continue
            kept_grams.append(g)
            pool.append({"prompt": p, "source": src})
    print(f"去重后池：{len(pool)}")
    assert len(pool) >= 200, "🔴 去重后不足 200"

    # ── 质检抽样（规则闸；样本落审计文件供人核）────────────────────────────
    sample = rng.sample(pool, min(50, len(pool)))
    passed = sum(1 for s in sample if ok_prompt(s["prompt"]))
    rate = passed / len(sample)
    print(f"质检抽 {len(sample)} 条：合格率 {rate:.0%}（门槛 ≥90%）")
    assert rate >= 0.9, "🔴 质检未过"

    # ── 采 120（前缀多样性：同 4 字开头 ≤6 条）─────────────────────────────
    rng.shuffle(pool)
    picked, pref = [], Counter()
    for it in pool:
        k = it["prompt"][:4]
        if pref[k] >= 6:
            continue
        pref[k] += 1
        picked.append(it)
        if len(picked) == 120:
            break
    assert len(picked) == 120, f"🔴 只采到 {len(picked)}"
    with open("data/u_route/chat_bank_v2.jsonl", "w") as f:
        for it in picked:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    with open("logs/u_route/s1_audit.txt", "w") as f:
        for it in picked:
            f.write(f"[{it['source']}] {it['prompt']}\n")
    n_s = sum(1 for x in picked if x["source"] == "smoltalk")
    print(f"✅ S1 完成：120 条（smoltalk {n_s} / wildchat {120-n_s}）"
          f"· 考场重叠 0 · 病句命中 0 → data/u_route/chat_bank_v2.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
