#!/usr/bin/env python
"""v14.5 · exam_v2 生成器（`24 §4-P2` 考卷 v2；生成后**冻结**，改内容=新版本号）。

    .venv/bin/python scripts/u_make_exams_v2.py   # → data/u_route/context_exam_v2.jsonl

对 v1 的修复（审计八条 → 24 §4-P2「考卷 v2」）：
  L1 50 题 = in-vocab 24（v14.5 训练词表内）+ OOV held-out 26（锁死永不进训练词表，
     清单落 data/u_route/oov_holdout_terms.json，L-族门禁引用）；judge=零工具+病句负正则
  L2 25 题：judge 增 expect_value（沙盒真值，读数在场判据）
  L3/L4：沿用 v1 原题（设计健康），原样拷贝
  句式：L1 第二轮从 S2 模式库 exam 侧 substitution 模板 ∪ v1 保底句式（跨版本可比）
talk_exam 不改版（P0-1 盲评自一致 100%），复核闸在打分流程加。
"""

from __future__ import annotations

import json
import re
import random
from pathlib import Path

rng = random.Random(1450)
OUT = Path("data/u_route")
STATE = json.load(open("data/demo/platform_state.json"))
CAMPS = STATE["campaigns"]

# ── L1 词表：iv 从训练 GLOSSARY 抽；oov = held-out 锁死清单 ──────────────────
import sys
sys.path.insert(0, "scripts")
from u_build_v14 import GLOSSARY  # noqa: E402  61 词训练词表（v14.5 沿用并扩改写版本）

OOV_TERMS = ["eCPM", "填充率", "竞得率", "跑量", "落地页", "转化窗口", "自归因",
             "混合变现", "内购", "广告变现", "激励视频", "插屏广告", "开屏广告",
             "信息流广告", "试玩广告", "版位", "出价上限", "日耗", "爆量",
             "素材迭代", "用户画像", "种子用户", "假量", "素材审核", "分包", "换皮"]
assert len(OOV_TERMS) == 26 and not set(OOV_TERMS) & set(GLOSSARY), "OOV 与训练词表相交"

# 第二轮句式：S2 exam 侧 substitution + v1 保底（跨版本可比）
pat = json.load(open("data/u_route/ellipsis_patterns.json"))
s2_exam_sub = [t["template"] for t in pat["templates"]
               if t["split"] == "exam" and t["kind"] == "substitution"
               and t["template"].count("{X}") == 1
               # 语境依赖模板（{X}中呢/{X}年呢 只在代码/年份语境合法）套名词会成病句，排除
               and not re.search(r"\{X\}[中年里在]", t["template"])]
FORMS2 = list(dict.fromkeys(
    [t.replace("{X}", "{b}") for t in s2_exam_sub] +
    ["那{b}呢", "{b}呢？", "那{b}又是什么"]))

iv = rng.sample(sorted(GLOSSARY), 24)
first_terms = sorted(GLOSSARY)
L1 = []
for i, (b, tag) in enumerate([(t, "iv") for t in iv] + [(t, "oov") for t in OOV_TERMS]):
    a = rng.choice([t for t in first_terms if t != b])
    L1.append({
        "id": f"L1_{i:02d}", "level": "L1", "vocab": tag,
        "turns": [f"{a}是什么意思？", rng.choice(FORMS2).format(b=b)],
        "judge": {"type": "definition_followup_v2", "term": b},
        "note": "v2：零工具（任何 tool/proposal 即挂）+ 回复含该词 + 定义性且无病句",
    })
rng.shuffle(L1)
for i, item in enumerate(L1):
    item["id"] = f"L1_{i:02d}"

# ── L2：v1 结构 + expect_value ───────────────────────────────────────────────
METRICS = [("消耗", "spend_7d"), ("安装量", "installs_7d"), ("ROAS", "roas_d7"),
           ("CPI", "cpi"), ("点击率", "ctr"), ("频次", "frequency")]
cids = list(CAMPS.keys())
L2 = []
for i in range(25):
    cid = cids[i % len(cids)]
    m1, _ = METRICS[i % len(METRICS)]
    m2, k2 = METRICS[(i + 1) % len(METRICS)]
    ref = rng.choice(["它", "这个 campaign", "那它", "这条计划"])
    L2.append({
        "id": f"L2_{i:02d}", "level": "L2", "campaign": cid,
        "turns": [f"帮我查一下 {cid} 最近的{m1}", f"{ref}的{m2}呢？"],
        "judge": {"type": "same_object_tool_v2", "campaign": cid,
                  "tools": ["campaign.get_metrics", "metrics.get_freshness",
                            "creative.get_metrics_by_asset"],
                  "metric_name": m2, "expect_value": CAMPS[cid]["metrics"][k2]},
        "note": "v2：指代解析正确 + 回复里把查到的数字告诉用户（读数在场）",
    })

# ── L3/L4：v1 原样拷贝 ──────────────────────────────────────────────────────
v1 = [json.loads(x) for x in open("data/u_route/context_exam.jsonl")]
L34 = [r for r in v1 if r["level"] in ("L3", "L4")]

rows = L1 + L2 + L34
with open(OUT / "context_exam_v2.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
json.dump({"terms": OOV_TERMS, "locked": "2026-08-29",
           "rule": "这些词永不进任何训练词表/训练数据（L-族门禁；v16_build_sft 建库断言）"},
          open(OUT / "oov_holdout_terms.json", "w"), ensure_ascii=False, indent=1)
print(f"context_exam_v2.jsonl  {len(rows)} 题（L1 50=iv24+oov26 · L2 25 · L3/L4 {len(L34)}）")
print(f"oov_holdout_terms.json  26 词锁死")
print(f"L1 第二轮句式池：{FORMS2}")
