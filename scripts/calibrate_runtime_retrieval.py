"""标定 runtime 侧检索的阈值。**可复现**，不是把结论写进注释就完事。

    python scripts/calibrate_runtime_retrieval.py
    python scripts/calibrate_runtime_retrieval.py --corpus <other.json>

★ 为什么 runtime 要单独标定（不能直接用沙盒的 0.35）

打分函数两边**是同一个**（`corpus.overlap_score`），但**候选集不是**：

    沙盒    每条 case 自带 1–2 篇、手写、构造上不会互撞     ⇒ 0.35 够用
    runtime 整个语料库一起参与打分，篇数只会越来越多       ⇒ 撞得上

⇒ **相同的打分函数 + 不同的候选集 = 操作点必然不同。**
这不是调参，是结构性的：误召回概率随语料条数单调上升。

⚠️ 本文件里的评测集**是我自己写的**，条数很少（同 `calibrate_retrieval.py` 当年那句
"远不够定案"）。真实查询要等 M10 影子模式。**在那之前这个数是暂定的**，
但至少它是量出来的、可重跑的、会随语料变化而变化 —— 而不是一个注释里的传说。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from syncopate.domains.adcampaign.corpus import overlap_score
from syncopate.runtime.retrieval import _clause_text

DEFAULT_CORPUS = Path("data/external/policy_corpus.seed.json")

# 应命中：自然问法 → 应该被召回的那一**族**条款（版本对算同一族，见下）
SHOULD_HIT: dict[str, set[str]] = {
    "单日预算涨幅上限": {"POL_BUDGET_DAILY_V1", "POL_BUDGET_DAILY_V2"},
    "单日涨幅限制": {"POL_BUDGET_DAILY_V1", "POL_BUDGET_DAILY_V2"},
    "日预算能涨多少": {"POL_BUDGET_DAILY_V1", "POL_BUDGET_DAILY_V2"},
    "素材审核要多久": {"POL_CREATIVE_REVIEW"},
    "视频素材审核时长": {"POL_CREATIVE_REVIEW"},
    "东南亚博彩素材能投吗": {"POL_SEA_GAMBLING"},
}
# ★ 版本对必须算同一族：V1/V2 是同一条政策的两个版本，**两个都该被召回**，
#   "哪一版现行"由 valid_to / supersedes 精确算（设计文档 §4），不归检索管。

# 应留空：语料里确实没有的问题
SHOULD_EMPTY = [
    "量子计算加速广告投放",
    "如何申请企业信用贷款",
    "服务器机房温度标准",
    "今天天气怎么样",
    "公司年会在哪里办",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    args = ap.parse_args()

    payload = json.loads(args.corpus.read_text(encoding="utf-8"))
    docs = {c["clause_id"]: _clause_text(c) for c in payload["policy_clauses"]}
    print(f"语料 {len(docs)} 条 ← {args.corpus}\n")

    print("=== 应命中：族内最高分 vs 族外最高分 ===")
    hit_min = 1.0
    for query, family in SHOULD_HIT.items():
        inside = max(overlap_score(query, d) for k, d in docs.items() if k in family)
        outside = max((overlap_score(query, d) for k, d in docs.items()
                       if k not in family), default=0.0)
        hit_min = min(hit_min, inside)
        flag = "✅" if inside > outside else "🔴 族外反超"
        print(f"  {query:22} 族内={inside:.3f}  族外={outside:.3f}  {flag}")

    print("\n=== 应留空：最高分（越低越好）===")
    empty_max = 0.0
    for query in SHOULD_EMPTY:
        best_score, best_key = max((overlap_score(query, d), k) for k, d in docs.items())
        empty_max = max(empty_max, best_score)
        print(f"  {query:22} 最高={best_score:.3f}  ({best_key})")

    print(f"\n★ 分离带：应命中最低 {hit_min:.3f}  vs  应留空最高 {empty_max:.3f}")
    if empty_max >= hit_min:
        print("  🔴 **没有分离带** —— 任何阈值都会同时产生漏召回和误召回。")
        print("     这时候调阈值是在错误的维度上使劲，该换的是打分函数或语料切分。")
        return 1
    lo, hi = empty_max, hit_min
    mid = round((lo + hi) / 2, 2)
    print(f"  可用区间 ({lo:.3f}, {hi:.3f}]   ⇒ 取中点 **{mid}**")
    print(f"\n  沙盒阈值 0.35 —— {'也落在带内' if lo < 0.35 <= hi else '🔴 落在带外，runtime 不能直接用'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
