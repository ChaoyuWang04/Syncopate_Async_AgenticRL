"""本机供给核对（2026-09-04 run26 教训：DRY 只验结构，数量闸的"供给够不够"没人在本机算过）。
不调教师、不造行：只数当前 SFT 桶里每类底题的供给，逐条对照建库脚本里注册的数量下限；供给 < 下限 ⇒ 红。
    python scripts/check_supply_vs_floors.py
被谁调：runbook `gates` 之后（sft-data 之前）· check_pipeline_invariants data 组。
⚠️ 下限数字从 u_build_v14_5 / u_build_v15_multiturn 的常量读，不在这里抄第二份。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, "."); sys.path.insert(0, "scripts")
from syncopate.pipeline.split import DEFAULT_BATCH_DIR, DEFAULT_SPLIT_DIR, load_split_bundles  # noqa: E402


def main() -> int:
    import u_build_v15_multiturn as MT
    S = load_split_bundles(Path(DEFAULT_BATCH_DIR), Path(DEFAULT_SPLIT_DIR), "sft")
    ok = ("tool_call", "answer")
    q = [c for c, b in S.items() if b.gold and b.gold.actions and b.gold.actions[0]["tool"] == "campaign.get_metrics"
         and MT.campaign_of(b) and b.verifier.expected_behavior in ok]
    z = [c for c, b in S.items() if b.gold and not b.gold.actions and b.verifier.expected_behavior in ok]
    fresh_defer = [c for c, b in S.items() if c.split("_")[0] in ("FRESH", "RELN", "FRCP") and b.verifier.expected_behavior == "defer"]
    fresh_ok = [c for c, b in S.items() if c.split("_")[0] in ("FRESH", "RELN", "FRCP") and b.verifier.expected_behavior == "tool_call"]
    rej_un = [c for c, b in S.items() if c.startswith("REJ") and (b.gold.final_answer or {}).get("reject_reason") == "unauthorized"]
    bud = [c for c, b in S.items() if c.split("_")[0] in ("BUD", "BCUT") and MT.campaign_of(b)]
    hard = [c for c in S if c.split("_")[0] in ("BUD", "DIA", "FAIL", "RAG", "SCALE")]
    chat_bank = sum(1 for _ in open("data/u_route/chat_bank_v2.jsonl"))
    # 下限：与 u_build_v14_5 里的断言同源（读源码常量，防两份）
    src = Path("scripts/u_build_v14_5.py").read_text(encoding="utf-8")
    import re
    l2_floor = int(re.search(r"assert len\(l2\) >= \((\d+) if IS_V15", src).group(1))
    l1_floor = int(re.search(r"assert len\(l1\) >= (\d+)", src).group(1))
    cot_floor = int(re.search(r"_cot_floor = (\d+) if IS_V15", src).group(1))
    l1_reuse = True   # L1 造法：z 底题循环复用作历史（Chaoyu 09-04 待裁：现状允许）
    rows = [
        ("L2 多轮（一题一行）", len(q), l2_floor),
        ("L1 定义行底题（复用作历史）", len(z) * (18 if l1_reuse else 1), l1_floor),
        ("CoT 难例池（候选行数上界）", len(hard), cot_floor),
        ("闲聊素材库", chat_bank, 80),
        ("六族 DEFF-still 底题", len(fresh_defer), 1), ("六族 DEFF-recheck 底题", len(fresh_ok), 1),
        ("六族 REJF-still 越权底题", len(rej_un), 1), ("六族 CLAF 底题", len(bud), 1),
    ]
    bad = 0
    print(f"[供给] SFT 桶 {len(S)} 道 · 对照建库数量闸")
    for name, supply, floor in rows:
        flag = "✅" if supply >= floor else "🔴"
        bad += supply < floor
        print(f"  {flag} {name:<24} 供给 {supply:>4} · 下限 {floor}")
    print(f"[供给] 六族分支各造 min(20, 供给)：DEFF {min(20, len(fresh_defer))}+{min(20, len(fresh_ok))} · REJF {min(20, len(q))}+{min(20, len(rej_un))} · CLAF {min(20, len(bud))}×2")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
