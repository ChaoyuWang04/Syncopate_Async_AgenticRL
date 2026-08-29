#!/usr/bin/env python
"""v14.5 · exam_v2 机判（`24 §4-P2` 考卷 v2 判据执行件）。

    .venv/bin/python scripts/u_exam_judge_v2.py --context logs/u_route/run_<arm>_ctxv2.jsonl

对 v1 judge 的修复：
  L1 v2  零工具判据（任何 tool/proposal 即挂——修 v1 黑名单漏 mmp./memory./policy. 前缀）
         + 回复含该词 + 定义性正词表（去掉单字「指」）+ 病句负正则（「指指」即挂）
  L2 v2  原判据 + 读数在场（reply 含 expect_value 的数字形态之一）
  行为读数四件套（不进 pass/fail，单独报）：话术复读率 ≤15% · 病句率 ≤2% ·
         L2 工具浪费率 ≤20% · 回复长度 p50/p95
L1 分 iv/oov 两线报（门槛 L1-iv ≥90% 且 L1-oov ≥70%）。
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict

# 单字「指」可安全保留：病句负正则（SICK）在其之前单独拦「指指」——
# 08-29 人核：删掉它误杀 6 条正常定义句（「内购指用户…」）
DEF_WORDS = r"是指|指的是|指|表示|衡量|用于|含义|意思是|定义|计算|叫做|就是|即|是[一让用将按把]"
SICK = re.compile(r"指指|的的|是是|了了")


def _tool_args(turn):
    out = []
    for t in turn.get("tools") or []:
        a = t.get("arguments")
        if isinstance(a, str):
            try:
                a = json.loads(a)
            except json.JSONDecodeError:
                a = {}
        out.append((t["tool"], a or {}))
    p = turn.get("proposal")
    if p:
        out.append((p.get("tool") or "proposal", p.get("params") or {}))
    return out


def _flat(d, acc=None):
    acc = acc if acc is not None else []
    if isinstance(d, dict):
        for v in d.values():
            _flat(v, acc)
    elif isinstance(d, list):
        for v in d:
            _flat(v, acc)
    else:
        acc.append(str(d))
    return acc


def value_forms(v) -> list[str]:
    """一个沙盒真值在回复里可能出现的书写形态。"""
    forms = {str(v)}
    if isinstance(v, float):
        forms |= {f"{v:g}", f"{v:.2f}".rstrip("0").rstrip("."), f"{v:.1f}"}
        if v < 1:                       # 比率类可能写成百分数
            forms |= {f"{v*100:g}", f"{v*100:.1f}".rstrip("0").rstrip("."),
                      f"{v*100:.0f}"}
    if isinstance(v, (int, float)) and float(v) == int(v):
        iv = int(v)
        forms |= {str(iv), f"{iv:,}"}
        if iv >= 10000 and iv % 10000 == 0:
            forms.add(f"{iv//10000}万")   # 中文万位
        elif iv >= 10000:
            forms.add(f"{iv/10000:g}万")
    return [f for f in forms if f]


def judge_item(item, spec):
    j = spec["judge"]
    t2 = item["turns"][-1]
    t1 = item["turns"][0]
    kind = j["type"]
    rep = t2.get("reply") or ""
    if kind == "definition_followup_v2":
        calls = _tool_args(t2)
        if calls:
            return False, f"概念追问动了工具/提案 {[t for t, _ in calls]}（v2=零动作判据）"
        if j["term"].lower() not in rep.lower():
            return False, "回复未提到该词"
        if SICK.search(rep):
            return False, f"病句（{SICK.search(rep).group()}）"
        if not re.search(DEF_WORDS, rep):
            return False, "回复无定义性措辞"
        return True, "ok"
    if kind == "same_object_tool_v2":
        hit = any(tool in j["tools"] and j["campaign"] in _flat(args)
                  for tool, args in _tool_args(t2))
        if not hit:
            return False, f"第二轮未对 {j['campaign']} 调数据工具（tools={[t for t, _ in _tool_args(t2)]}）"
        clean = rep.replace(",", "").replace("，", "").replace(" ", "")
        if not any(f in clean for f in value_forms(j["expect_value"])):
            return False, f"查了不读数（回复未含 {j['metric_name']}≈{j['expect_value']}）"
        return True, "ok"
    # L3/L4 沿用 v1 判据
    from u_exam_judge import judge_item as v1_judge
    return v1_judge(item, spec)


def behavior_readouts(rows):
    reps = [t.get("reply") or "" for r in rows for t in r["turns"] if t.get("reply")]
    tails = Counter(rep[-10:] for rep in reps if len(rep) >= 10)
    top_tail, top_n = (tails.most_common(1) or [("", 0)])[0]
    sick = sum(1 for rep in reps if SICK.search(rep))
    l2rows = [r for r in rows if r["level"] == "L2"]
    waste = sum(1 for r in l2rows if len(r["turns"][-1].get("tools") or []) > 2)
    lens = sorted(len(r) for r in reps) or [0]
    return {
        "canned_rate": round(top_n / max(1, len(reps)), 3),
        "canned_tail": top_tail,
        "sick_rate": round(sick / max(1, len(reps)), 3),
        "l2_tool_waste_rate": round(waste / max(1, len(l2rows)), 3),
        "reply_len_p50": lens[len(lens) // 2],
        "reply_len_p95": lens[int(len(lens) * 0.95) - 1],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--context", required=True)
    args = ap.parse_args()
    import sys
    sys.path.insert(0, "scripts")
    spec = {json.loads(x)["id"]: json.loads(x)
            for x in open("data/u_route/context_exam_v2.jsonl")}
    rows = [json.loads(x) for x in open(args.context)]
    by = defaultdict(lambda: [0, 0])
    fails = []
    for r in rows:
        s = spec[r["id"]]
        ok, why = judge_item(r, s)
        key = r["level"] if r["level"] != "L1" else f"L1-{s.get('vocab', '?')}"
        by[key][0] += ok
        by[key][1] += 1
        if not ok:
            fails.append((r["id"], why))
    print(f"== {args.context}")
    for lv in sorted(by):
        a, n = by[lv]
        print(f"  {lv}: {a}/{n} = {a/n:.0%}")
    ro = behavior_readouts(rows)
    print(f"  [行为读数] 话术复读率={ro['canned_rate']:.0%}（尾={ro['canned_tail']!r}，门槛≤15%）"
          f" 病句率={ro['sick_rate']:.1%}（≤2%） L2工具浪费={ro['l2_tool_waste_rate']:.0%}（≤20%）"
          f" 答长p50/p95={ro['reply_len_p50']}/{ro['reply_len_p95']}")
    for fid, why in fails[:25]:
        print(f"   ✗ {fid}: {why}")
    out = args.context.replace("run_", "judged_")
    json.dump({"file": args.context,
               "levels": {lv: {"pass": a, "n": n} for lv, (a, n) in by.items()},
               "readouts": ro, "fails": fails}, open(out, "w"), ensure_ascii=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
