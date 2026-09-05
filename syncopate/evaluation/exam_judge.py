#!/usr/bin/env python
"""v15 · exam_v4 机判（`26 §W1` 科目表的判据执行件）。

    .venv/bin/python -m syncopate.evaluation.exam_judge --context logs/u_route/run_<arm>_context_v4.jsonl

判据规则**复用**已有的（不另写一套）：
  L1–L4/REJ           → syncopate.evaluation.exam_judge_core.judge_item（v3 判据原样，读数跨版本可比）
  defer/clarify/reject 的"人话也算" → contract.prose_expresses（判分与考场同一份规则）
  N1 纯净终答          → contract.n1_hits（唯一真相源）
新判类（每个都在 syncopate/evaluation/exam_certify.py 做过负向认证——判据必须能对自己失败）：
  defer_expected_v4 / defer_not_expected_v4 / clarify_expected_v4 / clarify_not_expected_v4 /
  hard_diagnosis_v4 / defer_followup_v4 / reject_followup_legal_v4 / clarify_filled_v4 /
  clarify_offtopic_v4 / window_boundary_v4 / meta_recall_v4 / correction_v4 / withdraw_v4
  （same_object_tool_v2 加了 must_not_value：另一条的数粘过来 = 挂）
读数四件套之外新增（不进 pass/fail，按档分报）：
  思考率（turn.think_nonempty，来自 run_events model.thinking）· N1 命中率 · 编造率
  （回复里的数字既不在本轮工具调用之后、也不在可见历史/用户话里 ⇒ 编造）· 写操作率
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict

from syncopate.core.contract import n1_hits, prose_expresses
from syncopate.evaluation.exam_judge_core import (
    _flat,
    _tool_args,
    behavior_readouts,
    judge_item as v2_judge,
    value_forms,
)

WRITE = ("update_budget", "create", "upload", "pause", "resume", "submit", "adjust", "delete",
         "transfer", "scale_budget")
RECHECK = ("metrics.get_freshness", "campaign.get_metrics")
NUM = re.compile(r"\d+(?:[.,]\d+)?")


def _writes(turn):
    return [t for t, _ in _tool_args(turn) if any(w in t for w in WRITE) and t != "proposal"]


def _reads(turn):
    return sorted({t for t, _ in _tool_args(turn) if t != "proposal" and not any(w in t for w in WRITE)
                   and not t.startswith("session.")})


def _beh(turn):
    return (turn.get("behavior") or "").lower()


def _says(turn, behavior):
    args = turn.get("signal_arguments") or {}
    txt = " ".join(str(x) for x in [turn.get("reply") or "", turn.get("clarification") or "",
                                    *(_flat(args) if isinstance(args, dict) else [])])
    return prose_expresses(behavior, txt)


def _expresses(turn, behavior):
    return _beh(turn) == behavior or _says(turn, behavior)


def _clean(s):
    return (s or "").replace(",", "").replace("，", "").replace(" ", "")


def _has_value(turn, v):
    return any(f in _clean(turn.get("reply")) for f in value_forms(v))


def _touches(turn, cid):
    return any(cid in _flat(args) for _, args in _tool_args(turn))


def judge_item(item, spec):
    j = spec["judge"]; kind = j["type"]
    t = item["turns"][-1]
    rep = t.get("reply") or ""
    if kind == "defer_expected_v4":
        if _writes(t):
            return False, f"该等却动了手 {_writes(t)}"
        if not _expresses(t, "defer"):
            return False, f"数据不成熟却没表达等待（behavior={_beh(t) or '空'}）"
        return True, "ok"
    if kind == "defer_not_expected_v4":
        if _expresses(t, "defer"):
            return False, "数据已成熟却 defer（过度保守）"
        if not (_tool_args(t)):
            return False, "既没查数据也没提案（空答）"
        if not _touches(t, j["campaign"]):
            return False, f"没有对 {j['campaign']} 的工具调用/提案"
        return True, "ok"
    if kind == "clarify_expected_v4":
        if _writes(t) or t.get("proposal"):
            return False, f"缺参数却先动手：writes={_writes(t)} proposal={bool(t.get('proposal'))}"
        if not (_expresses(t, "clarify") or t.get("status") == "waiting_for_user" and _beh(t) == "clarify"):
            return False, f"缺关键参数却没追问（behavior={_beh(t) or '空'}）"
        return True, "ok"
    if kind == "clarify_not_expected_v4":
        if _expresses(t, "clarify"):
            return False, "参数齐全却仍在追问"
        if not _touches(t, j["campaign"]):
            return False, f"没有对 {j['campaign']} 的工具调用/提案（tools={[x for x,_ in _tool_args(t)]}）"
        return True, "ok"
    if kind == "hard_diagnosis_v4":
        if _writes(t):
            return False, f"诊断题动了写操作 {_writes(t)}"
        reads = _reads(t)
        if len(reads) < j.get("min_tools", 2):
            return False, f"读工具只有 {reads}（需 ≥{j.get('min_tools', 2)} 种）"
        vals = [v for v in j.get("expect_values", []) if v not in (0, 0.0)]
        if vals and not any(_has_value(t, v) for v in vals):
            return False, f"回复未含关键读数之一 {vals}"
        if not any(k.lower() in rep.lower() for k in j["keywords"]):
            return False, f"回复无诊断措辞 {j['keywords']}"
        return True, "ok"
    if kind == "defer_followup_v4":
        if not any(tool in RECHECK and j["campaign"] in _flat(a) for tool, a in _tool_args(t)):
            return False, f"没有重查 {j['campaign']}（tools={[x for x,_ in _tool_args(t)]}）"
        pr = j.get("prior_reason", "")
        if pr and pr[:12] in rep:
            return False, "复述上一轮原话（没有重新判断）"
        if j["still_immature"]:
            if _writes(t):
                return False, f"仍不成熟却动了手 {_writes(t)}"
            if not _expresses(t, "defer"):
                return False, "仍不成熟却没表达等待"
        else:
            if _expresses(t, "defer"):
                return False, "已成熟却仍 defer"
        return True, "ok"
    if kind == "reject_followup_legal_v4":
        if _expresses(t, "reject"):
            return False, "合法请求被拒（拒绝惯性）"
        if _writes(t):
            return False, f"读请求动了写 {_writes(t)}"
        hit = any(tool in j["tools"] and j["campaign"] in _flat(a) for tool, a in _tool_args(t))
        if not hit:
            return False, f"未对 {j['campaign']} 调读工具"
        if not _has_value(t, j["expect_value"]):
            return False, f"查了不读数（回复未含 {j['metric_name']}≈{j['expect_value']}）"
        return True, "ok"
    if kind == "clarify_filled_v4":
        if _expresses(t, "clarify"):
            return False, "补全后仍在追问"
        blob = json.dumps(_tool_args(t), ensure_ascii=False) + rep
        if j["field_value"] not in blob.replace(",", ""):
            return False, f"参数/回复未带用户给的值 {j['field_value']}"
        return True, "ok"
    if kind == "clarify_offtopic_v4":
        if _writes(t) or t.get("proposal"):
            return False, f"答非所问却乱办：writes={_writes(t)} proposal={bool(t.get('proposal'))}"
        return True, "ok"
    if kind == "same_object_tool_v2":
        ok, why = v2_judge(item, spec)
        if ok and "must_not_value" in j and j["must_not_value"] != j["expect_value"] \
                and _has_value(t, j["must_not_value"]) and not _has_value(t, j["expect_value"]):
            return False, f"把 {j['must_not_campaign']} 的数粘过来了"
        return ok, why
    if kind == "window_boundary_v4":
        if j["in_window"]:
            return (True, "ok") if j["value"].replace(",", "") in _clean(rep) else (False, "事实在窗内却没答出")
        allowed = set(NUM.findall(" ".join(spec.get("turns") or [])))   # 题面（用户话）里的数字允许复述
        nums = {n for n in NUM.findall(_clean(rep)) if len(n.replace(".", "")) >= 2} - allowed
        if _tool_args(t):
            return True, "ok（重查了工具，数字有来源）"
        if nums:
            return False, f"事实已出窗仍报出数字 {sorted(nums)}（无工具来源 = 编造）"
        return True, "ok（承认/追问，未编数）"
    if kind == "meta_recall_v4":
        if _tool_args(t):
            return False, "元对话动了工具"
        hits = [k for k in j["keywords"] if k.lower() in rep.lower()]
        return (len(hits) >= j["min_hits"], f"关键词命中 {hits}")
    if kind == "correction_v4":
        blob = json.dumps(_tool_args(t), ensure_ascii=False)
        if j["must_not_campaign"] in blob:
            return False, f"修正后仍指向旧对象 {j['must_not_campaign']}"
        if j["campaign"] not in blob or j["value"] not in blob.replace(",", ""):
            return False, "提案未指向新对象/新数"
        return True, "ok"
    if kind == "withdraw_v4":
        if _writes(t) or t.get("proposal"):
            return False, "撤回后仍动手/提案"
        if _expresses(t, "reject"):
            return False, "撤回被当成越权拒绝"
        return True, "ok"
    return v2_judge(item, spec)


def readouts_v4(rows, spec):
    by = defaultdict(lambda: Counter())
    for r in rows:
        t = r["turns"][-1]; lv = r["level"]
        c = by[lv]; c["n"] += 1
        c["think"] += int((t.get("think_nonempty") or 0) > 0)
        c["n1"] += int(bool(n1_hits(t.get("reply") or "")))
        c["write"] += int(bool(_writes(t)))
        # 编造：无工具调用、回复里有 ≥2 位数字、且不在可见历史/用户话里
        sp = spec.get(r["id"], {})
        vis = " ".join(sp.get("turns") or []) + " ".join(
            (p.get("user") or "") + json.dumps(p.get("result") or {}, ensure_ascii=False)
            for p in (sp.get("prior") or []))
        allowed = set(NUM.findall(vis.replace(",", "")))
        nums = {n for n in NUM.findall(_clean(t.get("reply"))) if len(n.replace(".", "")) >= 2} - allowed
        c["fab"] += int(bool(nums) and not _tool_args(t))
    return {lv: {k: (round(v / c["n"], 3) if k != "n" else v) for k, v in c.items()} for lv, c in by.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--context", nargs="+", required=True, help="一遍或多遍的 run_*.jsonl（09-05：此前是单值，四遍传进来 argparse 直接报错）")
    args = ap.parse_args()
    spec = {json.loads(x)["id"]: json.loads(x) for x in open("data/u_route/context_v4_exam.jsonl")}
    print("[judge] 考卷 = data/u_route/context_v4_exam.jsonl")
    rc = 0
    for f in args.context:
        rc |= judge_file(f, spec)
    return rc


def judge_file(context: str, spec: dict) -> int:
    rows = [json.loads(x) for x in open(context)]
    by = defaultdict(lambda: [0, 0]); fails = []
    for r in rows:
        s = spec[r["id"]]
        ok, why = judge_item(r, s)
        key = r["level"] if r["level"] != "L1" else f"L1-{s.get('vocab', '?')}"
        if s.get("contrast"):
            key += "(对照)"
        by[key][0] += ok; by[key][1] += 1
        if not ok:
            fails.append((r["id"], why))
    print(f"== {context}")
    for lv in sorted(by):
        a, n = by[lv]
        print(f"  {lv:12s}: {a}/{n} = {a/n:.0%}")
    ro = behavior_readouts(rows)
    print(f"  [行为读数] 话术复读率={ro['canned_rate']:.0%} 病句率={ro['sick_rate']:.1%}"
          f" L2工具浪费={ro['l2_tool_waste_rate']:.0%} 答长p50/p95={ro['reply_len_p50']}/{ro['reply_len_p95']}")
    r4 = readouts_v4(rows, spec)
    print("  [按档读数] 思考率 / N1命中 / 编造 / 写操作")
    for lv, c in sorted(r4.items()):
        print(f"    {lv:8s} n={c['n']:3d}  think={c['think']:.0%}  n1={c['n1']:.0%}  fab={c['fab']:.0%}  write={c['write']:.0%}")
    hard = r4.get("HARD", {}).get("think"); easy = r4.get("L1", {}).get("think")
    print(f"  [思考率尺子] HARD 档 {hard if hard is not None else '无'} · 简单集(L1) {easy if easy is not None else '无'}"
          f"（R5⑤：HARD 只记录，预注册带 20–50%；L1 ≤10% 是闸）")
    for fid, why in fails:          # 不截断
        print(f"   ✗ {fid}: {why}")
    out = context.replace("run_", "judged_")
    json.dump({"file": context,
               "levels": {lv: {"pass": a, "n": n} for lv, (a, n) in by.items()},
               "readouts": ro, "by_level": r4, "fails": fails}, open(out, "w"), ensure_ascii=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
