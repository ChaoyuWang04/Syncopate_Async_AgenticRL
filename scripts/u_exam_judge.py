#!/usr/bin/env python
"""U 路 P0 · 承接考场机判 + 说人话考场盲评包生成（`24 §4-P0` 的判据执行件）。

    .venv/bin/python scripts/u_exam_judge.py --context logs/u_route/run_cand_context.jsonl
    .venv/bin/python scripts/u_exam_judge.py --blind logs/u_route/run_cand_talk.jsonl \
        logs/u_route/run_base_talk.jsonl        # 多臂混洗出盲评包（不带臂标签）

机判规则（与题目携带的 judge 字段一一对应；宁可报 fail 不猜）：
  L1 definition_followup   第二轮：无数据类工具调用 + behavior=answer + 回复含定义性措辞与该词
  L2 same_object_tool      第二轮：调了允许工具之一，且 campaign 参数=先行词
  L3 budget_proposal       第二轮：提案/调用参数 ≈ 上文预算×factor（±tol）；halted 且提案对=过
  L4 clarify_then_proceed  第一轮 clarify；第二轮不再 clarify 同一件事且参数含用户给的值
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import defaultdict


def _tool_args(turn: dict) -> list[tuple[str, dict]]:
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
        params = p.get("params")
        if isinstance(params, str):
            try:
                params = json.loads(params)
            except json.JSONDecodeError:
                params = {}
        out.append((p.get("tool") or "proposal", params or {}))
    return out


def _flat_values(d, acc=None):
    acc = acc if acc is not None else []
    if isinstance(d, dict):
        for v in d.values():
            _flat_values(v, acc)
    elif isinstance(d, list):
        for v in d:
            _flat_values(v, acc)
    else:
        acc.append(str(d))
    return acc


def judge_item(item: dict, spec: dict) -> tuple[bool, str]:
    j = spec["judge"]; t2 = item["turns"][-1]; t1 = item["turns"][0]
    kind = j["type"]
    if kind == "definition_followup":
        data_tools = [t for t, _ in _tool_args(t2)
                      if t.startswith(("campaign.", "metrics.", "analysis.", "creative."))]
        if data_tools:
            return False, f"调了数据工具 {data_tools}（把概念追问当查询）"
        # ⚠️ 不卡 behavior 标签：空壳 tool_call+有效 reply 的形态（解析器家族坑余波）
        #   内容上承接正确应 PASS——P0-2 人核 20 条唯一分歧点，判据按内容修正（08-28）
        rep = t2.get("reply") or ""
        if j["term"].lower() not in rep.lower():
            return False, "回复未提到该词"
        if not re.search(j["def_regex"], rep):
            return False, "回复无定义性措辞"
        return True, "ok"
    if kind == "same_object_tool":
        for tool, args in _tool_args(t2):
            if tool in j["tools"] and j["campaign"] in _flat_values(args):
                return True, "ok"
        return False, f"第二轮未对 {j['campaign']} 调数据工具（tools={[t for t,_ in _tool_args(t2)]}）"
    if kind == "budget_proposal":
        want = j["old_budget"] * j["factor"]
        for tool, args in _tool_args(t2):
            if tool not in j["tools"] and tool != "proposal":
                continue
            for v in _flat_values(args):
                try:
                    x = float(str(v).replace(",", ""))
                except ValueError:
                    continue
                if abs(x - want) <= j["tol"] * want:
                    return True, "ok"
            # ⛔ 08-30：判据只认**绝对值**（old×factor），但 `campaign.scale_budget` 的
            #   语义本来就是**倍数**（工具说明：「扩量决策用这个」，factor=1.3 表示提到 1.3 倍）。
            #   模型用对了工具、提对了幅度，却因为"没写出 55000 这个数"被判 0 ——
            #   L3 整档 0/25 是这么来的。⇒ 同一件事的另一种单位也要认。
            #   ⚠️ 不是放宽：campaign 必须对得上，倍数也必须在同一个 tol 内。
            if tool == "campaign.scale_budget":
                f = args.get("factor")
                same_cmp = str(args.get("campaign_id") or "") == str(j["campaign"])
                try:
                    f = float(f)
                except (TypeError, ValueError):
                    f = None
                if same_cmp and f is not None and abs(f - j["factor"]) <= j["tol"] * j["factor"]:
                    return True, "ok（按倍数提案）"
        return False, f"未见 ≈{want:.0f} 的提案参数（也没有 factor≈{j['factor']} 的倍数提案）"
    if kind == "clarify_then_proceed":
        # ★ Chaoyu 08-30 裁定：人话追问也算 clarify（同 REJ，复用同一份规则）
        from syncopate.core.contract import prose_expresses
        c1 = ((t1.get("behavior") == "clarify") or bool(t1.get("clarification"))
              or prose_expresses("clarify", t1.get("reply") or ""))
        if not c1:
            return False, f"第一轮未 clarify（behavior={t1.get('behavior')}）"
        if t2.get("behavior") == "clarify":
            return False, "第二轮仍在 clarify"
        blob = json.dumps(_tool_args(t2), ensure_ascii=False) + (t2.get("reply") or "")
        if j["field_value"] not in blob:
            return False, "第二轮参数/回复未带用户给的值"
        return True, "ok"
    return False, f"未知判类 {kind}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--context")
    ap.add_argument("--blind", nargs="*", default=[])
    args = ap.parse_args()

    if args.context:
        spec = {json.loads(x)["id"]: json.loads(x)
                for x in open("data/u_route/context_exam.jsonl")}
        rows = [json.loads(x) for x in open(args.context)]
        by = defaultdict(lambda: [0, 0]); fails = []
        for r in rows:
            ok, why = judge_item(r, spec[r["id"]])
            lv = r["level"]
            by[lv][0] += ok; by[lv][1] += 1
            if not ok:
                fails.append((r["id"], why))
        print(f"== {args.context}")
        for lv in sorted(by):
            a, n = by[lv]
            print(f"  {lv}: {a}/{n} = {a/n:.0%}")
        for fid, why in fails[:20]:
            print(f"   ✗ {fid}: {why}")
        json.dump({"file": args.context,
                   "levels": {lv: {"pass": a, "n": n} for lv, (a, n) in by.items()},
                   "fails": fails},
                  open(args.context.replace("run_", "judged_"), "w"), ensure_ascii=False)

    if args.blind:
        pool = []
        for f in args.blind:
            arm = f.split("run_")[1].split("_")[0]
            for x in open(f):
                r = json.loads(x)
                rep = r["turns"][-1].get("reply") or ""
                pool.append({"key": hashlib.md5(f"{arm}|{r['id']}".encode()).hexdigest()[:10],
                             "cat": r.get("cat"), "prompt": " ⇢ ".join(
                                 t for t in ([*map(str, [])] or [])) or None,
                             "turns_user": None, "reply": rep, "_arm": arm, "_id": r["id"]})
        rng = random.Random(int(hashlib.md5(
            "".join(sorted(p["key"] for p in pool)).encode()).hexdigest()[:8], 16))
        rng.shuffle(pool)
        keymap = {p["key"]: {"arm": p.pop("_arm"), "id": p.pop("_id")} for p in pool}
        with open("logs/u_route/blind_pack.jsonl", "w") as f:
            for p in pool:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        json.dump(keymap, open("logs/u_route/blind_key.json", "w"))
        print(f"✅ 盲评包 {len(pool)} 条 → logs/u_route/blind_pack.jsonl（钥匙另存，评完再开）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
