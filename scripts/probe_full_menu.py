#!/usr/bin/env python
"""全量 30 工具菜单探针：模型自己选工具，选得对不对、格式还守不守。

    python scripts/probe_full_menu.py

★ 背景（22 §I 后续，Chaoyu 2026-08-20）：意图选择改成**模型自选**，
  prompt 里直接给全量 34 个工具（修剪后工具块 5468 tok；上限 9216+8192=18432，09-02）。
  代价是训练分布外——训练 case 的菜单是 12–16 个工具。**这个探针就是量那个代价。**

★ 三个判据（都不需要人判）：
  ① 格式保持     每条任务正常收尾（不 parse_error、不空转）
  ② 不编工具     `unknown_tool` 事件为 0（全量菜单下工具名更容易串）
  ③ 该调的调了   每类任务的"必调工具"至少命中一个（选工具的准确性）
⚠️ 判据③刻意宽（"至少一个"）：解法多样性是我们要保的东西，
  钉死工具序列等于把判据写成"必须照抄 gold"，那会为错误的理由失败（守则③）。
"""

from __future__ import annotations

import asyncio
import sys
import time
import uuid

import httpx

BASE = "http://127.0.0.1:8000"
HDRS = {"Authorization": "Bearer dev-token-demo", "Content-Type": "application/json"}

CASES = [
    ("查一下 CMP_1 昨天的花费和转化", "I01",
     {"campaign.get_metrics"}),
    ("CMP_1 最近转化下滑，帮我做一次归因分析", "I07",
     {"campaign.detect_anomalies", "analysis.feature_lift", "mmp.get_attribution",
      "campaign.get_metrics", "playbook.get_optimization"}),
    ("CMP_1 表现不错，帮我评估扩量，预算提高 20%", "I09",
     {"policy.get_budget_rule", "risk.check_account", "benchmark.get_safety_line",
      "metrics.get_freshness", "campaign.get_metrics"}),
    ("为 CMP_1 找相似的高表现素材并给上新建议", "I11",
     {"creative.search_similar", "creative.get_asset_tags",
      "creative.get_metrics_by_asset", "campaign.get_metrics"}),
    ("CMP_1 是三天前刚上的，现在能扩量吗", "I09",
     {"metrics.get_freshness", "campaign.get_metrics"}),
]


async def run_case(c: httpx.AsyncClient, cid: str, msg: str, intent: str) -> dict:
    run = (await c.post(f"/conversations/{cid}/messages",
                        json={"user_message": msg, "intent": intent,
                              "automation_tier": "C"},
                        headers={**HDRS,
                                 "Idempotency-Key": f"fm-{uuid.uuid4().hex}"})).json()
    rid = run["run_id"]
    t0 = time.monotonic()
    while time.monotonic() - t0 < 180:
        msgs = (await c.get(f"/conversations/{cid}/messages")).json()
        row = next((m for m in msgs if m["run_id"] == rid), None)
        if row and row["status"] in ("succeeded", "failed", "cancelled",
                                     "waiting_for_user"):
            break
        await asyncio.sleep(1.0)
    else:
        return {"run_id": rid, "status": "deadline", "tools": [], "unknown": 0}
    ev = (await c.get(f"/runs/{rid}/events")).text
    tools, unknown = [], 0
    for block in ev.split("\n\n"):
        if "event: tool.result" not in block:
            continue
        data = block.split("data: ", 1)[-1]
        import json as _j
        try:
            p = _j.loads(data)
        except ValueError:
            continue
        tools.append(p.get("tool"))
        if p.get("error") == "unknown_tool":
            unknown += 1
    return {"run_id": rid, "status": row["status"], "tools": tools, "unknown": unknown}


async def main() -> int:
    fmt_ok = hit_ok = 0
    unknown_total = 0
    async with httpx.AsyncClient(base_url=BASE, headers=HDRS, timeout=60) as c:
        cid = (await c.post("/conversations",
                            json={"title": "probe-full-menu"})).json()["conversation_id"]
        print(f"会话 {cid}（全量 30 工具模式）\n" + "=" * 74)
        for msg, intent, expected in CASES:
            r = await run_case(c, cid, msg, intent)
            terminal_ok = r["status"] in ("succeeded", "waiting_for_user")
            hit = bool(set(r["tools"]) & expected)
            fmt_ok += terminal_ok
            hit_ok += hit
            unknown_total += r["unknown"]
            print(f"\n  {msg[:34]}")
            print(f"    终态 {r['status']:<18} {'✅' if terminal_ok else '🔴'}"
                  f"   编工具 {r['unknown']} {'✅' if r['unknown'] == 0 else '🔴'}")
            print(f"    调了：{' → '.join(t for t in r['tools'] if t) or '(无)'}")
            print(f"    该调的至少命中一个 {'✅' if hit else '🔴'}"
                  f"（期望其一：{'/'.join(sorted(expected))[:60]}）")

    n = len(CASES)
    print("\n" + "=" * 74)
    print(f"① 格式保持   {fmt_ok}/{n}")
    print(f"② 不编工具   unknown_tool 共 {unknown_total} 次（要 0）")
    print(f"③ 选工具对   {hit_ok}/{n}")
    ok = fmt_ok == n and unknown_total == 0 and hit_ok == n
    print(f"\n{'✅ 全量菜单可用' if ok else '🔴 不合格 ⇒ SYNCOPATE_TOOL_MENU=intent 打回按意图裁剪'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
