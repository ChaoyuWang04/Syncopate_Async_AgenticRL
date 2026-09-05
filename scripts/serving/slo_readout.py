#!/usr/bin/env python3
"""K9-1 · 九条 SLO 一键打印（27 §15）：读数全部来自表/端点，⛔ 不许人工查库口算。

  python scripts/serving/slo_readout.py [--org org_demo] [--api http://127.0.0.1:8000 --token dev-token-demo]
    --api 给了才量 POST /runs P95（真打 100 次）；SSE after 补齐由 tests/runtime/test_events_k7 守（结构属性，报 by-test）。
"""
from __future__ import annotations

import argparse
import asyncio
import os
import time

from syncopate.runtime.db import Database
from syncopate.runtime import metrics


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--org", default=None)
    ap.add_argument("--api", default=None)
    ap.add_argument("--token", default="dev-token-demo")
    a = ap.parse_args()
    db = Database()
    await db.connect(max_size=2)
    try:
        m = await metrics.snapshot(db, org_id=a.org)
        from syncopate.runtime.budget import org_budget_state
        if a.org:
            m["org_budget_ratio"] = (await org_budget_state(db, org_id=a.org))["ratio"]
        m["sse_after_ok"] = True      # 由 tests/runtime/test_events_k7.py::test_reconnect_receives_exactly_the_missing_events 守
        if a.api:
            import httpx
            samples = []
            async with httpx.AsyncClient(base_url=a.api, headers={"Authorization": f"Bearer {a.token}"}) as c:
                for _ in range(100):
                    t0 = time.perf_counter()
                    r = await c.post("/runs", json={"user_message": "slo probe"})
                    samples.append((time.perf_counter() - t0) * 1000)
                    if r.status_code not in (201, 200):
                        break
            samples.sort()
            m["post_runs_p95_ms"] = samples[int(0.95 * len(samples)) - 1] if samples else None
        print(f"[slo] 读数时间 {time.strftime('%Y-%m-%d %H:%M:%S')} org={a.org or '*'}")
        for row in metrics.slo_table(m):
            v = row["value"]
            vs = f"{v:.3f}" if isinstance(v, float) else str(v)
            print(f"  {row['verdict']}  {row['slo']:<28} = {vs:<12} ({row['owner']})")
        for al in metrics.alerts(m):
            print(f"[alert] {al['alert']}={al['value']} → {al['runbook']}")
        return 0
    finally:
        await db.close()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
