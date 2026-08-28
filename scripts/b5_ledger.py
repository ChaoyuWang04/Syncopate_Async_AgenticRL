#!/usr/bin/env python
"""B-5/E33 S0 · 分账表拼装：goodput 逐 run 明细 × PG 事件时间线 × worker 分账行 → 每站份额。

    .venv/bin/python scripts/b5_ledger.py --goodput logs/b5/gp_c8.json \
        --worker-log logs/b5/stack/worker.log --out logs/b5/ledger_c8.json

站的定义（e2e = t_queue + t_exec + t_sse_tail 按构造成立）：
    t_queue     run 创建 → run.started 事件（排队等领单）
    t_exec      run.started → 终态事件（worker 手里的时间），再拆：
                llm / tool / db_wait / db_tx / rest（rest 显式打出，覆盖率的分母）
    t_sse_tail  终态事件落库 → 客户端收到（SSE 推送延迟的直接读数）
门槛（E33 S0）：六项加总 ÷ e2e ≥ 0.90；同档重跑份额漂移 < 5pt。
时钟：客户端与 PG 同机；仍实测 clock_timestamp 与 time.time 偏移并校正。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import time

import asyncpg

DSN = os.environ.get("SYNCOPATE_PG_DSN",
                     "postgresql://syncopate:syncopate@127.0.0.1:5432/syncopate")


def load_stage_lines(path: str) -> dict[str, dict]:
    out = {}
    pat = re.compile(r"\[stage-timing\] (\{.*\})")
    for line in open(path, errors="replace"):
        m = pat.search(line)
        if m:
            try:
                d = json.loads(m.group(1))
                out[d["run_id"]] = d
            except json.JSONDecodeError:
                pass
    return out


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--goodput", required=True)
    ap.add_argument("--worker-log", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    gp = json.load(open(args.goodput))
    stages = load_stage_lines(args.worker_log)
    conn = await asyncpg.connect(DSN)
    skew = await conn.fetchval("SELECT extract(epoch FROM clock_timestamp())") - time.time()

    ledgers = []
    for lv in gp["levels"]:
        runs = lv.get("runs") or []
        rids = [r["run_id"] for r in runs]
        rows = await conn.fetch(
            """SELECT r.run_id,
                      extract(epoch FROM r.created_at)                       AS created,
                      extract(epoch FROM min(e.created_at) FILTER (WHERE e.kind='run.started'))  AS started,
                      extract(epoch FROM max(e.created_at) FILTER (WHERE e.kind IN
                        ('run.succeeded','run.failed','run.cancelled','run.waiting_for_user'))) AS finished
               FROM agent_runs r JOIN run_events e
                 ON e.run_id=r.run_id AND e.org_id=r.org_id
               WHERE r.run_id = ANY($1::text[]) GROUP BY r.run_id, r.created_at""", rids)
    # ⤷ 注意：终态事件名以 kind 前缀 run. 记录（与 TERMINAL 集一致）
        tl = {r["run_id"]: r for r in rows}
        agg: dict[str, float] = {}
        n_cov = 0
        e2e_sum = 0.0
        miss_stage = 0
        for r in runs:
            t = tl.get(r["run_id"])
            s = stages.get(r["run_id"])
            if not t or not t["started"] or not t["finished"] or not r.get("t_done"):
                continue
            if s is None:
                miss_stage += 1
                continue
            e2e = (r["t_done"] + skew) - t["created"]
            parts = {
                "t_queue": t["started"] - t["created"],
                "llm": s.get("llm", 0.0), "tool": s.get("tool", 0.0),
                "db_wait": s.get("db_wait", 0.0), "db_tx": s.get("db_tx", 0.0),
                "t_sse_tail": (r["t_done"] + skew) - t["finished"],
            }
            t_exec = t["finished"] - t["started"]
            parts["rest"] = max(0.0, t_exec - (parts["llm"] + parts["tool"]
                                               + parts["db_wait"] + parts["db_tx"]))
            for k, v in parts.items():
                agg[k] = agg.get(k, 0.0) + max(0.0, v)
            e2e_sum += e2e
            n_cov += 1
        if not n_cov:
            ledgers.append({"concurrency": lv["concurrency"], "error": "no joined runs",
                            "missing_stage_lines": miss_stage})
            continue
        shares = {k: round(v / e2e_sum, 4) for k, v in agg.items()}
        known = sum(v for k, v in shares.items() if k != "rest")
        ledgers.append({
            "concurrency": lv["concurrency"], "n": n_cov,
            "missing_stage_lines": miss_stage,
            "e2e_mean_s": round(e2e_sum / n_cov, 3),
            "sec_per_run": {k: round(v / n_cov, 3) for k, v in agg.items()},
            "share": shares,
            "coverage_incl_rest": round(known + shares.get("rest", 0), 4),
            "coverage_known": round(known, 4),
        })
    await conn.close()
    out = {"goodput_file": args.goodput, "clock_skew_s": round(skew, 4), "ledgers": ledgers}
    json.dump(out, open(args.out, "w"), ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
