#!/usr/bin/env python
"""E32 S2.4 · goodput@SLO 阶梯（after 主口径，Chaoyu 08-28 认可）。

    .venv/bin/python scripts/b4_goodput.py --levels 8,16,24,32,48,64 \
        --out logs/b4/goodput_<tag>.json

定义：**混合意图（I01/I07/I09/I11 轮转）闭环并发 C 下，四个意图的 P95 全部守住
§19 各自门槛、且全部到终态的最大 C**。阶梯从低到高，首个破线级即停，goodput=前一级。

前置（口径的一部分，跑前自查）：
- PG + API(:8000) + vLLM(:8100，被测拓扑) 在跑；
- **worker 用 org_acme + --concurrency ≥ 最高级别**（默认 8 会封顶——§19 的"并发 8"
  原口径就是 worker 槽位数；goodput 量的是 serving 层，编排层必须先不设瓶颈，
  worker 槽位数与级别一起记进产物 = 同尺可复现）。
- 门槛/意图/判定全部 import runtime_loadtest（守则⑨：不抄数）。

每级样本量 = max(2×C, 24) 条（闭环补位），每意图 ≥ C/2 条才有资格算 P95。
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import sys
import time

import httpx

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
import runtime_loadtest as lt  # noqa: E402  （INTENTS/§19 门槛/终态判定的唯一来源）


async def level(client: httpx.AsyncClient, conc: int) -> dict:
    n_total = max(2 * conc, 24)
    intents = itertools.cycle(["I01", "I07", "I09", "I11"])
    sem = asyncio.Semaphore(conc)
    rows: list[tuple[str, dict]] = []

    async def one(intent: str) -> None:
        async with sem:
            r = await lt.run_to_terminal(client, intent, deadline=300)
            rows.append((intent, r))

    t0 = time.monotonic()
    await asyncio.gather(*[one(next(intents)) for _ in range(n_total)])
    wall = time.monotonic() - t0

    out: dict = {"concurrency": conc, "n": n_total, "wall_s": round(wall, 1),
                 "runs_per_min": round(n_total / wall * 60, 1), "intents": {}, "pass": True}
    for intent in ("I01", "I07", "I09", "I11"):
        grp = [r for it, r in rows if it == intent]
        good = [r for r in grp if r["terminal"] in lt.TERMINAL]
        lat = [r["e2e_ms"] for r in good]
        p95 = lt.pct(lat, .95)
        thr = lt.INTENTS[intent]["p95_ms"]
        ok = bool(len(good) == len(grp) and lat and p95 <= thr)
        out["intents"][intent] = {"n": len(grp), "terminal_ok": len(good),
                                  "p95_ms": round(p95, 0) if lat else None,
                                  "thr_ms": thr, "pass": ok}
        out["pass"] = out["pass"] and ok
    return out


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", default="8,16,24,32,48,64")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    levels = [int(x) for x in args.levels.split(",")]

    results = []
    goodput = 0
    async with httpx.AsyncClient(base_url=lt.BASE, timeout=330,
                                 headers={"Authorization": f"Bearer {lt.TOKEN_ACME}"}) as client:
        health = (await client.get("/healthz")).json()
        if health.get("status") != "ok":
            print(f"🔴 /healthz = {health}"); return 1
        for c in levels:
            print(f"== 级别 C={c} ==")
            r = await level(client, c)
            results.append(r)
            print(json.dumps(r, ensure_ascii=False))
            if not r["pass"]:
                print(f"🔴 C={c} 破线，停梯")
                break
            goodput = c

    out = {"tag": args.tag, "goodput_at_slo": goodput, "levels": results}
    with open(args.out, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n★ goodput@SLO = {goodput}   ✅ {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
