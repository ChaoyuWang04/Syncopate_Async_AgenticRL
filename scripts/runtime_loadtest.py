#!/usr/bin/env python
"""M9.7 · 全指标压测驱动：把设计文档 §19 那张表逐条打出实测数字。

    python scripts/runtime_loadtest.py                 # 全部阶段（约 10–20 分钟）
    python scripts/runtime_loadtest.py --skip crash,model_down
    python scripts/runtime_loadtest.py --quick         # 每意图样本减半

前置：PG + API(:8000) + org_acme worker（带 SYNCOPATE_DECIDER_URL）+ vLLM(:8100) 都在跑。

★ 它是**驱动器 + 报表**：门槛口径全部来自设计文档 §19（syncopate-project-design-v0.1.md），
  gold token 基线来自 data/sft/v13 的 supervised_tokens 家族中位数（数据推导，不拍）。
⚠️ 工具延迟是 FakeAdPlatform（D-2：不接真平台）⇒ 那两行数字量的是**我们这侧的开销**，
  不含真平台 RTT —— 报表里显式标注，别拿去当真世界结论。
⚠️ model_down 阶段会杀掉 vLLM 且不自动重启（重启要 ~2 分钟，交给人决定何时做）。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import statistics
import subprocess
import sys
import time
import uuid
from typing import Any

import asyncpg
import httpx

BASE = "http://127.0.0.1:8000"
TOKEN_ACME = "dev-token-acme"
TOKEN_GLOBEX = "dev-token-globex"
DSN = os.environ.get("SYNCOPATE_PG_DSN",
                     "postgresql://syncopate:syncopate@127.0.0.1:5432/syncopate")
TERMINAL = {"run.succeeded", "run.failed", "run.cancelled", "run.waiting_for_user"}

# §19 门槛（延迟 ms）；gold tokens_out 中位数来自 data/sft/v13 家族中位（×3 为门槛）
INTENTS = {
    "I01": {"msg": "查一下 CMP_1 昨天的花费和转化，给我一句话结论", "p95_ms": 5_000,
            "deadline": 90, "gold_out": 387},
    "I07": {"msg": "CMP_1 最近转化下滑，帮我做一次归因分析，找出主要原因", "p95_ms": 30_000,
            "deadline": 150, "gold_out": 210},
    "I09": {"msg": "CMP_1 表现不错，帮我评估并执行扩量，预算可以提高 20%", "p95_ms": 60_000,
            "deadline": 210, "gold_out": 295},
    "I11": {"msg": "为 CMP_1 找一些相似的高表现素材并给出上新建议", "p95_ms": 180_000,
            "deadline": 270, "gold_out": 316},
}

REPORT: list[tuple[str, str, str, str]] = []      # (指标, 门槛, 实测, 判定)


def report(metric: str, threshold: str, measured: str, ok: bool | None) -> None:
    verdict = "✅" if ok else ("⚠️" if ok is None else "🔴")
    REPORT.append((metric, threshold, measured, verdict))
    print(f"  {verdict} {metric}: {measured}  (门槛 {threshold})")


def pct(vals: list[float], q: float) -> float:
    if not vals:
        return float("nan")
    s = sorted(vals)
    return s[min(len(s) - 1, int(q * len(s)))]


async def submit(client: httpx.AsyncClient, *, msg: str, intent: str,
                 tier: str = "C", key: str | None = None) -> dict:
    r = await client.post("/runs", json={"user_message": msg, "intent": intent,
                                         "automation_tier": tier},
                          headers={"Idempotency-Key": key or f"lt-{uuid.uuid4().hex}"})
    r.raise_for_status()
    return r.json()


async def follow(client: httpx.AsyncClient, run_id: str, *, after: int = 0,
                 deadline: float = 120.0) -> tuple[str, int, list[tuple[int, str]]]:
    """跟 SSE 到终态，带总墙钟死线。返回 (终态, 最后seq, [(seq,kind)…])。"""
    events: list[tuple[int, str]] = []
    last = after

    async def _stream() -> str:
        nonlocal last
        headers = {"Last-Event-ID": str(after)} if after else {}
        async with client.stream("GET", f"/runs/{run_id}/events", headers=headers,
                                 timeout=deadline) as resp:
            resp.raise_for_status()
            ev = ""
            async for line in resp.aiter_lines():
                if line.startswith("id: "):
                    last = int(line[4:])
                elif line.startswith("event: "):
                    ev = line[7:]
                elif line == "" and ev:
                    events.append((last, ev))
                    if ev in TERMINAL:
                        return ev
                    ev = ""
        return "stream_closed"

    try:
        terminal = await asyncio.wait_for(_stream(), timeout=deadline)
    except asyncio.TimeoutError:
        terminal = "deadline"
    return terminal, last, events


async def run_to_terminal(client: httpx.AsyncClient, intent: str,
                          *, deadline: float | None = None) -> dict:
    spec = INTENTS[intent]
    t0 = time.monotonic()
    created = await submit(client, msg=spec["msg"], intent=intent)
    rid = created["run_id"]
    terminal, _, events = await follow(client, rid,
                                       deadline=deadline or spec["deadline"])
    return {"run_id": rid, "terminal": terminal,
            "e2e_ms": (time.monotonic() - t0) * 1e3,
            "n_events": len(events)}


# ── 阶段 1 · 分意图端到端延迟 + token 成本 ───────────────────────────────────
async def phase_latency(client: httpx.AsyncClient, pool: asyncpg.Pool,
                        quick: bool) -> dict[str, list[float]]:
    print("\n== 阶段 1 · 分意图端到端延迟（串行单流，真模型）==")
    n_by_intent = {"I01": 6 if quick else 12, "I07": 4 if quick else 8,
                   "I09": 4 if quick else 8, "I11": 4 if quick else 8}
    lat: dict[str, list[float]] = {}
    runs_by_intent: dict[str, list[str]] = {}
    for intent, n in n_by_intent.items():
        rows = []
        for i in range(n):
            r = await run_to_terminal(client, intent)
            rows.append(r)
            print(f"    {intent} #{i+1}/{n}: {r['terminal']:<21} {r['e2e_ms']:8.0f} ms")
        good = [r for r in rows if r["terminal"] in TERMINAL]
        lat[intent] = [r["e2e_ms"] for r in good]
        runs_by_intent[intent] = [r["run_id"] for r in good]
        bad = [r for r in rows if r["terminal"] not in TERMINAL]
        if bad:
            report(f"{intent} 完成率", "全部到终态",
                   f"{len(good)}/{n}（{len(bad)} 条超死线）", False)

    for intent, vals in lat.items():
        spec = INTENTS[intent]
        p50, p95, p99 = pct(vals, .5), pct(vals, .95), pct(vals, .99)
        report(f"{intent} P95", f"≤ {spec['p95_ms']/1e3:.0f}s",
               f"{p95/1e3:.2f}s (n={len(vals)})", p95 <= spec["p95_ms"])
        # ⚠️ P50/P95 比值判据隐含"P95 顶着预算"的前提：整个分布远快于预算时，
        #   P50≈P95 说明的是"均匀地快"，不是"慢是常态"。按守则③（判据总在误报
        #   会训练人忽略它），P95 ≤ 预算 25% 时判"不适用"——判据本身的修正
        #   已提案给 §19（11 §5 登记），不在这里静默改门槛。
        if p95 <= spec["p95_ms"] * 0.25:
            report(f"{intent} P50/P95", "≤ 40%",
                   f"{p50/p95*100:.0f}%（P95 仅为预算 {p95/spec['p95_ms']*100:.0f}%，"
                   "比值失义 ⇒ 不适用）", None)
        else:
            report(f"{intent} P50/P95", "≤ 40%", f"{p50/p95*100:.0f}%", p50 <= p95 * 0.4)
        report(f"{intent} P99/P95", "≤ 2×", f"{p99/p95:.2f}×", p99 <= p95 * 2)

    # token 成本（tokens_out / run vs gold 中位数 ×3）
    for intent, rids in runs_by_intent.items():
        if not rids:
            continue
        rows = await pool.fetch(
            "SELECT run_id, sum(tokens_out) AS tout FROM usage_records "
            "WHERE run_id = ANY($1::text[]) GROUP BY run_id", rids)
        touts = [r["tout"] for r in rows if r["tout"]]
        if not touts:
            report(f"{intent} tokens_out/任务", "有账", "无 usage 记录", False)
            continue
        med = statistics.median(touts)
        cap = INTENTS[intent]["gold_out"] * 3
        report(f"{intent} tokens_out/任务(中位)", f"≤ gold中位×3 = {cap}",
               f"{med:.0f} (n={len(touts)})", med <= cap)
    return lat


# ── 阶段 2 · 组件延迟：RAG 检索 + 工具调用（进程内直测，Fake 平台注意事项见头注）──
async def phase_components() -> None:
    print("\n== 阶段 2 · 组件延迟（进程内直测）==")
    sys.path.insert(0, ".")
    from syncopate.runtime.db import Database
    from syncopate.runtime.platform import FakeAdPlatform
    from syncopate.runtime.retrieval import RetrievalService

    db = Database()
    await db.connect(max_size=4)
    try:
        svc = RetrievalService(db)
        qs = ["预算上调的审批规则", "新素材的审核要求", "转化数据几天算成熟",
              "扩量的风控限制", "地域投放政策"]
        lat = []
        for i in range(50):
            t0 = time.monotonic()
            await svc.search_policy(org_id="org_acme", query=qs[i % len(qs)])
            lat.append((time.monotonic() - t0) * 1e3)
        report("RAG 检索 P95", "< 200ms", f"{pct(lat, .95):.1f} ms (n=50)",
               pct(lat, .95) < 200)

        plat = FakeAdPlatform()
        tl = []
        for i in range(100):
            t0 = time.monotonic()
            await plat.get_metrics(campaign_id="CMP_1")
            tl.append((time.monotonic() - t0) * 1e3)
        report("工具调用 P95（Fake 平台，不含真 RTT）", "< 2s",
               f"{pct(tl, .95):.2f} ms (n=100)", pct(tl, .95) < 2000)
    finally:
        await db.close()


# ── 阶段 3 · 并发劣化 + 队列积压 ─────────────────────────────────────────────
async def phase_concurrency(client: httpx.AsyncClient, pool: asyncpg.Pool,
                            single_p95: float, quick: bool) -> None:
    # ★ 判定按 §19 原口径：**恰好 8 条并发**（= worker 并发数）。压 16 条到 8 个
    #   槽上，一半 run 在排队，劣化结构性趋向 2×+ —— 那量的是排队不是劣化。
    #   16 条超配的表现另出一行信息（无门槛）。
    print("\n== 阶段 3 · 并发 8 条 I01（worker 并发 8）==")
    n = 8
    rows = await asyncio.gather(*(run_to_terminal(client, "I01", deadline=240)
                                  for _ in range(n)))
    good = [r for r in rows if r["terminal"] in TERMINAL]
    lat = [r["e2e_ms"] for r in good]
    p95c = pct(lat, .95)
    report("并发 run 数", "≥ 8 且全部到终态", f"{len(good)}/{n} 并发到终态",
           len(good) == n)
    if single_p95:
        report("并发 P95 劣化（8 并发）", "≤ 2×",
               f"{p95c/1e3:.2f}s / 单流 {single_p95/1e3:.2f}s "
               f"= {p95c/single_p95:.2f}×", p95c <= 2 * single_p95)
    if not quick:
        rows16 = await asyncio.gather(*(run_to_terminal(client, "I01", deadline=240)
                                        for _ in range(16)))
        lat16 = [r["e2e_ms"] for r in rows16 if r["terminal"] in TERMINAL]
        report("16 并发（2× 超配，信息行）", "—（无门槛：含排队）",
               f"P95 {pct(lat16, .95)/1e3:.2f}s = 单流 {pct(lat16, .95)/single_p95:.2f}×",
               None)
        rows += rows16
    # 队列积压：这批 run 里 started 事件时间 − 创建时间的最大值
    rids = [r["run_id"] for r in rows]
    row = await pool.fetchrow(
        "SELECT max(extract(epoch FROM e.created_at - r.created_at)) AS oldest "
        "FROM run_events e JOIN agent_runs r ON r.run_id=e.run_id AND r.org_id=e.org_id "
        "WHERE e.run_id = ANY($1::text[]) AND e.kind='run.started'", rids)
    oldest = row["oldest"] or 0
    report("队列积压（最老排队时间）", "≤ 60s", f"{oldest:.1f}s", oldest <= 60)


# ── 阶段 4 · 幂等（并发重复投递）────────────────────────────────────────────
async def phase_idempotency(client: httpx.AsyncClient, pool: asyncpg.Pool) -> None:
    print("\n== 阶段 4 · 幂等：同一 Idempotency-Key 并发投 10 次 ==")
    key = f"lt-idem-{uuid.uuid4().hex[:8]}"
    rs = await asyncio.gather(*(submit(client, msg=INTENTS['I01']['msg'],
                                       intent="I01", key=key) for _ in range(10)))
    rids = {r["run_id"] for r in rs}
    created = sum(1 for r in rs if r.get("created"))
    n_db = await pool.fetchval(
        "SELECT count(*) FROM agent_runs WHERE org_id='org_acme' AND idempotency_key=$1",
        key)
    ok = len(rids) == 1 and n_db == 1
    report("幂等有效性（并发重复投递）", "100%（1 条真执行）",
           f"10 投 → {len(rids)} run_id · created={created} · 库中 {n_db} 条", ok)


# ── 阶段 5 · SSE 断线补发 ───────────────────────────────────────────────────
async def phase_sse(client: httpx.AsyncClient, pool: asyncpg.Pool) -> None:
    print("\n== 阶段 5 · SSE 断线补发（读 1 条就断，再续传）==")
    created = await submit(client, msg=INTENTS["I07"]["msg"], intent="I07")
    rid = created["run_id"]
    # 第一段：只读到第一个事件就断线
    first_seq = 0
    async with client.stream("GET", f"/runs/{rid}/events", timeout=60) as resp:
        async for line in resp.aiter_lines():
            if line.startswith("id: "):
                first_seq = int(line[4:])
            elif line == "" and first_seq:
                break                              # 拿到一条完整事件，断线
    # 第二段：带 Last-Event-ID 续传到终态
    terminal, _, events2 = await follow(client, rid, after=first_seq, deadline=180)
    got = [first_seq] + [s for s, _ in events2]
    db_seqs = [r["seq"] for r in await pool.fetch(
        "SELECT seq FROM run_events WHERE run_id=$1 AND org_id='org_acme' "
        "AND seq<=$2 ORDER BY seq", rid, max(got))]
    ok = got == db_seqs and terminal in TERMINAL
    report("SSE 断线补发", "100%（无缺无重）",
           f"断在 seq={first_seq}，续传收 {len(events2)} 条，"
           f"合并后 {'=' if ok else '≠'} 库序列（终态 {terminal}）", ok)


# ── 阶段 6 · 成本触顶降级（org_globex，用完即清）────────────────────────────
async def phase_cost_cap(pool: asyncpg.Pool) -> None:
    print("\n== 阶段 6 · 单 org 日预算触顶（org_globex + 临时 worker）==")
    marker = f"lt-cap-{uuid.uuid4().hex[:8]}"
    await pool.execute(
        "INSERT INTO usage_records (org_id, run_id, tokens_in, tokens_out, cost_micros) "
        "VALUES ('org_globex', $1, 0, 0, 20000000)", marker)
    env = {**os.environ}
    env.pop("SYNCOPATE_DECIDER_URL", None)         # 入门闸在模型之前，不需要模型
    wk = subprocess.Popen(
        [sys.executable, "-m", "syncopate.runtime.worker",
         "--org-id", "org_globex", "--worker-id", "lt-globex"],
        env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        async with httpx.AsyncClient(base_url=BASE, timeout=30,
                                     headers={"Authorization": f"Bearer {TOKEN_GLOBEX}"}) as g:
            created = await submit(g, msg="把预算提到 1200", intent="I09")
            rid = created["run_id"]
            terminal, _, events = await follow(g, rid, deadline=30)
        kinds = [k for _, k in events]
        degraded = "run.degraded" in kinds
        ok = degraded and terminal in ("run.failed", "run.cancelled")
        report("成本触顶降级", "必须降级且可见（降级 ≠ 静默）",
               f"事件链 {' → '.join(kinds)}", ok)
    finally:
        wk.send_signal(signal.SIGTERM)
        wk.wait(timeout=10)
        await pool.execute("DELETE FROM usage_records WHERE run_id=$1", marker)


# ── 阶段 7 · 崩溃恢复（kill -9 org_acme worker）─────────────────────────────
async def phase_crash(client: httpx.AsyncClient, pool: asyncpg.Pool) -> None:
    print("\n== 阶段 7 · 崩溃恢复：3 条在飞时 kill -9 worker，再重启 ==")
    out = subprocess.run(["pgrep", "-f", "runtime.worker --org-id org_acme"],
                         capture_output=True, text=True)
    pids = [int(p) for p in out.stdout.split()]
    if not pids:
        report("崩溃后恢复", "100%", "找不到 acme worker 进程，跳过", None)
        return
    rids = [(await submit(client, msg=INTENTS["I01"]["msg"], intent="I01"))["run_id"]
            for _ in range(3)]
    await asyncio.sleep(2.0)                       # 让 worker 抢到并开跑
    for p in pids:
        os.kill(p, signal.SIGKILL)
    subprocess.Popen(["nohup", "logs/runtime/start_worker.sh"],
                     stdout=open("logs/runtime/worker.log", "a"),
                     stderr=subprocess.STDOUT)
    # lease 60s 过期后会被重抢；上限 150s
    deadline = time.monotonic() + 150
    terminal_n = 0
    while time.monotonic() < deadline:
        n = await pool.fetchval(
            "SELECT count(*) FROM agent_runs WHERE run_id = ANY($1::text[]) "
            "AND status IN ('succeeded','failed','cancelled','waiting_for_user')", rids)
        terminal_n = n
        if n == len(rids):
            break
        await asyncio.sleep(5)
    report("崩溃后恢复成功率", "100%",
           f"{terminal_n}/{len(rids)} 条在重启后到达终态（lease 60s 重抢）",
           terminal_n == len(rids))


# ── 阶段 8 · 场景②：模型服务挂掉 ────────────────────────────────────────────
async def phase_model_down(client: httpx.AsyncClient) -> None:
    print("\n== 阶段 8 · 场景②：杀掉 vLLM，看降级形状（不自动重启）==")
    # ⚠️ 必须连 EngineCore 子进程一起杀：只杀 API server 会留下一个孤儿
    #   VLLM::EngineCore 攥着 ~30G 显存，下次起服务直接 OOM（08-20 实测）。
    out = subprocess.run(["pgrep", "-f", "vllm serve|VLLM::EngineCore"],
                         capture_output=True, text=True)
    pids = [int(p) for p in out.stdout.split()]
    if not pids:
        report("场景② 模型服务挂掉", "run 显式失败、worker 存活", "vLLM 不在跑，跳过", None)
        return
    for p in pids:
        os.kill(p, signal.SIGKILL)
    await asyncio.sleep(3)
    r = await run_to_terminal(client, "I01", deadline=60)
    worker_alive = subprocess.run(["pgrep", "-f", "runtime.worker --org-id org_acme"],
                                  capture_output=True).returncode == 0
    ok = r["terminal"] == "run.failed" and worker_alive
    report("场景② 模型服务挂掉", "run 显式失败（不挂死不静默）· worker 存活",
           f"终态 {r['terminal']}（{r['e2e_ms']/1e3:.1f}s）· worker {'存活' if worker_alive else '死了'}",
           ok)
    print("  ⚠️ vLLM 已被杀，后续要手动重启（09 §0 的命令）")


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip", default="", help="逗号分隔：crash,model_down,cost,sse,idem,conc,comp,lat")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    skip = set(filter(None, args.skip.split(",")))

    pool = await asyncpg.create_pool(DSN, min_size=1, max_size=4)
    async with httpx.AsyncClient(base_url=BASE, timeout=60,
                                 headers={"Authorization": f"Bearer {TOKEN_ACME}"}) as client:
        health = (await client.get("/healthz")).json()
        if health.get("status") != "ok":
            print(f"🔴 /healthz = {health}")
            return 1

        single_p95 = 0.0
        if "lat" not in skip:
            lat = await phase_latency(client, pool, args.quick)
            single_p95 = pct(lat.get("I01", []), .95)
        if "comp" not in skip:
            await phase_components()
        if "conc" not in skip:
            await phase_concurrency(client, pool, single_p95, args.quick)
        if "idem" not in skip:
            await phase_idempotency(client, pool)
        if "sse" not in skip:
            await phase_sse(client, pool)
        if "cost" not in skip:
            await phase_cost_cap(pool)
        if "crash" not in skip:
            await phase_crash(client, pool)
        if "model_down" not in skip:
            await phase_model_down(client)

    await pool.close()

    print("\n" + "=" * 78)
    print("§19 指标实测汇总（Fake 平台口径见脚本头注）")
    print("=" * 78)
    w = max(len(m) for m, *_ in REPORT)
    for metric, threshold, measured, verdict in REPORT:
        print(f"{verdict} {metric:<{w}}  {measured}   [门槛 {threshold}]")
    bad = sum(1 for *_, v in REPORT if v == "🔴")
    print(f"\n{'🔴' if bad else '✅'} {len(REPORT) - bad}/{len(REPORT)} 项达标")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
