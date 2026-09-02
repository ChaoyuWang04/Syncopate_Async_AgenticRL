"""K8 · 两个不跑业务的旁观者（课件 CH8 §6）：**Sweeper 修 run 的状态，Reconciliation 修副作用的真相。**

都不跑 loop、不调模型、不执行业务工具；四原则（§8.3）：条件必须能写成 SQL · 修复动作必写事件 ·
修复动作可审计 · 不绕状态机（一律 transition_run）。

    python -m syncopate.runtime.sweeper            # 常驻：每 SWEEP_INTERVAL_S 扫一轮；对账每 RECONCILE_EVERY 轮

扫描类（§8.2 + 我们定死的两条）：
  A  running ∧ lease 过期            → sweep_expired_run 三分支（取消→次数→重投），重投写 outbox
  B  waiting_for_user 超龄           → run.waiting_too_long 事件（提醒；不改状态）
  C  queued 超龄 ∧ outbox 无 pending  → run.stuck_queued 告警事件；⛔ 不自动重投（掩盖投递层病根），
                                        人工经 requeue_outbox 重投（H37 我们定死）
  D  写类 tool_calls running 超龄     → response_lost（按治理表 expected_max_ms 逐工具判，禁全局常量）
  E  outbox dispatched > 30 天        → 归档删除（H108）
对账（§11.7）：response_lost 的写调用 → 按幂等键查 platform_ledger → 命中回填 succeeded + tool.repaired
  / 未命中 failed / 账本不可用 → tool.manual_review（事件，不是状态）；三写入同一事务。
"""
from __future__ import annotations

import asyncio
import os
import time
from typing import Any

from syncopate.runtime.db import (Database, MAX_RUN_ATTEMPTS, append_event, enqueue_outbox,
                                  sweep_expired_run)

SWEEP_INTERVAL_S = float(os.environ.get("SYNCOPATE_SWEEP_INTERVAL_S", "10"))
RECONCILE_EVERY = int(os.environ.get("SYNCOPATE_RECONCILE_EVERY", "30"))     # 10s×30 = 5 分钟兜底
WAITING_TOO_LONG_S = float(os.environ.get("SYNCOPATE_WAITING_TOO_LONG_S", str(6 * 3600)))
STUCK_QUEUED_S = float(os.environ.get("SYNCOPATE_STUCK_QUEUED_S", "300"))
OUTBOX_ARCHIVE_DAYS = int(os.environ.get("SYNCOPATE_OUTBOX_ARCHIVE_DAYS", "30"))


async def sweep_once(db: Database, *, actor_id: str = "sweeper", org_id: str | None = None,
                     waiting_too_long_s: float = WAITING_TOO_LONG_S,
                     stuck_queued_s: float = STUCK_QUEUED_S,
                     archive_days: int = OUTBOX_ARCHIVE_DAYS) -> dict[str, int]:
    """一轮扫描。`org_id` = 只扫一个租户（测试隔离 / 按租户切 sweeper，同 worker/dispatcher 的 --org-id）。"""
    counts = {"requeued": 0, "cancelled": 0, "failed": 0, "waiting_too_long": 0,
              "stuck_queued": 0, "response_lost": 0, "outbox_archived": 0}
    # ---- A · running ∧ lease 过期 ----
    async with db.tx() as conn:
        rows = await conn.fetch(
            """
            SELECT org_id, run_id, attempts, cancel_requested_at, lease_owner
              FROM agent_runs
             WHERE status='running' AND lease_expires_at IS NOT NULL AND lease_expires_at < now()
               AND ($1::text IS NULL OR org_id = $1)
             ORDER BY lease_expires_at LIMIT 100
             FOR UPDATE SKIP LOCKED
            """, org_id)
        for r in rows:
            outcome = await sweep_expired_run(conn, org_id=r["org_id"], run_id=r["run_id"],
                                              attempts=r["attempts"], cancel_requested_at=r["cancel_requested_at"],
                                              lease_owner_was=r["lease_owner"], actor_id=actor_id, enqueue=True)
            counts[outcome] += 1
            print(f"[sweeper] run={r['run_id']} lease 过期 ⇒ {outcome}（attempts={r['attempts']}）", flush=True)
    # ---- B · waiting_for_user 超龄（提醒一次）----
    async with db.tx() as conn:
        rows = await conn.fetch(
            """
            SELECT r.org_id, r.run_id, extract(epoch FROM (now()-r.updated_at)) AS age
              FROM agent_runs r
             WHERE r.status='waiting_for_user' AND r.updated_at < now() - make_interval(secs => $1)
               AND ($2::text IS NULL OR r.org_id = $2)
               AND NOT EXISTS (SELECT 1 FROM run_events e WHERE e.org_id=r.org_id AND e.run_id=r.run_id
                                  AND e.kind='run.waiting_too_long')
             LIMIT 100
            """, waiting_too_long_s, org_id)
        for r in rows:
            await append_event(conn, org_id=r["org_id"], run_id=r["run_id"], kind="run.waiting_too_long",
                               payload={"age_seconds": int(r["age"])})
            counts["waiting_too_long"] += 1
    # ---- C · queued 超龄 ∧ outbox 无 pending（僵尸 queued，H37）----
    async with db.tx() as conn:
        rows = await conn.fetch(
            """
            SELECT r.org_id, r.run_id, extract(epoch FROM (now()-r.updated_at)) AS age
              FROM agent_runs r
             WHERE r.status='queued' AND r.updated_at < now() - make_interval(secs => $1)
               AND ($2::text IS NULL OR r.org_id = $2)
               AND NOT EXISTS (SELECT 1 FROM outbox_jobs o WHERE o.org_id=r.org_id
                                  AND o.payload->>'run_id'=r.run_id AND o.status='pending')
               AND NOT EXISTS (SELECT 1 FROM run_events e WHERE e.org_id=r.org_id AND e.run_id=r.run_id
                                  AND e.kind='run.stuck_queued')
             LIMIT 100
            """, stuck_queued_s, org_id)
        for r in rows:
            await append_event(conn, org_id=r["org_id"], run_id=r["run_id"], kind="run.stuck_queued",
                               payload={"age_seconds": int(r["age"]), "action": "manual requeue_outbox"})
            counts["stuck_queued"] += 1
            print(f"[sweeper] 🔴 run={r['run_id']} queued {int(r['age'])}s 且 outbox 无 pending ⇒ 告警（不自动重投）",
                  flush=True)
    # ---- D · 写类 tool_calls running 超龄 → response_lost（逐工具 expected_max_ms）----
    from syncopate.runtime.tool_governance import GOVERNANCE
    async with db.tx() as conn:
        rows = await conn.fetch(
            """
            SELECT id, org_id, run_id, tool, extract(epoch FROM (now()-created_at))*1000 AS age_ms
              FROM tool_calls
             WHERE side_effect AND status='running' AND ($1::text IS NULL OR org_id = $1)
             ORDER BY created_at LIMIT 200
            """, org_id)
        for r in rows:
            g = GOVERNANCE.get(r["tool"])
            limit = g.expected_max_ms if g else 60_000
            if r["age_ms"] < limit:
                continue
            await conn.execute("UPDATE tool_calls SET status='response_lost', ended_at=now(), "
                               "error=COALESCE(error, 'response_lost: worker died mid-call') WHERE id=$1", r["id"])
            await append_event(conn, org_id=r["org_id"], run_id=r["run_id"], kind="tool.response_lost",
                               payload={"tool": r["tool"], "tool_call_id": r["id"], "age_ms": int(r["age_ms"])})
            counts["response_lost"] += 1
    # ---- E · outbox 归档 ----
    async with db.tx() as conn:
        tag = await conn.execute(
            "DELETE FROM outbox_jobs WHERE status='dispatched' AND dispatched_at < now() - make_interval(days => $1)",
            archive_days)
        counts["outbox_archived"] = int(tag.split()[-1]) if tag.split()[-1].isdigit() else 0
    return counts


async def requeue_outbox(db: Database, *, org_id: str, run_id: str, operator: str) -> int:
    """僵尸 queued 的人工重投（不自动）：写 outbox + 审计。"""
    async with db.tx() as conn:
        st = await conn.fetchval("SELECT status FROM agent_runs WHERE org_id=$1 AND run_id=$2", org_id, run_id)
        if st != "queued":
            raise ValueError(f"run {run_id} 不在 queued（{st}），不能重投")
        job = await enqueue_outbox(conn, org_id=org_id, run_id=run_id)
        await append_event(conn, org_id=org_id, run_id=run_id, kind="run.requeued_manually",
                           payload={"outbox_job_id": job, "operator": operator})
        await conn.execute("INSERT INTO audit_logs (run_id, org_id, action, param_source, detail) "
                           "VALUES ($1,$2,'run.requeue_outbox','user',$3)", run_id, org_id,
                           {"operator": operator, "outbox_job_id": job})
        return job


async def reconcile_once(db: Database, ledger: Any, *, actor_id: str = "reconciler",
                         org_id: str | None = None) -> dict[str, int]:
    """对账（§11.7 六步）：只盯 tool_calls 里结果未知的**写**调用。三种结果的写入都在同一事务里。"""
    counts = {"repaired_succeeded": 0, "repaired_failed": 0, "manual_review": 0}
    async with db.tx() as conn:
        rows = await conn.fetch(
            """
            SELECT id, org_id, run_id, tool, external_idempotency_key
              FROM tool_calls
             WHERE side_effect AND status='response_lost' AND external_idempotency_key IS NOT NULL
               AND ($1::text IS NULL OR org_id = $1)
             ORDER BY id LIMIT 200
            """, org_id)
    for r in rows:
        try:
            downstream = await ledger.get(r["external_idempotency_key"])
            available = True
        except Exception as exc:                       # noqa: BLE001 —— 账本不可用：不猜，转人工
            downstream, available = None, False
            print(f"[reconcile] 账本不可用 {exc!r} ⇒ tool_call={r['id']} manual_review", flush=True)
            from syncopate.runtime.log import log_event
            log_event("reconcile", "ledger_unavailable", level="error", run_id=r["run_id"], org_id=r["org_id"],
                      tool_call_id=r["id"], tool=r["tool"], error=repr(exc)[:300])
        async with db.tx() as conn:
            if not available:
                await append_event(conn, org_id=r["org_id"], run_id=r["run_id"], kind="tool.manual_review",
                                   payload={"tool": r["tool"], "tool_call_id": r["id"], "reason": "ledger_unavailable"})
                counts["manual_review"] += 1
                continue
            if downstream is not None:
                await conn.execute(
                    "UPDATE tool_calls SET status='succeeded', ok=TRUE, result=$2, error=NULL, error_json=NULL, "
                    "ended_at=now() WHERE id=$1", r["id"], downstream)
                resolved = "succeeded"
                counts["repaired_succeeded"] += 1
            else:
                await conn.execute(
                    "UPDATE tool_calls SET status='failed', ok=FALSE, error='reconciled: 下游无此键，副作用未发生', "
                    "error_json=$2, ended_at=now() WHERE id=$1", r["id"],
                    {"code": "reconciled_not_found", "message": "下游账本无此幂等键", "retryable": True, "alert": False})
                resolved = "failed"
                counts["repaired_failed"] += 1
            await append_event(conn, org_id=r["org_id"], run_id=r["run_id"], kind="tool.repaired",
                               payload={"tool": r["tool"], "tool_call_id": r["id"], "resolved_status": resolved})
            await conn.execute("INSERT INTO audit_logs (run_id, org_id, action, param_source, detail) "
                               "VALUES ($1,$2,'tool.reconcile','system',$3)", r["run_id"], r["org_id"],
                               {"tool_call_id": r["id"], "resolved_status": resolved, "actor": actor_id,
                                "key": r["external_idempotency_key"]})
            print(f"[reconcile] tool_call={r['id']} run={r['run_id']} ⇒ {resolved}", flush=True)
    return counts


async def _serve(interval: float) -> None:
    import signal
    from syncopate.runtime.platform import PlatformLedger

    db = Database()
    await db.connect(max_size=int(os.environ.get("SYNCOPATE_SWEEPER_DB_POOL", "3")))
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)
    ledger = PlatformLedger(db)
    print(f"[sweeper] 就位 interval={interval}s reconcile_every={RECONCILE_EVERY} "
          f"waiting_too_long={WAITING_TOO_LONG_S}s stuck_queued={STUCK_QUEUED_S}s max_attempts={MAX_RUN_ATTEMPTS}",
          flush=True)
    tick = 0
    try:
        while not stop.is_set():
            t0 = time.monotonic()
            c = await sweep_once(db)
            if any(c.values()):
                print(f"[sweeper] {c}", flush=True)
            tick += 1
            if tick % RECONCILE_EVERY == 0 or c["response_lost"]:
                rc = await reconcile_once(db, ledger)
                if any(rc.values()):
                    print(f"[reconcile] {rc}", flush=True)
            try:
                await asyncio.wait_for(stop.wait(), timeout=max(0.0, interval - (time.monotonic() - t0)))
            except asyncio.TimeoutError:
                pass
    finally:
        await db.close()


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Syncopate sweeper + reconciliation")
    ap.add_argument("--interval", type=float, default=SWEEP_INTERVAL_S)
    a = ap.parse_args()
    asyncio.run(_serve(a.interval))


if __name__ == "__main__":
    main()
