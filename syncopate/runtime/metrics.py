"""M9.7 前置 · 延迟与吞吐的分位数查询（§19 的门槛靠它判定）。

★★★ 为什么单独写一个模块：**门槛量不了就等于没写**

设计文档 §19 的门槛是「端到端 **P95，按意图分**」「工具调用 P95（排除 480s 慢工具）」。
在这之前 runtime 里没有任何东西能算出这两个数 —— 门槛写着，尺子不存在。
这正是 M7 那六条毕业条件废掉的方式（`11-runtime-acceptance.md` §0）。

★ 三个口径上的选择，都会影响结论：

1. **端到端 = `ended_at − started_at`，不是 `updated_at − created_at`。**
   后者混进了**排队时间**。"用户等了多久"和"我们跑了多久"是两个问题，
   混在一起两个都答不了。排队时间单独由 `queue_wait` 报。
2. **按意图分组**：各意图的工具链深度差很远（I01 两步 vs I11 十几步），
   一个总体 P95 对谁都不对 —— 同 §21「不能合并」的那三处。
3. **分位数在 SQL 里算**（`percentile_disc`），不是取回来在 Python 里排序：
   压测时行数会到十万级。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from syncopate.runtime.db import Database

# §19 的门槛（按意图，单位秒）。数从设计文档来，不在这里另立一套。
E2E_P95_BUDGET_SECONDS: dict[str, float] = {
    "I01": 5.0, "I07": 30.0, "I09": 60.0, "I11": 180.0,
}
# 端到端 P50 ≤ P95 × 0.4、P99 ≤ P95 × 2（§19）
P50_RATIO, P99_RATIO = 0.4, 2.0
TOOL_P95_BUDGET_SECONDS = 2.0

# ★ 排除慢工具：§19 明写「排除 480s 慢工具」。这些是**设计上就慢**的，
#   混进来会把 P95 拉到毫无意义 —— 但它们**不能被静默丢掉**，
#   所以单独报一行 `slow_tools`，而不是当作不存在。
SLOW_TOOLS = frozenset({"creative.review", "system.wait"})


@dataclass(frozen=True)
class LatencyRow:
    key: str
    n: int
    p50_ms: int
    p95_ms: int
    p99_ms: int


async def _quantiles(db: Database, sql: str, *args: Any) -> list[LatencyRow]:
    async with db.tx() as conn:
        rows = await conn.fetch(sql, *args)
    return [LatencyRow(key=r["k"] or "(未标注)", n=r["n"],
                       p50_ms=int(r["p50"] or 0), p95_ms=int(r["p95"] or 0),
                       p99_ms=int(r["p99"] or 0)) for r in rows]


async def end_to_end_by_intent(db: Database, *, org_id: str | None = None) -> list[LatencyRow]:
    """端到端延迟，按意图。**只算真的跑完的**（ended_at 非空）。"""
    return await _quantiles(db, """
        SELECT intent AS k, count(*) AS n,
               percentile_disc(0.50) WITHIN GROUP (
                   ORDER BY extract(epoch FROM (ended_at - started_at))*1000) AS p50,
               percentile_disc(0.95) WITHIN GROUP (
                   ORDER BY extract(epoch FROM (ended_at - started_at))*1000) AS p95,
               percentile_disc(0.99) WITHIN GROUP (
                   ORDER BY extract(epoch FROM (ended_at - started_at))*1000) AS p99
          FROM agent_runs
         WHERE started_at IS NOT NULL AND ended_at IS NOT NULL
           AND ($1::text IS NULL OR org_id = $1)
         GROUP BY intent ORDER BY intent
    """, org_id)


async def queue_wait_seconds(db: Database, *, org_id: str | None = None) -> float:
    """★ 最老的一条还在排队的 run 等了多久（§19 的积压门槛：≤ 60s）。

    用**时间**而不是**条数**：条数受批量影响，时间才是用户感受到的东西。
    """
    async with db.tx() as conn:
        val = await conn.fetchval("""
            SELECT COALESCE(max(extract(epoch FROM (now() - created_at))), 0)
              FROM agent_runs
             WHERE status='queued' AND ($1::text IS NULL OR org_id=$1)
        """, org_id)
    return float(val or 0.0)


async def tool_latency(db: Database, *, org_id: str | None = None,
                       include_slow: bool = False) -> list[LatencyRow]:
    """工具调用延迟。默认**排除慢工具**（§19），但它们由 `slow_tools` 单独可查。

    ⚠️ 只算真的执行过的：`replayed_from IS NULL`。被幂等挡下那次没打平台，
    把它算进去会**把 P95 拉低成一个假象**。
    """
    slow = list(SLOW_TOOLS)
    return await _quantiles(db, """
        SELECT tool AS k, count(*) AS n,
               percentile_disc(0.50) WITHIN GROUP (ORDER BY latency_ms) AS p50,
               percentile_disc(0.95) WITHIN GROUP (ORDER BY latency_ms) AS p95,
               percentile_disc(0.99) WITHIN GROUP (ORDER BY latency_ms) AS p99
          FROM tool_calls
         WHERE latency_ms IS NOT NULL AND replayed_from IS NULL
           AND ($1::bool OR NOT (tool = ANY($2::text[])))
           AND ($3::text IS NULL OR org_id = $3)
         GROUP BY tool ORDER BY tool
    """, include_slow, slow, org_id)


async def read_write_split(db: Database, *, org_id: str | None = None) -> dict[str, Any]:
    """★ 读 ⊥ 写分桶（§21：混在一起，大量读操作会稀释掉写操作的风险）。

    eval 侧 2026-08-16 补上这把尺子，一量就发现写桶成功率只有 20% ——
    runtime 侧一直没有对应物。分组依据取自 `tools.WRITE_TOOLS`，**不另立一套清单**。
    """
    from syncopate.runtime.tools import WRITE_TOOLS
    async with db.tx() as conn:
        rows = await conn.fetch("""
            SELECT (tool = ANY($1::text[])) AS is_write,
                   count(*) AS n,
                   count(*) FILTER (WHERE ok IS TRUE) AS ok_n,
                   percentile_disc(0.95) WITHIN GROUP (ORDER BY latency_ms) AS p95
              FROM tool_calls
             WHERE replayed_from IS NULL AND ($2::text IS NULL OR org_id=$2)
             GROUP BY 1
        """, list(WRITE_TOOLS), org_id)
    out: dict[str, Any] = {}
    for r in rows:
        bucket = "write" if r["is_write"] else "read"
        n = r["n"] or 0
        out[bucket] = {"n": n, "ok": r["ok_n"] or 0,
                       "success_rate": round((r["ok_n"] or 0) / n, 4) if n else None,
                       "p95_ms": int(r["p95"] or 0)}
    return out


def verdict(rows: list[LatencyRow], budgets: dict[str, float]) -> list[str]:
    """把分位数对成 §19 的判定。**没有门槛的分组显式说"无门槛"，不当成通过。**"""
    out = []
    for row in rows:
        budget = budgets.get(row.key)
        if budget is None:
            out.append(f"{row.key}: 无门槛（不可判定，不是通过）  n={row.n} p95={row.p95_ms}ms")
            continue
        p95_ok = row.p95_ms <= budget * 1000
        p50_ok = row.p50_ms <= budget * 1000 * P50_RATIO
        p99_ok = row.p99_ms <= budget * 1000 * P99_RATIO
        mark = "✅" if (p95_ok and p50_ok and p99_ok) else "🔴"
        out.append(f"{mark} {row.key}: n={row.n} "
                   f"p50={row.p50_ms}/{int(budget*1000*P50_RATIO)} "
                   f"p95={row.p95_ms}/{int(budget*1000)} "
                   f"p99={row.p99_ms}/{int(budget*1000*P99_RATIO)} ms")
    return out


# --------------------------------------------------------------------------
# K3-11 · 积压（课件 §12）：告警挂用户视角的 oldest_job_age，不挂系统视角的 queue_length
# --------------------------------------------------------------------------

OLDEST_JOB_AGE_ALERT_S = 60.0     # 27 §15 queue lag SLO 的告警线（P95 < 10s 是目标，60s 是"用户已经在等"）


async def queue_backlog(db: Database, *, org_id: str | None = None,
                        redis_client: Any | None = None) -> dict[str, Any]:
    """八指标里本阶段先落地五个 + Redis 队列长度（有客户端时）。
    `oldest_queued_run_age_s` = 最老一条 queued run 从创建到现在（用户等了多久）——告警判据。"""
    async with db.tx() as conn:
        row = await conn.fetchrow("""
            SELECT
              (SELECT count(*) FROM outbox_jobs WHERE status='pending'
                 AND ($1::text IS NULL OR org_id=$1))                                     AS outbox_pending,
              (SELECT COALESCE(max(extract(epoch FROM (now()-created_at))),0) FROM outbox_jobs
                WHERE status='pending' AND ($1::text IS NULL OR org_id=$1))                AS oldest_pending_age_s,
              (SELECT count(*) FROM agent_runs WHERE status='queued'
                 AND ($1::text IS NULL OR org_id=$1))                                     AS queued_runs,
              (SELECT COALESCE(max(extract(epoch FROM (now()-created_at))),0) FROM agent_runs
                WHERE status='queued' AND ($1::text IS NULL OR org_id=$1))                 AS oldest_queued_run_age_s,
              (SELECT count(*) FROM dead_letter_jobs WHERE reprocessed_at IS NULL
                 AND ($1::text IS NULL OR org_id=$1))                                     AS dead_letter_open
        """, org_id)
    out = {k: (float(v) if k.endswith("_s") else int(v)) for k, v in dict(row).items()}
    if redis_client is not None:
        out["redis_queue_lengths"] = {q: int(redis_client.llen(q))
                                      for q in ("interactive", "batch", "maintenance")}
    out["alert"] = out["oldest_queued_run_age_s"] > OLDEST_JOB_AGE_ALERT_S
    return out


# --------------------------------------------------------------------------
# K9-3 · 指标面板（~10 项，课件 15 项裁剪）+ 告警绑定 runbook；K9-1 · 九条 SLO 自动读数
# --------------------------------------------------------------------------

RUNBOOK = {
    "queue_lag": "27 §13 K11-2 · RUNBOOK 01 queue lag 持续升高",
    "stuck_runs": "RUNBOOK 02 卡死 run",
    "write_tool_errors": "RUNBOOK 03 写工具报错",
    "dead_letter": "RUNBOOK 01/03（死信 = 病历）",
    "response_lost": "RUNBOOK 03（对账）",
    "budget": "RUNBOOK 06 token 消耗异常",
}


async def snapshot(db: Database, *, org_id: str | None = None) -> dict[str, Any]:
    """一次查全（读数来源全部是表，不许人工口算）。"""
    async with db.tx() as conn:
        by_status = {r["status"]: int(r["n"]) for r in await conn.fetch(
            "SELECT status, count(*) AS n FROM agent_runs WHERE ($1::text IS NULL OR org_id=$1) GROUP BY status", org_id)}
        row = await conn.fetchrow("""
            SELECT
              (SELECT count(*) FROM agent_runs WHERE status='running' AND lease_expires_at < now()
                 AND ($1::text IS NULL OR org_id=$1))                                            AS stuck_running,
              (SELECT count(*) FROM run_events e JOIN agent_runs r ON r.org_id=e.org_id AND r.run_id=e.run_id
                WHERE e.kind='run.stuck_queued' AND r.status='queued'
                  AND ($1::text IS NULL OR e.org_id=$1))                                          AS stuck_queued,
              (SELECT count(*) FROM tool_calls WHERE status='skipped_duplicate'
                 AND ($1::text IS NULL OR org_id=$1))                                            AS duplicate_prevented_total,
              (SELECT count(*) FROM tool_calls WHERE side_effect AND status='response_lost'
                 AND ($1::text IS NULL OR org_id=$1))                                            AS response_lost_open,
              (SELECT count(*) FROM dead_letter_jobs WHERE reprocessed_at IS NULL
                 AND ($1::text IS NULL OR org_id=$1))                                            AS dead_letter_open,
              (SELECT count(*) FROM tool_calls WHERE side_effect AND blocked_by IS NULL
                 AND created_at > now()-interval '24 hours' AND ($1::text IS NULL OR org_id=$1)) AS write_calls_24h,
              (SELECT count(*) FROM tool_calls WHERE side_effect AND blocked_by IS NULL AND status IN ('failed','response_lost')
                 AND created_at > now()-interval '24 hours' AND ($1::text IS NULL OR org_id=$1)) AS write_errors_24h,
              (SELECT count(*) FROM agent_runs WHERE created_at > now()-interval '24 hours'
                 AND status IN ('succeeded','failed','cancelled') AND ($1::text IS NULL OR org_id=$1)) AS finished_24h,
              (SELECT count(*) FROM agent_runs WHERE created_at > now()-interval '24 hours' AND status='failed'
                 AND ($1::text IS NULL OR org_id=$1))                                            AS failed_24h,
              (SELECT count(*) FROM agent_runs WHERE created_at > now()-interval '24 hours'
                 AND ($1::text IS NULL OR org_id=$1))                                            AS created_24h,
              (SELECT count(DISTINCT run_id) FROM run_events WHERE kind='run.created'
                 AND created_at > now()-interval '24 hours' AND ($1::text IS NULL OR org_id=$1)) AS created_events_24h,
              (SELECT count(*) FROM agent_runs WHERE budget_exceeded_at IS NOT NULL
                 AND ($1::text IS NULL OR org_id=$1))                                            AS budget_waiting_total,
              (SELECT COALESCE(percentile_disc(0.95) WITHIN GROUP (ORDER BY extract(epoch FROM (started_at-created_at))),0)
                 FROM agent_runs WHERE started_at IS NOT NULL AND created_at > now()-interval '24 hours'
                 AND ($1::text IS NULL OR org_id=$1))                                            AS queue_lag_p95_s,
              (SELECT COALESCE(percentile_disc(0.95) WITHIN GROUP (ORDER BY extract(epoch FROM (ended_at-created_at))),0)
                 FROM agent_runs WHERE ended_at IS NOT NULL AND created_at > now()-interval '24 hours'
                 AND ($1::text IS NULL OR org_id=$1))                                            AS completion_p95_s
        """, org_id)
    m = {k: (float(v) if k.endswith("_s") else int(v)) for k, v in dict(row).items()}
    backlog = await queue_backlog(db, org_id=org_id)
    m.update({f"runs_{k}": v for k, v in by_status.items()})
    m["outbox_pending"] = backlog["outbox_pending"]
    m["oldest_queued_run_age_s"] = backlog["oldest_queued_run_age_s"]
    m["stuck_runs"] = m["stuck_running"] + m["stuck_queued"]
    m["write_tool_error_rate"] = (m["write_errors_24h"] / m["write_calls_24h"]) if m["write_calls_24h"] else 0.0
    m["run_failed_ratio"] = (m["failed_24h"] / m["finished_24h"]) if m["finished_24h"] else 0.0
    m["run_created_success_rate"] = (m["created_events_24h"] / m["created_24h"]) if m["created_24h"] else 1.0
    return m


def alerts(m: dict[str, Any]) -> list[dict[str, str]]:
    """告警绑定行动（课件 §12：告警正文带 Runbook 引用）。单一指标不判根因，这里只是"该看了"。"""
    out = []
    if m["oldest_queued_run_age_s"] > OLDEST_JOB_AGE_ALERT_S:
        out.append({"alert": "queue_lag", "value": f"{m['oldest_queued_run_age_s']:.0f}s", "runbook": RUNBOOK["queue_lag"]})
    if m["stuck_runs"] >= 10:
        out.append({"alert": "stuck_runs", "value": str(m["stuck_runs"]), "runbook": RUNBOOK["stuck_runs"]})
    if m["write_tool_error_rate"] > 0.001 and m["write_calls_24h"] >= 20:
        out.append({"alert": "write_tool_errors", "value": f"{m['write_tool_error_rate']:.2%}", "runbook": RUNBOOK["write_tool_errors"]})
    if m["dead_letter_open"] > 0:
        out.append({"alert": "dead_letter", "value": str(m["dead_letter_open"]), "runbook": RUNBOOK["dead_letter"]})
    if m["response_lost_open"] > 0:
        out.append({"alert": "response_lost", "value": str(m["response_lost_open"]), "runbook": RUNBOOK["response_lost"]})
    return out


def render_prometheus(m: dict[str, Any]) -> str:
    lines = []
    for k, v in sorted(m.items()):
        if isinstance(v, (int, float)):
            lines.append(f"syncopate_{k} {v}")
    return "\n".join(lines) + "\n"


SLO_SPEC = (   # 27 §15 九条：名 · 读数键 · 判据（lambda 读数→bool）· 归属
    ("POST /runs P95 < 300ms", "post_runs_p95_ms", lambda v: v is not None and v < 300, "K1"),
    ("run.created 成功率 > 99.9%", "run_created_success_rate", lambda v: v > 0.999, "K2/K3"),
    ("普通任务 P95 完成 < 60s", "completion_p95_s", lambda v: v < 60, "整链"),
    ("run.failed 比例 < 1%", "run_failed_ratio", lambda v: v < 0.01, "K5"),
    ("写类工具错误率 < 0.1%", "write_tool_error_rate", lambda v: v < 0.001, "K6"),
    ("queue lag P95 < 10s", "queue_lag_p95_s", lambda v: v < 10, "K3"),
    ("SSE 断线后可通过 after 补齐", "sse_after_ok", lambda v: v is True, "K7"),
    ("stuck run 数量 < 10", "stuck_runs", lambda v: v < 10, "K8"),
    ("单 org 每日成本不超预算", "org_budget_ratio", lambda v: v is not None and v < 1.0, "K9"),
)


def slo_table(m: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for name, key, ok, owner in SLO_SPEC:
        v = m.get(key)
        rows.append({"slo": name, "value": v, "ok": (ok(v) if v is not None else None), "owner": owner,
                     "verdict": "✅" if v is not None and ok(v) else ("⬜ 无读数" if v is None else "🔴")})
    return rows


async def by_version(db: Database, *, key: str = "prompt_version", org_id: str | None = None,
                     window_hours: int = 24 * 30) -> list[dict[str, Any]]:
    """K10-5：任一指标按版本切片（contract/prompt/model）。不切只能报警，切了才能定位。"""
    assert key in ("contract_version", "prompt_version", "model_version")
    async with db.tx() as conn:
        rows = await conn.fetch(f"""
            SELECT COALESCE({key}, '<unset>') AS version, count(*) AS runs,
                   sum(CASE WHEN status='failed' THEN 1 ELSE 0 END) AS failed,
                   sum(CASE WHEN status='succeeded' THEN 1 ELSE 0 END) AS succeeded
              FROM agent_runs
             WHERE created_at > now() - make_interval(hours => $1) AND ($2::text IS NULL OR org_id=$2)
             GROUP BY 1 ORDER BY runs DESC
        """, window_hours, org_id)
    return [{"version": r["version"], "runs": int(r["runs"]), "failed": int(r["failed"]),
             "succeeded": int(r["succeeded"]),
             "failed_ratio": (int(r["failed"]) / int(r["runs"])) if r["runs"] else 0.0} for r in rows]
