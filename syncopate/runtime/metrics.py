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
