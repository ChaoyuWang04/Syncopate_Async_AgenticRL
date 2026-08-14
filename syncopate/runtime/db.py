"""M9 · 数据库接入层：连接池 + 三层幂等的物理实现。

★★★ 幂等是整个 runtime 里唯一一个"错了就是真金白银"的东西

其余组件出问题最多是服务不可用；**重复扣款是不可逆损失**。所以这一层的判据
必须是**实测重复投递**（tests/runtime/test_idempotency.py），不能是代码 review。

三层各自靠什么兑现（设计文档 §38）：

    请求级  用户点两次        agent_runs 上的 UNIQUE(org_id, idempotency_key)
    任务级  队列重投          agent_runs 的状态机 + lease（claim_run 里的原子 UPDATE）
    工具级  同一次预算变更两次  tool_calls 上的 UNIQUE(org_id, external_idempotency_key)

⚠️ **只有第三层是外部系统认的。** 实查过：Meta Marketing API 本身没有幂等机制，
所以这层保证得由我们自己兑现 —— 就是那个唯一索引 + 命中后**返回原结果而不是重放**。

★ 为什么唯一索引不按 run_id 分组：跨 run 重试同一个动作同样是重复扣款。
"""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import asyncpg

DSN = os.environ.get(
    "SYNCOPATE_PG_DSN", "postgresql://syncopate:syncopate@127.0.0.1:5432/syncopate")


class Database:
    """asyncpg 连接池的薄封装。JSONB 统一走 json.dumps/loads，不留隐式转换。"""

    def __init__(self, dsn: str | None = None) -> None:
        self.dsn = dsn or DSN
        self._pool: asyncpg.Pool | None = None

    async def connect(self, *, min_size: int = 1, max_size: int = 10) -> None:
        async def init(conn: asyncpg.Connection) -> None:
            # 不设的话取出来是 str，业务侧到处 json.loads，早晚漏一处。
            await conn.set_type_codec("jsonb", encoder=json.dumps, decoder=json.loads,
                                      schema="pg_catalog")
        self._pool = await asyncpg.create_pool(self.dsn, min_size=min_size, max_size=max_size,
                                               init=init)

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    @property
    def pool(self) -> asyncpg.Pool:
        if self._pool is None:
            raise RuntimeError("连接池没建 —— 先 await db.connect()")
        return self._pool

    @asynccontextmanager
    async def tx(self):
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                yield conn


# --------------------------------------------------------------------------
# 第一层 · 请求级幂等
# --------------------------------------------------------------------------


@dataclass
class RunHandle:
    run_id: str
    created: bool          # False = 命中了已有的那次请求，**没有新建**


async def create_run(db: Database, *, org_id: str, run_id: str, user_message: str,
                     idempotency_key: str | None = None, intent: str | None = None,
                     automation_tier: str | None = None) -> RunHandle:
    """建一次 run。带 Idempotency-Key 时**同一个 org 内重复请求返回原来那次**。

    ★ 用 `ON CONFLICT DO NOTHING` + 回查，而不是"先查再插" ——
    后者在并发下有竞态窗口（两个请求同时查到"不存在"，然后都插）。
    唯一索引是**数据库替我们保证的**，应用层只负责识别冲突。
    """
    async with db.tx() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO agent_runs (run_id, org_id, idempotency_key, user_message,
                                    intent, automation_tier, status)
            VALUES ($1, $2, $3, $4, $5, $6, 'queued')
            ON CONFLICT (org_id, idempotency_key) WHERE idempotency_key IS NOT NULL
            DO NOTHING
            RETURNING run_id
            """,
            run_id, org_id, idempotency_key, user_message, intent, automation_tier)
        if row is not None:
            return RunHandle(run_id=row["run_id"], created=True)
        # 冲突 ⇒ 把原来那次捞出来返回。**不报错** —— 重复提交是正常现象，不是错误。
        existing = await conn.fetchrow(
            "SELECT run_id FROM agent_runs WHERE org_id=$1 AND idempotency_key=$2",
            org_id, idempotency_key)
        if existing is None:                      # 理论上到不了：冲突了却查不到
            raise RuntimeError("幂等冲突但找不到原记录，索引和查询条件不一致？")
        return RunHandle(run_id=existing["run_id"], created=False)


# --------------------------------------------------------------------------
# 第二层 · 任务级幂等（状态机 + lease）
# --------------------------------------------------------------------------


async def claim_run(db: Database, *, worker_id: str, lease_seconds: int = 60) -> dict | None:
    """抢一个待跑的 run。**原子**：同一条 run 不可能被两个 worker 同时抢到。

    ★ `FOR UPDATE SKIP LOCKED` 是这里的关键：没有它，多个 worker 会锁在同一行上
    互相等待（吞吐塌成串行）；有了它，抢不到的直接跳过看下一条。

    ★ lease 过期才能被重抢 —— worker 崩了不会让任务永远卡住，
    而正常在跑的任务也不会被别人偷走。这就是"队列重投不重复执行"的实现。
    """
    async with db.tx() as conn:
        row = await conn.fetchrow(
            """
            WITH claimable AS (
                SELECT id FROM agent_runs
                WHERE status = 'queued'
                   OR (status = 'running' AND lease_expires_at < now())
                ORDER BY created_at
                LIMIT 1
                FOR UPDATE SKIP LOCKED
            )
            UPDATE agent_runs r
               SET status = 'running',
                   lease_owner = $1,
                   lease_expires_at = now() + make_interval(secs => $2),
                   attempt = r.attempt + 1,
                   updated_at = now()
              FROM claimable c
             WHERE r.id = c.id
            RETURNING r.run_id, r.org_id, r.user_message, r.attempt
            """,
            worker_id, lease_seconds)
        return dict(row) if row else None


async def finish_run(db: Database, *, org_id: str, run_id: str, status: str,
                     result: dict | None = None, error: str | None = None) -> None:
    async with db.tx() as conn:
        await conn.execute(
            """
            UPDATE agent_runs SET status=$3, result=$4, error=$5,
                   lease_owner=NULL, lease_expires_at=NULL, updated_at=now()
             WHERE org_id=$1 AND run_id=$2
            """, org_id, run_id, status, result, error)


# --------------------------------------------------------------------------
# 第三层 · 工具级幂等 —— ★ 唯一被外部系统认的那层
# --------------------------------------------------------------------------


@dataclass
class ToolCallResult:
    ok: bool
    data: dict[str, Any] | None
    error: str | None
    replayed: bool         # True = 命中幂等键，**没有真的再执行一次**
    call_id: int


async def record_tool_call(
    db: Database, *, org_id: str, run_id: str, step: int, tool: str,
    arguments: dict[str, Any], external_idempotency_key: str | None,
    execute,                                   # async () -> (ok, data, error)
) -> ToolCallResult:
    """执行一次工具调用，带工具级幂等。

    ★★ 顺序是刻意的：**先占坑，再执行**。

    反过来（先执行再记账）的话，进程在"执行完但还没记账"的窗口里崩掉，
    重试就会**真的再扣一次钱** —— 而那正是这层要防的东西。
    先占坑意味着：占坑成功才执行；占坑冲突说明别人已经执行过，直接返回原结果。

    ⚠️ 占坑用**独立事务**提交，不能和执行放在一个事务里：
    执行是外部副作用（HTTP 调用），事务回滚**撤销不了它**。
    """
    if external_idempotency_key is not None:
        async with db.tx() as conn:
            prior = await conn.fetchrow(
                """
                SELECT id, ok, result, error FROM tool_calls
                 WHERE org_id=$1 AND external_idempotency_key=$2 AND replayed_from IS NULL
                """, org_id, external_idempotency_key)
            if prior is not None:
                # 命中 ⇒ **返回原结果，不重放**。同时记一条"这次被幂等挡下了"的痕迹，
                # 否则"到底重试了几次"在事后完全不可见。
                await conn.execute(
                    """
                    INSERT INTO tool_calls (run_id, org_id, step, tool, arguments,
                                            ok, result, error, replayed_from)
                    VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)
                    """, run_id, org_id, step, tool, arguments,
                    prior["ok"], prior["result"], prior["error"], prior["id"])
                return ToolCallResult(ok=prior["ok"], data=prior["result"],
                                      error=prior["error"], replayed=True,
                                      call_id=prior["id"])

    # 占坑（独立事务，先提交）
    async with db.tx() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO tool_calls (run_id, org_id, step, tool, arguments,
                                    external_idempotency_key)
            VALUES ($1,$2,$3,$4,$5,$6) RETURNING id
            """, run_id, org_id, step, tool, arguments, external_idempotency_key)
        call_id = row["id"]

    ok, data, error = await execute()

    async with db.tx() as conn:
        await conn.execute("UPDATE tool_calls SET ok=$2, result=$3, error=$4 WHERE id=$1",
                           call_id, ok, data, error)
    return ToolCallResult(ok=ok, data=data, error=error, replayed=False, call_id=call_id)
