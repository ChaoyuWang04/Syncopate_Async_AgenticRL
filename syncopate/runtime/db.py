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
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import hashlib
import secrets
import uuid

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
            # ⚠️ `default=str`：写 JSONB 的值里可能混着 date/datetime/Decimal
            #   （工具结果直接进 tool_calls.result）—— 少了它就是下面那条 bug 的另一半。
            await conn.set_type_codec("jsonb",
                                      encoder=lambda v: json.dumps(v, default=str),
                                      decoder=json.loads, schema="pg_catalog")
            # ★★★ NUMERIC → float（2026-08-20 实测抓到，代价很大的一条）
            #
            # asyncpg 默认把 NUMERIC 解成 `Decimal`，而 **Decimal 不能 JSON 序列化**。
            # 后果链：任何返回 NUMERIC 列的工具（安全线 cpi_d7_max、行业基准 p50、
            # 记忆 confidence、地域 roas、feature lift…）→ 记账写 JSONB 时
            # `json.dumps` 抛 TypeError → 收口按 `tool_crashed` 处理 →
            # **模型收到"这个工具暂时不可用"，于是如实回答"查不到"**。
            #
            # ⚠️ 它伪装成"模型能力差"：人看到的是一串 no_data / unavailable，
            #   而真相是半数工具在真实路径上根本跑不通。测试没抓到，因为测试里
            #   插的数据走的是同一条 Decimal 路径、又极少断言"观测能被渲染给模型"。
            # ⇒ 判据在 `tests/runtime/test_tool_impls.py::test_..._json_serializable`。
            #
            # ★ 为什么在这里改而不是在 30 个工具里各转一次：**该一致的值不该有 30 份副本**。
            # ⚠️ 精度：钱一律是 INTEGER（微单位），NUMERIC 只用于比率/系数，float 够用。
            await conn.set_type_codec("numeric", encoder=str, decoder=float,
                                      schema="pg_catalog", format="text")
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
        # B-5 分账插桩（stage_timing.ENABLED=False 时只多一次 bool 判断）
        from syncopate.runtime import stage_timing as st
        if not st.ENABLED:
            async with self.pool.acquire() as conn:
                async with conn.transaction():
                    yield conn
            return
        t0 = time.perf_counter()
        async with self.pool.acquire() as conn:
            st.add("db_wait", time.perf_counter() - t0)
            t1 = time.perf_counter()
            try:
                async with conn.transaction():
                    yield conn
            finally:
                st.add("db_in_tool" if st.in_tool() else "db_tx",
                       time.perf_counter() - t1)


# --------------------------------------------------------------------------
# 第一层 · 请求级幂等
# --------------------------------------------------------------------------


def new_run_id() -> str:
    """不可枚举 id（课件 H18）：`run_` + 48 位随机十六进制。api 与测试只许从这里拿。"""
    return f"run_{uuid.uuid4().hex[:12]}"


def new_conversation_id() -> str:
    return f"conv_{uuid.uuid4().hex[:12]}"


def input_hash(**fields: Any) -> str:
    """幂等第二把锁（课件 H11）：同 key 不同 input ⇒ 409。规范化 = 排序键 + 不转义。"""
    canon = json.dumps(fields, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


async def next_seq(conn: asyncpg.Connection, *, org_id: str, run_id: str) -> int:
    """seq 领号器（课件 C9 / H13，27 K2-2）：同事务 `last_seq+1`，行锁持到 COMMIT
    ⇒ 分配顺序 = 可见顺序，两个写者（API 写取消、worker 写工具事件）不可能撞号，
    SSE 补发无空洞。⛔ 全库禁止 `SELECT max(seq)+1`（K2 门槛②的负向认证就是它）。"""
    seq = await conn.fetchval(
        "UPDATE agent_runs SET last_seq = last_seq + 1 "
        " WHERE org_id=$1 AND run_id=$2 RETURNING last_seq", org_id, run_id)
    if seq is None:
        raise LookupError(f"run {run_id!r} 不存在于 org {org_id!r}：无法分配 seq")
    return int(seq)


async def append_event(conn: asyncpg.Connection, *, org_id: str, run_id: str,
                       kind: str, payload: dict[str, Any] | None = None) -> int:
    """在**调用方的事务里**写一条事件（领号 + INSERT 同事务）。返回 seq。"""
    seq = await next_seq(conn, org_id=org_id, run_id=run_id)
    await conn.execute(
        "INSERT INTO run_events (run_id, org_id, seq, kind, payload) VALUES ($1,$2,$3,$4,$5)",
        run_id, org_id, seq, kind, payload or {})
    return seq


@dataclass
class RunHandle:
    run_id: str
    created: bool               # False = 命中了已有的那次请求，**没有新建**
    input_matches: bool = True  # 幂等命中时：同 key 的 input 是否相同（False ⇒ K1-3 判 409）


# --------------------------------------------------------------------------
# F-1 · 会话（chatbox 壳的载体，22 §I-1）——只是组织方式，不改 run 语义
# --------------------------------------------------------------------------


async def create_conversation(db: Database, *, org_id: str, conversation_id: str,
                              title: str | None = None, model: str = "rl") -> None:
    # model：dev mode 的会话级模型锁定（rl/sft/base，Chaoyu 08-29）。创建即定，永不 UPDATE。
    async with db.tx() as conn:
        await conn.execute(
            "INSERT INTO conversations (conversation_id, org_id, title, model) "
            "VALUES ($1, $2, $3, $4) ON CONFLICT (org_id, conversation_id) DO NOTHING",
            conversation_id, org_id, title, model)


async def prior_turns(db: Database, *, org_id: str, conversation_id: str,
                      before_run_id: str, limit: int = 6) -> list[dict]:
    """同一会话里**这条 run 之前**已经收尾的轮次（最近 limit 条，按时间正序）。

    ★ 只取有结论的（succeeded）：还在跑的、被取消的没有可复述的内容，
      塞进去只会让模型看到半截东西。
    ⚠️ 取 limit 条不是"省事"，是**预算纪律**：历史无上限地长下去必然撞截断，
      而截断这件事在本项目有前科（budget-truncation-family）。宁可少喂，不可静默砍。
    """
    # ★ 09-02（`26 §2.5` Ⓑ）：信令收场的轮次**也是历史**。此前只取 succeeded ⇒
    #   clarify 轮（停在 running）与 reject 轮（cancelled + result=None）都不进历史，
    #   模型在线上看不到自己上一轮问了什么/拒了什么——L4「clarify 后接着办」结构上不可能。
    #   现在：succeeded（含 defer、含被 close_parked_clarify_runs 收尾的 clarify 轮）
    #   ∪ cancelled 且 error='session_reject'（result = 信令自己的话，worker 现在会存）。
    #   仍然排除：failed / 其它 cancelled（没有可复述的内容）/ 还在跑的。
    async with db.tx() as conn:
        rows = await conn.fetch(
            """
            SELECT run_id, user_message, result
              FROM agent_runs
             WHERE org_id=$1 AND conversation_id=$2 AND run_id <> $3
               AND result IS NOT NULL
               AND (status='succeeded'
                    OR (status='cancelled' AND error='session_reject'))
             ORDER BY created_at DESC
             LIMIT $4
            """, org_id, conversation_id, before_run_id, limit)
    return [dict(r) for r in reversed(rows)]


async def park_run_for_user(db: Database, *, org_id: str, run_id: str,
                            result: dict | None, payload: dict | None = None) -> None:
    """没有审批单的挂起（v15 `session.clarify`）：run 置 waiting_for_user 等用户补充。

    ★ 09-02 之前这条路径**不存在**：agent_loop 对 clarify 返回 halted 且 case_ref=None，
      worker 当"审批单已开"直接 return ⇒ run 停在 running，60 s lease 过期后被 claim_run
      当崩溃 run 重抢重跑（R5 考场 L4 第一轮 8/25 正是 status=running）。
    ★ 形状按 K 线 D25 统一映射：running→waiting_for_user **必须清 lease**；
      终态事件 run.waiting_for_user 与状态在**同一事务**（同 gateway.open_approval_case）。
    ★ result 现在就存（信令自己的话），这样 prior_turns 收尾后能直接复述。
    """
    async with db.tx() as conn:
        await conn.execute(
            """
            UPDATE agent_runs SET status='waiting_for_user', result=$3,
                   lease_owner=NULL, lease_expires_at=NULL, resume_token=$4, updated_at=now()
             WHERE org_id=$1 AND run_id=$2
            """, org_id, run_id, result, new_resume_token())
        await append_event(conn, org_id=org_id, run_id=run_id,
                           kind="run.waiting_for_user", payload=payload or {})


async def close_parked_clarify_runs(db: Database, *, org_id: str,
                                    conversation_id: str) -> list[str]:
    """同会话来了下一条用户消息 ⇒ 之前等补充的 clarify 轮**收尾为 succeeded**。

    「一条消息 = 一个 run」不变：用户的回答是新 run，这一轮的追问作为历史进新 run 的 prompt
    （prior_turns 只取 succeeded ⇒ 收尾是它进历史的前提）。
    ⛔ 只收 requires_approval=FALSE 的：等审批的 run 由 POST /approvals 裁决，不许被一条
      聊天消息顺手关掉（负向认证在 tests/runtime/test_clarify_turns_enter_history.py）。
    """
    async with db.tx() as conn:
        rows = await conn.fetch(
            """
            UPDATE agent_runs SET status='succeeded', ended_at=now(), updated_at=now()
             WHERE org_id=$1 AND conversation_id=$2
               AND status='waiting_for_user' AND requires_approval=FALSE
            RETURNING run_id, result
            """, org_id, conversation_id)
        for r in rows:
            await append_event(conn, org_id=org_id, run_id=r["run_id"],
                               kind=_TERMINAL_EVENT["succeeded"], payload=dict(r["result"] or {}))
    return [r["run_id"] for r in rows]


async def conversation_exists(db: Database, *, org_id: str,
                              conversation_id: str) -> bool:
    """★ 越权同 run：`WHERE org_id=` 在 SQL 里挡 —— 别人的会话和不存在的会话不可区分。"""
    async with db.tx() as conn:
        return bool(await conn.fetchval(
            "SELECT 1 FROM conversations WHERE org_id=$1 AND conversation_id=$2",
            org_id, conversation_id))


async def create_run(db: Database, *, org_id: str, run_id: str, user_message: str,
                     idempotency_key: str | None = None, intent: str | None = None,
                     automation_tier: str | None = None,
                     conversation_id: str | None = None,
                     run_type: str = "chat") -> RunHandle:
    """建一次 run。带 Idempotency-Key 时**同一个 org 内重复请求返回原来那次**。

    ★ 用 `ON CONFLICT DO NOTHING` + 回查，而不是"先查再插" ——
    后者在并发下有竞态窗口（两个请求同时查到"不存在"，然后都插）。
    唯一索引是**数据库替我们保证的**，应用层只负责识别冲突。
    """
    ihash = input_hash(user_message=user_message, intent=intent,
                       automation_tier=automation_tier, conversation_id=conversation_id,
                       run_type=run_type)
    async with db.tx() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO agent_runs (run_id, org_id, idempotency_key, user_message,
                                    intent, automation_tier, status, conversation_id,
                                    run_type, input_hash)
            VALUES ($1, $2, $3, $4, $5, $6, 'queued', $7, $8, $9)
            ON CONFLICT (org_id, idempotency_key) WHERE idempotency_key IS NOT NULL
            DO NOTHING
            RETURNING run_id
            """,
            run_id, org_id, idempotency_key, user_message, intent, automation_tier,
            conversation_id, run_type, ihash)
        if row is not None:
            # K2-6：run 与 run.created **同生共死**（课件 C6"让系统失败得干净"）。
            # 事务里没有任何 publish；outbox 行 K3-2 加进来。
            await append_event(conn, org_id=org_id, run_id=run_id, kind="run.created",
                               payload={"run_type": run_type, "intent": intent})
            return RunHandle(run_id=row["run_id"], created=True)
        # 冲突 ⇒ 把原来那次捞出来返回。**不报错** —— 重复提交是正常现象，不是错误。
        existing = await conn.fetchrow(
            "SELECT run_id, input_hash FROM agent_runs WHERE org_id=$1 AND idempotency_key=$2",
            org_id, idempotency_key)
        if existing is None:                      # 理论上到不了：冲突了却查不到
            raise RuntimeError("幂等冲突但找不到原记录，索引和查询条件不一致？")
        # 第二把锁：同 key 不同 input。这里只**判定**，409 由 API 层（K1-3）决定。
        same = existing["input_hash"] is None or existing["input_hash"] == ihash
        return RunHandle(run_id=existing["run_id"], created=False, input_matches=same)


# --------------------------------------------------------------------------
# 第二层 · 任务级幂等（状态机 + lease）
# --------------------------------------------------------------------------


async def claim_run(db: Database, *, worker_id: str, lease_seconds: int = 60,
                    org_id: str | None = None) -> dict | None:
    """抢一个待跑的 run。**原子**：同一条 run 不可能被两个 worker 同时抢到。

    ★ `FOR UPDATE SKIP LOCKED` 是这里的关键：没有它，多个 worker 会锁在同一行上
    互相等待（吞吐塌成串行）；有了它，抢不到的直接跳过看下一条。

    ★ lease 过期才能被重抢 —— worker 崩了不会让任务永远卡住，
    而正常在跑的任务也不会被别人偷走。这就是"队列重投不重复执行"的实现。

    ★★ `org_id` = **把这个 worker 限定在一个租户上**（None = 全局，老行为）。

    ⚠️ 它不只是测试用的开关，是一条真实的生产能力：worker 池按租户切分
      （大客户独占一组 worker）是标准做法。
    ⚠️⚠️ 但它**首先修的是一个探针污染问题**：队列是全局的 ⇒ 任何调 `run_once`
      的测试/探针都会抢走别人遗留的活。2026-08-19 我自己就中过一次 ——
      探针报「C 档没走审批」，实际是它抢到了别的 run，**得出了一个完全错误的结论**。
      ⇒ 此前的修法是"每处记得先排空"（`test_worker._drain`），
        而**手动步骤一定会被忘** —— `test_retrieval.py` 就没排，它正是那条偶发红的来源。
      ⇒ 结构上拿不到别人的活，比"记得排空"可靠。
    """
    async with db.tx() as conn:
        row = await conn.fetchrow(
            """
            WITH inflight AS (
                -- 每个 org 当前有多少条在跑（含 lease 还没过期的）
                SELECT org_id, count(*) AS n FROM agent_runs
                 WHERE status='running' AND lease_expires_at >= now()
                 GROUP BY org_id
            ), claimable AS (
                SELECT r.id FROM agent_runs r
                LEFT JOIN inflight i ON i.org_id = r.org_id
                WHERE ($3::text IS NULL OR r.org_id = $3)     -- ★ 可选的租户限定，见下
                  AND (r.status = 'queued'
                       OR (r.status = 'running' AND r.lease_expires_at < now()))
                -- ★★ 公平分配：先按"这个 org 手上已经有几条在跑"排，再按先来后到。
                -- 之前是纯全局 FIFO ⇒ **一个 org 灌满队列就把别人饿死**
                -- （压测场景⑤「单 org 刷爆预算」考的正是这个）。
                -- 它同时也是"长任务不阻塞其他任务"的一半：另一半是并发（见 Worker.serve）。
                ORDER BY COALESCE(i.n, 0), r.created_at
                LIMIT 1
                FOR UPDATE SKIP LOCKED
            )
            UPDATE agent_runs r
               SET status = 'running',
                   started_at = COALESCE(r.started_at, now()),   -- ★ 只记第一次
                   lease_owner = $1,
                   lease_expires_at = now() + make_interval(secs => $2),
                   attempts = r.attempts + 1,
                   updated_at = now()
              FROM claimable c
             WHERE r.id = c.id
            RETURNING r.run_id, r.org_id, r.user_message, r.attempts,
                      r.intent, r.automation_tier, r.requires_approval,
                      -- ★ 多轮要它：不在 RETURNING 里 = worker 拿不到 = 历史永远为空
                      --   （「字段全程有效，只是没有消费者」那条的镜像）
                      r.conversation_id
            """,
            worker_id, lease_seconds, org_id)
        return dict(row) if row else None


# 终态 status → 终态事件。SSE 的关流判据（api.TERMINAL）必须与这张表一致。
_TERMINAL_EVENT = {"succeeded": "run.succeeded",
                   "failed": "run.failed",
                   "cancelled": "run.cancelled"}


async def finish_run(db: Database, *, org_id: str, run_id: str, status: str,
                     result: dict | None = None, error: str | None = None) -> None:
    """收尾 run，并在**同一事务里**发终态事件。

    ★★ 2026-08-20 冒烟实测抓到的结构 bug：此前终态事件由各调用方自己补发，
    而 release_gate / 成本闸 / refused 这几条退出路径**谁都没发** ——
    run 在库里已经 cancelled，SSE 客户端却永远等不到关流事件，挂死。
    ⇒ 修法同「审批单和 run 状态同一事务」那条：**状态翻终态 = 必有终态事件**，
    做成结构保证，调用方不再各自补发（补了就是重复）。
    """
    async with db.tx() as conn:
        await conn.execute(
            """
            UPDATE agent_runs SET status=$3, result=$4, error=$5,
                   ended_at=now(),
                   lease_owner=NULL, lease_expires_at=NULL, updated_at=now()
             WHERE org_id=$1 AND run_id=$2
            """, org_id, run_id, status, result, error)
        kind = _TERMINAL_EVENT.get(status)
        if kind is not None:
            payload = dict(result) if (status == "succeeded" and result) else (
                {"error": error} if error else {})
            await append_event(conn, org_id=org_id, run_id=run_id, kind=kind, payload=payload)


async def resume_after_approval(db: Database, *, org_id: str, run_id: str) -> None:
    """审批有了结论之后，把 run 放回可抢队列。

    ★★★ 这条函数补的是一个**断掉的闭环**：以前 `POST /approvals/{case_ref}`
    只 UPDATE 审批单，而 `claim_run` 不认 `waiting_for_user`
    ⇒ **人点了同意，run 永远不会继续**，飞轮回路 2 的 `modified_params`
    落了库却没有任何东西会去执行它。

    ⛔ **原本选的是"从头重跑"，2026-08-19 已改成"从 transcript 续"** ——
       原理由是「重跑一遍读操作，读是便宜的那一侧，代价可以接受」。
       **那个前提不成立了**（三条都变了）：

        ① 平台加了 BUC 积分制（B-1a）⇒ **读也扣配额**，不再免费
        ② 编排改成模型驱动（B-3b）⇒ 重跑要**重新花模型调用的钱**
        ③ 重跑会重新踩改动频次上限 —— 一小时只有 4 格

       ⇒ `agent_loop.save_transcript` 给 `checkpoints` 补上了它一直缺的写入路径
         （原文说"那张表现在没人写"，现在有了）。
       ⚠️ **幂等键那条兜底没有撤** —— transcript 只是省掉重复劳动，
         不是正确性的唯一依赖。两条都在，才敢在生产上恢复。

    ⇒ 记在这里，因为"为什么不做断点续"以后一定会被再问一次。
    """
    async with db.tx() as conn:
        tag = await conn.execute(
            """
            UPDATE agent_runs
               SET status='queued', lease_owner=NULL, lease_expires_at=NULL,
                   resume_token=NULL, updated_at=now()
             WHERE org_id=$1 AND run_id=$2 AND status='waiting_for_user'
            """, org_id, run_id)
        if tag.endswith(" 1"):
            # K1-5：恢复留痕（课件 CH4 事件名 run.resumed）；回到 queued 不是 running——
            # 必须重新走队列被领取，保住"同一时刻只有一个 worker 持有"
            await append_event(conn, org_id=org_id, run_id=run_id, kind="run.resumed",
                               payload={"actor": "approval"})



# --------------------------------------------------------------------------
# K1（课件 CH1）：六状态 · 协作式取消 · resume 四道检查 · trace 聚合
# --------------------------------------------------------------------------

RUN_STATUSES = ("queued", "running", "waiting_for_user", "succeeded", "failed", "cancelled")
TERMINAL_STATUSES = frozenset({"succeeded", "failed", "cancelled"})


def new_resume_token() -> str:
    """resume 的凭证（课件 CH1 §5.4）：run 进 waiting_for_user 时生成，用一次即清。"""
    return secrets.token_urlsafe(24)


async def request_cancel(db: Database, *, org_id: str, run_id: str, reason: str,
                         actor: str = "api") -> str | None:
    """协作式取消（课件 CH1 §5.2 / CH3 §11，27 K1-4）。返回：

        None          run 不在这个 org（API 按 404 处理，防枚举）
        "terminal"    已是终态 ⇒ API 409 RUN_ALREADY_TERMINAL
        "cancelled"   queued / waiting_for_user：没有 worker 在跑，API 直接迁 cancelled
        "requested"   running：只写 cancel_requested_at + run.cancel_requested 事件；
                      由 worker 在安全点自己迁（ActionGate 入口是第一个安全点）

    ⛔ `cancel_requested` 永不进状态枚举——它是意图不是状态（H104）。
    """
    async with db.tx() as conn:
        row = await conn.fetchrow(
            "SELECT status FROM agent_runs WHERE org_id=$1 AND run_id=$2 FOR UPDATE",
            org_id, run_id)
        if row is None:
            return None
        st = row["status"]
        if st in TERMINAL_STATUSES:
            return "terminal"
        if st == "running":
            await conn.execute(
                "UPDATE agent_runs SET cancel_requested_at = COALESCE(cancel_requested_at, now()) "
                " WHERE org_id=$1 AND run_id=$2", org_id, run_id)
            await append_event(conn, org_id=org_id, run_id=run_id, kind="run.cancel_requested",
                               payload={"reason": reason, "actor": actor})
            return "requested"
        await conn.execute(
            """
            UPDATE agent_runs SET status='cancelled', error=$3, ended_at=now(),
                   lease_owner=NULL, lease_expires_at=NULL, resume_token=NULL,
                   cancel_requested_at = COALESCE(cancel_requested_at, now())
             WHERE org_id=$1 AND run_id=$2
            """, org_id, run_id, f"cancelled_by_{actor}")
        await append_event(conn, org_id=org_id, run_id=run_id, kind=_TERMINAL_EVENT["cancelled"],
                           payload={"reason": reason, "actor": actor, "from": st})
        await conn.execute(
            "INSERT INTO audit_logs (run_id, org_id, action, param_source, detail) "
            "VALUES ($1,$2,'run.cancelled','user',$3)",
            run_id, org_id, {"reason": reason, "from": st, "actor": actor})
        return "cancelled"


async def cancel_requested(db: Database, *, org_id: str, run_id: str) -> bool:
    """安全点读取消意图（ActionGate 入口调用）。"""
    async with db.tx() as conn:
        return bool(await conn.fetchval(
            "SELECT cancel_requested_at IS NOT NULL FROM agent_runs WHERE org_id=$1 AND run_id=$2",
            org_id, run_id))


async def resume_run(db: Database, *, org_id: str, run_id: str, resume_token: str,
                     input: dict | None = None) -> str | None:
    """resume 四道检查（课件 CH1 §5.4，27 K1-5）：404 → 409 状态 → token → 回 queued。

    返回 None / "terminal" / "not_waiting" / "bad_token" / "resumed"。
    ⛔ 回到 queued 不是 running：必须重新走队列被领取。
    ⚠️ input 只落进 run.resumed 事件（K5-2 让 loop 消费它），K1 不假装它已进 prompt。
    """
    async with db.tx() as conn:
        row = await conn.fetchrow(
            "SELECT status, resume_token FROM agent_runs WHERE org_id=$1 AND run_id=$2 FOR UPDATE",
            org_id, run_id)
        if row is None:
            return None
        if row["status"] in TERMINAL_STATUSES:
            return "terminal"
        if row["status"] != "waiting_for_user":
            return "not_waiting"
        if not row["resume_token"] or not secrets.compare_digest(row["resume_token"], resume_token):
            return "bad_token"
        await conn.execute(
            """
            UPDATE agent_runs SET status='queued', lease_owner=NULL, lease_expires_at=NULL,
                   resume_token=NULL
             WHERE org_id=$1 AND run_id=$2
            """, org_id, run_id)
        await append_event(conn, org_id=org_id, run_id=run_id, kind="run.resumed",
                           payload={"actor": "api", "input": input or {}})
        return "resumed"


_TRACE_TABLES = ("run_events", "agent_steps", "model_calls", "tool_calls",
                 "checkpoints", "usage_records", "audit_logs", "approval_cases")


async def trace(db: Database, *, org_id: str, run_id: str) -> dict | None:
    """八表按 run 聚合（课件 CH8 §12 / 27 K1-8·K8-4）。每张子表查询**直接带 org_id**
    （子表已冗余 org_id，课件 H17 的写法废除）。"""
    async with db.tx() as conn:
        run = await conn.fetchrow(
            "SELECT * FROM agent_runs WHERE org_id=$1 AND run_id=$2", org_id, run_id)
        if run is None:
            return None
        out: dict = {"run": dict(run)}
        for t in _TRACE_TABLES:
            rows = await conn.fetch(
                f"SELECT * FROM {t} WHERE org_id=$1 AND run_id=$2 ORDER BY id", org_id, run_id)
            out[t] = [dict(r) for r in rows]
        return out

async def approved_action(db: Database, *, org_id: str, run_id: str) -> dict | None:
    """这条 run 有没有一个**已经被人裁决过**的动作。

    ★ 返回 `modified_params`（人改过的）优先于 `proposed_params`（agent 提的）——
    **人改了什么就执行什么**，否则"人工修正"这条飞轮回路只是记账，不影响世界。
    """
    async with db.tx() as conn:
        row = await conn.fetchrow(
            """
            SELECT case_ref, action_type, status, proposed_params, modified_params
              FROM approval_cases
             WHERE org_id=$1 AND run_id=$2 AND status IN ('approved','modified','rejected')
             ORDER BY reviewed_at DESC NULLS LAST, id DESC
             LIMIT 1
            """, org_id, run_id)
    if row is None:
        return None
    out = dict(row)
    out["params"] = out["modified_params"] or out["proposed_params"] or {}
    return out


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


async def _await_settled_prior(db: Database, org_id: str, key: str,
                               *, timeout_seconds: float = 5.0,
                               poll_seconds: float = 0.05):
    """查同一幂等键的上一次调用；如果它**还在执行中**，等它落定（有上限）。

    ★★★ 为什么需要这个：**"命中一条还在执行中的记录"是第三种情况**

    设计文档 §38 只定义了"已完成"的命中。并发重复投递时（压测场景①、队列重投），
    第二次命中的是一条**占了坑但 ok/result 还是 NULL** 的行。
    第一版直接把它当"原结果"返回 ⇒ 调用方拿到 `ok=None`：

        worker 判 `not written.ok` ⇒ run 记成 failed（error 还是 None）
        **而钱其实已经花出去了** —— 用户被告知失败，账单上却有这一笔。

    ⚠️ 这是**返回空比报错更毒**的又一例（同 flash-attn 反向恒 0 那条）：
    没有异常、没有日志、状态机看着也正常。

    ⇒ 处理办法两段：① 有界等待，让绝大多数并发重复投递拿到**真的原结果**；
    ② 等不到就**如实说"处理中"**（`ok=False` + 明确的 error），
    **绝不冒充成功，也绝不冒充失败**。
    """
    import asyncio as _asyncio
    import time as _time

    deadline = _time.monotonic() + timeout_seconds
    while True:
        async with db.tx() as conn:
            prior = await conn.fetchrow(
                """
                SELECT id, ok, result, error FROM tool_calls
                 WHERE org_id=$1 AND external_idempotency_key=$2 AND replayed_from IS NULL
                """, org_id, key)
        if prior is None or prior["ok"] is not None:
            return prior
        if _time.monotonic() >= deadline:
            return prior                     # 仍未落定 ⇒ 上面会返回"处理中"
        await _asyncio.sleep(poll_seconds)


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
    from asyncpg.exceptions import UniqueViolationError

    prior = None
    if external_idempotency_key is not None:
        prior = await _await_settled_prior(db, org_id, external_idempotency_key)
    if prior is None and external_idempotency_key is not None:
        # 课件 H01 的工具级形态："先查再插"有竞态——两个并发调用都查到"没有"，都去占坑，
        # 第二个撞 tool_calls_external_idem_uniq。救你的是约束不是查询：撞了就当命中，回到等原结果那条路。
        try:
            async with db.tx() as conn:
                row = await conn.fetchrow(
                    """
                    INSERT INTO tool_calls (run_id, org_id, step, tool, arguments,
                                            external_idempotency_key)
                    VALUES ($1,$2,$3,$4,$5,$6) RETURNING id
                    """, run_id, org_id, step, tool, arguments, external_idempotency_key)
                call_id = row["id"]
        except UniqueViolationError:
            prior = await _await_settled_prior(db, org_id, external_idempotency_key)
            assert prior is not None, "撞了唯一键却查不到原记录：索引与查询条件不一致？"
    if prior is not None:
        async with db.tx() as conn:
            # 命中 ⇒ **返回原结果，不重放**。同时记一条"这次被幂等挡下了"的痕迹，
            # 否则"到底重试了几次"在事后完全不可见。
            await conn.execute(
                """
                INSERT INTO tool_calls (run_id, org_id, step, tool, arguments,
                                        ok, result, error, replayed_from)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)
                """, run_id, org_id, step, tool, arguments,
                prior["ok"], prior["result"], prior["error"], prior["id"])
        # ok 仍是 NULL ⇒ 那一次至今没跑完（等超时了）。**如实上报"处理中"**，
        # 绝不冒充成功 —— 见 _await_settled_prior 的说明。
        if prior["ok"] is None:
            return ToolCallResult(
                ok=False, data=None, replayed=True, call_id=prior["id"],
                error="tool_call_in_progress: 同一幂等键的上一次调用仍在执行中，"
                      "本次**没有**重复执行；结果未知，不要当成失败处理")
        return ToolCallResult(ok=prior["ok"], data=prior["result"],
                              error=prior["error"], replayed=True,
                              call_id=prior["id"])

    # 占坑（独立事务，先提交）—— 带幂等键的已在上面占过坑，这里只剩只读工具（无键）
    if external_idempotency_key is None:
        async with db.tx() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO tool_calls (run_id, org_id, step, tool, arguments,
                                        external_idempotency_key)
                VALUES ($1,$2,$3,$4,$5,$6) RETURNING id
                """, run_id, org_id, step, tool, arguments, external_idempotency_key)
            call_id = row["id"]

    # ★ 计时从**执行**开始，不含占坑那次写库 —— §19 量的是"打平台花了多久"。
    import time as _time
    t0 = _time.perf_counter()
    ok, data, error = await execute()
    latency_ms = int((_time.perf_counter() - t0) * 1000)

    async with db.tx() as conn:
        await conn.execute(
            "UPDATE tool_calls SET ok=$2, result=$3, error=$4, latency_ms=$5 WHERE id=$1",
            call_id, ok, data, error, latency_ms)
    return ToolCallResult(ok=ok, data=data, error=error, replayed=False, call_id=call_id)
