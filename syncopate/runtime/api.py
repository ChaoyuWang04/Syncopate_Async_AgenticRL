"""M9.3 · API 层：资源式接口 + org_id 注入 + response_model 白名单。

★★★ 三条不能妥协的（设计文档 §36）

1. **`org_id` 从鉴权注入，永不从请求体读。**
   多租户系统里"前端传 org_id"等于把越权做成了 feature。这里用 `Depends(current_org)`，
   路由函数**拿不到**请求体里的 org_id —— Pydantic 模型里根本没有这个字段。

2. **`response_model` 是白名单。** 内部字段（lease_owner / attempt / 别的 org 的东西）
   不能因为"顺手 return 了整行"就漏出去。白名单意味着**加字段要显式**，
   而不是"忘了删就泄漏"。

3. **写接口必须收 `Idempotency-Key`。** 用户点两次是常态不是异常，
   见 db.py 里三层幂等的第一层。

⚠️ 鉴权这里用的是最小实现（Bearer token → org 映射表）。真上线要换成真 OIDC/JWT，
但**注入的形状不能变** —— 换实现时只改 `current_org`，路由一行都不用动。
"""

from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path
from typing import Annotated, Any, AsyncIterator

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from syncopate.runtime.db import (Database, conversation_exists, create_conversation,
                                  create_run, resume_after_approval)

# --------------------------------------------------------------------------
# 鉴权：token → org。最小实现，形状是对的。
# --------------------------------------------------------------------------

# ⚠️ 只是开发期占位。真上线换 OIDC/JWT —— 但换的是这个函数，不是路由签名。
_DEV_TOKENS: dict[str, str] = {
    "dev-token-acme": "org_acme",
    "dev-token-globex": "org_globex",
}


async def current_org(authorization: Annotated[str | None, Header()] = None) -> str:
    """从 Authorization 头解析出 org_id。**这是 org_id 的唯一来源。**"""
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "缺少 Bearer token")
    org = _DEV_TOKENS.get(authorization.split(" ", 1)[1].strip())
    if org is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "token 无效")
    return org


OrgId = Annotated[str, Depends(current_org)]


def get_db(request: Request) -> Database:
    return request.app.state.db


# ⚠️ **必须定义在模块级**，不能放进 create_app 里当局部变量。
# `from __future__ import annotations` 让所有注解变成字符串，FastAPI 解析时
# 在**模块作用域**里查名字 —— 查不到就退化成"这是个查询参数"，
# 表现是所有接口一律 422 `missing query param: db`，和依赖注入八竿子打不着。
DB = Annotated[Database, Depends(get_db)]


# --------------------------------------------------------------------------
# 请求 / 响应模型
# --------------------------------------------------------------------------


class RunCreate(BaseModel):
    """⚠️ **刻意没有 org_id 字段。** 前端传了也进不来 —— 这是第 1 条纪律的物理保证，
    不是靠"记得别读它"。"""

    user_message: str = Field(min_length=1, max_length=4000)
    intent: str | None = None
    automation_tier: str | None = Field(default=None, pattern="^[ABCD]$")


class RunView(BaseModel):
    """★ 白名单：只有这些字段会出去。lease_owner / attempt / error 细节留在内部。"""

    run_id: str
    status: str
    intent: str | None = None
    automation_tier: str | None = None
    requires_approval: bool = False
    created: bool = Field(default=True, description="False = 幂等命中，返回的是原来那次")


class ApprovalView(BaseModel):
    case_ref: str
    run_id: str
    action_type: str
    proposed_params: dict[str, Any]
    rationale: str | None = None
    trigger_reason: str | None = None
    status: str


class UsageView(BaseModel):
    tokens_in: int
    tokens_out: int
    cost_micros: int
    runs: int


class ApprovalDecision(BaseModel):
    decision: str = Field(pattern="^(approved|rejected|modified)$")
    modified_params: dict[str, Any] | None = None
    reviewer_id: str


class ConversationCreate(BaseModel):
    title: str | None = Field(default=None, max_length=120)


class ConversationView(BaseModel):
    conversation_id: str
    title: str | None = None
    runs: int = 0
    last_activity: str | None = None


class MessageCreate(BaseModel):
    """会话里发一条消息 = 建一个挂在会话上的 run。

    ★ `automation_tier` 在这个**新入口必填**（09 §4.6.4 的缺口在此关掉：
      旧 /runs 不动以免破坏现有调用方，新入口不留"没声明"这种状态）。
    """

    user_message: str = Field(min_length=1, max_length=4000)
    intent: str | None = None
    automation_tier: str = Field(pattern="^[ABCD]$")


class MessageView(BaseModel):
    """会话历史里的一条：一问 +（跑完后的）一答。前端按它回放历史。"""

    run_id: str
    user_message: str
    status: str
    intent: str | None = None
    automation_tier: str | None = None
    result: dict[str, Any] | None = None
    error: str | None = None
    created_at: str


# --------------------------------------------------------------------------
# 应用
# --------------------------------------------------------------------------


def create_app(db: Database | None = None) -> FastAPI:
    app = FastAPI(title="Syncopate Runtime", version="0.1.0")
    app.state.db = db

    @app.on_event("startup")
    async def _startup() -> None:
        if app.state.db is None:
            app.state.db = Database()
            await app.state.db.connect()

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        # ⚠️ 关完必须**清掉引用**。只 close 不清空的话，同一个 app 再次 startup 时
        # `if app.state.db is None` 判成"已有"，于是攥着一个**已关闭的池**继续跑，
        # 表现是 "连接池没建"。生产里 uvicorn --reload 和测试里都会重进 lifespan。
        if app.state.db is not None:
            await app.state.db.close()
            app.state.db = None

    # ---- 控制台页面（静态壳，不鉴权 —— 数据接口各自带 org 鉴权；
    #      对公网的门在 Caddy 边界的 token，见 09 §0）----
    @app.get("/ui", include_in_schema=False)
    async def ui() -> FileResponse:
        return FileResponse(Path(__file__).parent / "ui.html", media_type="text/html")

    # ---- 健康检查（不鉴权，压测场景②要靠它判断服务活没活）----
    @app.get("/healthz")
    async def healthz(db: DB) -> dict[str, str]:
        try:
            async with db.tx() as conn:
                await conn.fetchval("SELECT 1")
            return {"status": "ok", "db": "up"}
        except Exception:
            # ★ 不抛 500：健康检查的作用是**如实报告**，不是自己也挂掉。
            return {"status": "degraded", "db": "down"}

    # ---- 资源式：POST /runs ----
    @app.post("/runs", response_model=RunView, status_code=status.HTTP_201_CREATED)
    async def create_run_endpoint(
        body: RunCreate,
        org_id: OrgId,
        db: DB,
        idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key")] = None,
    ) -> RunView:
        """★ 注意签名：`org_id` 来自 `Depends`，`body` 里没有它。

        ★ 幂等命中时返回 **201 + created=False**，而不是 409。
        重复提交是正常现象，报错会让客户端以为出事了从而**再重试一次**。
        """
        handle = await create_run(
            db, org_id=org_id, run_id=f"run_{uuid.uuid4().hex[:12]}",
            user_message=body.user_message, idempotency_key=idempotency_key,
            intent=body.intent, automation_tier=body.automation_tier)
        async with db.tx() as conn:
            row = await conn.fetchrow(
                "SELECT run_id,status,intent,automation_tier,requires_approval "
                "FROM agent_runs WHERE org_id=$1 AND run_id=$2", org_id, handle.run_id)
        return RunView(**dict(row), created=handle.created)

    # ---- GET /runs/{run_id} ----
    @app.get("/runs/{run_id}", response_model=RunView)
    async def get_run(run_id: str, org_id: OrgId, db: DB) -> RunView:
        """★★ `WHERE org_id=$1` 是**越权防线**，不是过滤器。

        少了它，知道别人 run_id 的人就能读到别人的数据。所以这里不能写成
        "先查出来再判断 org 对不对" —— 那样一次 SQL 注入或一个 typo 就穿了。
        """
        async with db.tx() as conn:
            row = await conn.fetchrow(
                "SELECT run_id,status,intent,automation_tier,requires_approval "
                "FROM agent_runs WHERE org_id=$1 AND run_id=$2", org_id, run_id)
        if row is None:
            # ★ 别人的 run 和不存在的 run 返回**同一个** 404 ——
            # 区分开就成了一个探测别人 run_id 是否存在的接口。
            raise HTTPException(status.HTTP_404_NOT_FOUND, "run 不存在")
        return RunView(**dict(row))

    # ---- 审批网关：GET / POST ----
    @app.get("/approvals", response_model=list[ApprovalView])
    async def list_approvals(org_id: OrgId, db: DB,
                             status_filter: str = "pending") -> list[ApprovalView]:
        async with db.tx() as conn:
            rows = await conn.fetch(
                "SELECT case_ref,run_id,action_type,proposed_params,rationale,"
                "trigger_reason,status FROM approval_cases "
                "WHERE org_id=$1 AND status=$2 ORDER BY created_at", org_id, status_filter)
        return [ApprovalView(**dict(r)) for r in rows]

    @app.post("/approvals/{case_ref}", response_model=ApprovalView)
    async def decide_approval(case_ref: str, body: ApprovalDecision,
                              org_id: OrgId, db: DB) -> ApprovalView:
        """★ `modified_params` 是**飞轮回路 2 的燃料** —— 人改了什么必须落库（§37）。"""
        async with db.tx() as conn:
            row = await conn.fetchrow(
                """
                UPDATE approval_cases
                   SET status=$3, reviewer_id=$4, reviewed_at=now(), modified_params=$5
                 WHERE org_id=$1 AND case_ref=$2 AND status='pending'
                RETURNING case_ref,run_id,action_type,proposed_params,rationale,
                          trigger_reason,status
                """, org_id, case_ref, body.decision, body.reviewer_id, body.modified_params)
        if row is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "审批单不存在或已处理")
        # ★★ 裁决完必须把 run 放回队列 —— 否则它永远停在 waiting_for_user。
        # 「暂停就必须能恢复」：网关的输出是暂停不是拒绝，只做前半截等于永久卡死。
        await resume_after_approval(db, org_id=org_id, run_id=row["run_id"])
        return ApprovalView(**dict(row))

    # ---- F-1 · 会话门面（chatbox 壳的载体；run 语义原样不动）--------------

    @app.post("/conversations", response_model=ConversationView,
              status_code=status.HTTP_201_CREATED)
    async def create_conversation_endpoint(body: ConversationCreate, org_id: OrgId,
                                           db: DB) -> ConversationView:
        cid = f"conv_{uuid.uuid4().hex[:12]}"
        await create_conversation(db, org_id=org_id, conversation_id=cid,
                                  title=body.title)
        return ConversationView(conversation_id=cid, title=body.title)

    @app.get("/conversations", response_model=list[ConversationView])
    async def list_conversations(org_id: OrgId, db: DB) -> list[ConversationView]:
        async with db.tx() as conn:
            rows = await conn.fetch(
                """
                SELECT c.conversation_id, c.title,
                       count(r.run_id) AS runs,
                       to_char(max(coalesce(r.created_at, c.created_at)),
                               'YYYY-MM-DD"T"HH24:MI:SSZ') AS last_activity
                FROM conversations c
                LEFT JOIN agent_runs r ON r.org_id = c.org_id
                     AND r.conversation_id = c.conversation_id
                WHERE c.org_id = $1
                GROUP BY c.conversation_id, c.title, c.created_at
                ORDER BY max(coalesce(r.created_at, c.created_at)) DESC
                """, org_id)
        return [ConversationView(**dict(r)) for r in rows]

    @app.get("/conversations/{cid}/messages", response_model=list[MessageView])
    async def conversation_messages(cid: str, org_id: OrgId,
                                    db: DB) -> list[MessageView]:
        """会话历史：runs 按时间回放（前端渲染完历史后，对最新 run 开 SSE）。"""
        if not await conversation_exists(db, org_id=org_id, conversation_id=cid):
            raise HTTPException(status.HTTP_404_NOT_FOUND, "会话不存在")
        async with db.tx() as conn:
            rows = await conn.fetch(
                "SELECT run_id, user_message, status, intent, automation_tier, "
                "       result, error, "
                "       to_char(created_at, 'YYYY-MM-DD\"T\"HH24:MI:SSZ') AS created_at "
                "FROM agent_runs WHERE org_id=$1 AND conversation_id=$2 "
                "ORDER BY created_at", org_id, cid)
        out = []
        for r in rows:
            d = dict(r)
            if isinstance(d.get("result"), str):
                d["result"] = json.loads(d["result"])
            out.append(MessageView(**d))
        return out

    @app.post("/conversations/{cid}/messages", response_model=RunView,
              status_code=status.HTTP_201_CREATED)
    async def post_message(
        cid: str, body: MessageCreate, org_id: OrgId, db: DB,
        idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key")] = None,
    ) -> RunView:
        """一条消息 = 一个 run。幂等/审批/事件全部沿用 run 的既有语义。"""
        if not await conversation_exists(db, org_id=org_id, conversation_id=cid):
            raise HTTPException(status.HTTP_404_NOT_FOUND, "会话不存在")
        handle = await create_run(
            db, org_id=org_id, run_id=f"run_{uuid.uuid4().hex[:12]}",
            user_message=body.user_message, idempotency_key=idempotency_key,
            intent=body.intent, automation_tier=body.automation_tier,
            conversation_id=cid)
        async with db.tx() as conn:
            row = await conn.fetchrow(
                "SELECT run_id,status,intent,automation_tier,requires_approval "
                "FROM agent_runs WHERE org_id=$1 AND run_id=$2", org_id, handle.run_id)
        return RunView(**dict(row), created=handle.created)

    # ---- M9.6 · SSE：事件流 + 断线补发 ----------------------------------

    # 终态事件。收到它就关流 —— 否则客户端会**永远挂着等下一条**，
    # 而服务端也永远留着一个连接（压测场景①：连接数才是先撑爆的东西）。
    # ⚠️ 与 db._TERMINAL_EVENT 必须一致：库里翻终态 = 必发其中之一（同一事务，
    #   2026-08-20 冒烟抓到 cancelled 各路径不发事件 ⇒ SSE 挂死，已改成结构保证）。
    TERMINAL = {"run.succeeded", "run.failed", "run.cancelled", "run.waiting_for_user"}

    async def _event_stream(db: Database, org_id: str, run_id: str,
                            after_seq: int, request: Request) -> AsyncIterator[str]:
        last = after_seq
        idle = 0.0
        while True:
            # ★ 客户端断开就停 —— 不检查的话，浏览器关了标签页我们还在轮询数据库。
            if await request.is_disconnected():
                return
            async with db.tx() as conn:
                rows = await conn.fetch(
                    "SELECT seq, kind, payload FROM run_events "
                    "WHERE org_id=$1 AND run_id=$2 AND seq>$3 ORDER BY seq",
                    org_id, run_id, last)
            for row in rows:
                last = row["seq"]
                # ★ `id:` 就是客户端下次带回来的 Last-Event-ID。
                yield (f"id: {row['seq']}\n"
                       f"event: {row['kind']}\n"
                       f"data: {json.dumps(row['payload'], ensure_ascii=False)}\n\n")
                if row["kind"] in TERMINAL:
                    return
            if rows:
                idle = 0.0
            else:
                idle += 0.25
                # 心跳：注释行，客户端会忽略，但能让中间的代理不掐断空闲连接。
                if idle >= 5.0:
                    idle = 0.0
                    yield ": keepalive\n\n"
                await asyncio.sleep(0.25)

    @app.get("/runs/{run_id}/events")
    async def stream_events(
        run_id: str, org_id: OrgId, db: DB, request: Request,
        last_event_id: Annotated[str | None, Header(alias="Last-Event-ID")] = None,
    ) -> StreamingResponse:
        """★★ 断线补发：客户端带 `Last-Event-ID` 回来，我们从那之后接着推。

        这就是 `run_events.seq` 必须**由数据库分配且连续**的原因（见 worker.py）——
        内存计数器在 worker 重启后会从头开始，客户端就永远收不到中间那段。

        ⚠️ 越权同样在 SQL 里挡（`WHERE org_id=$1`），不是在应用层判断。
        """
        async with db.tx() as conn:
            exists = await conn.fetchval(
                "SELECT 1 FROM agent_runs WHERE org_id=$1 AND run_id=$2", org_id, run_id)
        if not exists:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "run 不存在")
        try:
            after = int(last_event_id) if last_event_id else 0
        except ValueError:
            after = 0          # 客户端给了脏值 ⇒ 从头推，总比 500 好
        return StreamingResponse(
            _event_stream(db, org_id, run_id, after, request),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    # ---- M9.6 · 观测：成本与用量 ----------------------------------------

    @app.get("/usage", response_model=UsageView)
    async def usage(org_id: OrgId, db: DB) -> UsageView:
        """★ 按 org 的当日用量。压测场景⑤（单 org 刷爆预算）要靠它判断降级对不对。"""
        async with db.tx() as conn:
            row = await conn.fetchrow(
                "SELECT COALESCE(sum(tokens_in),0) AS tin, COALESCE(sum(tokens_out),0) AS tout, "
                "COALESCE(sum(cost_micros),0) AS cost, count(DISTINCT run_id) AS runs "
                "FROM usage_records WHERE org_id=$1 AND day=CURRENT_DATE", org_id)
        return UsageView(tokens_in=row["tin"], tokens_out=row["tout"],
                         cost_micros=row["cost"], runs=row["runs"])

    return app


app = create_app()
