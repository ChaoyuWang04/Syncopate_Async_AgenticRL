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
from dataclasses import dataclass
import uuid
from pathlib import Path
from typing import Literal, Annotated, Any, AsyncIterator

from fastapi import Cookie, Depends, FastAPI, Header, HTTPException, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from pydantic import BaseModel, Field

from syncopate.runtime.db import (Database, close_parked_clarify_runs, conversation_exists,
                                  create_conversation, create_run, new_conversation_id,
                                  new_run_id, request_cancel, resume_after_approval,
                                  resume_run, trace, InvalidRunTransition, TERMINAL_STATUSES)

# --------------------------------------------------------------------------
# 鉴权：token → org。最小实现，形状是对的。
# --------------------------------------------------------------------------

# ⚠️ 只是开发期占位。真上线换 OIDC/JWT —— 但换的是这个函数，不是路由签名。
# ★ org_demo 是**给真人用的**（网页控制台/常驻 worker 挂它）；acme/globex 归测试。
#   分开是结构性的：常驻 worker 消费 org_acme 会抢走测试的 run（C-1 同族，
#   2026-08-20 SSE 测试就是这么被污染的——回放里多出 worker 真跑出来的事件）。
_DEV_TOKENS: dict[str, str] = {
    "dev-token-acme": "org_acme",
    "dev-token-globex": "org_globex",
    "dev-token-demo": "org_demo",
    # K1-8：/trace 含完整 prompt/参数，org 校验挡不住"组织内普通成员看别人全文"（课件 H08）
    # ⇒ 独立角色位 trace，默认关。角色随 token 来，不随请求体来。
    "dev-token-acme-trace": "org_acme",
}
_DEV_ROLES: dict[str, frozenset[str]] = {"dev-token-acme-trace": frozenset({"member", "trace"})}
_DEFAULT_ROLES = frozenset({"member"})


@dataclass(frozen=True)
class _Principal:
    org: str
    roles: frozenset[str]


AUTH_COOKIE = "syncopate_token"


def _parse_token(authorization: str | None, cookie_token: str | None = None) -> str:
    """凭证两条通道，**同一张 token 表**：Authorization: Bearer（API 调用方）或同域 Cookie
    `syncopate_token`（浏览器 SSE，K7-4：前端 dist 由 API 同源挂载 /app ⇒ 同域 Cookie 零成本）。
    ⛔ 长效 key 永不进 URL（CI grep 守着 frontend/src）。"""
    token: str | None = None
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization.split(" ", 1)[1].strip()
    elif cookie_token:
        token = cookie_token.strip()
    if not token:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "缺少 Bearer token 或同域 Cookie")
    if token not in _DEV_TOKENS:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "token 无效")
    return token


async def current_org(authorization: Annotated[str | None, Header()] = None,
                      syncopate_token: Annotated[str | None, Cookie()] = None) -> str:
    """从 Authorization 头（或同域 Cookie）解析出 org_id。**这是 org_id 的唯一来源。**"""
    return _DEV_TOKENS[_parse_token(authorization, syncopate_token)]


async def current_principal(authorization: Annotated[str | None, Header()] = None,
                            syncopate_token: Annotated[str | None, Cookie()] = None) -> _Principal:
    token = _parse_token(authorization, syncopate_token)
    return _Principal(org=_DEV_TOKENS[token], roles=_DEV_ROLES.get(token, _DEFAULT_ROLES))


OrgId = Annotated[str, Depends(current_org)]
Principal = Annotated[_Principal, Depends(current_principal)]


# --------------------------------------------------------------------------
# K1-7 双层错误码：统一信封 {error:{code,message,request_id}}
#   code 给前端代码判断（稳定，进注册表）；message 给人看（可改）；
#   request_id 让用户报错时能一句话定位到日志。
# --------------------------------------------------------------------------

ERROR_CODES = frozenset({
    "UNAUTHORIZED", "FORBIDDEN", "NOT_FOUND", "CONFLICT", "VALIDATION_ERROR", "INTERNAL_ERROR",
    "IDEMPOTENCY_CONFLICT", "IDEMPOTENCY_KEY_TOO_SHORT", "INVALID_RUN_INPUT",
    "RUN_ALREADY_TERMINAL", "RUN_NOT_WAITING_FOR_USER", "INVALID_RESUME_TOKEN",
    "TRACE_FORBIDDEN", "INVALID_RUN_TRANSITION", "RUN_NOT_TERMINAL",
})
_STATUS_DEFAULT_CODE = {401: "UNAUTHORIZED", 403: "FORBIDDEN", 404: "NOT_FOUND",
                        409: "CONFLICT", 422: "VALIDATION_ERROR"}


class ApiError(HTTPException):
    """带业务码的 HTTP 错误。⛔ code 必须在 ERROR_CODES 注册，否则前端没法稳定判断。"""

    def __init__(self, status_code: int, code: str, message: str) -> None:
        if code not in ERROR_CODES:
            raise ValueError(f"未注册的错误码 {code!r}")
        super().__init__(status_code, detail={"code": code, "message": message})


def _request_id(request: Request) -> str:
    return request.headers.get("X-Request-ID") or uuid.uuid4().hex[:16]


def _error_body(request: Request, status_code: int, detail: Any) -> dict[str, Any]:
    if isinstance(detail, dict) and "code" in detail:
        code, message = detail["code"], str(detail.get("message", ""))
    else:
        code, message = _STATUS_DEFAULT_CODE.get(status_code, "INTERNAL_ERROR"), str(detail)
    return {"error": {"code": code, "message": message, "request_id": _request_id(request)}}


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
    # K1-1：run_type 决定 input 按哪个子 schema 校验（课件 H06："格式对但字段缺"不许漏到 worker）。
    # 当前只有 chat（一条消息 = 一个 run）；新类型 = 在 RUN_INPUT_MODELS 加一行。
    run_type: Literal["chat"] = "chat"


class _ChatInput(BaseModel):
    user_message: str = Field(min_length=1, max_length=4000)


RUN_INPUT_MODELS: dict[str, type[BaseModel]] = {"chat": _ChatInput}


def _validate_run_input(run_type: str, body: dict[str, Any]) -> None:
    model = RUN_INPUT_MODELS.get(run_type)
    if model is None:
        raise ApiError(422, "INVALID_RUN_INPUT", f"未知 run_type {run_type!r}")
    try:
        model.model_validate(body)
    except Exception as exc:  # pydantic.ValidationError
        raise ApiError(422, "INVALID_RUN_INPUT", str(exc)[:300]) from exc


class RunView(BaseModel):
    """★ 白名单：只有这些字段会出去。lease_owner / attempts / error 细节留在内部。"""

    run_id: str
    status: str
    intent: str | None = None
    automation_tier: str | None = None
    requires_approval: bool = False
    created: bool = Field(default=True, description="False = 幂等命中，返回的是原来那次")
    run_type: str = "chat"
    cancel_requested: bool = False              # 意图，不是状态（K1-4）
    resume_token: str | None = None             # 只在 waiting_for_user 时给出（K1-5）


class CancelRunRequest(BaseModel):
    reason: str = Field(default="", max_length=500)


class RerunRequest(BaseModel):
    """rerun = 新建 run 串上 parent（课件 §5.4：终态后只能新建，不能把旧 run 改回 running）。"""

    reason: str = Field(min_length=1, max_length=500)
    user_message: str | None = Field(default=None, min_length=1, max_length=4000)


class ResumeRunRequest(BaseModel):
    """resume 是"带着新信息继续"，不只是继续（27 K1-1）。input 落进 run.resumed 事件。"""

    resume_token: str = Field(min_length=8)
    input: dict[str, Any] | None = None


_RUN_VIEW_SQL = (
    "SELECT run_id,status,intent,automation_tier,requires_approval,run_type,"
    " cancel_requested_at IS NOT NULL AS cancel_requested,"
    " CASE WHEN status='waiting_for_user' THEN resume_token END AS resume_token"
    " FROM agent_runs WHERE org_id=$1 AND run_id=$2")


async def _run_view(db: Database, org_id: str, run_id: str, *, created: bool = True) -> RunView | None:
    async with db.tx() as conn:
        row = await conn.fetchrow(_RUN_VIEW_SQL, org_id, run_id)
    return None if row is None else RunView(**dict(row), created=created)


def _check_idempotency_key(key: str | None) -> None:
    # 课件 H05：key 由客户端在点击那一刻生成，粒度「实体:动作:v版本」；min_length=8 挡住 "1"、"abc" 这类
    if key is not None and len(key) < 8:
        raise ApiError(422, "IDEMPOTENCY_KEY_TOO_SHORT", "Idempotency-Key 至少 8 个字符")


class ApprovalView(BaseModel):
    case_ref: str
    run_id: str
    action_type: str
    proposed_params: dict[str, Any]
    rationale: str | None = None
    trigger_reason: str | None = None
    status: str
    # ★ `evidence` 一直在落库却从没出过接口（"有生产者没消费者"）——
    #   而设计原话是「人看的是证据不是结论」。档位判定理由就在里面
    #   （tier / tier_reason），审批卡要显示"为什么判成这一档"。
    evidence: dict[str, Any] | None = None


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
    # dev mode 模型选择（Chaoyu 08-29）：会话创建即锁定，不提供修改接口
    model: Literal["rl", "sft", "base"] = "rl"


class ConversationView(BaseModel):
    conversation_id: str
    title: str | None = None
    runs: int = 0
    last_activity: str | None = None
    model: str = "rl"



class MessageCreate(BaseModel):
    """会话里发一条消息 = 建一个挂在会话上的 run。

    ⚠️ 2026-08-20 起 `automation_tier` 与 `intent` **都不再需要调用方给**：

      intent            工具菜单改成全量 30 个，模型自己选（`decider.FULL_MENU_MODE`）
                        ⇒ 这个字段只剩标注/统计用途
      automation_tier   档位由**动作本身**推导（`tier_policy.derive_tier`）
                        ⇒ 给了也只能**往严了拉**（org 说"这条只许看" = D），
                          放松是结构上做不到的（`more_cautious`）

    ★ 09 §4.6.4 那条"应当必填"的缺口就此**换了个解法关闭**：不是逼人填，
      而是让这个值有一个**不依赖任何人填写**的来源。
    """

    user_message: str = Field(min_length=1, max_length=4000)
    intent: str | None = None
    automation_tier: str | None = Field(default=None, pattern="^[ABCD]$")


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
    # B-5 S3 门铃：{f"{org}|{run}": {asyncio.Event,...}}；listener 断了 SSE 靠
    # 2s 兜底轮询照常活（[sse-bell] 判据行），铃只是把事件延迟从 ≤2s 压到 ~0。
    app.state.sse_waiters = {}
    app.state.sse_bell_task = None

    # K1-7：所有非 2xx 走同一个信封（含 Starlette 自己抛的 404/405 与 422 校验错误）
    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code,
                            content=_error_body(request, exc.status_code, exc.detail),
                            headers=getattr(exc, "headers", None))

    @app.exception_handler(InvalidRunTransition)
    async def _bad_transition(request: Request, exc: InvalidRunTransition) -> JSONResponse:
        # K4-1：非法迁移 → 409，message 带 from/to（"cannot transition from succeeded to running"）
        return JSONResponse(status_code=409, content=_error_body(
            request, 409, {"code": "INVALID_RUN_TRANSITION", "message": str(exc)}))

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        msg = "; ".join(f"{'.'.join(str(x) for x in e.get('loc', ()))}: {e.get('msg')}"
                        for e in exc.errors()[:3])
        return JSONResponse(status_code=422, content=_error_body(
            request, 422, {"code": "VALIDATION_ERROR", "message": msg}))

    async def _sse_bell() -> None:
        import asyncpg as _apg

        from syncopate.runtime.db import DSN as _dsn
        while True:
            try:
                conn = await _apg.connect(_dsn)

                def _cb(_c, _pid, _ch, payload) -> None:
                    for ev in tuple(app.state.sse_waiters.get(payload, ())):
                        ev.set()

                await conn.add_listener("run_events", _cb)
                print("[sse-bell] listener 就位", flush=True)
                while not conn.is_closed():
                    await asyncio.sleep(5)
                print("[sse-bell] listener 连接关闭，重建（兜底轮询在扛）", flush=True)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                print(f"[sse-bell] listener 异常 {exc!r}，5s 重连（兜底轮询在扛）", flush=True)
                await asyncio.sleep(5)

    @app.on_event("startup")
    async def _startup() -> None:
        if app.state.db is None:
            app.state.db = Database()
            # B-5 S1：池容量 env 可配（默认 10 不变）。S0 实测 C=96 时借连接等待
            # 占 e2e 24-29%（每单 1.9s）——10 条连接伺候不了高并发。
            import os
            await app.state.db.connect(
                max_size=int(os.environ.get("SYNCOPATE_API_DB_POOL", "10")))
        app.state.sse_bell_task = asyncio.create_task(_sse_bell())

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        if app.state.sse_bell_task is not None:
            app.state.sse_bell_task.cancel()
            app.state.sse_bell_task = None
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

    # ---- F-2/F-3 · chatbox 前端（assistant-ui 构建产物，挂 /app）----
    # ★ 条件挂载：dist 不存在（没构建过/CI 环境）时静默跳过，不影响 API 与测试。
    #   没有空闲外部端口 ⇒ 前端与 API 同源共用 8265 的 Caddy 边界（09 §0）。
    dist = Path(__file__).resolve().parents[2] / "frontend" / "dist"
    if dist.is_dir():
        from fastapi.staticfiles import StaticFiles
        app.mount("/app", StaticFiles(directory=dist, html=True), name="app")

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
        response: Response,
        idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key")] = None,
    ) -> RunView:
        """★ 注意签名：`org_id` 来自 `Depends`，`body` 里没有它。

        K1-3 幂等三态（课件 CH1 §4.2 / H01 / H04）：
            新 key                → 201 created=True
            同 key 同 input       → **200** created=False（分布式正常现象，不是错误）
            同 key 不同 input     → 409 IDEMPOTENCY_CONFLICT（第二把锁 input_hash）
        竞态由 UNIQUE 约束 + ON CONFLICT 兜住（db.create_run），不是"先查再插"。
        """
        _check_idempotency_key(idempotency_key)
        _validate_run_input(body.run_type, body.model_dump())
        handle = await create_run(
            db, org_id=org_id, run_id=new_run_id(),
            user_message=body.user_message, idempotency_key=idempotency_key,
            intent=body.intent, automation_tier=body.automation_tier,
            run_type=body.run_type)
        if not handle.created:
            if not handle.input_matches:
                raise ApiError(409, "IDEMPOTENCY_CONFLICT",
                               "同一个 Idempotency-Key 已用于不同的输入")
            response.status_code = status.HTTP_200_OK
        return await _run_view(db, org_id, handle.run_id, created=handle.created)

    # ---- GET /runs/{run_id} ----
    @app.get("/runs/{run_id}", response_model=RunView)
    async def get_run(run_id: str, org_id: OrgId, db: DB) -> RunView:
        """★★ `WHERE org_id=$1` 是**越权防线**，不是过滤器。

        少了它，知道别人 run_id 的人就能读到别人的数据。所以这里不能写成
        "先查出来再判断 org 对不对" —— 那样一次 SQL 注入或一个 typo 就穿了。
        """
        view = await _run_view(db, org_id, run_id)
        if view is None:
            # ★ 别人的 run 和不存在的 run 返回**同一个** 404 ——
            # 区分开就成了一个探测别人 run_id 是否存在的接口。
            raise HTTPException(status.HTTP_404_NOT_FOUND, "run 不存在")
        return view

    # ---- K1-4 · 协作式取消 ----
    @app.post("/runs/{run_id}/cancel", response_model=RunView)
    async def cancel_run(run_id: str, body: CancelRunRequest, org_id: OrgId, db: DB,
                         response: Response) -> RunView:
        """queued/waiting_for_user ⇒ 直接 cancelled（200）；running ⇒ 只登记意图（202），
        worker 在安全点自己迁；终态 ⇒ 409（不是 400、不是静默成功）。"""
        outcome = await request_cancel(db, org_id=org_id, run_id=run_id, reason=body.reason)
        if outcome is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "run 不存在")
        if outcome == "terminal":
            raise ApiError(409, "RUN_ALREADY_TERMINAL", "run 已结束，不能取消")
        if outcome == "requested":
            response.status_code = status.HTTP_202_ACCEPTED
        return await _run_view(db, org_id, run_id)

    # ---- K1-5 · resume 四道检查 ----
    @app.post("/runs/{run_id}/resume", response_model=RunView)
    async def resume_run_endpoint(run_id: str, body: ResumeRunRequest, org_id: OrgId,
                                  db: DB) -> RunView:
        outcome = await resume_run(db, org_id=org_id, run_id=run_id,
                                   resume_token=body.resume_token, input=body.input)
        if outcome is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "run 不存在")
        if outcome == "terminal":
            raise ApiError(409, "RUN_ALREADY_TERMINAL", "run 已结束，不能恢复")
        if outcome == "not_waiting":
            raise ApiError(409, "RUN_NOT_WAITING_FOR_USER", "run 不在等待用户，不能恢复")
        if outcome == "bad_token":
            raise ApiError(403, "INVALID_RESUME_TOKEN", "resume_token 不匹配")
        return await _run_view(db, org_id, run_id)

    # ---- K4-5 · rerun 通道：终态永不回队，只能新建并用 parent_run_id 串起来 ----
    @app.post("/runs/{run_id}/rerun", response_model=RunView, status_code=status.HTTP_201_CREATED)
    async def rerun(run_id: str, body: RerunRequest, org_id: OrgId, db: DB) -> RunView:
        async with db.tx() as conn:
            parent = await conn.fetchrow(
                "SELECT status, user_message, intent, automation_tier, conversation_id, run_type "
                "FROM agent_runs WHERE org_id=$1 AND run_id=$2", org_id, run_id)
        if parent is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "run 不存在")
        if parent["status"] not in TERMINAL_STATUSES:
            raise ApiError(409, "RUN_NOT_TERMINAL", "只有已结束的 run 才能 rerun（进行中的请用 cancel/resume）")
        handle = await create_run(
            db, org_id=org_id, run_id=new_run_id(),
            user_message=body.user_message or parent["user_message"] or "",
            intent=parent["intent"], automation_tier=parent["automation_tier"],
            conversation_id=parent["conversation_id"], run_type=parent["run_type"] or "chat",
            parent_run_id=run_id, rerun_reason=body.reason)
        return await _run_view(db, org_id, handle.run_id, created=True)

    # ---- K1-8 · trace（独立角色位；跨租户 404 先于角色 403）----
    @app.get("/runs/{run_id}/trace")
    async def get_trace(run_id: str, principal: Principal, db: DB) -> dict[str, Any]:
        data = await trace(db, org_id=principal.org, run_id=run_id)
        if data is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "run 不存在")
        if "trace" not in principal.roles:
            raise ApiError(403, "TRACE_FORBIDDEN", "需要 trace 角色")
        return json.loads(json.dumps(data, ensure_ascii=False, default=str))

    # ---- 审批网关：GET / POST ----
    @app.get("/approvals", response_model=list[ApprovalView])
    async def list_approvals(org_id: OrgId, db: DB,
                             status_filter: str = "pending") -> list[ApprovalView]:
        async with db.tx() as conn:
            rows = await conn.fetch(
                "SELECT case_ref,run_id,action_type,proposed_params,rationale,"
                "trigger_reason,status,evidence FROM approval_cases "
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
                          trigger_reason,status,evidence
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
        cid = new_conversation_id()
        await create_conversation(db, org_id=org_id, conversation_id=cid,
                                  title=body.title, model=body.model)
        return ConversationView(conversation_id=cid, title=body.title, model=body.model)

    @app.get("/conversations", response_model=list[ConversationView])
    async def list_conversations(org_id: OrgId, db: DB) -> list[ConversationView]:
        async with db.tx() as conn:
            rows = await conn.fetch(
                """
                SELECT c.conversation_id, c.title, c.model,
                       count(r.run_id) AS runs,
                       to_char(max(coalesce(r.created_at, c.created_at)),
                               'YYYY-MM-DD"T"HH24:MI:SSZ') AS last_activity
                FROM conversations c
                LEFT JOIN agent_runs r ON r.org_id = c.org_id
                     AND r.conversation_id = c.conversation_id
                WHERE c.org_id = $1
                GROUP BY c.conversation_id, c.title, c.model, c.created_at
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
        cid: str, body: MessageCreate, org_id: OrgId, db: DB, response: Response,
        idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key")] = None,
    ) -> RunView:
        """一条消息 = 一个 run。幂等/审批/事件全部沿用 run 的既有语义（含 K1-3 三态）。"""
        _check_idempotency_key(idempotency_key)
        if not await conversation_exists(db, org_id=org_id, conversation_id=cid):
            raise HTTPException(status.HTTP_404_NOT_FOUND, "会话不存在")
        # ★ 09-02（`26 §2.5` Ⓐ）：这条消息可能就是对上一轮 clarify 的回答 ⇒ 先把等补充的
        #   clarify 轮收尾为 succeeded，它才会作为历史进新 run 的 prompt。等审批的不动。
        await close_parked_clarify_runs(db, org_id=org_id, conversation_id=cid)
        handle = await create_run(
            db, org_id=org_id, run_id=new_run_id(),
            user_message=body.user_message, idempotency_key=idempotency_key,
            intent=body.intent, automation_tier=body.automation_tier,
            conversation_id=cid)
        if not handle.created:
            if not handle.input_matches:
                raise ApiError(409, "IDEMPOTENCY_CONFLICT", "同一个 Idempotency-Key 已用于不同的输入")
            response.status_code = status.HTTP_200_OK
        return await _run_view(db, org_id, handle.run_id, created=handle.created)

    # ---- M9.6 · SSE：事件流 + 断线补发 ----------------------------------

    # 终态事件。收到它就关流 —— 否则客户端会**永远挂着等下一条**，
    # 而服务端也永远留着一个连接（压测场景①：连接数才是先撑爆的东西）。
    # ⚠️ 与 db._TERMINAL_EVENT 必须一致：库里翻终态 = 必发其中之一（同一事务，
    #   2026-08-20 冒烟抓到 cancelled 各路径不发事件 ⇒ SSE 挂死，已改成结构保证）。
    TERMINAL = {"run.completed", "run.failed", "run.cancelled", "run.waiting_for_user"}   # K4-2：run.completed

    async def _event_stream(db: Database, org_id: str, run_id: str,
                            after_seq: int, request: Request) -> AsyncIterator[str]:
        # B-5 S3 门铃：先挂铃再查库（挂→查的顺序消掉"查完刚好来事件"的窗口；
        # 铃只是加速，2s 兜底轮询保住原语义——listener 死了 SSE 照常活）。
        last = after_seq
        idle = 0.0
        waiters: dict[str, set[asyncio.Event]] = request.app.state.sse_waiters
        wkey = f"{org_id}|{run_id}"
        # SSE 特殊行（课件 §6.1.1）：retry 告诉客户端重连间隔；放在注释块里，不算一条事件
        yield ": retry\nretry: 3000\n\n"
        from syncopate.runtime.event_layer import public_view
        while True:
            # ★ 客户端断开就停 —— 不检查的话，浏览器关了标签页我们还在轮询数据库。
            if await request.is_disconnected():
                return
            bell = asyncio.Event()
            waiters.setdefault(wkey, set()).add(bell)
            try:                                 # 每圈自挂自摘（挂→查消竞态窗）
                async with db.tx() as conn:
                    rows = await conn.fetch(
                        "SELECT seq, kind, payload FROM run_events "
                        "WHERE org_id=$1 AND run_id=$2 AND seq>$3 ORDER BY seq",
                        org_id, run_id, last)
                for row in rows:
                    last = row["seq"]
                    # K7-2 事件分层：internal/audit/未登记的不外推（游标照样推进：补发无空洞靠 seq，
                    # 客户端拿到的是"下一条 public"之前最后一个 seq）
                    data = public_view(row["kind"], row["payload"])
                    if data is not None:
                        # ★ `id:` 就是客户端下次带回来的 Last-Event-ID。
                        yield (f"id: {row['seq']}\n"
                               f"event: {row['kind']}\n"
                               f"data: {json.dumps(data, ensure_ascii=False)}\n\n")
                    if row["kind"] in TERMINAL:
                        return
                if rows:
                    idle = 0.0
                else:
                    try:
                        await asyncio.wait_for(bell.wait(), timeout=2.0)
                    except asyncio.TimeoutError:
                        pass                     # 兜底轮询（拔铃仍活）
                    idle += 2.0
                    # 心跳：注释行，客户端会忽略，但能让中间的代理不掐断空闲连接。
                    if idle >= 4.0:
                        idle = 0.0
                        yield ": keepalive\n\n"
            finally:
                s = waiters.get(wkey)
                if s is not None:
                    s.discard(bell)
                    if not s:
                        waiters.pop(wkey, None)

    @app.get("/runs/{run_id}/events")
    async def stream_events(
        run_id: str, org_id: OrgId, db: DB, request: Request,
        last_event_id: Annotated[str | None, Header(alias="Last-Event-ID")] = None,
        after: int | None = None,
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
            # K7-1 双路续传：query `after` 优先于 `Last-Event-ID`（课件：query > header）
            after = int(after) if after is not None else (int(last_event_id) if last_event_id else 0)
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
