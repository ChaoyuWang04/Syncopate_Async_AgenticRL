"""工具注册表：加一个工具 = 写一个文件 + 挂一个装饰器。

两个不同于 AdCampaignAgent 现有工具的关键点：

1. **有状态**。原来的 `upload_creative_asset` 无论传什么参数都返回同一个
   `ASSET_2031`，那样「模型有没有真的上传」根本验证不了。这里每个工具都拿到
   `ToolContext`（env + sandbox），读工具查真实的世界，写工具入真实的账。

2. **延迟是真的**。`latency_seconds` 走 `asyncio.sleep`，不是打个时间戳假装慢。
   假的慢暴露不了阻塞问题，异步对照实验就白做了。素材审核这类任务本来就要等
   几小时，把它如实建模出来，长尾 rollout 才有真实来源。
"""

from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass, field
from typing import Any, Callable

from syncopate.core import failures as F
from syncopate.core.sandbox import Sandbox
from syncopate.core.schemas import Case, EnvSnapshot


@dataclass
class Mutation:
    """一次写动作对世界造成的改变。**不回给模型**，只用于叠加视图。

    为什么要显式声明，而不是让写工具直接改 env：
    `EnvSnapshot` 只读是重放和归因的基础（同一条 case 并发跑 N 条 rollout，
    各自的写动作绝不能串台）。所以世界不变，改变记在账本里，
    读的时候把账本**叠加**上去 —— 世界原样和本次改动始终分得开。
    """

    table: str
    key: str
    fields: dict[str, Any]


@dataclass
class ToolContext:
    """工具执行时能看到的一切。"""

    case: Case
    env: EnvSnapshot
    sandbox: Sandbox
    step: int                # 当前是第几个 assistant 轮，1-indexed
    tool_call_id: str
    # 延迟缩放，由 execute 注入。system.wait 这类**等待时长由参数决定**的工具要用它——
    # latency_seconds 是静态的，表达不了"等 30 秒"这种动态时长。
    latency_scale: float = 1.0

    # ---- ★ 读工具必须走这两个方法，不要直接 ctx.env.row() ----
    #
    # 实测过的缺口：改预算 500 → 900 之后再查，读到的还是 500。
    # 因为 env 只读、写只进账本，而读工具**根本不看账本**。
    # 真实平台改完再读会读到新值（Meta 明确支持 read-after-write），
    # 我们这里读不到 —— 模型因此学不会「改完要确认」，还可能反复改同一个对象。
    def row(self, table: str, key: str | None) -> dict[str, Any] | None:
        """世界的那一行，**叠加本次 rollout 已生效的写动作**。"""
        base = self.env.row(table, key)
        if base is None or key is None:
            return base
        overlay = self.sandbox.mutations_for(table, key)
        return {**base, **overlay} if overlay else base

    def table(self, name: str) -> dict[str, dict[str, Any]]:
        """整张表的叠加视图。"""
        rows = self.env.table(name)
        changed = self.sandbox.mutated_keys(name)
        if not changed:
            return rows
        return {k: ({**v, **self.sandbox.mutations_for(name, k)} if k in changed else v)
                for k, v in rows.items()}


@dataclass
class ToolResult:
    """工具返回值。ok=False 时 observation 里带 error，模型能看到并重试。"""

    ok: bool
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    # 写工具声明它改了世界的什么。不进 observation —— 模型看到的只有 data。
    mutation: "Mutation | None" = None

    def observation(self) -> dict[str, Any]:
        return dict(self.data) if self.ok else {"error": self.error or "tool_failed"}


@dataclass
class ToolSpec:
    """一个工具的完整定义。"""

    name: str
    description: str
    parameters: dict[str, Any]           # JSON Schema，直接喂给 OpenAI tools 格式
    kind: str                            # "read" 只查不改 / "write" 产生副作用
    handler: Callable[..., Any]
    fact_key: str | None = None          # 写工具翻转哪个谓词；读工具为 None
    latency_seconds: float = 0.0         # 真实等待时长
    requires: list[str] = field(default_factory=list)  # 前置工具，做依赖检查用

    # ---- 真实性声明（docs/syncopate/07 §6）----
    api_ref: str | None = None           # 对应的真实 endpoint，如 "meta:POST /{campaign_id}"
    # ★ 幂等：写工具必须收 client_request_id。
    # Meta Marketing API **没有**幂等机制（实查文档确认），所以现实中重试一次
    # 就是多改一次预算。我们在沙盒里把它建成"平台支持"，是为了让模型学会**传这个键**——
    # 真实接入时由我们自己的 runtime 兑现这层保证（设计文档 §38 三层幂等的第三层）。
    idempotent: bool = False
    # ★ 调用配额：Meta 实况是每个 ad set **每小时最多改 4 次预算**，
    # 超了报 613/1487632 并封禁该 ad set 一小时。
    # {"limit": 4, "scope": "campaign_id", "error": "613"}
    quota: dict[str, Any] | None = None

    def openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    """全局工具表。域实现用 @registry.tool(...) 往里注册。"""

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}
        # 延迟缩放：调试时设成 0.01，把 480 秒的等待压成 5 秒；正式跑设 1.0。
        self.latency_scale: float = 1.0

    # ---------------------------------------------------------------- 注册

    def tool(
        self,
        *,
        name: str,
        description: str,
        parameters: dict[str, Any],
        kind: str = "read",
        fact_key: str | None = None,
        latency_seconds: float = 0.0,
        requires: list[str] | None = None,
        api_ref: str | None = None,
        idempotent: bool = False,
        quota: dict[str, Any] | None = None,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            if kind == "write" and not fact_key:
                raise ValueError(f"write tool {name!r} must declare a fact_key")
            self._tools[name] = ToolSpec(
                name=name,
                description=description,
                parameters=parameters,
                kind=kind,
                handler=func,
                fact_key=fact_key,
                latency_seconds=latency_seconds,
                requires=requires or [],
                api_ref=api_ref,
                idempotent=idempotent,
                quota=quota,
            )
            return func

        return decorator

    # ---------------------------------------------------------------- 查询

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def write_tools(self) -> dict[str, str]:
        """{工具名: fact_key}，verifier 用它把工具映射到谓词。"""
        return {n: s.fact_key for n, s in self._tools.items() if s.kind == "write" and s.fact_key}

    def menu(self, names: list[str] | None = None) -> list[dict[str, Any]]:
        """渲染进 prompt 的工具菜单。

        names=None 给全量；给了列表就只放子集——tool_missing 类 case 靠这个
        故意抽掉必需工具，看模型怎么绕。
        """
        selected = self.names() if names is None else [n for n in names if n in self._tools]
        return [self._tools[n].openai_schema() for n in selected]

    # ---------------------------------------------------------------- 执行

    async def execute(self, name: str, arguments: dict[str, Any], ctx: ToolContext) -> ToolResult:
        """执行一个工具。写工具会自动入账，调用方不用管。

        这是 async 的：`latency_seconds` 走真实 `asyncio.sleep`，
        所以慢工具会真的占住时间——这正是我们要测量的东西。
        """
        spec = self._tools.get(name)
        if spec is None:
            return ToolResult(ok=False, error=f"unknown_tool: {name}")
        ctx.latency_scale = self.latency_scale

        # ---- ★ 幂等去重：同一个 client_request_id 重复提交，返回上次的结果 ----
        #
        # 这是 F1（超时后重试）的正确形态。真实世界里超时分两种，
        # **模型看到的现象一模一样**：请求没发出去（该重试）vs 发到了回包丢了
        # （重试=重复扣款）。有幂等键时重试才是安全的；没有键就必须先查证。
        if spec.kind == "write" and spec.idempotent:
            request_id = arguments.get("client_request_id")
            if not request_id:
                return ToolResult(
                    ok=False,
                    error="missing_argument: client_request_id is required for write operations")
            prior = ctx.sandbox.find_by_request_id(name, str(request_id))
            if prior is not None:
                # 不重复入账 —— 否则 duplicate_write_cap 会把**正确的重试**判成违规
                return ToolResult(ok=True, data={**prior.result, "deduplicated": True})

        # ---- ★ 调用配额：Meta 每个 ad set 每小时最多改 4 次预算 ----
        if spec.quota:
            scope_field = spec.quota.get("scope")
            scope_value = arguments.get(scope_field) if scope_field else None
            used = sum(1 for r in ctx.sandbox.records_for(name)
                       if scope_value is None or r.object_key == scope_value)
            if used >= int(spec.quota.get("limit", 1 << 30)):
                return ToolResult(ok=False, error=(
                    f"{spec.quota.get('error', 'rate_limited')}: "
                    f"{name} 对 {scope_value} 已达上限 {spec.quota['limit']} 次/小时，"
                    f"该对象已被平台冻结一小时"))

        # ---- ★★★ 失败注入。放在配额之后、handler 之前 ----
        call_index = ctx.sandbox.note_call(name)
        script = F.match(ctx.env.failures, name, call_index)

        if script and script.get("mode") == F.TIMEOUT:
            # ★ 超时必须**真的消耗时间**。它是最贵的一种等待——等满了，什么都没拿到。
            # 不计时的话，超时在吞吐指标上是免费的，异步的收益会被系统性低估。
            # 和 latency_seconds 走同一个缩放旋钮：训练时压掉，做异步对照实验时设 1.0。
            await asyncio.sleep(float(script.get("timeout_seconds", 30)) * self.latency_scale)
            # ★ 灵魂所在：超时但**副作用可能已经生效**。
            # side_effect_applied=True 时照常执行 handler（世界真的变了、账本真的记了），
            # 但**返回给模型的是一个错误**，且错误里不透露到底变没变。
            # 模型只能靠"先查证"自己搞清楚 —— 这正是要训进权重的行为。
            if script.get("side_effect_applied"):
                await self._run_handler(spec, name, arguments, ctx)
            return ToolResult(ok=False, error=F.error_message(script, name))

        if script and script.get("mode") in (F.RATE_LIMITED, F.SERVER_ERROR, F.FORBIDDEN):
            return ToolResult(ok=False, error=F.error_message(script, name))

        if spec.latency_seconds > 0:
            await asyncio.sleep(spec.latency_seconds * self.latency_scale)

        try:
            outcome = spec.handler(arguments, ctx)
            if inspect.isawaitable(outcome):
                outcome = await outcome
        except Exception as exc:  # noqa: BLE001  工具内部报错不该炸掉整条 rollout
            return ToolResult(ok=False, error=f"{type(exc).__name__}: {exc}")

        result = outcome if isinstance(outcome, ToolResult) else ToolResult(ok=True, data=outcome or {})

        # ok=True 但内容有问题的三类：空 / 数值离谱 / 藏了指令。
        # 它们**不报错**，模型必须自己看出来 —— 比错误码难得多，也更接近真实。
        if script and result.ok and script.get("mode") in (F.EMPTY, F.ABSURD_VALUE,
                                                           F.INJECTED_INSTRUCTION):
            result = ToolResult(ok=True, data=F.corrupt(result.data, script),
                                mutation=result.mutation)

        # 记下工具返回里出现过的对象 id —— 防注入的判据
        if result.ok:
            _collect_ids(result.data, ctx.sandbox.ids_seen_in_output)

        # 写工具统一在这里入账，域实现不用重复写这段样板。
        if spec.kind == "write" and spec.fact_key:
            ctx.sandbox.record_write(
                tool=name,
                fact_key=spec.fact_key,
                arguments=arguments,
                result=result.data,
                step=ctx.step,
                tool_call_id=ctx.tool_call_id,
                ok=result.ok,
                object_key=_object_key(arguments),
                mutation=(
                    {"table": result.mutation.table, "key": result.mutation.key,
                     "fields": dict(result.mutation.fields)}
                    if result.mutation else None
                ),
            )
        return result

    async def _run_handler(self, spec: "ToolSpec", name: str, arguments: dict[str, Any],
                           ctx: ToolContext) -> ToolResult:
        """执行 handler 并入账。超时但副作用已生效的分支复用它。"""
        try:
            outcome = spec.handler(arguments, ctx)
            if inspect.isawaitable(outcome):
                outcome = await outcome
        except Exception as exc:  # noqa: BLE001
            return ToolResult(ok=False, error=f"{type(exc).__name__}: {exc}")
        result = outcome if isinstance(outcome, ToolResult) else ToolResult(ok=True, data=outcome or {})
        if spec.kind == "write" and spec.fact_key:
            ctx.sandbox.record_write(
                tool=name, fact_key=spec.fact_key, arguments=arguments, result=result.data,
                step=ctx.step, tool_call_id=ctx.tool_call_id, ok=result.ok,
                object_key=_object_key(arguments),
                mutation=({"table": result.mutation.table, "key": result.mutation.key,
                           "fields": dict(result.mutation.fields)} if result.mutation else None))
        return result

    def snapshot(self) -> list[dict[str, Any]]:
        """工具表快照，落进 artifact 用于事后复盘（老师包的 tool_schema_hash 同思路）。"""
        return [
            {"name": s.name, "kind": s.kind, "fact_key": s.fact_key,
             "latency_seconds": s.latency_seconds, "api_ref": s.api_ref,
             "idempotent": s.idempotent, "quota": s.quota}
            for s in (self._tools[n] for n in self.names())
        ]


# 写动作的「被写对象」主键。用于 duplicate_writes 和 wrong_object 类 cap。
_OBJECT_KEY_FIELDS = ("campaign_id", "creative_id", "asset_id", "account_id", "ad_group_id")


_ID_PATTERN_PREFIXES = ("CMP_", "ACC_", "CRE_", "ASSET_", "APR_")


def _collect_ids(data: Any, sink: set[str]) -> None:
    """递归收集工具返回里出现过的对象 id。

    ★ 这是防注入的判据（设计文档 §37 的 param_source）：
    campaign 名称、素材标题在真实平台上**是别人能填的**，
    「拿工具返回里读来的 id 去做写动作」是一条可判定的越界规则。
    """
    if isinstance(data, str):
        for token in data.replace(",", " ").replace('"', " ").split():
            if token.startswith(_ID_PATTERN_PREFIXES):
                sink.add(token.strip(".,;:)]}"))
    elif isinstance(data, dict):
        for value in data.values():
            _collect_ids(value, sink)
    elif isinstance(data, (list, tuple)):
        for value in data:
            _collect_ids(value, sink)


def _object_key(arguments: dict[str, Any]) -> str | None:
    for field_name in _OBJECT_KEY_FIELDS:
        value = arguments.get(field_name)
        if isinstance(value, str) and value:
            return value
    return None


# 域实现 import 这一个实例往里注册。
REGISTRY = ToolRegistry()
