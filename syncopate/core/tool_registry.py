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

from syncopate.core.sandbox import Sandbox
from syncopate.core.schemas import Case, EnvSnapshot


@dataclass
class ToolContext:
    """工具执行时能看到的一切。"""

    case: Case
    env: EnvSnapshot
    sandbox: Sandbox
    step: int                # 当前是第几个 assistant 轮，1-indexed
    tool_call_id: str


@dataclass
class ToolResult:
    """工具返回值。ok=False 时 observation 里带 error，模型能看到并重试。"""

    ok: bool
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

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

        if spec.latency_seconds > 0:
            await asyncio.sleep(spec.latency_seconds * self.latency_scale)

        try:
            outcome = spec.handler(arguments, ctx)
            if inspect.isawaitable(outcome):
                outcome = await outcome
        except Exception as exc:  # noqa: BLE001  工具内部报错不该炸掉整条 rollout
            return ToolResult(ok=False, error=f"{type(exc).__name__}: {exc}")

        result = outcome if isinstance(outcome, ToolResult) else ToolResult(ok=True, data=outcome or {})

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
            )
        return result

    def snapshot(self) -> list[dict[str, Any]]:
        """工具表快照，落进 artifact 用于事后复盘（老师包的 tool_schema_hash 同思路）。"""
        return [
            {"name": s.name, "kind": s.kind, "fact_key": s.fact_key, "latency_seconds": s.latency_seconds}
            for s in (self._tools[n] for n in self.names())
        ]


# 写动作的「被写对象」主键。用于 duplicate_writes 和 wrong_object 类 cap。
_OBJECT_KEY_FIELDS = ("campaign_id", "creative_id", "asset_id", "account_id", "ad_group_id")


def _object_key(arguments: dict[str, Any]) -> str | None:
    for field_name in _OBJECT_KEY_FIELDS:
        value = arguments.get(field_name)
        if isinstance(value, str) and value:
            return value
    return None


# 域实现 import 这一个实例往里注册。
REGISTRY = ToolRegistry()
