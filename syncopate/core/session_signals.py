"""session.* 信令工具的注册（v15 契约，`25 §3.1`）。

三条终止性信令 + 一条非终止的 report。**handler 全是零副作用 ack** ——
它们不改变世界，只是把"我要等/我要问/我要拒/我要报数"变成一个**可被系统编排的动作**
（N4「行为即动作」）。终止语义由 runner/runtime 特判，不在 handler 里。

⚠️ 为什么必须注册进 ToolRegistry 而不是特判：
不注册的话 rollout 循环会把它判成 `tool_not_available`，于是
①它变成一条失败 action（污染 tool_errors 与步数）②模型在 prompt 里也看不到 schema。
—— 这正是本项目第一失效形状（机制在但没接上）的形状。
"""

from __future__ import annotations

from typing import Any

from syncopate.core.contract import REPORT_TOOL, SESSION_TOOL_SPECS, TERMINAL_SIGNALS
from syncopate.core.tool_registry import REGISTRY, ToolContext, ToolRegistry, ToolResult

SESSION_TOOL_NAMES = frozenset(TERMINAL_SIGNALS) | {REPORT_TOOL}


def register_session_tools(registry: ToolRegistry | None = None) -> None:
    """把信令族注册进表。幂等——重复调用不报错（域模块可能被 import 多次）。"""
    reg = registry or REGISTRY
    for spec in SESSION_TOOL_SPECS:
        fn = spec["function"]
        if reg.get(fn["name"]) is not None:
            continue

        # ⚠️ handler 的签名是 `(args: dict, ctx: ToolContext)` —— 与全部业务工具同一份约定
        #   （tool_registry 调的是 `spec.handler(arguments, ctx)`）。
        #   ⛔ 2026-08-30：初版写成 `(ctx, **kwargs)`，注册过、schema 也对，
        #     但**一次都没被真的执行过** ⇒ 直到 gold 回放才炸出 TypeError。
        #     「登记 ≠ 实现」的第 N 次：注册表是最像证据的东西。
        def _ack(args: dict[str, Any], ctx: ToolContext,
                 _name: str = fn["name"]) -> ToolResult:
            return ToolResult(ok=True, data={"acknowledged": True, "signal": _name,
                                             "arguments": dict(args)})

        reg.tool(name=fn["name"], description=fn["description"],
                 parameters=fn["parameters"], kind="read")(_ack)

    if reg.get(REPORT_TOOL) is None:
        def _report(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
            return ToolResult(ok=True, data={"acknowledged": True,
                                             "reported_fields": sorted(args)})

        reg.tool(name=REPORT_TOOL,
                 description="给出本轮结论里机器需要核对的结构化字段（非终止，报完继续收尾）",
                 parameters={"type": "object", "properties": {}, "additionalProperties": True},
                 kind="read")(_report)
