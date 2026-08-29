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
from syncopate.core.tool_registry import REGISTRY, ToolContext, ToolRegistry

SESSION_TOOL_NAMES = frozenset(TERMINAL_SIGNALS) | {REPORT_TOOL}


def register_session_tools(registry: ToolRegistry | None = None) -> None:
    """把信令族注册进表。幂等——重复调用不报错（域模块可能被 import 多次）。"""
    reg = registry or REGISTRY
    for spec in SESSION_TOOL_SPECS:
        fn = spec["function"]
        if reg.get(fn["name"]) is not None:
            continue

        async def _ack(ctx: ToolContext, _name: str = fn["name"], **kwargs: Any) -> dict[str, Any]:
            return {"acknowledged": True, "signal": _name, "arguments": kwargs}

        reg.tool(name=fn["name"], description=fn["description"],
                 parameters=fn["parameters"], kind="read")(_ack)

    if reg.get(REPORT_TOOL) is None:
        async def _report(ctx: ToolContext, **kwargs: Any) -> dict[str, Any]:
            return {"acknowledged": True, "reported_fields": sorted(kwargs)}

        reg.tool(name=REPORT_TOOL,
                 description="给出本轮结论里机器需要核对的结构化字段（非终止，报完继续收尾）",
                 parameters={"type": "object", "properties": {}, "additionalProperties": True},
                 kind="read")(_report)
