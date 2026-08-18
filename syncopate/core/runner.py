"""轨迹执行器：把一串工具调用跑成一条完整轨迹。

两个调用方共用它：

    1. gold 验证 —— 把 gold 的 actions 喂进来，确认它真能拿到高分
       （老师包里 gold 的 2737 条分数全部恰好 = 1.0，是**预烤**进文件的，
        不是跑出来的。我们要求 gold 必须真跑一遍才算数。）
    2. 单元测试 —— 手写一条「错误路径」验证 cap 会不会命中

真正跑模型的 AgentLoop 走的是同一套 `ToolContext` / `Trajectory` 结构，
区别只在于「下一步调什么」是模型决定还是脚本给定。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from syncopate.core.sandbox import Sandbox
from syncopate.core.schemas import CaseBundle
from syncopate.core.tool_registry import ToolContext, ToolRegistry
from syncopate.core.trajectory import Action, Observation, Trajectory


@dataclass
class PlannedCall:
    """脚本里的一次调用。step 留空则按顺序自增；显式给同一个 step 可模拟并行/违规。"""

    tool: str
    arguments: dict[str, Any]
    step: int | None = None


def plan(*calls: tuple[str, dict[str, Any]]) -> list[PlannedCall]:
    """便捷写法：plan(("a", {...}), ("b", {...}))"""
    return [PlannedCall(tool=tool, arguments=args) for tool, args in calls]


async def run_plan(
    bundle: CaseBundle,
    registry: ToolRegistry,
    calls: list[PlannedCall],
    *,
    final_answer: dict[str, Any] | None = None,
    behavior: str = "tool_call",
    rollout_id: str = "r0",
    run_id: str = "local",
) -> tuple[Trajectory, Sandbox]:
    """按计划执行，返回 (轨迹, 沙盒)。

    namespace_id 是 run:case:rollout 三段式——同一个 case 并发跑 N 条 rollout 时
    靠它保证各自的写动作绝不串台。
    """
    namespace_id = f"{run_id}:{bundle.case_id}:{rollout_id}"
    sandbox = Sandbox(bundle.env, namespace_id=namespace_id)
    trajectory = Trajectory(
        case_id=bundle.case_id,
        rollout_id=rollout_id,
        namespace_id=namespace_id,
        behavior=behavior,
        final_answer=dict(final_answer or {}),
    )

    auto_step = 0
    # 同一步内的第几个调用，用于生成 tc_{step}_{index} 形式的 id
    per_step_index: dict[int, int] = {}

    for call in calls:
        if call.step is None:
            auto_step += 1
            step = auto_step
        else:
            step = call.step
            auto_step = max(auto_step, step)

        index = per_step_index.get(step, 0)
        per_step_index[step] = index + 1
        tool_call_id = f"tc_{step}" if index == 0 else f"tc_{step}_{index}"

        ctx = ToolContext(
            case=bundle.case, env=bundle.env, sandbox=sandbox, step=step, tool_call_id=tool_call_id
        )
        result = await registry.execute(call.tool, call.arguments, ctx)

        trajectory.actions.append(
            Action(step=step, tool_call_id=tool_call_id, name=call.tool, arguments=dict(call.arguments))
        )
        trajectory.observations.append(
            Observation(
                tool_call_id=tool_call_id, tool=call.tool, ok=result.ok, data=result.data, error=result.error
            )
        )

        if step >= bundle.case.max_steps:
            trajectory.truncated = True
            trajectory.truncation_reason = "turns"       # 计划回放只有这一种截断
            break

    return trajectory, sandbox
