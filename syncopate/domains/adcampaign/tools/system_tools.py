"""`system.wait` —— 让「退避」成为一个可示范、可验证的动作。

★ 为什么必须是一个工具

429 限流的正确处理核心是「按 retry_after 等够了再重试」。但如果沙盒里没有
「等待」这个动作，**退避这件事就不存在**：模型无法示范它，gold 无法写它，
verifier 也无从判断模型是等够了才重试、还是立刻就重试了。

★★ 真停还是假停：都是，靠 latency_scale 一个旋钮

`await asyncio.sleep()` 挂起的是**这一条 rollout 的协程**，事件循环立刻去跑
同一个 worker 里的其它 rollout —— 所以它既是真停（这条 rollout 确实在等），
又不阻塞别人。实测 head_of_line_ratio = 2.5~4.4 就是这个机制的证据；
假停的话这个比值会恒等于 1。

    latency_scale=1.00   如实计入（做异步对照实验时）
    latency_scale=0.01   压掉（训练时，不浪费 GPU）

⚠️ 缩放**不影响学习信号**：评分看的是 `seconds` 这个参数够不够大，
不是墙钟真的走了多久。
"""

from __future__ import annotations

import asyncio
from typing import Any

from syncopate.core.tool_registry import REGISTRY, ToolContext, ToolResult

# 一次等待的上限。真实 agent 不该在一次 rollout 里睡半天——
# 超过这个量级就该走审批/上报，而不是干等。
MAX_WAIT_SECONDS = 600


@REGISTRY.tool(
    name="system.wait",
    description=(
        "等待指定秒数后继续。\n"
        "· 收到 429/限流且返回里带 retry_after 时，应当先等**不少于** retry_after 秒再重试。\n"
        "· 不要用它等待数据成熟（那是几天的量级，应当用 defer）。\n"
        f"· 单次上限 {MAX_WAIT_SECONDS} 秒；需要等更久说明这件事不该在本次会话里做完。"
    ),
    parameters={
        "type": "object",
        "properties": {"seconds": {"type": "integer", "description": "等待秒数"}},
        "required": ["seconds"],
    },
    kind="read",
    api_ref="runtime:sleep",
    # ★ 不计平台配额：等待是**本地**操作，不是 API 调用。
    # 计费的话，额度耗尽时连"等一下让额度恢复"都做不了——
    # 那条唯一正确的出路会被自己堵死（实测踩到过）。
    cost_points=0,
)
async def wait(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    try:
        seconds = int(args["seconds"])
    except (KeyError, TypeError, ValueError):
        return ToolResult(ok=False, error="invalid_argument: seconds must be an integer")
    if seconds <= 0:
        return ToolResult(ok=False, error="invalid_argument: seconds must be positive")
    if seconds > MAX_WAIT_SECONDS:
        return ToolResult(
            ok=False,
            error=f"wait_too_long: {seconds}s > {MAX_WAIT_SECONDS}s。"
                  "需要等这么久的事情应当走审批或改期，不要在会话里干等")
    await asyncio.sleep(seconds * ctx.latency_scale)
    ctx.sandbox.waited_seconds += seconds
    # ★ 等待让 API 积分按衰减窗口回落 —— 这才是 429 时"等一下"真正有用的原因。
    # 不建模衰减的话，等待只是浪费时间，模型学到的会是"等也没用"。
    budget = ctx.env.api_budget
    if budget and ctx.sandbox.api_points_spent > 0:
        decay = float(budget.get("decay_seconds", 300)) or 300.0
        recovered = int(int(budget.get("limit", 0)) * min(1.0, seconds / decay))
        ctx.sandbox.api_points_spent = max(0, ctx.sandbox.api_points_spent - recovered)
    return ToolResult(ok=True, data={"waited_seconds": seconds,
                                     "total_waited_seconds": ctx.sandbox.waited_seconds})
