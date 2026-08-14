"""M9.4 · Tool Runtime：超时 / 重试 / 权限 / 幂等键。

★★★ 这里是「沙盒是 runtime 的子集，契约由 runtime 定义」的落点

沙盒（`syncopate/core`）的工具是纯函数、无网络、确定性；这里的要处理超时、重试、
限流、部分失败。**两边行为不一致，训出来的策略在线上就不成立。**

⇒ 具体到三条：

1. **写工具必须带外部幂等键。** 沙盒里 `idempotent=True` 的工具要求模型传
   `client_request_id`；这里把它变成真的保证 —— 生成键、落库、重放时返回原结果。

2. **重试只在平台明确说可重试时才做，而且必须带同一个幂等键。**
   ⚠️ **超时不能无脑重试**：超时分两种（没发出去 / 回包丢了），现象一模一样。
   带幂等键重试是安全的（平台或我们的库会挡住第二次执行）；
   不带键的重试就是在赌，赌输了是重复扣款。

3. **重试次数有上限，且失败要如实上报。** 沙盒的 `MAX_ATTEMPTS` 是同一条。
   ⚠️ 绝不能"重试到成功为止" —— 那会让沙盒里学到的"失败之后怎么办"变成死代码。

★ 权限校验放在这一层而不是 API 层：同一个动作可能来自 API、也可能来自 worker
自己的编排，两条路都得过同一道闸。
"""

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from syncopate.runtime.db import Database, ToolCallResult, record_tool_call
from syncopate.runtime.platform import PlatformError

# 写工具白名单 → 需要的权限。不在表里的工具一律当读工具。
WRITE_TOOLS: dict[str, str] = {
    "campaign.update_budget": "budget:write",
    "campaign.create": "campaign:write",
    "campaign.scale_budget": "budget:write",
    "approval.create_case": "approval:write",
}

MAX_ATTEMPTS = 3          # 和沙盒 core/failures.py 的 MAX_ATTEMPTS 保持一致
RETRY_BACKOFF = (0.05, 0.15)


class PermissionDenied(Exception):
    pass


@dataclass
class ToolOutcome:
    ok: bool
    data: dict[str, Any] | None
    error: str | None
    attempts: int
    replayed: bool
    idempotency_key: str | None


def derive_idempotency_key(*, org_id: str, run_id: str, tool: str,
                           arguments: dict[str, Any]) -> str:
    """从 (org, run, 工具, 参数) 推出稳定的外部幂等键。

    ★ **必须是确定性的**：重试时要能推出**同一个键**，否则幂等等于没有。
    所以不能用 uuid4，也不能带时间戳。

    ★ 带上参数的哈希：同一个 run 里"把预算改成 900"和"改成 1200"是两个不同的动作，
    共用一个键会让第二个被误挡。
    """
    payload = repr(sorted(arguments.items())).encode()
    digest = hashlib.sha256(payload).hexdigest()[:16]
    return f"{org_id}:{run_id}:{tool}:{digest}"


class ToolRuntime:
    """执行一次工具调用的全套：权限 → 幂等键 → 超时重试 → 落库。"""

    def __init__(self, db: Database, *, permissions: set[str] | None = None,
                 timeout_seconds: float = 30.0) -> None:
        self.db = db
        self.permissions = permissions if permissions is not None else set(WRITE_TOOLS.values())
        self.timeout_seconds = timeout_seconds

    def _check_permission(self, tool: str) -> None:
        needed = WRITE_TOOLS.get(tool)
        if needed is not None and needed not in self.permissions:
            raise PermissionDenied(f"缺少权限 {needed}（工具 {tool}）")

    async def call(self, *, org_id: str, run_id: str, step: int, tool: str,
                   arguments: dict[str, Any],
                   invoke: Callable[..., Awaitable[dict[str, Any]]]) -> ToolOutcome:
        """`invoke` 是真正打平台的那个协程，签名 `invoke(**arguments, idempotency_key=...)`。"""
        self._check_permission(tool)

        is_write = tool in WRITE_TOOLS
        key = derive_idempotency_key(org_id=org_id, run_id=run_id, tool=tool,
                                     arguments=arguments) if is_write else None

        attempts = 0

        async def execute() -> tuple[bool, dict[str, Any] | None, str | None]:
            nonlocal attempts
            last_error: str | None = None
            for attempt in range(1, MAX_ATTEMPTS + 1):
                attempts = attempt
                try:
                    kwargs = dict(arguments)
                    if is_write:
                        kwargs["idempotency_key"] = key
                    data = await asyncio.wait_for(invoke(**kwargs),
                                                  timeout=self.timeout_seconds)
                    return True, data, None
                except PlatformError as exc:
                    last_error = str(exc)
                    # ★★ 只在平台**明确说可重试**时才重试，且写动作重试必须带同一个键。
                    # 不带键就重试 = 在赌"刚才那次没生效"，赌输了是重复扣款。
                    if not exc.retriable or (is_write and key is None):
                        return False, None, last_error
                except asyncio.TimeoutError:
                    # 我们这侧的超时。和平台超时一样：**分不出副作用有没有发生**。
                    last_error = "client_timeout: 本地等待超时"
                if attempt < MAX_ATTEMPTS:
                    await asyncio.sleep(RETRY_BACKOFF[min(attempt - 1, len(RETRY_BACKOFF) - 1)])
            # ★ 用尽重试就如实上报失败，**不是重试到成功为止** ——
            # 否则沙盒里教的"失败之后怎么办"在线上永远用不上。
            return False, None, last_error

        result: ToolCallResult = await record_tool_call(
            self.db, org_id=org_id, run_id=run_id, step=step, tool=tool,
            arguments=arguments, external_idempotency_key=key, execute=execute)

        return ToolOutcome(ok=result.ok, data=result.data, error=result.error,
                           attempts=attempts, replayed=result.replayed,
                           idempotency_key=key)
