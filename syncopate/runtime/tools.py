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

# 写工具白名单 → 需要的权限：**从治理表派生**（K6-3），不再手抄一份。
# 08-19 的老病：这里登记了 8 个写工具、实现只有 2 个——登记表是最像证据的名单。
# 现在 `tool_governance.assert_governance_complete()` 在导入时对着沙盒 REGISTRY 断言：
# 已登记 == 全集、side_effect == kind=="write"。缺一个、多一个、标错一个都在导入时炸。
from syncopate.runtime.tool_governance import (assert_governance_complete, check_output,
                                               classify_platform_error, error_json, governance,
                                               write_permissions)

assert_governance_complete()
WRITE_TOOLS: dict[str, str] = write_permissions()

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


def derive_idempotency_key(*, org_id: str, run_id: str | None = None, tool: str,
                           arguments: dict[str, Any]) -> str:
    """写工具的外部幂等键。**必须确定性**（重试要推出同一个键），**不含 run_id**（K5-6，课件 §11.4）：

    带 run_id 的键在"同 run 内崩溃重试"下完全正常，一遇 rerun（新 run_id）就变成"新的一笔"——
    下游认不出、本地 UNIQUE 也拦不住 ⇒ 双重扣款。键标识的是"这是哪一笔业务"，不是"哪一次执行"。
    业务实体 = 参数本身（沙盒 spec 的 client_request_id 就是模型给的业务级请求号，训练里教过它传）。
    `run_id` 参数只为兼容旧调用方，**不参与**键的计算。
    """
    payload = repr(sorted(arguments.items())).encode()
    digest = hashlib.sha256(payload).hexdigest()[:16]
    return f"{org_id}:{tool}:{digest}"


class ToolRuntime:
    """执行一次工具调用的全套：权限 → 幂等键 → 超时重试 → 落库。"""

    def __init__(self, db: Database, *, permissions: set[str] | None = None,
                 timeout_seconds: float = 30.0) -> None:
        self.db = db
        # ⚠️ **默认不是"全给"。** 第一版默认值是 `set(WRITE_TOOLS.values())`，
        # 而 worker 用的正是这个默认值 ⇒ **权限闸在真实路径上从不拒绝**，
        # 那道闸等于不存在。改成最小权限：不传就只有当前编排真正需要的那一个。
        # 要更多权限得**显式**要 —— 忘了要会报 PermissionDenied，比悄悄放行好。
        self.permissions = set(permissions) if permissions is not None else {"budget:write"}
        self.timeout_seconds = timeout_seconds

    def _check_permission(self, tool: str) -> None:
        needed = WRITE_TOOLS.get(tool)
        if needed is not None and needed not in self.permissions:
            raise PermissionDenied(f"缺少权限 {needed}（工具 {tool}）")

    async def call(self, *, org_id: str, run_id: str, step: int, tool: str,
                   arguments: dict[str, Any],
                   invoke: Callable[..., Awaitable[dict[str, Any]]]) -> ToolOutcome:
        """`invoke` 是真正打平台的那个协程，签名 `invoke(**arguments, idempotency_key=...)`。

        K6：超时 / 可重试错误 / 输出键 全部从治理表读（禁全局常量）；失败一律结构化成
        error_json {code, message, retryable, alert}（课件 CH6 分诊三字段）。
        """
        self._check_permission(tool)
        gov = governance(tool)
        is_write = gov.side_effect
        key = derive_idempotency_key(org_id=org_id, run_id=run_id, tool=tool,
                                     arguments=arguments) if is_write else None
        timeout = gov.timeout_seconds

        attempts = 0

        async def execute() -> tuple[bool, dict[str, Any] | None, str | None, dict[str, Any] | None]:
            nonlocal attempts
            last_error: str | None = None
            last_json: dict[str, Any] | None = None
            for attempt in range(1, MAX_ATTEMPTS + 1):
                attempts = attempt
                try:
                    kwargs = dict(arguments)
                    if is_write:
                        kwargs["idempotency_key"] = key
                    data = await asyncio.wait_for(invoke(**kwargs), timeout=timeout)
                    bad = check_output(tool, data)
                    if bad:
                        # 反向污染：脏返回不进 context。副作用（若有）已发生 ⇒ 结果照存、观测报错、不重试
                        return False, None, bad, error_json("output_schema_violation", bad)
                    return True, data, None, None
                except PlatformError as exc:
                    last_error = str(exc)
                    last_json = classify_platform_error(tool, code=str(exc.code), message=str(exc))
                    # ★★ 只在治理表登记为可重试的错误码上重试，且写动作重试必须带同一个键。
                    if not last_json["retryable"] or (is_write and key is None):
                        return False, None, last_error, last_json
                except asyncio.TimeoutError:
                    # 我们这侧的超时。和平台超时一样：**分不出副作用有没有发生**。
                    last_error = "client_timeout: 本地等待超时"
                    # 读工具：超时可重试（无副作用）；写工具：结果未知 ⇒ 不重试，由 record_tool_call 记 response_lost
                    last_json = error_json("client_timeout", last_error, retryable=not is_write)
                    if is_write:
                        return False, None, last_error, last_json
                if attempt < MAX_ATTEMPTS:
                    await asyncio.sleep(RETRY_BACKOFF[min(attempt - 1, len(RETRY_BACKOFF) - 1)])
            # ★ 用尽重试就如实上报失败，**不是重试到成功为止** ——
            # 否则沙盒里教的"失败之后怎么办"在线上永远用不上。
            return False, None, last_error, last_json

        result: ToolCallResult = await record_tool_call(
            self.db, org_id=org_id, run_id=run_id, step=step, tool=tool,
            arguments=arguments, external_idempotency_key=key, execute=execute,
            side_effect=is_write)

        return ToolOutcome(ok=result.ok, data=result.data, error=result.error,
                           attempts=attempts, replayed=result.replayed,
                           idempotency_key=key)
