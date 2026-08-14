"""M9.4 · 假广告平台：可注入故障的外部系统替身。

★ 为什么不接真 Meta API（2026-08-14 定）

真 API 会**真的烧钱**，而 M9 要验的是**我们这侧的正确性**。真接入留到 M10 影子模式。
但假平台必须**如实建模真实世界的坏脾气**，否则 runtime 的降级路径就是假的。

★★★ 一条从沙盒继承过来、这里必须兑现的纪律：**超时分两种**

    请求没发出去    重试是安全的
    到了但回包丢了  **重试 = 重复扣款**

⚠️ 而模型/客户端看到的现象**一模一样**（都是超时）。沙盒里这条靠
`side_effect_applied` 建模（`EnvSnapshot.failures` 的注释：
"构造不出后者，模型学到的就是'超时=没做成'，那是错的"）。
这里必须同样建出来 —— 而且**错误文本要逐字相同**，否则 runtime 就能靠文本区分，
那它学到的东西在真平台上不成立。

⇒ 区分只能靠**幂等键**：重试时带同一个键，平台告诉你"这个键我见过"。
这就是三层幂等第三层存在的全部理由。

★ 实查过的事实（记在 `core/tool_registry.py`）：**Meta Marketing API 本身没有幂等机制**。
所以这里的 `_seen_keys` 是我们**希望平台有**的东西；真接入时这层保证得由
`runtime/db.py` 的 `record_tool_call` 兑现。假平台把它建出来，是为了让
runtime 的代码路径和真实接入时一致。
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any


class PlatformError(Exception):
    """平台侧错误。`retriable` 是**平台说的**，不是我们猜的。"""

    def __init__(self, message: str, *, code: str, retriable: bool) -> None:
        super().__init__(message)
        self.code = code
        self.retriable = retriable


# ★ 超时的错误文本**只有一份**。两种超时（没发出去 / 回包丢了）用同一句话，
# 因为真实世界里客户端确实分不出来。谁想分辨，只能靠幂等键回查。
TIMEOUT_MESSAGE = "upstream_timeout: 平台在 30s 内没有响应"


@dataclass
class FaultPlan:
    """故障剧本。**由调用方声明，不随机** —— 和沙盒 `EnvSnapshot.failures` 同一条纪律：
    随机故障会让"这次失败"和"模型做得不同"混在一起，压测的归因就糊了。"""

    # 第 n 次调用超时（1-indexed）
    timeout_at: set[int] = field(default_factory=set)
    # ★ 超时的那次，副作用**到底发生了没有**。这是两种超时的唯一区别。
    side_effect_applied: bool = False
    rate_limit_at: set[int] = field(default_factory=set)
    server_error_at: set[int] = field(default_factory=set)
    latency_seconds: float = 0.0


@dataclass
class FakeAdPlatform:
    """内存态的假平台。**不是 mock**：它有真实的状态，写动作真的改数字。"""

    budgets: dict[str, int] = field(default_factory=dict)
    faults: FaultPlan = field(default_factory=FaultPlan)
    calls: int = 0
    # 平台侧记住见过的幂等键 → 原结果。真 Meta 没有这个，见模块 docstring。
    _seen_keys: dict[str, dict[str, Any]] = field(default_factory=dict)

    async def update_budget(self, *, campaign_id: str, new_budget: int,
                            idempotency_key: str | None = None) -> dict[str, Any]:
        self.calls += 1
        n = self.calls

        if self.faults.latency_seconds:
            await asyncio.sleep(self.faults.latency_seconds)

        # 幂等键命中 ⇒ 直接返回原结果，**不重复执行**
        if idempotency_key and idempotency_key in self._seen_keys:
            return {**self._seen_keys[idempotency_key], "deduped_by_platform": True}

        if n in self.faults.rate_limit_at:
            raise PlatformError("rate_limited: 请求过于频繁", code="rate_limited", retriable=True)
        if n in self.faults.server_error_at:
            raise PlatformError("server_error: 平台内部错误", code="server_error", retriable=True)

        if n in self.faults.timeout_at:
            # ★★★ 关键：副作用**先按剧本决定要不要真的发生**，再抛同一句超时。
            # side_effect_applied=True 就是"到了但回包丢了" —— 这时重试会重复扣款。
            if self.faults.side_effect_applied:
                self.budgets[campaign_id] = new_budget
                if idempotency_key:
                    self._seen_keys[idempotency_key] = {
                        "campaign_id": campaign_id, "new_budget": new_budget}
            raise PlatformError(TIMEOUT_MESSAGE, code="timeout", retriable=True)

        self.budgets[campaign_id] = new_budget
        result = {"campaign_id": campaign_id, "new_budget": new_budget}
        if idempotency_key:
            self._seen_keys[idempotency_key] = result
        return result

    async def get_metrics(self, *, campaign_id: str) -> dict[str, Any]:
        self.calls += 1
        if self.faults.latency_seconds:
            await asyncio.sleep(self.faults.latency_seconds)
        if self.calls in self.faults.timeout_at:
            raise PlatformError(TIMEOUT_MESSAGE, code="timeout", retriable=True)
        return {"campaign_id": campaign_id,
                "daily_budget": self.budgets.get(campaign_id, 50_000),
                "cpi": 2.1, "roas_d7": 0.42}
