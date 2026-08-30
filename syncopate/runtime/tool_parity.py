"""沙盒与 runtime 的**工具对齐账本**（B-5）。

★★★ 为什么这份必须在补工具**之前**建，而不是补完再说

`09 §1-③` 的纪律：**沙盒是 runtime 的子集，且契约由 runtime 定义。**
同一个工具两边行为不一致，**训出来的策略在线上就不成立** —— 而且不会报错。

⚠️⚠️ 2026-08-19 实测（**这个数字我先前报错过，是本判据自己纠正的**）：
训练侧 **30** 个工具，runtime 真正实现的只有 **2** 个。
`tools.WRITE_TOOLS` 里登记了 **8** 个写工具的权限与幂等 —— **其中 7 个根本没有实现**。
⇒ **登记 ≠ 实现。** 看 `WRITE_TOOLS` 有 8 个就以为支持 8 个写工具，正是这条判据要防的误读。
⇒ 这个缺口存在了很久，**没有任何东西在喊** —— 因为"没实现"不会报错，
  它只是让某个工具在 runtime 里不存在，而在训练里存在。

★ 所以这份账本的判据是「**某集合应当完整**」型（守则①）：

    已实现 ∪ 已登记的缺口  ==  沙盒的全部工具
    已实现 ∩ 已登记的缺口  ==  空

⇒ 新增一个沙盒工具而不在这里登记 ⇒ **判据红**。
⇒ 声称实现了但签名对不上 ⇒ **判据红**。
⇒ **缺口本身不红** —— 它是被承认的债，不是失败。但它必须**写下来才算被承认**。

⚠️ 反面教材就在隔壁：`tools.WRITE_TOOLS` 那张表漏登记一个写工具的代价不是报错，
   而是**那个写动作悄悄没有幂等保护**。同一个形状。
"""

from __future__ import annotations

import inspect
from typing import Any, Callable

# ── runtime 已实现的工具（由 worker 的绑定表登记）────────────────────────
#
# ⚠️ 这里只写名字。实现在 `worker._bindings()`，签名一致性由下面的判据检查。
IMPLEMENTED: set[str] = {
    "campaign.get_metrics",
    "campaign.update_budget",
    # B-2 第一批（2026-08-19）：平台/检索服务已经支持的那些
    "campaign.list",
    "metrics.get_freshness",
    "policy.search",
    "insight.search_claims",
    # B-2 第二批：记忆库 + 安全线
    "memory.read",
    "memory.search",
    "memory.write_proposal",
    "memory.invalidate",
    "memory.conflict_resolve",
    "benchmark.get_safety_line",
    # B-2 第三批：素材库
    "creative.upload",
    "creative.poll_review",
    "creative.get_asset_tags",
    "creative.get_metrics_by_asset",
    "creative.search_similar",
    "system.wait",
    # B-2 第四批：写工具
    "approval.create_case",
    "campaign.create",
    "campaign.scale_budget",
    # B-2 第五批：数据源类 —— 至此账本清零
    "analysis.feature_lift",
    "analysis.geo_breakdown",
    "benchmark.get_industry_baseline",
    "calendar.get_seasonal_context",
    "campaign.detect_anomalies",
    "mmp.get_attribution",
    "playbook.get_optimization",
    "policy.get_budget_rule",
    "risk.check_account",
}

# ── 还没实现的缺口：**必须写清楚"为什么还没做"** ──────────────────────────
#
# ★ 只写工具名不够 —— 那会变成一张永远不会被清空的名单。
#   写清楚归属，才知道它在等谁。
KNOWN_GAPS: dict[str, str] = {
    # 平台形状已就位（分页 / 显式字段 / 异步任务），差实现
}

# ── ★ 第三类：契约信令族（v15）────────────────────────────────────────────
#
# `session.*` 在沙盒里是**注册进表的工具**（不注册模型就看不见 schema，
# 调用会被判 tool_not_available —— R0 已经为此作废过一次读数）；
# 但在 runtime 侧它们**不走工具绑定**，而是由 agent_loop 的状态机接住：
#   defer   → 挂起并记 recheck_after_days
#   clarify → waiting_for_user
#   reject  → 终止并审计
#   report  → 收下机器字段，轨迹继续
# ⇒ 既不算「已实现的工具」（worker._bindings() 里没有、也不该有），
#   也不算「缺口」（它没有欠任何东西）。硬塞进任何一类都会让账本说谎。
# ⇒ 单开一类，并且**一样进完整性等式** —— 新增一个信令而不登记在这里，判据照样红。
CONTRACT_SIGNALS: dict[str, str] = {
    "session.defer": "runtime 由 agent_loop 状态机处理：挂起 + recheck_after_days（R4①）",
    "session.clarify": "runtime 由 agent_loop 状态机处理：转 waiting_for_user（R4①）",
    "session.reject": "runtime 由 agent_loop 状态机处理：终止并写审计（R4①）",
    "session.report": "非终止信令：收下机器可核字段，轨迹继续（判分取数来源，25 §3.3）",
}


def sandbox_tools() -> dict[str, Any]:
    """沙盒的全部工具 spec。**这是唯一真相来源**，不在这里另抄一份名单。"""
    import syncopate.domains.adcampaign  # noqa: F401  触发注册
    from syncopate.core.tool_registry import REGISTRY
    return {name: REGISTRY.get(name) for name in REGISTRY.names()}


def coverage_report() -> dict[str, Any]:
    """账本对账：已实现 / 已登记缺口 / **两边都没有的**。"""
    names = set(sandbox_tools())
    # ★ 信令族按契约存在与否入账：v14 沙盒里没有它们，不该被判成「账本写错了名字」。
    #   （intersect 而不是并入，否则 v14 下 stale_ledger 会红——判据要在两个契约下都成立。）
    ledgered = IMPLEMENTED | set(KNOWN_GAPS) | (set(CONTRACT_SIGNALS) & names)
    return {
        "sandbox_total": len(names),
        "implemented": sorted(IMPLEMENTED),
        "gaps": sorted(set(KNOWN_GAPS)),
        "contract_signals": sorted(set(CONTRACT_SIGNALS) & names),
        # ★ 沙盒有、账本里一个字都没提 —— 这才是真正危险的那一类
        "unledgered": sorted(names - ledgered),
        # 账本里有、沙盒却没有 ⇒ 名字写错了，或者沙盒删了工具
        "stale_ledger": sorted(ledgered - names),
        "both": sorted(IMPLEMENTED & set(KNOWN_GAPS)),
    }


def signature_mismatch(name: str, impl: Callable[..., Any]) -> str | None:
    """runtime 的实现能不能接住沙盒 spec 的**必填参数**。

    ⚠️ 判据刻意只查**必填参数**，不查可选参数和返回值：
      - 必填参数对不上 ⇒ 调用直接 TypeError，是硬错
      - 返回字段对不上 ⇒ 也是错，但那要跑起来才看得见（留给 B-5 的行为对齐）
      ⇒ **先立能立住的那半，不要因为立不全就一条都不立。**

    ★ 写工具额外要求能接 `idempotency_key` —— 那是 `ToolRuntime` 会传的。
      接不住的话，幂等保护在这个工具上是**静默失效**的。
    """
    spec = sandbox_tools().get(name)
    if spec is None:
        return f"沙盒里没有这个工具：{name}"
    required = list(spec.parameters.get("required", []))
    sig = inspect.signature(impl)
    params = sig.parameters
    has_kwargs = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
    if has_kwargs:
        return None                      # **kwargs 接得住任何东西
    missing = [r for r in required if r not in params]
    if missing:
        return f"{name}: 实现接不住必填参数 {missing}"
    from syncopate.runtime.tools import WRITE_TOOLS
    if name in WRITE_TOOLS and "idempotency_key" not in params:
        return (f"{name}: 是写工具但实现接不住 idempotency_key "
                f"⇒ 幂等保护在这个工具上**静默失效**")
    return None
