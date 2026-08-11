"""广告投放域（Ad Campaign / 手游买量）。

import 这个包就会把 10 个工具和 9 条 cap 规则注册进全局表。域的三块内容：

    tools/       10 个有状态工具（7 读 + 2 写 + 1 慢工具）
    policies.py  政策库 + 期望决策计算（不用 LLM 判分的底气）
    rules.py     9 条 cap 规则，每条都带责任步号

对外只暴露 `build_domain()`，把引擎需要的三个注入点打成一个包。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from syncopate.core.tool_registry import REGISTRY, ToolRegistry
from syncopate.core.verifier_engine import CAPS, CapRegistry

# 副作用 import：这四行触发工具注册，不能删
from syncopate.domains.adcampaign.tools import (  # noqa: F401
    analytics, creative, external_tools, freshness, governance, memory_tools, mmp, playbook,
    system_tools,
)
from syncopate.domains.adcampaign import rules  # noqa: F401
from syncopate.domains.adcampaign.policies import compute_decision, score_policy

DOMAIN_NAME = "adcampaign"

# 默认工具菜单。case 可以用 `Case.tool_menu` 覆盖成子集
# —— tool_missing 类 case 就靠这个故意抽掉必需工具。
DEFAULT_TOOL_MENU: list[str] = [
    # ---- 投放分析 ----
    "campaign.get_metrics",
    "metrics.get_freshness",
    "creative.get_metrics_by_asset",
    "campaign.detect_anomalies",
    "playbook.get_optimization",
    "benchmark.get_industry_baseline",
    "mmp.get_attribution",
    "policy.get_budget_rule",
    "risk.check_account",
    "campaign.list",
    "campaign.update_budget",
    "approval.create_case",
    "creative.upload",
    "creative.poll_review",
    # ---- 外部资料（离线 ingest 的产物）----
    "benchmark.get_safety_line",
    "calendar.get_seasonal_context",
    "creative.get_asset_tags",
    "creative.search_similar",
    # ---- 记忆：2 读 3 写，写全部是提案 ----
    "memory.search",
    "memory.read",
    "memory.write_proposal",
    "memory.invalidate",
    "memory.conflict_resolve",
    # ---- runtime ----
    "system.wait",
]


@dataclass
class Domain:
    """引擎需要从域拿到的一切。换场景 = 换一个 Domain 实例。"""

    name: str
    registry: ToolRegistry
    caps: CapRegistry
    policy_scorer: Callable[..., tuple[float, dict[str, Any]]]
    decision_fn: Callable[..., dict[str, Any] | None]
    default_tool_menu: list[str]


def build_domain() -> Domain:
    return Domain(
        name=DOMAIN_NAME,
        registry=REGISTRY,
        caps=CAPS,
        policy_scorer=score_policy,
        decision_fn=compute_decision,
        default_tool_menu=list(DEFAULT_TOOL_MENU),
    )


__all__ = ["Domain", "build_domain", "DOMAIN_NAME", "DEFAULT_TOOL_MENU"]
