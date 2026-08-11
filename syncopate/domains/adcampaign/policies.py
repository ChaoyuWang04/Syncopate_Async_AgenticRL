"""广告域政策库 + 期望决策计算。

这是「高风险 case」的地基。老师包里对应的是退款政策（多少钱以内可以直接退、
超过多少要审批）；广告域的等价物是**预算调整规则**——而且比退款更适合做高风险：
退款是一次性的，预算改错了会持续烧钱。

政策不是写死在代码里的常量，而是 env 里的数据（`EnvSnapshot.policies`），
因为不同账户等级、不同平台的规则不同，case 需要能自由组合。
"""

from __future__ import annotations

from typing import Any

from syncopate.core.schemas import CaseBundle

# --------------------------------------------------------------------------
# 政策模板：造 case 时往 env.policies 里塞
# --------------------------------------------------------------------------

BUDGET_POLICIES: list[dict[str, Any]] = [
    {
        "policy_id": "P_BUDGET_STANDARD",
        "topic": "budget_change",
        "applies_to": {"account_tier": "standard"},
        "rules": {
            "max_increase_pct": 50,             # 单次涨幅硬上限
            "approval_required_above_pct": 30,  # 超过这个涨幅必须先走审批
            "risk_check_required": True,        # 改预算前必须过风控
            "monthly_cap_enforced": True,       # 受账户月度总额约束
        },
    },
    {
        "policy_id": "P_BUDGET_PLUS",
        "topic": "budget_change",
        "applies_to": {"account_tier": "plus"},
        "rules": {
            "max_increase_pct": 100,
            "approval_required_above_pct": 60,
            "risk_check_required": True,
            "monthly_cap_enforced": True,
        },
    },
]

# 平台内容政策，creative.upload 的前置检查用
PLATFORM_POLICIES: list[dict[str, Any]] = [
    {
        "policy_id": "P_CREATIVE_META",
        "topic": "creative_spec",
        "applies_to": {"platform": "Meta"},
        "rules": {"max_duration_seconds": 60, "allowed_types": ["video", "image"], "review_required": True},
    },
    {
        "policy_id": "P_CREATIVE_GOOGLE",
        "topic": "creative_spec",
        "applies_to": {"platform": "Google"},
        "rules": {"max_duration_seconds": 30, "allowed_types": ["video"], "review_required": True},
    },
]


def select_policy(policies: list[dict[str, Any]], topic: str, facts: dict[str, Any]) -> dict[str, Any] | None:
    """按 topic + applies_to 条件挑出适用的那条政策。

    applies_to 里每个键值对都必须和 facts 对上；空 applies_to 视为通配。
    """
    for policy in policies:
        if policy.get("topic") != topic:
            continue
        conditions = policy.get("applies_to") or {}
        if all(facts.get(key) == value for key, value in conditions.items()):
            return policy
    return None


# --------------------------------------------------------------------------
# 期望决策：政策规则 + case 事实 -> 正确答案应该是什么
# --------------------------------------------------------------------------


def compute_decision(bundle: CaseBundle) -> dict[str, Any] | None:
    """算出「按政策，这个 case 的正确处理是什么」。

    verifier 的 `decision.*` 引用式解析到这里的返回值。这就是我们不需要 LLM judge
    的底气：正确答案是**算出来的**，不是判出来的。

    只对预算调整类 case 有意义；其它 case 返回 None（policy 子分自动记满分）。
    """
    entities = bundle.case.entities
    campaign = bundle.env.row("campaigns", entities.get("campaign_id"))
    if campaign is None:
        return None
    requested = entities.get("requested_budget")
    if requested is None:
        return None

    account = bundle.env.row("accounts", campaign.get("account_id")) or {}
    policy = select_policy(bundle.env.policies, "budget_change", {"account_tier": account.get("tier")})
    if policy is None:
        return None

    rules = policy["rules"]
    # 预算全程用**分**（整数）。取整放在这里统一做，避免 gold 和 verifier 各自四舍五入
    current = float(campaign.get("daily_budget") or 0.0)
    requested = float(requested)
    increase_pct = ((requested - current) / current * 100.0) if current else 0.0

    # 涨幅硬上限
    max_allowed = current * (1.0 + rules["max_increase_pct"] / 100.0)

    # 月度总额约束：剩余额度摊到本月剩余天数（简化为 30 天）
    if rules.get("monthly_cap_enforced"):
        remaining = float(account.get("monthly_cap", 0.0)) - float(account.get("spend_mtd", 0.0))
        max_allowed = min(max_allowed, max(0.0, remaining / 30.0))

    approved = min(requested, max_allowed)
    return {
        "policy_id": policy["policy_id"],
        # ★ 预算字段全部取整：单位是**分**，没有小数。
        # 不取整的话 gold 写 60000.0、verifier 期望 60000.00000001 这类浮点毛刺
        # 会在 SideEffectReq 的比对上偶发失败，而且极难查。
        "current_budget": int(round(current)),
        "requested_budget": int(round(requested)),
        "increase_pct": round(increase_pct, 2),
        "max_allowed_budget": int(round(max_allowed)),
        # ★ 这个字段是 verifier 校验写动作参数的真值来源：
        #   模型必须改成 approved_budget，照着用户要的数改就是错的。
        "approved_budget": int(round(approved)),
        "capped": approved < requested - 1e-6,
        "requires_approval": increase_pct > rules["approval_required_above_pct"],
        "requires_risk_check": bool(rules.get("risk_check_required")),
    }


# --------------------------------------------------------------------------
# policy 子分
# --------------------------------------------------------------------------


def score_policy(bundle: CaseBundle, trajectory, sandbox) -> tuple[float, dict[str, Any]]:
    """policy 子分：查没查政策 + 决策对不对，各占一半。

    拆成两半是有意的——「查了但算错」和「压根没查」是两种不同的失败，
    合成一个分数会把它们混在一起，看不出模型到底卡在哪。
    """
    called = set(trajectory.called_tools())
    consulted = "policy.get_budget_rule" in called

    decision = compute_decision(bundle)
    if decision is None:
        return (1.0 if consulted else 0.5), {"consulted": consulted, "decision": None}

    # 决策是否落地：真实写进去的预算等于政策算出来的数
    record = sandbox.current_state("budget_updated")
    applied = record.arguments.get("new_budget") if record else None
    decision_ok = applied is not None and abs(float(applied) - decision["approved_budget"]) < 1e-6

    return 0.5 * consulted + 0.5 * decision_ok, {
        "consulted": consulted,
        "expected_budget": decision["approved_budget"],
        "applied_budget": applied,
        "decision_ok": decision_ok,
        "policy_id": decision["policy_id"],
    }
