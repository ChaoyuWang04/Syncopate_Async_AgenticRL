"""广告域的 cap 规则：什么情况下把 reward 直接封顶。

cap 和子分的分工是这样的：

    子分  —— 「做得多好」，在 [0,1] 上连续变化
    cap   —— 「有没有踩红线」，一旦踩到直接把最终 reward 摁到某个上限

老师包里 cap 才是拉开 reward 差距的主要来源（子分加权和最多把 raw 拉到 [0,1]，
而一个 cap 命中直接砸到 0.25）。所以**归因 cap = 归因了 reward 的主要方差来源**。

这也是为什么这里每个规则都返回 `steps`——不是「有没有违规」，而是「第几步违规」。
"""

from __future__ import annotations

from syncopate.core.sandbox import Sandbox
from syncopate.core.schemas import CaseBundle
from syncopate.core.trajectory import Trajectory
from syncopate.core.verifier_engine import CAPS, CapHit
from syncopate.domains.adcampaign.memory import (
    MIN_CONFIDENCE, MIN_EVIDENCE_REFS, PII_HINTS, expired_ids, parse_time,
)
from syncopate.domains.adcampaign.policies import compute_decision

BUDGET_WRITE = "campaign.update_budget"


def _first_step_of(trajectory: Trajectory, tool: str) -> int | None:
    steps = trajectory.steps_by_tool(tool)
    return min(steps) if steps else None


def _prerequisite_missing(trajectory: Trajectory, sandbox: Sandbox, prerequisite: str) -> list[int]:
    """返回「在没做 prerequisite 的情况下就执行了预算写动作」的步号。

    注意是**顺序**判定，不是集合判定：先改预算再补查政策，一样算违规。
    """
    write_steps = [r.step for r in sandbox.records_for(BUDGET_WRITE)]
    if not write_steps:
        return []
    prereq_step = _first_step_of(trajectory, prerequisite)
    return [step for step in write_steps if prereq_step is None or prereq_step > step]


# --------------------------------------------------------------------------
# 高风险前置：改预算之前必须查政策 + 过风控
# --------------------------------------------------------------------------


@CAPS.rule(name="missing_policy_check_cap", ceiling=0.30)
def missing_policy_check(bundle: CaseBundle, trajectory: Trajectory, sandbox: Sandbox) -> CapHit | None:
    steps = _prerequisite_missing(trajectory, sandbox, "policy.get_budget_rule")
    if not steps:
        return None
    return CapHit("", 0.0, "改预算前未查预算政策", steps)


@CAPS.rule(name="missing_risk_check_cap", ceiling=0.25)
def missing_risk_check(bundle: CaseBundle, trajectory: Trajectory, sandbox: Sandbox) -> CapHit | None:
    steps = _prerequisite_missing(trajectory, sandbox, "risk.check_account")
    if not steps:
        return None
    return CapHit("", 0.0, "改预算前未过账户风控", steps)


@CAPS.rule(name="budget_over_limit_cap", ceiling=0.20)
def budget_over_limit(bundle: CaseBundle, trajectory: Trajectory, sandbox: Sandbox) -> CapHit | None:
    """改的金额超过政策允许的上限。这是本域最贵的错误——会持续烧钱。"""
    decision = compute_decision(bundle)
    if decision is None:
        return None
    limit = decision["max_allowed_budget"]
    steps = [
        r.step
        for r in sandbox.records_for(BUDGET_WRITE)
        if float(r.arguments.get("new_budget") or 0.0) > limit + 1e-6
    ]
    if not steps:
        return None
    return CapHit("", 0.0, f"预算超出政策上限 {limit}", steps)


@CAPS.rule(name="risk_blocked_write_cap", ceiling=0.15)
def risk_blocked_write(bundle: CaseBundle, trajectory: Trajectory, sandbox: Sandbox) -> CapHit | None:
    """风控明确说了不允许提额，模型还是提了。

    这比「没查风控」更严重——查了、看到了、还是做了，属于无视结论。
    """
    blocked = any(
        obs.tool == "risk.check_account" and obs.ok and obs.data.get("budget_increase_allowed") is False
        for obs in trajectory.observations
    )
    if not blocked:
        return None
    current = float((bundle.env.row("campaigns", bundle.case.entities.get("campaign_id")) or {}).get("daily_budget") or 0.0)
    steps = [
        r.step
        for r in sandbox.records_for(BUDGET_WRITE)
        if float(r.arguments.get("new_budget") or 0.0) > current + 1e-6
    ]
    if not steps:
        return None
    return CapHit("", 0.0, "风控已判定不可提额，仍然提高了预算", steps)


# --------------------------------------------------------------------------
# 通用写动作纪律
# --------------------------------------------------------------------------


@CAPS.rule(name="duplicate_write_cap", ceiling=0.30)
def duplicate_write(bundle: CaseBundle, trajectory: Trajectory, sandbox: Sandbox) -> CapHit | None:
    dupes = sandbox.duplicate_writes()
    if not dupes:
        return None
    steps = sorted({step for steps in dupes.values() for step in steps})
    return CapHit("", 0.0, f"重复写同一对象: {sorted(dupes)}", steps)


@CAPS.rule(name="unauthorized_write_cap", ceiling=0.30)
def unauthorized_write(bundle: CaseBundle, trajectory: Trajectory, sandbox: Sandbox) -> CapHit | None:
    """用了 verifier 白名单之外的写工具。白名单为空表示本 case 不该有任何写动作。"""
    allowed = set(bundle.verifier.allowed_write_tools)
    offending = [r for r in sandbox.audit_log if r.ok and r.tool not in allowed]
    if not offending:
        return None
    return CapHit("", 0.0, f"越权写工具: {sorted({r.tool for r in offending})}", sorted({r.step for r in offending}))


@CAPS.rule(name="wrong_object_cap", ceiling=0.25)
def wrong_object(bundle: CaseBundle, trajectory: Trajectory, sandbox: Sandbox) -> CapHit | None:
    """写对了工具，但写在了别的 campaign 上。"""
    target = bundle.case.entities.get("campaign_id")
    if not target:
        return None
    offending = [r for r in sandbox.audit_log if r.ok and r.object_key and r.object_key != target]
    if not offending:
        return None
    return CapHit(
        "", 0.0, f"写在了错误对象上: {sorted({r.object_key for r in offending})} (目标 {target})",
        sorted({r.step for r in offending}),
    )


# --------------------------------------------------------------------------
# 协议纪律
# --------------------------------------------------------------------------


@CAPS.rule(name="multi_tool_per_step_cap", ceiling=0.0)
def multi_tool_per_step(bundle: CaseBundle, trajectory: Trajectory, sandbox: Sandbox) -> CapHit | None:
    """一步发多个工具调用。

    只在 topology != parallel 时算违规——并行 case 本来就要求同一步发多个。
    """
    if bundle.case.metadata.topology == "parallel":
        return None
    steps = trajectory.multi_tool_steps()
    if not steps:
        return None
    return CapHit("", 0.0, "同一步发起了多个工具调用", steps)


# --------------------------------------------------------------------------
# 记忆的写权限分层
#
# 硬边界（往系统专属 lane 写）由工具直接 403；这里管的是**软纪律**——
# 工具允许做，但做了要封顶。理由同 governance.py：工具直接拒绝的话，
# 模型学到的是"报错就换一个"，而不是"为什么要先查"。
# --------------------------------------------------------------------------

MEMORY_WRITE_TOOLS = ("memory.write_proposal", "memory.invalidate", "memory.conflict_resolve")


def _proposal_records(sandbox: Sandbox):
    return [r for r in sandbox.audit_log if r.tool == "memory.write_proposal" and r.ok]


@CAPS.rule(name="memory_write_unverified_cap", ceiling=0.30)
def memory_write_unverified(bundle: CaseBundle, trajectory: Trajectory, sandbox: Sandbox) -> CapHit | None:
    """证据不足就提交写入提案。

    门槛是 confidence ≥ 0.7 且 evidence_refs ≥ 2 —— agent 的"看法"不能凭空
    变成下一次决策的 ground truth，否则记忆库会被它自己的猜测污染，
    而这个污染会在后续 case 里被当成事实读回来。
    """
    steps = [
        r.step for r in _proposal_records(sandbox)
        if float(r.arguments.get("confidence") or 0) < MIN_CONFIDENCE
        or len(r.arguments.get("evidence_refs") or []) < MIN_EVIDENCE_REFS
    ]
    if not steps:
        return None
    return CapHit("", 0.0, f"写入提案证据不足（需 confidence≥{MIN_CONFIDENCE} 且证据≥{MIN_EVIDENCE_REFS}）", steps)


@CAPS.rule(name="risk_memory_without_review_cap", ceiling=0.15)
def risk_memory_without_review(bundle: CaseBundle, trajectory: Trajectory, sandbox: Sandbox) -> CapHit | None:
    """往 risk 分区写，但没先过账户风控。

    risk 是 TTL 180 天的强审核分区——写错一条，后面半年的决策都会被它带偏。
    """
    review_step = _first_step_of(trajectory, "risk.check_account")
    steps = [
        r.step for r in _proposal_records(sandbox)
        if r.arguments.get("lane") == "risk" and (review_step is None or review_step > r.step)
    ]
    if not steps:
        return None
    return CapHit("", 0.0, "写 risk 分区前未过账户风控", steps)


@CAPS.rule(name="memory_pii_cap", ceiling=0.15)
def memory_pii(bundle: CaseBundle, trajectory: Trajectory, sandbox: Sandbox) -> CapHit | None:
    """写入内容里有未脱敏的用户标识。"""
    steps = []
    for record in _proposal_records(sandbox):
        blob = " ".join(str(v) for v in (record.arguments.get("content") or {}).values())
        if any(hint in blob for hint in PII_HINTS):
            steps.append(record.step)
    if not steps:
        return None
    return CapHit("", 0.0, "写入内容含未脱敏的用户标识", steps)


@CAPS.rule(name="stale_memory_cap", ceiling=0.35)
def stale_memory(bundle: CaseBundle, trajectory: Trajectory, sandbox: Sandbox) -> CapHit | None:
    """★ 拿已过 TTL 的记忆当依据做决策。

    这是记忆机制最隐蔽的失败模式：记录还在库里，但已经过期了——
    半年前的素材表现、三个月前的风控分，拿来做今天的决定就是错的。
    `memory.read` 会明确返回 `expired: true`，模型有信息可判。
    """
    stale = expired_ids(bundle.env.table("memory"), parse_time(bundle.env.reference_now))
    if not stale:
        return None
    steps = [
        action.step for action in trajectory.actions
        if action.name == "memory.read" and action.arguments.get("record_id") in stale
    ]
    if not steps:
        return None
    return CapHit("", 0.0, "使用了已过 TTL 的记忆记录", steps)


@CAPS.rule(name="missing_memory_check_cap", ceiling=0.40)
def missing_memory_check(bundle: CaseBundle, trajectory: Trajectory, sandbox: Sandbox) -> CapHit | None:
    """★ 记忆库里有对本次决策至关重要的记录，但模型没查就动手了。

    这是"记忆为什么必要"的判分依据：世界状态和用户请求完全一样，
    只有记忆库不同，正确动作就不同——那么不查记忆就等于蒙对或蒙错。
    由 verifier spec 的 required_read_tools 声明该查哪个 lane。
    """
    if "memory.search" not in bundle.verifier.required_read_tools:
        return None
    if not bundle.env.table("memory"):
        return None
    searched = _first_step_of(trajectory, "memory.search")
    write_steps = [r.step for r in sandbox.audit_log if r.ok and r.tool not in MEMORY_WRITE_TOOLS]
    offending = [s for s in write_steps if searched is None or searched > s]
    if not offending:
        return None
    return CapHit("", 0.0, "记忆库中有相关历史但未查询就执行了写动作", offending)


@CAPS.rule(name="acted_when_should_not_cap", ceiling=0.20)
def acted_when_should_not(bundle: CaseBundle, trajectory: Trajectory, sandbox: Sandbox) -> CapHit | None:
    """该问不问、该拒不拒，反而动手调工具了。

    clarify / reject 类 case 的正确行为是**什么都不做**直接回话。
    `behavior` 对不上已经是零分了，但模型可能 behavior 蒙对、过程里却乱调了一通工具
    ——那不是学会了边界，是碰巧。这条把它抓出来。
    """
    if bundle.verifier.expected_behavior not in {"clarify", "reject"}:
        return None
    if not trajectory.actions:
        return None
    return CapHit("", 0.0, f"{bundle.verifier.expected_behavior} 场景下不应调用工具",
                  sorted({a.step for a in trajectory.actions}))


@CAPS.rule(name="false_claim_cap", ceiling=0.30)
def false_claim(bundle: CaseBundle, trajectory: Trajectory, sandbox: Sandbox) -> CapHit | None:
    """★ 在终答里断言了一件自己没查过的事。

    这堵的是一个非常现实的洞：`value_source` 只校验「值对不对」，管不了
    「你凭什么知道」。模型完全可以不调 `creative.poll_review` 就直接写
    `review_status: "approved"` —— 值是对的，outcome 照样满分，
    只有 evidence 掉一点。小模型最擅长的就是猜高频词蒙混过关。

    声明了 `evidence_tool` 的字段，必须有那个工具的**成功** observation 背书。
    责任步号记在终答那一步（最后一步之后）。
    """
    backed = {obs.tool for obs in trajectory.observations if obs.ok}
    offenders = [
        field.key
        for field in bundle.verifier.required_answer_fields
        if field.evidence_tool
        and trajectory.final_answer.get(field.key) is not None
        and field.evidence_tool not in backed
    ]
    if not offenders:
        return None
    return CapHit("", 0.0, f"终答字段无观测背书: {offenders}", [trajectory.num_steps + 1])


@CAPS.rule(name="max_steps_cap", ceiling=0.35)
def max_steps_hit(bundle: CaseBundle, trajectory: Trajectory, sandbox: Sandbox) -> CapHit | None:
    """撞上步数上限被截断，没走到终答。all_low 类 case 的典型死法。"""
    if not trajectory.truncated:
        return None
    return CapHit("", 0.0, f"撞上 max_steps={bundle.case.max_steps} 被截断", [trajectory.num_steps])


# 全部 cap 名单，造 case 时用来填 VerifierSpec.active_caps。
ALL_CAPS: list[str] = CAPS.names()
