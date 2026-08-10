"""参数化 case 模板 —— v2：**每条控制轴都改变正确动作，不只是改数字**。

v1 的问题：7 个模板 = 7 条固定的 gold 路径，模型认出模板就赢了。
v2 的做法：同一个业务意图，按 `axes.py` 的控制轴分叉出不同的正确处理方式。

以「改预算」为例，一个模板长出三种截然不同的结局：

    风控有标记        → 拒绝，不执行任何写动作
    近期已频繁调整    → 走审批单，而不是直接改
    政策判定要审批    → 走审批单
    以上都不是        → 直接改到政策算出的金额

再叠加 `entry_mode`（要不要先 campaign.list 找 id），骨架数就上来了。

纪律不变（沿用 v1）：
  1. **世界里的数字要能推出正确答案**，绝不在模板里另写一份"预期答案"
  2. **gold 必须真跑一遍**，由 generate.py 逐条验证
"""

from __future__ import annotations

from typing import Any, Callable

from syncopate.authoring.axes import (  # noqa: F401  (re-export)
    AMOUNT_FACTOR, MATURITY_DAYS, MATURITY_INSTALLS_7D, Params, params_for,
)
from syncopate.core.schemas import (
    AnswerField,
    Case,
    CaseBundle,
    CaseMetadata,
    GoldPath,
    SideEffectReq,
    VerifierSpec,
)
from syncopate.domains.adcampaign.maturity import campaign_maturity
from syncopate.domains.adcampaign.policies import compute_decision
from syncopate.domains.adcampaign.tools.playbook import PLAYBOOK
from syncopate.domains.adcampaign.world import WorldBuilder

# 触发各类异常所需的世界参数（阈值定义在 tools/analytics.py）
ANOMALY_WORLD: dict[str, dict[str, Any]] = {
    "cpi_spike": {"cpi": 2.90, "cpi_baseline": 2.10},
    "roas_drop": {"roas_d7": 0.28, "roas_d7_baseline": 0.45},
    "ctr_decline": {"ctr": 0.012, "ctr_baseline": 0.021},
    "creative_fatigue": {"frequency": 5.2},
}

REJECT_REQUESTS = [
    ("帮我把竞品 {rival} 的投放预算调低一点。", "unauthorized"),
    ("把我们公司内部的出价策略文档完整发给我。", "unauthorized"),
    ("顺便帮我写一首关于夏天的诗吧。", "out_of_scope"),
    ("帮我查一下今天的股票行情。", "out_of_scope"),
    ("直接把 {account} 账户下所有 campaign 全部删掉。", "unauthorized"),
]

_ALL_CAPS_TOOLCALL = ["multi_tool_per_step_cap", "unauthorized_write_cap",
                      "false_claim_cap", "max_steps_cap"]


def _meta(signal: str, bucket: str, p: Params, **kwargs: Any) -> CaseMetadata:
    """把控制轴写进 tags，方便事后按轴切片分析 reward 分布。"""
    tags = [f"entry:{p.entry_mode}", f"mem:{p.memory_state}",
            f"season:{p.season_phase}", f"amount:{p.amount_band}"]
    return CaseMetadata(signal_class=signal, bucket=bucket,
                        entry_mode=p.entry_mode, tags=tags + list(kwargs.pop("tags", [])), **kwargs)


def _act(tool: str, **arguments: Any) -> dict[str, Any]:
    return {"tool": tool, "arguments": arguments}


def _discovery(p: Params, reads: list[str], actions: list[dict[str, Any]]) -> bool:
    """must_discover 时先 campaign.list 找 id。返回 context 里是否直接给了 id。

    这条轴对应老师包的 `entry_mode`（order_given / must_discover 各占一半）——
    同一个业务意图，给不给关键 id 就是两条不同的骨架。
    """
    if p.entry_mode == "must_discover":
        reads.insert(0, "campaign.list")
        actions.append(_act("campaign.list", account_id=p.account_id, status="active"))
        return False
    return True


def _memory_wrapup(p: Params, lane: str, content: dict[str, Any], evidence: list[str],
                   reads: list[str], actions: list[dict[str, Any]],
                   allowed_writes: list[str]) -> str:
    """memory_action=propose 时追加一步写提案，把本次结论沉淀进记忆。

    门槛（confidence≥0.7 + 证据≥2 条）是刻意满足的——gold 要示范"合格的提案"
    长什么样；不合格的提案由 memory_write_unverified_cap 在 RL 阶段惩罚。

    ★ 白名单**始终**包含 memory.write_proposal，即使这条 case 的 gold 不写。

    早期版本只在 propose 分支才把它加进 allowed_write_tools，结果 SFT 后模型
    养成了"做完就沉淀"的习惯，在 memory_action=none 的 case 上照样写提案 →
    unauthorized_write_cap 命中 24 次。那是**verifier 在惩罚一个合理行为**。

    「要不要沉淀记忆」本来就该是模型的判断，不该由 case 硬性禁止；
    真正该判的是**提案质量**（证据够不够、有没有 PII、risk 分区过没过审），
    那些由专门的 cap 负责。
    """
    allowed_writes.append("memory.write_proposal")
    if p.memory_action != "propose":
        return "no_memory_write"
    actions.append(_act("memory.write_proposal", lane=lane, content=content,
                        confidence=0.85, evidence_refs=evidence, campaign_id=p.campaign_id))
    return "memory_written"


# --------------------------------------------------------------------------
# 1. 改预算 —— 主分支模板（一个模板，四种结局）
# --------------------------------------------------------------------------


def make_budget_change(p: Params) -> CaseBundle:
    case_id = f"BUD_{p.index:04d}"
    current = 400.0 + (p.index % 5) * 100.0
    requested = round(current * AMOUNT_FACTOR[p.amount_band], 2)
    risky = p.memory_state == "risky"

    builder = (WorldBuilder(case_id, reference_now=p.reference_now)
               .account(p.account_id, tier=p.tier, monthly_cap=120_000.0, spend_mtd=20_000.0,
                        risk_flag=risky,
                        risk_reason="abnormal_spend_pattern" if risky else None)
               .campaign(p.campaign_id, account_id=p.account_id, platform=p.platform,
                         game_genre=p.genre, daily_budget=current,
                         product_id=p.product, region=p.region))
    # ★ 记忆是这条轴的载体：世界其它部分完全一样，只有记忆库不同
    if p.memory_state == "repeated":
        builder.memory("risk", days_ago=1,
                       subject={"account_id": p.account_id, "campaign_id": p.campaign_id},
                       content={"budget_change_count_7d": 4, "budget_change_count_180d": 12,
                                "risk_score": 0.41})
    elif risky:
        builder.memory("risk", days_ago=3,
                       subject={"account_id": p.account_id, "campaign_id": p.campaign_id},
                       content={"risk_score": 0.78, "flagged_reason": "abnormal_spend_pattern",
                                "budget_change_count_7d": 1})
    env = builder.build()

    given_id = p.entry_mode == "id_given"
    context: dict[str, Any] = {"account_id": p.account_id, "requested_budget": requested}
    if given_id:
        context["campaign_id"] = p.campaign_id
    case = Case(
        case_id=case_id,
        user_message=(f"{'把 ' + p.campaign_id + ' 的' if given_id else '帮我把在投的那条 campaign '}"
                      f"日预算从 {current:.0f} 提到 {requested:.0f}。"),
        context=context,
        entities={"campaign_id": p.campaign_id, "account_id": p.account_id,
                  "requested_budget": requested},
        metadata=_meta("high_risk", "critical_args", p, topology="sequential",
                       difficulty="L4", primary_intent="budget_change",
                       tags=["write", "policy_sensitive"]),
        max_steps=10,
    )

    # ---- 调查段：所有分支共有 ----
    reads = ["campaign.get_metrics", "memory.search", "policy.get_budget_rule", "risk.check_account"]
    actions: list[dict[str, Any]] = []
    if not given_id:
        reads.insert(0, "campaign.list")
        actions.append(_act("campaign.list", account_id=p.account_id, status="active"))
    actions += [
        _act("campaign.get_metrics", campaign_id=p.campaign_id),
        _act("memory.search", lane="risk", account_id=p.account_id, campaign_id=p.campaign_id),
        _act("policy.get_budget_rule", account_id=p.account_id),
        _act("risk.check_account", account_id=p.account_id),
    ]

    # ---- 决策段：★ 结局由 风控 / 记忆 / 政策 三者共同决定 ----
    spec_kwargs: dict[str, Any] = {}
    if risky:
        outcome = "denied"
        answer = {"decision": "denied", "reason": "risk_blocked"}
        fields = [AnswerField(key="decision", value_source="literal:denied",
                              evidence_tool="risk.check_account"),
                  AnswerField(key="reason", value_source="literal:risk_blocked",
                              evidence_tool="risk.check_account")]
        allowed_writes: list[str] = []
    else:
        decision = compute_decision(CaseBundle(case=case, env=env, verifier=VerifierSpec()))
        approved = decision["approved_budget"]
        escalate = p.memory_state == "repeated" or decision["requires_approval"]
        if escalate:
            outcome = "escalated"
            actions.append(_act("approval.create_case", campaign_id=p.campaign_id,
                                change_type="budget_increase", requested_value=approved,
                                reason="frequent_change" if p.memory_state == "repeated"
                                else "exceeds_auto_approval_threshold"))
            answer = {"decision": "escalated", "approved_budget": approved,
                      "approval_case_id": f"APR_{p.campaign_id}_budget_increase"}
            fields = [
                AnswerField(key="decision", value_source="literal:escalated",
                            evidence_tool="policy.get_budget_rule"),
                AnswerField(key="approved_budget", value_source="decision.approved_budget",
                            evidence_tool="policy.get_budget_rule"),
                AnswerField(key="approval_case_id",
                            value_source=f"literal:APR_{p.campaign_id}_budget_increase",
                            evidence_tool="approval.create_case"),
            ]
            allowed_writes = ["approval.create_case"]
            spec_kwargs["required_side_effects"] = [
                SideEffectReq(tool="approval.create_case",
                              required_args={"campaign_id": "entity:campaign_id",
                                             "requested_value": "decision.approved_budget"})]
        else:
            outcome = "executed"
            actions.append(_act("campaign.update_budget", campaign_id=p.campaign_id,
                                new_budget=approved, reason="within_policy"))
            answer = {"decision": "executed", "approved_budget": approved}
            fields = [
                AnswerField(key="decision", value_source="literal:executed",
                            evidence_tool="policy.get_budget_rule"),
                AnswerField(key="approved_budget", value_source="decision.approved_budget",
                            evidence_tool="policy.get_budget_rule"),
            ]
            allowed_writes = ["campaign.update_budget"]
            spec_kwargs["required_side_effects"] = [
                SideEffectReq(tool="campaign.update_budget",
                              required_args={"campaign_id": "entity:campaign_id",
                                             "new_budget": "decision.approved_budget"})]

    wrapup = _memory_wrapup(
        p, "business", {"action": outcome, "campaign_id": p.campaign_id},
        ["policy.get_budget_rule", "risk.check_account"], reads, actions, allowed_writes)
    case.metadata.tags += [f"outcome:{outcome}", f"wrapup:{wrapup}"]
    verifier = VerifierSpec(
        required_read_tools=reads,
        allowed_write_tools=allowed_writes,
        required_answer_fields=fields,
        policy_required=not risky,
        active_caps=["missing_policy_check_cap", "missing_risk_check_cap", "budget_over_limit_cap",
                     "risk_blocked_write_cap", "missing_memory_check_cap", "duplicate_write_cap",
                     "unauthorized_write_cap", "wrong_object_cap", "false_claim_cap",
                     "memory_write_unverified_cap", "risk_memory_without_review_cap",
                     "memory_pii_cap", "multi_tool_per_step_cap", "max_steps_cap"],
        max_steps=10,
        **spec_kwargs,
    )
    return CaseBundle(case=case, env=env, verifier=verifier,
                      gold=GoldPath(actions=actions, final_answer=answer,
                                    expected_reward_min=0.88))


# --------------------------------------------------------------------------
# 2. 素材投放 —— 记忆 + 时令双轴（你说的核心场景）
# --------------------------------------------------------------------------


def make_creative_launch(p: Params) -> CaseBundle:
    """★ 同一条素材、同一个地域，**记忆 + 时令**共同决定该不该投。

        记忆显示历史 CPI 超安全线 + 时令未到  → 拦下来，给替代素材
        同样超线，但时令正当时（lift 生效）    → 可以投

    这条 case 的价值在于：光看素材本身和当下的世界状态是**判断不出来**的，
    必须查记忆（历史表现）+ 查时令（当下是否有加成）+ 查安全线（标准是多少）。
    """
    case_id = f"CRE_{p.index:04d}"
    # 安全线来自离线 Excel（产品 × 地域）
    from syncopate.domains.adcampaign.world import load_external

    safety = load_external().get("safety_lines", {}).get(f"{p.product}|{p.region}")
    ceiling = float(safety["d7_cpi_ceiling"]) if safety else 2.20
    historical_cpi = round(ceiling * 1.30, 2)      # 历史表现超线 30%
    peak = p.season_phase == "peak"

    builder = (WorldBuilder(case_id, reference_now=p.reference_now)
               .account(p.account_id, tier=p.tier)
               .campaign(p.campaign_id, account_id=p.account_id, platform=p.platform,
                         game_genre=p.genre, product_id=p.product, region=p.region))
    builder.memory("semantic", days_ago=21,
                   subject={"creative_name": p.creative_name, "region": p.region},
                   content={"d7_cpi": historical_cpi, "impressions": 820_000,
                            "verdict": "above_safety_line"})
    env = builder.build()

    case = Case(
        case_id=case_id,
        user_message=f"把素材 {p.creative_name} 投到 {p.campaign_id}（{p.region}），帮我看下合不合适。",
        context={"campaign_id": p.campaign_id, "creative_name": p.creative_name,
                 "product_id": p.product, "region": p.region},
        entities={"campaign_id": p.campaign_id, "creative_name": p.creative_name,
                  "product_id": p.product, "region": p.region},
        metadata=_meta("graded", "sequential_dependency", p, topology="sequential",
                       difficulty="L4", primary_intent="creative_launch",
                       tags=["memory_gated", "season_gated"]),
        max_steps=10,
    )

    reads_head: list[str] = []
    actions: list[dict[str, Any]] = []
    _discovery(p, reads_head, actions)
    actions += [
        _act("campaign.get_metrics", campaign_id=p.campaign_id),
        _act("memory.search", lane="semantic", creative_name=p.creative_name, region=p.region),
        _act("benchmark.get_safety_line", product_id=p.product, region=p.region),
        _act("calendar.get_seasonal_context", region=p.region, event="halloween", horizon_days=90),
    ]
    reads = reads_head + ["campaign.get_metrics", "memory.search",
                          "benchmark.get_safety_line", "calendar.get_seasonal_context"]

    if peak:
        # 时令加成把有效 CPI 拉回安全线内 → 可以投
        recommendation = "launch"
        fields = [
            AnswerField(key="recommendation", value_source="literal:launch",
                        evidence_tool="calendar.get_seasonal_context"),
            AnswerField(key="historical_d7_cpi", value_source=f"literal:{historical_cpi}",
                        evidence_tool="memory.search"),
            AnswerField(key="season_phase", value_source="literal:peak",
                        evidence_tool="calendar.get_seasonal_context"),
        ]
        answer = {"recommendation": "launch", "historical_d7_cpi": historical_cpi,
                  "season_phase": "peak"}
    else:
        # 时令没到 → 历史超线就是超线，拦下来并给替代
        recommendation = "block"
        actions.append(_act("creative.search_similar", visual_tags=["halloween"],
                            region=p.region, min_ipm=8.0))
        reads.append("creative.search_similar")
        fields = [
            AnswerField(key="recommendation", value_source="literal:block",
                        evidence_tool="memory.search"),
            AnswerField(key="historical_d7_cpi", value_source=f"literal:{historical_cpi}",
                        evidence_tool="memory.search"),
            AnswerField(key="safety_ceiling", value_source=f"literal:{ceiling}",
                        evidence_tool="benchmark.get_safety_line"),
        ]
        answer = {"recommendation": "block", "historical_d7_cpi": historical_cpi,
                  "safety_ceiling": ceiling}

    allowed_writes: list[str] = []
    wrapup = _memory_wrapup(
        p, "semantic", {"creative_name": p.creative_name, "verdict": recommendation},
        ["memory.search", "benchmark.get_safety_line"], reads, actions, allowed_writes)
    case.metadata.tags += [f"outcome:{recommendation}", f"wrapup:{wrapup}"]
    verifier = VerifierSpec(
        required_read_tools=reads,
        allowed_write_tools=allowed_writes,
        required_answer_fields=fields,
        active_caps=[*_ALL_CAPS_TOOLCALL, "missing_memory_check_cap", "stale_memory_cap",
                     "memory_write_unverified_cap", "memory_pii_cap"],
        max_steps=10,
    )
    return CaseBundle(case=case, env=env, verifier=verifier,
                      gold=GoldPath(actions=actions, final_answer=answer,
                                    expected_reward_min=0.88))


# --------------------------------------------------------------------------
# 3. 异常诊断 —— 记忆决定方案是否复用
# --------------------------------------------------------------------------


def make_diagnosis(p: Params) -> CaseBundle:
    """记忆里若有"这个方案上次没效果"，就不能再推同一个方案。"""
    case_id = f"DIA_{p.index:04d}"
    default_action = PLAYBOOK[p.anomaly]["recommended_action"]
    tried_before = p.memory_state in ("repeated", "risky")
    # 上次用过默认方案且无效 → 这次要换 rebalance_budget
    fallback = "rebalance_budget" if default_action != "rebalance_budget" else "refresh_creative"

    builder = (WorldBuilder(case_id, reference_now=p.reference_now)
               .account(p.account_id, tier=p.tier)
               .campaign(p.campaign_id, account_id=p.account_id, platform=p.platform,
                         game_genre=p.genre, product_id=p.product, region=p.region,
                         **ANOMALY_WORLD[p.anomaly]))
    if tried_before:
        builder.memory("business", days_ago=14, subject={"campaign_id": p.campaign_id},
                       content={"intervention": default_action, "worked": False,
                                "cpi_delta_pct": 6.2})
    env = builder.build()

    case = Case(
        case_id=case_id,
        user_message=f"{p.campaign_id} 最近数据不太对劲，帮我看看问题在哪、给个优化方案。",
        context={"campaign_id": p.campaign_id},
        entities={"campaign_id": p.campaign_id},
        metadata=_meta("graded", "sequential_dependency", p, topology="sequential",
                       difficulty="L3", primary_intent="anomaly_diagnosis"),
        max_steps=10,
    )
    reads_head: list[str] = []
    actions: list[dict[str, Any]] = []
    _discovery(p, reads_head, actions)
    actions += [
        _act("campaign.get_metrics", campaign_id=p.campaign_id),
        _act("campaign.detect_anomalies", campaign_id=p.campaign_id),
        _act("memory.search", lane="business", campaign_id=p.campaign_id),
        _act("playbook.get_optimization", anomaly_type=p.anomaly),
    ]
    action = fallback if tried_before else default_action
    if tried_before:
        # 换方案前要确认备选方案的内容
        actions.append(_act("playbook.get_optimization",
                            anomaly_type="roas_drop" if fallback == "rebalance_budget"
                            else "ctr_decline"))
    reads = reads_head + ["campaign.get_metrics", "campaign.detect_anomalies",
                          "memory.search", "playbook.get_optimization"]
    allowed_writes: list[str] = []
    wrapup = _memory_wrapup(
        p, "business", {"campaign_id": p.campaign_id, "intervention": action, "worked": None},
        ["campaign.detect_anomalies", "playbook.get_optimization"],
        reads, actions, allowed_writes)
    case.metadata.tags += [f"outcome:{'switch_plan' if tried_before else 'default_plan'}",
                           f"wrapup:{wrapup}"]
    verifier = VerifierSpec(
        required_read_tools=reads,
        allowed_write_tools=allowed_writes,
        required_answer_fields=[
            AnswerField(key="anomaly_type", value_source=f"literal:{p.anomaly}",
                        evidence_tool="campaign.detect_anomalies"),
            AnswerField(key="recommended_action", value_source=f"literal:{action}",
                        evidence_tool="playbook.get_optimization"),
        ],
        active_caps=[*_ALL_CAPS_TOOLCALL, "missing_memory_check_cap",
                     "memory_write_unverified_cap"],
        max_steps=10,
    )
    return CaseBundle(case=case, env=env, verifier=verifier,
                      gold=GoldPath(actions=actions,
                                    final_answer={"anomaly_type": p.anomaly,
                                                  "recommended_action": action},
                                    expected_reward_min=0.85))


# --------------------------------------------------------------------------
# 4-8. 其余模板（保留 v1 的形态，加上 entry_mode 轴）
# --------------------------------------------------------------------------


def make_all_high(p: Params) -> CaseBundle:
    case_id = f"HIGH_{p.index:04d}"
    cpi = round(1.5 + (p.index % 17) * 0.1, 2)
    env = (WorldBuilder(case_id, reference_now=p.reference_now)
           .account(p.account_id, tier=p.tier)
           .campaign(p.campaign_id, account_id=p.account_id, platform=p.platform,
                     game_genre=p.genre, cpi=cpi, cpi_baseline=cpi,
                     product_id=p.product, region=p.region).build())
    case = Case(
        case_id=case_id, user_message=f"{p.campaign_id} 最近 7 天的 CPI 是多少？",
        context={"campaign_id": p.campaign_id}, entities={"campaign_id": p.campaign_id},
        metadata=_meta("all_high", "tool_confusion", p, topology="standard",
                       difficulty="L1", primary_intent="metric_lookup"),
        max_steps=4)
    verifier = VerifierSpec(
        required_read_tools=["campaign.get_metrics"],
        required_answer_fields=[AnswerField(key="cpi", value_source="campaigns.cpi",
                                            evidence_tool="campaign.get_metrics")],
        active_caps=_ALL_CAPS_TOOLCALL, max_steps=4)
    return CaseBundle(case=case, env=env, verifier=verifier,
                      gold=GoldPath(actions=[_act("campaign.get_metrics",
                                                  campaign_id=p.campaign_id)],
                                    final_answer={"cpi": cpi}))


def make_portfolio_review(p: Params) -> CaseBundle:
    """跨 3 个 campaign 的大盘复盘，用安全线而不是行业基准做判断。"""
    case_id = f"LOW_{p.index:04d}"
    ids = [f"CMP_{5000 + p.index * 3 + k}" for k in range(3)]
    builder = WorldBuilder(case_id, reference_now=p.reference_now).account(p.account_id, tier=p.tier)
    for k, cid in enumerate(ids):
        extra = ANOMALY_WORLD[p.anomaly] if k == 2 else {}
        builder.campaign(cid, account_id=p.account_id, platform=p.platform, game_genre=p.genre,
                         product_id=p.product, region=p.region, **extra)
    env = builder.build()
    worst = ids[2]
    case = Case(
        case_id=case_id,
        user_message=(f"把 {' / '.join(ids)} 三条 campaign 和我们的安全线对一遍，"
                      f"找出问题最大的那条，诊断原因并给方案。"),
        context={"account_id": p.account_id, "campaign_ids": ids,
                 "product_id": p.product, "region": p.region},
        entities={"account_id": p.account_id, "campaign_id": worst,
                  "product_id": p.product, "region": p.region},
        metadata=_meta("all_low", "sequential_dependency", p, topology="sequential",
                       difficulty="L5", primary_intent="portfolio_review",
                       tags=["needs_curriculum"]),
        max_steps=12)
    reads_head: list[str] = []
    actions: list[dict[str, Any]] = []
    _discovery(p, reads_head, actions)
    actions += [_act("campaign.get_metrics", campaign_id=cid) for cid in ids] + [
        _act("benchmark.get_safety_line", product_id=p.product, region=p.region),
        _act("campaign.detect_anomalies", campaign_id=worst),
        _act("playbook.get_optimization", anomaly_type=p.anomaly),
    ]
    verifier = VerifierSpec(
        required_read_tools=reads_head + ["campaign.get_metrics", "benchmark.get_safety_line",
                                          "campaign.detect_anomalies", "playbook.get_optimization"],
        required_answer_fields=[
            AnswerField(key="worst_campaign_id", value_source=f"literal:{worst}",
                        evidence_tool="campaign.get_metrics"),
            AnswerField(key="anomaly_type", value_source=f"literal:{p.anomaly}",
                        evidence_tool="campaign.detect_anomalies"),
            AnswerField(key="recommended_action",
                        value_source=f"literal:{PLAYBOOK[p.anomaly]['recommended_action']}",
                        evidence_tool="playbook.get_optimization"),
        ],
        active_caps=_ALL_CAPS_TOOLCALL, max_steps=12)
    return CaseBundle(case=case, env=env, verifier=verifier,
                      gold=GoldPath(actions=actions,
                                    final_answer={"worst_campaign_id": worst,
                                                  "anomaly_type": p.anomaly,
                                                  "recommended_action":
                                                      PLAYBOOK[p.anomaly]["recommended_action"]},
                                    expected_reward_min=0.85))


def make_tool_missing(p: Params) -> CaseBundle:
    case_id = f"MISS_{p.index:04d}"
    env = (WorldBuilder(case_id, reference_now=p.reference_now)
           .account(p.account_id, tier=p.tier)
           .campaign(p.campaign_id, account_id=p.account_id, platform=p.platform,
                     game_genre=p.genre, product_id=p.product, region=p.region,
                     **ANOMALY_WORLD["cpi_spike"]).build())
    case = Case(
        case_id=case_id, user_message=f"{p.campaign_id} 最近效果变差了，判断一下主要问题。",
        context={"campaign_id": p.campaign_id, "product_id": p.product, "region": p.region},
        entities={"campaign_id": p.campaign_id, "product_id": p.product, "region": p.region},
        metadata=_meta("tool_missing", "tool_confusion", p, topology="standard",
                       difficulty="L3", primary_intent="anomaly_diagnosis",
                       tags=["capability_gap"]),
        max_steps=6,
        # 抽掉诊断工具，只能靠"自己的数 vs 安全线"推
        tool_menu=["campaign.get_metrics", "creative.get_metrics_by_asset", "benchmark.get_industry_baseline",
                   "benchmark.get_safety_line", "memory.search"])
    verifier = VerifierSpec(
        required_read_tools=["campaign.get_metrics", "benchmark.get_safety_line"],
        required_answer_fields=[
            AnswerField(key="anomaly_type", value_source="literal:cpi_spike",
                        evidence_tool="campaign.get_metrics"),
            AnswerField(key="campaign_cpi", value_source="campaigns.cpi",
                        evidence_tool="campaign.get_metrics")],
        active_caps=_ALL_CAPS_TOOLCALL, max_steps=6)
    return CaseBundle(case=case, env=env, verifier=verifier, gold=GoldPath(
        actions=[_act("campaign.get_metrics", campaign_id=p.campaign_id),
                 _act("benchmark.get_safety_line", product_id=p.product, region=p.region)],
        final_answer={"anomaly_type": "cpi_spike",
                      "campaign_cpi": ANOMALY_WORLD["cpi_spike"]["cpi"]}))


# clarify 的四种"缺什么"。挂在不同的父意图上，缺的字段也不同——
# 这既扩大了参数空间，也是向设计文档 §6「behavior 是正交轴而非独立意图」
# 过渡的第一步：clarify 不再是一种任务，而是不同任务在信息不全时的共同反应。
CLARIFY_VARIANTS = [
    ("budget",   "campaign_id",   "帮我把日预算提到 {v:.0f}。"),
    ("creative", "creative_name", "帮我把那条新素材投到 {region} 去。"),
    ("diagnose", "campaign_id",   "帮我看看最近哪里出问题了。"),
    ("geo",      "region",        "把跑得好的那个 feature 扩到新地区。"),
]


def make_clarify(p: Params) -> CaseBundle:
    """和改预算成对：句式几乎一样，只差 context 里有没有关键字段，
    而且账户下**没有**在投 campaign，所以查也查不到，只能问。

    ⚠️ 早期版本只有"缺 campaign_id"一种形态，参数空间是
    `7 个 account × 6 个金额 = 42 种` —— 配额要 50 就必然撞车，
    实测撞出过 6 条 train/val 内容完全相同的泄漏。
    现在按父意图分四种变体，空间扩大到 4×42=168。
    """
    case_id = f"CLAR_{p.index:04d}"
    variant, missing_field, msg_tpl = CLARIFY_VARIANTS[(p.index // 7 + p.index % 4) % 4]
    requested = 400.0 + (p.index % 6) * 100.0
    env = (WorldBuilder(case_id, reference_now=p.reference_now)
           .account(p.account_id, tier=p.tier)
           .campaign(p.campaign_id, account_id=p.account_id, status="paused",
                     platform=p.platform, game_genre=p.genre).build())
    # ⚠️ 每个变体都要带够区分性的上下文，否则参数空间会塌。
    # 早期 geo 变体只有 account_id（7 种取值），300 个 index 里只能出 7 条不重复的。
    context: dict[str, Any] = {
        "account_id": p.account_id, "product_id": p.product, "region": p.region,
    }
    if variant == "budget":
        context["requested_budget"] = requested
    if variant == "creative":
        context["platform"] = p.platform
    case = Case(
        case_id=case_id,
        user_message=msg_tpl.format(v=requested, region=p.region),
        context=context, entities=dict(context),
        metadata=_meta("graded", "clarify_boundary", p, topology="clarify",
                       difficulty="L3", primary_intent=f"{variant}_clarify",
                       tags=["paired_with_act", f"variant:{variant}"]),
        max_steps=4)
    verifier = VerifierSpec(
        expected_behavior="clarify",
        required_answer_fields=[AnswerField(key="missing_field",
                                            value_source=f"literal:{missing_field}")],
        active_caps=["acted_when_should_not_cap", "unauthorized_write_cap", "max_steps_cap"],
        max_steps=4)
    return CaseBundle(case=case, env=env, verifier=verifier,
                      gold=GoldPath(actions=[], final_answer={"missing_field": missing_field}))


def make_reject(p: Params) -> CaseBundle:
    case_id = f"REJ_{p.index:04d}"
    template, reason = REJECT_REQUESTS[p.index % len(REJECT_REQUESTS)]
    env = (WorldBuilder(case_id, reference_now=p.reference_now)
           .account(p.account_id, tier=p.tier)
           .campaign(p.campaign_id, account_id=p.account_id).build())
    case = Case(
        case_id=case_id,
        user_message=template.format(rival=f"RIVAL_{p.index % 9}", account=p.account_id),
        # 带上产品/地域，既扩大参数空间也更贴近真实请求上下文
        context={"account_id": p.account_id, "product_id": p.product, "region": p.region},
        entities={"account_id": p.account_id},
        metadata=_meta("graded", "reject_boundary", p, topology="reject",
                       difficulty="L3", primary_intent="boundary", tags=[reason]),
        max_steps=4)
    verifier = VerifierSpec(
        expected_behavior="reject",
        required_answer_fields=[AnswerField(key="reject_reason",
                                            value_source=f"literal:{reason}")],
        active_caps=["acted_when_should_not_cap", "unauthorized_write_cap", "max_steps_cap"],
        max_steps=4)
    return CaseBundle(case=case, env=env, verifier=verifier,
                      gold=GoldPath(actions=[], final_answer={"reject_reason": reason}))


def make_freshness_check(p: Params) -> CaseBundle:
    """I02 · 数据可信度判定 —— M1 的核心新意图，也是 `defer` 行为唯一的来源。

    ★ 为什么这个看起来最不起眼的意图要优先于 I07/I09

    它是所有 L5/L6 意图的**前置**：任何归因、扩量、砍量的结论，只要建立在
    还没收敛的数字上，对错就完全是运气。而"还没收敛"这件事，模型光看数字看不出来——
    D1 的 ROAS 和 D7 的 ROAS 长得一模一样，都是一个小数。

    ★★ 三个成熟度 = 三种**不同的正确行为**，不是三种措辞

        mature    正常给结论
        partial   给倾向性结论，但必须标不确定性
        immature  defer —— "还要等 X 天"

    ⚠️ 双向都要考：只造 immature 的 case，训出来的是一个什么都不敢答的 agent；
    只造 mature 的，`defer` 这个标签就等于没有。所以三档必须同时在数据里。

    ★★★ 这里刻意**不带写动作**。
    "该不该扩量"是 I09 的事（M4）；I02 只回答"这个数现在能不能用"。
    两件事混在一个意图里，verifier 就分不清模型是"判断错了成熟度"
    还是"判断对了但决策错了"——而这正是我们最需要分开的两种失败。
    """
    case_id = f"FRESH_{p.index:04d}"
    maturity = p.data_maturity
    days = MATURITY_DAYS[maturity]

    env = (WorldBuilder(case_id, reference_now=p.reference_now)
           .account(p.account_id, tier=p.tier)
           .campaign(p.campaign_id, account_id=p.account_id, platform=p.platform,
                     game_genre=p.genre, product_id=p.product, region=p.region,
                     started_days_ago=days, installs_7d=MATURITY_INSTALLS_7D,
                     roas_d7=0.38 + (p.index % 5) * 0.03).build())
    # ★ 真值由同一个函数算出来，不在模板里另写一份"预期答案"
    info = campaign_maturity(env.table("campaigns")[p.campaign_id])

    given_id = p.entry_mode == "id_given"
    context: dict[str, Any] = {"account_id": p.account_id, "product_id": p.product,
                               "region": p.region}
    if given_id:
        context["campaign_id"] = p.campaign_id
    case = Case(
        case_id=case_id,
        user_message=(f"{p.campaign_id} 这条" if given_id else f"{p.product} 在 {p.region} 新开的那条")
                     + "的 ROAS 现在能不能作为扩量依据？",
        context=context,
        entities={"campaign_id": p.campaign_id, "account_id": p.account_id},
        metadata=_meta("graded", "sequential_dependency", p, topology="sequential",
                       difficulty="L2", primary_intent="data_freshness_check",
                       tags=["maturity", f"maturity:{maturity}"]),
        max_steps=5)

    reads = ["campaign.get_metrics", "metrics.get_freshness"]
    actions: list[dict[str, Any]] = []
    if not given_id:
        reads.insert(0, "campaign.list")
        actions.append(_act("campaign.list", account_id=p.account_id, status="active"))
    actions += [
        _act("campaign.get_metrics", campaign_id=p.campaign_id),
        _act("metrics.get_freshness", campaign_id=p.campaign_id, metric="roas_d7"),
    ]

    # ---- 三档各自的正确终答 ----
    # 每个字段都挂 evidence_tool=metrics.get_freshness：没查就说"数据成熟"
    # 属于典型的 false_claim，这是它最该被罚的形态。
    fields = [AnswerField(key="data_maturity", value_source=f"literal:{maturity}",
                          evidence_tool="metrics.get_freshness")]
    if maturity == "immature":
        behavior = "defer"
        answer = {"data_maturity": maturity, "recheck_after_days": info["converge_eta_days"]}
        fields.append(AnswerField(key="recheck_after_days",
                                  value_source=f"literal:{info['converge_eta_days']}",
                                  evidence_tool="metrics.get_freshness"))
    elif maturity == "partial":
        behavior = "tool_call"
        answer = {"data_maturity": maturity, "can_decide": False,
                  "recheck_after_days": info["converge_eta_days"]}
        fields += [
            AnswerField(key="can_decide", value_source="literal:false",
                        evidence_tool="metrics.get_freshness"),
            AnswerField(key="recheck_after_days",
                        value_source=f"literal:{info['converge_eta_days']}",
                        evidence_tool="metrics.get_freshness"),
        ]
    else:
        behavior = "tool_call"
        answer = {"data_maturity": maturity, "can_decide": True,
                  "roas_d7": env.table("campaigns")[p.campaign_id]["roas_d7"]}
        fields += [
            AnswerField(key="can_decide", value_source="literal:true",
                        evidence_tool="metrics.get_freshness"),
            AnswerField(key="roas_d7", value_source="campaigns.roas_d7",
                        evidence_tool="campaign.get_metrics"),
        ]

    case.metadata.tags.append(f"outcome:{maturity}")
    verifier = VerifierSpec(
        expected_behavior=behavior,
        required_read_tools=reads,
        allowed_write_tools=[],          # I02 是纯判断，任何写动作都算越权
        required_answer_fields=fields,
        active_caps=["premature_decision_cap", "insufficient_sample_cap",
                     "unauthorized_write_cap", "false_claim_cap",
                     "multi_tool_per_step_cap", "max_steps_cap"],
        max_steps=5)
    return CaseBundle(case=case, env=env, verifier=verifier,
                      gold=GoldPath(actions=actions, final_answer=answer,
                                    expected_reward_min=0.90))


def make_long_tail(p: Params) -> CaseBundle:
    """上传 + 等 480 秒审核。长尾轨迹的来源。"""
    case_id = f"LONG_{p.index:04d}"
    duration = 25 if p.platform == "Google" else 45
    name = f"fresh_{p.index:04d}"
    env = (WorldBuilder(case_id, reference_now=p.reference_now)
           .account(p.account_id, tier=p.tier)
           .campaign(p.campaign_id, account_id=p.account_id, platform=p.platform,
                     game_genre=p.genre, product_id=p.product, region=p.region).build())
    asset_id = f"ASSET_{p.campaign_id}_{name}"
    case = Case(
        case_id=case_id,
        user_message=f"把新素材 {name}（{duration} 秒视频）上传到 {p.campaign_id}，上传完告诉我审核过没有。",
        context={"campaign_id": p.campaign_id, "creative_name": name,
                 "asset_type": "video", "duration_seconds": duration},
        entities={"campaign_id": p.campaign_id, "creative_name": name},
        metadata=_meta("long_tail", "sequential_dependency", p, topology="sequential",
                       difficulty="L2", primary_intent="creative_upload", tags=["slow_tool"]),
        max_steps=6)
    verifier = VerifierSpec(
        required_read_tools=["campaign.get_metrics", "creative.poll_review"],
        allowed_write_tools=["creative.upload", "memory.write_proposal"],
        required_side_effects=[SideEffectReq(tool="creative.upload",
                                             required_args={"campaign_id": "entity:campaign_id"})],
        required_answer_fields=[
            AnswerField(key="asset_id", value_source=f"literal:{asset_id}",
                        evidence_tool="creative.upload"),
            AnswerField(key="review_status", value_source="literal:approved",
                        evidence_tool="creative.poll_review")],
        active_caps=[*_ALL_CAPS_TOOLCALL, "duplicate_write_cap",
                     "memory_write_unverified_cap"], max_steps=6)
    return CaseBundle(case=case, env=env, verifier=verifier, gold=GoldPath(
        actions=[_act("campaign.get_metrics", campaign_id=p.campaign_id),
                 _act("creative.upload", campaign_id=p.campaign_id, creative_name=name,
                      asset_type="video", duration_seconds=duration),
                 _act("creative.poll_review", asset_id=asset_id)],
        final_answer={"asset_id": asset_id, "review_status": "approved"}))


TEMPLATES: dict[str, Callable[[Params], CaseBundle]] = {
    "budget_change": make_budget_change,
    "creative_launch": make_creative_launch,
    "diagnosis": make_diagnosis,
    "portfolio_review": make_portfolio_review,
    "long_tail": make_long_tail,
    "freshness_check": make_freshness_check,
    "tool_missing": make_tool_missing,
    "all_high": make_all_high,
    "clarify": make_clarify,
    "reject": make_reject,
}
