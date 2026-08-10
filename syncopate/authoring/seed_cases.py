"""6 条种子 case，覆盖五类 reward 信号形态 + 工具缺失。

这些是**手写**的模板 case，作用是把「引擎 + 域」端到端验证一遍，
并且给后面批量生成 case 的 generator 当参照。规模化生成是下一步的事。

设计纪律（对应 AdCampaignAgent 的 RL_Data_Design §9）：

  1. **世界里的数字要能推出正确答案**。不要在 case 里另写一份「预期答案」，
     那样世界和答案会不同步。比如 approved_budget=750 是政策规则套上
     daily_budget=500 算出来的，改任何一边另一边自动跟着变。
  2. **gold 只约束 reward 真正在乎的东西**，不强求用户没说、context 没给的字段。
  3. **gold 必须真跑一遍**。老师包里 2737 条 gold 的分数全部恰好 = 1.0，
     是预烤进文件的；我们要求跑出来才算数（见 tests/domains/test_seed_cases.py）。
"""

from __future__ import annotations

from syncopate.core.runner import PlannedCall
from syncopate.core.schemas import (
    AnswerField,
    Case,
    CaseBundle,
    CaseMetadata,
    GoldPath,
    SideEffectReq,
    VerifierSpec,
)
from syncopate.domains.adcampaign.world import WorldBuilder


def _bundle(case: Case, env, verifier: VerifierSpec, gold: GoldPath) -> CaseBundle:
    return CaseBundle(case=case, env=env, verifier=verifier, gold=gold)


# --------------------------------------------------------------------------
# 1. all_high —— 弱信号反例
# --------------------------------------------------------------------------


def case_all_high() -> CaseBundle:
    """单步查询。任何会调工具的模型都能拿满分。

    它存在的意义是**反例**：组内 8 条 rollout 分数全一样 → GRPO 的 advantage ≈ 0
    → 这条 case 贡献的梯度接近于零。用来实测「弱信号 case 占比多少会拖垮训练」。
    """
    case_id = "SIG_HIGH_001"
    env = (WorldBuilder(case_id)
           .account("ACC_01")
           .campaign("CMP_1024", account_id="ACC_01", platform="Meta", game_genre="puzzle", cpi=2.10)
           .build())
    case = Case(
        case_id=case_id,
        user_message="CMP_1024 最近 7 天的 CPI 是多少？",
        context={"campaign_id": "CMP_1024"},
        entities={"campaign_id": "CMP_1024"},
        metadata=CaseMetadata(signal_class="all_high", bucket="tool_confusion",
                              topology="standard", difficulty="L1", primary_intent="metric_lookup"),
        max_steps=4,
    )
    verifier = VerifierSpec(
        required_read_tools=["campaign.get_metrics"],
        required_answer_fields=[AnswerField(key="cpi", value_source="campaigns.cpi")],
        active_caps=["multi_tool_per_step_cap", "unauthorized_write_cap", "max_steps_cap"],
        max_steps=4,
    )
    gold = GoldPath(
        actions=[{"tool": "campaign.get_metrics", "arguments": {"campaign_id": "CMP_1024"}}],
        final_answer={"cpi": 2.10},
    )
    return _bundle(case, env, verifier, gold)


# --------------------------------------------------------------------------
# 2. graded —— RL 主力
# --------------------------------------------------------------------------


def case_graded() -> CaseBundle:
    """诊断 → 方案。这是最有价值的一类：4 条 rollout 会自然分出 4 个档位。

        全对                      -> 1.00
        跳过 get_metrics          -> 0.93（evidence 2/3）
        不诊断直接猜方案          -> 更低（playbook 参数猜错会报错）
        一步发多个工具            -> 0.00（cap）

    分数拉得开 = advantage 拉得开 = 梯度信号强。
    """
    case_id = "SIG_GRADED_001"
    # cpi 2.90 / baseline 2.10 = 1.38 > 1.15 -> detect_anomalies 会报 cpi_spike
    env = (WorldBuilder(case_id)
           .account("ACC_01")
           .campaign("CMP_2048", account_id="ACC_01", platform="Meta", game_genre="puzzle",
                     cpi=2.90, cpi_baseline=2.10, roas_d7=0.45, roas_d7_baseline=0.45, frequency=2.4)
           .build())
    case = Case(
        case_id=case_id,
        user_message="CMP_2048 最近数据不太对劲，帮我看看是什么问题，然后给个优化方案。",
        context={"campaign_id": "CMP_2048"},
        entities={"campaign_id": "CMP_2048"},
        metadata=CaseMetadata(signal_class="graded", bucket="sequential_dependency",
                              topology="sequential", difficulty="L2", primary_intent="anomaly_diagnosis"),
        max_steps=8,
    )
    verifier = VerifierSpec(
        required_read_tools=["campaign.get_metrics", "campaign.detect_anomalies", "playbook.get_optimization"],
        required_answer_fields=[
            AnswerField(key="anomaly_type", value_source="literal:cpi_spike",
                        evidence_tool="campaign.detect_anomalies"),
            AnswerField(key="recommended_action", value_source="literal:narrow_targeting",
                        evidence_tool="playbook.get_optimization"),
        ],
        active_caps=["multi_tool_per_step_cap", "unauthorized_write_cap",
                     "false_claim_cap", "max_steps_cap"],
        max_steps=8,
    )
    gold = GoldPath(
        actions=[
            {"tool": "campaign.get_metrics", "arguments": {"campaign_id": "CMP_2048"}},
            {"tool": "campaign.detect_anomalies", "arguments": {"campaign_id": "CMP_2048"}},
            {"tool": "playbook.get_optimization", "arguments": {"anomaly_type": "cpi_spike"}},
        ],
        final_answer={"anomaly_type": "cpi_spike", "recommended_action": "narrow_targeting"},
    )
    return _bundle(case, env, verifier, gold)


# --------------------------------------------------------------------------
# 3. long_tail —— 慢任务
# --------------------------------------------------------------------------


def case_long_tail() -> CaseBundle:
    """上传素材 + 等审核。正确率不低，但 `creative.poll_review` 要真等 480 秒。

    这是我们研究异步的燃料：一批 rollout 里只要有一条落到这类 case 上，
    整批的完成时间就被它拖住——而阻塞式 verifier 会让情况更糟。
    """
    case_id = "SIG_LONGTAIL_001"
    env = (WorldBuilder(case_id)
           .account("ACC_01")
           .campaign("CMP_3072", account_id="ACC_01", platform="Meta", game_genre="puzzle")
           .build())
    case = Case(
        case_id=case_id,
        user_message="把新素材 hook_b_v1（45 秒竖版视频）上传到 CMP_3072，上传完告诉我审核过了没有。",
        context={"campaign_id": "CMP_3072", "creative_name": "hook_b_v1",
                 "asset_type": "video", "duration_seconds": 45},
        entities={"campaign_id": "CMP_3072", "creative_name": "hook_b_v1"},
        metadata=CaseMetadata(signal_class="long_tail", bucket="sequential_dependency",
                              topology="sequential", difficulty="L2", primary_intent="creative_upload"),
        max_steps=6,
    )
    verifier = VerifierSpec(
        required_read_tools=["campaign.get_metrics", "creative.poll_review"],
        allowed_write_tools=["creative.upload"],
        required_side_effects=[
            SideEffectReq(tool="creative.upload", required_args={"campaign_id": "entity:campaign_id"})
        ],
        required_answer_fields=[
            AnswerField(key="asset_id", value_source="literal:ASSET_CMP_3072_hook_b_v1",
                        evidence_tool="creative.upload"),
            AnswerField(key="review_status", value_source="literal:approved",
                        evidence_tool="creative.poll_review"),
        ],
        active_caps=["multi_tool_per_step_cap", "unauthorized_write_cap", "duplicate_write_cap",
                     "false_claim_cap", "max_steps_cap"],
        max_steps=6,
    )
    gold = GoldPath(
        actions=[
            {"tool": "campaign.get_metrics", "arguments": {"campaign_id": "CMP_3072"}},
            {"tool": "creative.upload", "arguments": {
                "campaign_id": "CMP_3072", "creative_name": "hook_b_v1",
                "asset_type": "video", "duration_seconds": 45}},
            {"tool": "creative.poll_review", "arguments": {"asset_id": "ASSET_CMP_3072_hook_b_v1"}},
        ],
        final_answer={"asset_id": "ASSET_CMP_3072_hook_b_v1", "review_status": "approved"},
    )
    return _bundle(case, env, verifier, gold)


# --------------------------------------------------------------------------
# 4. high_risk —— 必须过前置检查
# --------------------------------------------------------------------------


def case_high_risk() -> CaseBundle:
    """改预算。★ 用户要 900，但按政策只能到 750 —— **不查政策就不可能知道 750**。

    算式：standard 等级单次涨幅上限 50% -> 500 × 1.5 = 750
          月度剩余额度 (60000-18000)/30 = 1400，不构成更紧的约束
          涨幅 (900-500)/500 = 80% > 30% -> 需要审批

    照着用户说的 900 改就是错的，会撞 budget_over_limit_cap（0.20）。
    这是本域最贵的错误——预算改错会持续烧钱，比一次性退款严重。
    """
    case_id = "SIG_RISK_001"
    env = (WorldBuilder(case_id)
           .account("ACC_01", tier="standard", monthly_cap=60_000.0, spend_mtd=18_000.0, risk_flag=False)
           .campaign("CMP_4096", account_id="ACC_01", platform="Meta", game_genre="puzzle", daily_budget=500.0)
           .build())
    case = Case(
        case_id=case_id,
        user_message="CMP_4096 跑得不错，把日预算从 500 提到 900 吧。",
        context={"campaign_id": "CMP_4096", "account_id": "ACC_01", "requested_budget": 900},
        entities={"campaign_id": "CMP_4096", "account_id": "ACC_01", "requested_budget": 900},
        metadata=CaseMetadata(signal_class="high_risk", bucket="critical_args",
                              topology="sequential", difficulty="L4", primary_intent="budget_change",
                              tags=["write", "policy_sensitive", "risk_gated"]),
        max_steps=8,
    )
    verifier = VerifierSpec(
        required_read_tools=["campaign.get_metrics", "policy.get_budget_rule", "risk.check_account"],
        allowed_write_tools=["campaign.update_budget"],
        required_side_effects=[
            SideEffectReq(tool="campaign.update_budget",
                          required_args={"campaign_id": "entity:campaign_id",
                                         "new_budget": "decision.approved_budget"})
        ],
        required_answer_fields=[
            AnswerField(key="approved_budget", value_source="decision.approved_budget",
                        evidence_tool="policy.get_budget_rule"),
            AnswerField(key="requires_approval", value_source="decision.requires_approval",
                        evidence_tool="policy.get_budget_rule"),
        ],
        policy_required=True,
        expected_policy_id="P_BUDGET_STANDARD",
        active_caps=["missing_policy_check_cap", "missing_risk_check_cap", "budget_over_limit_cap",
                     "risk_blocked_write_cap", "duplicate_write_cap", "unauthorized_write_cap", "false_claim_cap",
                     "wrong_object_cap", "multi_tool_per_step_cap", "max_steps_cap"],
        max_steps=8,
    )
    gold = GoldPath(
        actions=[
            {"tool": "campaign.get_metrics", "arguments": {"campaign_id": "CMP_4096"}},
            {"tool": "policy.get_budget_rule", "arguments": {"account_id": "ACC_01"}},
            {"tool": "risk.check_account", "arguments": {"account_id": "ACC_01"}},
            {"tool": "campaign.update_budget", "arguments": {
                "campaign_id": "CMP_4096", "new_budget": 750.0, "reason": "policy_capped_increase"}},
        ],
        final_answer={"approved_budget": 750.0, "requires_approval": True},
    )
    return _bundle(case, env, verifier, gold)


# --------------------------------------------------------------------------
# 5. all_low —— 链路太长，基本做不完
# --------------------------------------------------------------------------


def case_all_low() -> CaseBundle:
    """跨 3 个 campaign 的大盘复盘：最少 8 步，max_steps=10，几乎没有容错空间。

    小模型大概率走不完，撞 max_steps_cap（0.35）。组内全低分 → advantage 也接近 0，
    和 all_high 是同一种病的另一端。**这类 case 的正确用法是 curriculum 拆分**：
    先训「单个 campaign 对基准」，再训「三个一起」。
    """
    case_id = "SIG_LOW_001"
    builder = WorldBuilder(case_id).account("ACC_02", tier="standard")
    # 三个 campaign 各有各的毛病，其中 CMP_5003 最严重（两个异常）
    builder.campaign("CMP_5001", account_id="ACC_02", platform="Meta", game_genre="puzzle", cpi=2.30)
    builder.campaign("CMP_5002", account_id="ACC_02", platform="Google", game_genre="casual", cpi=1.70)
    builder.campaign("CMP_5003", account_id="ACC_02", platform="Meta", game_genre="puzzle",
                     cpi=3.40, cpi_baseline=2.10, frequency=4.8)
    env = builder.build()
    case = Case(
        case_id=case_id,
        user_message=("把 ACC_02 下面 CMP_5001 / CMP_5002 / CMP_5003 三条 campaign 的表现"
                      "都和行业基准对一遍，找出问题最大的那一条，诊断原因并给出优化方案。"),
        context={"account_id": "ACC_02", "campaign_ids": ["CMP_5001", "CMP_5002", "CMP_5003"]},
        entities={"account_id": "ACC_02", "campaign_id": "CMP_5003"},
        metadata=CaseMetadata(signal_class="all_low", bucket="sequential_dependency",
                              topology="sequential", difficulty="L5", primary_intent="portfolio_review",
                              tags=["needs_curriculum"]),
        max_steps=10,
    )
    verifier = VerifierSpec(
        required_read_tools=["campaign.get_metrics", "benchmark.get_industry_baseline",
                             "campaign.detect_anomalies", "playbook.get_optimization"],
        required_answer_fields=[
            AnswerField(key="worst_campaign_id", value_source="literal:CMP_5003"),
            AnswerField(key="anomaly_type", value_source="literal:cpi_spike"),
            AnswerField(key="recommended_action", value_source="literal:narrow_targeting"),
        ],
        active_caps=["multi_tool_per_step_cap", "unauthorized_write_cap", "max_steps_cap"],
        max_steps=10,
    )
    gold = GoldPath(
        actions=[
            {"tool": "campaign.get_metrics", "arguments": {"campaign_id": "CMP_5001"}},
            {"tool": "campaign.get_metrics", "arguments": {"campaign_id": "CMP_5002"}},
            {"tool": "campaign.get_metrics", "arguments": {"campaign_id": "CMP_5003"}},
            {"tool": "benchmark.get_industry_baseline", "arguments": {"platform": "Meta", "game_genre": "puzzle", "metric": "cpi"}},
            {"tool": "benchmark.get_industry_baseline", "arguments": {"platform": "Google", "game_genre": "casual", "metric": "cpi"}},
            {"tool": "campaign.detect_anomalies", "arguments": {"campaign_id": "CMP_5003"}},
            {"tool": "playbook.get_optimization", "arguments": {"anomaly_type": "cpi_spike"}},
        ],
        final_answer={"worst_campaign_id": "CMP_5003", "anomaly_type": "cpi_spike",
                      "recommended_action": "narrow_targeting"},
        expected_reward_min=0.90,   # 7 步 > expected 4 步，efficiency 会扣一点
    )
    return _bundle(case, env, verifier, gold)


# --------------------------------------------------------------------------
# 6. tool_missing —— 故意抽掉必需工具
# --------------------------------------------------------------------------


def case_tool_missing() -> CaseBundle:
    """和 graded 同一个任务，但菜单里**没有 campaign.detect_anomalies**。

    模型必须退而求其次：拉自己的指标 + 拉行业基准，自己比出来是 CPI 偏高。
    这条 case 的用途是**度量能力缺口**——先看模型在缺工具时能不能绕过去，
    再用一轮 SFT 专门教这个 fallback 路径，然后复测。
    """
    case_id = "SIG_TOOLMISS_001"
    env = (WorldBuilder(case_id)
           .account("ACC_01")
           .campaign("CMP_6144", account_id="ACC_01", platform="Meta", game_genre="puzzle",
                     cpi=2.90, cpi_baseline=2.10)
           .build())
    case = Case(
        case_id=case_id,
        user_message="CMP_6144 最近效果变差了，帮我判断一下主要问题出在哪。",
        context={"campaign_id": "CMP_6144"},
        entities={"campaign_id": "CMP_6144"},
        metadata=CaseMetadata(signal_class="tool_missing", bucket="tool_confusion",
                              topology="standard", difficulty="L3", primary_intent="anomaly_diagnosis",
                              tags=["capability_gap"]),
        max_steps=6,
        # ★ 菜单里没有 campaign.detect_anomalies 和 playbook.get_optimization
        tool_menu=["campaign.get_metrics", "creative.get_metrics_by_asset", "benchmark.get_industry_baseline",
                   "policy.get_budget_rule", "risk.check_account"],
    )
    verifier = VerifierSpec(
        required_read_tools=["campaign.get_metrics", "benchmark.get_industry_baseline"],
        required_answer_fields=[
            AnswerField(key="anomaly_type", value_source="literal:cpi_spike"),
            AnswerField(key="campaign_cpi", value_source="campaigns.cpi"),
        ],
        active_caps=["multi_tool_per_step_cap", "unauthorized_write_cap", "max_steps_cap"],
        max_steps=6,
    )
    gold = GoldPath(
        actions=[
            {"tool": "campaign.get_metrics", "arguments": {"campaign_id": "CMP_6144"}},
            {"tool": "benchmark.get_industry_baseline", "arguments": {"platform": "Meta", "game_genre": "puzzle", "metric": "cpi"}},
        ],
        final_answer={"anomaly_type": "cpi_spike", "campaign_cpi": 2.90},
    )
    return _bundle(case, env, verifier, gold)


# --------------------------------------------------------------------------

SEED_BUILDERS = {
    "SIG_HIGH_001": case_all_high,
    "SIG_GRADED_001": case_graded,
    "SIG_LONGTAIL_001": case_long_tail,
    "SIG_RISK_001": case_high_risk,
    "SIG_LOW_001": case_all_low,
    "SIG_TOOLMISS_001": case_tool_missing,
}


def build_all() -> list[CaseBundle]:
    return [builder() for builder in SEED_BUILDERS.values()]


def gold_plan(bundle: CaseBundle) -> list[PlannedCall]:
    """把 gold 的 actions 转成执行器认识的计划。"""
    assert bundle.gold is not None, f"{bundle.case_id} has no gold"
    return [PlannedCall(tool=a["tool"], arguments=a.get("arguments", {})) for a in bundle.gold.actions]
