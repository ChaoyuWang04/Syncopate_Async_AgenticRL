"""种子 case 的端到端验证。

最重要的一条：**gold 必须真跑一遍拿到高分**。
老师包里 2737 条 gold 的分数全部恰好 = 1.0，是构造时预烤进文件的，
没有任何一条被真的执行过——所以「gold 走得通」这件事其实从未被验证。
我们反过来：gold 不跑通就不算数。
"""

from __future__ import annotations

import asyncio

import pytest

from syncopate.authoring.seed_cases import SEED_BUILDERS, build_all, gold_plan
from syncopate.core.runner import PlannedCall, run_plan
from syncopate.core.verifier_engine import score_trajectory
from syncopate.domains.adcampaign import build_domain

DOMAIN = build_domain()


@pytest.fixture(autouse=True)
def fast_latency():
    """把 480 秒的审核等待压成毫秒级，否则测试跑不完。"""
    original = DOMAIN.registry.latency_scale
    DOMAIN.registry.latency_scale = 0.0002      # 480s -> 0.096s
    yield
    DOMAIN.registry.latency_scale = original


def _score(bundle, calls, final_answer, behavior="tool_call"):
    trajectory, sandbox = asyncio.run(
        run_plan(bundle, DOMAIN.registry, calls, final_answer=final_answer, behavior=behavior)
    )
    result = score_trajectory(
        bundle, trajectory, sandbox,
        policy_scorer=DOMAIN.policy_scorer, decision_fn=DOMAIN.decision_fn, caps=DOMAIN.caps,
    )
    return result, trajectory, sandbox


def _run_gold(bundle):
    return _score(bundle, gold_plan(bundle), bundle.gold.final_answer)


# --------------------------------------------------------------------------
# 1. gold 必须真跑通
# --------------------------------------------------------------------------


@pytest.mark.parametrize("case_id", sorted(SEED_BUILDERS))
def test_gold_actually_reaches_expected_reward(case_id):
    bundle = SEED_BUILDERS[case_id]()
    result, trajectory, _ = _run_gold(bundle)

    # 所有工具调用都得成功——gold 里出现工具报错说明 case 本身设计错了
    failed = [o for o in trajectory.observations if not o.ok]
    assert not failed, f"{case_id} gold 里有工具报错: {[(o.tool, o.error) for o in failed]}"

    assert not result.cap_hits, f"{case_id} gold 命中了 cap: {[h.name for h in result.cap_hits]}"
    assert result.reward >= bundle.gold.expected_reward_min, (
        f"{case_id} gold reward={result.reward} < {bundle.gold.expected_reward_min}\n"
        f"subscores={result.subscores}\ndetails={result.details}"
    )


def test_all_seeds_cover_five_signal_classes():
    classes = {b.case.metadata.signal_class for b in build_all()}
    assert classes == {"all_high", "graded", "long_tail", "high_risk", "all_low", "tool_missing"}


# --------------------------------------------------------------------------
# 2. graded case 必须真的分层（这是 RL 主力 case 的存在理由）
# --------------------------------------------------------------------------


def test_graded_case_actually_grades():
    """4 条不同质量的轨迹必须拿到 4 个不同的分数，且严格递减。"""
    bundle = SEED_BUILDERS["SIG_GRADED_001"]()
    cid = {"campaign_id": "CMP_2048"}
    answer = {"anomaly_type": "cpi_spike", "recommended_action": "narrow_targeting"}

    perfect, _, _ = _run_gold(bundle)

    skip_metrics, _, _ = _score(bundle, [
        PlannedCall("campaign.detect_anomalies", cid),
        PlannedCall("playbook.get_optimization", {"anomaly_type": "cpi_spike"}),
    ], answer)

    # 不诊断直接猜方案，猜错类型 -> 工具报错 -> 终答也填错
    wrong_guess, _, _ = _score(bundle, [
        PlannedCall("campaign.get_metrics", cid),
        PlannedCall("playbook.get_optimization", {"anomaly_type": "creative_fatigue"}),
    ], {"anomaly_type": "creative_fatigue", "recommended_action": "rotate_creative"})

    # 一步发俩工具 -> cap 直接砸到 0
    multi_tool, _, _ = _score(bundle, [
        PlannedCall("campaign.get_metrics", cid, step=1),
        PlannedCall("campaign.detect_anomalies", cid, step=1),
        PlannedCall("playbook.get_optimization", {"anomaly_type": "cpi_spike"}, step=2),
    ], answer)

    rewards = [perfect.reward, skip_metrics.reward, wrong_guess.reward, multi_tool.reward]
    assert rewards == sorted(rewards, reverse=True), f"分数没有单调递减: {rewards}"
    assert len(set(rewards)) == 4, f"分数没有拉开层次: {rewards}"
    assert multi_tool.reward == 0.0
    assert multi_tool.cap_steps["multi_tool_per_step_cap"] == [1]


def test_all_high_case_is_indeed_flat():
    """★ 弱信号反例：不同走法拿到几乎一样的分 -> GRPO advantage ≈ 0。"""
    bundle = SEED_BUILDERS["SIG_HIGH_001"]()
    cid = {"campaign_id": "CMP_1024"}

    direct, _, _ = _score(bundle, [PlannedCall("campaign.get_metrics", cid)], {"cpi": 2.10})
    # 绕一步再查，只有 efficiency 掉一点点
    detour, _, _ = _score(bundle, [
        PlannedCall("creative.get_metrics_by_asset", cid),
        PlannedCall("campaign.get_metrics", cid),
    ], {"cpi": 2.10})

    assert direct.reward == pytest.approx(1.0)
    assert abs(direct.reward - detour.reward) < 0.06, "这条 case 应该是弱信号的"


# --------------------------------------------------------------------------
# 3. 高风险 case：前置检查必须真的管用
# --------------------------------------------------------------------------


def test_high_risk_requires_policy_lookup_to_know_the_number():
    """★ 用户要 900 元(90000 分)，政策只允许 750 元(75000 分)。照用户说的改 -> 撞上限 cap。"""
    bundle = SEED_BUILDERS["SIG_RISK_001"]()
    decision = DOMAIN.decision_fn(bundle)
    assert decision["approved_budget"] == 75_000
    assert decision["requires_approval"] is True

    naive, _, _ = _score(bundle, [
        PlannedCall("campaign.get_metrics", {"campaign_id": "CMP_4096"}),
        PlannedCall("policy.get_budget_rule", {"account_id": "ACC_01"}),
        PlannedCall("risk.check_account", {"account_id": "ACC_01"}),
        PlannedCall("campaign.update_budget", {"campaign_id": "CMP_4096", "new_budget": 90_000, "client_request_id": "tk10"}),
    ], {"approved_budget": 90_000, "requires_approval": True})

    assert "budget_over_limit_cap" in naive.cap_steps
    assert naive.cap_steps["budget_over_limit_cap"] == [4]
    assert naive.reward <= 0.20


def test_skipping_risk_check_caps_reward_with_step_attribution():
    """跳过风控直接改 -> cap 命中，且知道是第几步违规的。"""
    bundle = SEED_BUILDERS["SIG_RISK_001"]()
    result, _, _ = _score(bundle, [
        PlannedCall("campaign.get_metrics", {"campaign_id": "CMP_4096"}),
        PlannedCall("policy.get_budget_rule", {"account_id": "ACC_01"}),
        PlannedCall("campaign.update_budget", {"campaign_id": "CMP_4096", "new_budget": 75_000, "client_request_id": "tk11"}),
    ], {"approved_budget": 75_000, "requires_approval": True})

    assert result.cap_steps["missing_risk_check_cap"] == [3]
    assert result.reward <= 0.25


def test_checking_after_writing_still_counts_as_violation():
    """先斩后奏也算违规——前置检查是顺序判定，不是集合判定。"""
    bundle = SEED_BUILDERS["SIG_RISK_001"]()
    result, _, _ = _score(bundle, [
        PlannedCall("campaign.get_metrics", {"campaign_id": "CMP_4096"}),
        PlannedCall("campaign.update_budget", {"campaign_id": "CMP_4096", "new_budget": 75_000, "client_request_id": "tk12"}),
        PlannedCall("policy.get_budget_rule", {"account_id": "ACC_01"}),
        PlannedCall("risk.check_account", {"account_id": "ACC_01"}),
    ], {"approved_budget": 75_000, "requires_approval": True})

    assert result.cap_steps["missing_policy_check_cap"] == [2]
    assert result.cap_steps["missing_risk_check_cap"] == [2]


def test_wrong_campaign_is_caught():
    """写在了别的 campaign 上。"""
    bundle = SEED_BUILDERS["SIG_RISK_001"]()
    bundle.env.readonly_tables["campaigns"]["CMP_9999"] = dict(
        bundle.env.readonly_tables["campaigns"]["CMP_4096"], campaign_id="CMP_9999")
    result, _, _ = _score(bundle, [
        PlannedCall("campaign.get_metrics", {"campaign_id": "CMP_4096"}),
        PlannedCall("policy.get_budget_rule", {"account_id": "ACC_01"}),
        PlannedCall("risk.check_account", {"account_id": "ACC_01"}),
        PlannedCall("campaign.update_budget", {"campaign_id": "CMP_9999", "new_budget": 75_000, "client_request_id": "tk13"}),
    ], {"approved_budget": 75_000, "requires_approval": True})

    assert result.cap_steps["wrong_object_cap"] == [4]


# --------------------------------------------------------------------------
# 4. 长尾 case：延迟是真的
# --------------------------------------------------------------------------


def test_long_tail_case_is_actually_slow():
    """★ poll_review 必须真的占住时间，否则异步对照实验没有意义。"""
    bundle = SEED_BUILDERS["SIG_LONGTAIL_001"]()

    async def timed():
        loop = asyncio.get_running_loop()
        start = loop.time()
        await run_plan(bundle, DOMAIN.registry, gold_plan(bundle),
                       final_answer=bundle.gold.final_answer)
        return loop.time() - start

    # latency_scale=0.0002 下 480s -> 0.096s；确认它确实等了
    assert asyncio.run(timed()) >= 0.09


def test_poll_before_upload_fails():
    """没上传就查审核 -> 报错。这是结构性的顺序依赖，不是 prompt 里的软约束。"""
    bundle = SEED_BUILDERS["SIG_LONGTAIL_001"]()
    result, trajectory, _ = _score(bundle, [
        PlannedCall("creative.poll_review", {"asset_id": "ASSET_CMP_3072_hook_b_v1"}),
    ], {})
    assert trajectory.observations[0].ok is False
    assert "upload it first" in trajectory.observations[0].error


# --------------------------------------------------------------------------
# 5. 工具缺失 case
# --------------------------------------------------------------------------


def test_tool_missing_case_hides_the_tool():
    """菜单里确实没有 detect_anomalies，但工具本身在注册表里（只是不给看）。"""
    bundle = SEED_BUILDERS["SIG_TOOLMISS_001"]()
    menu_names = [t["function"]["name"] for t in DOMAIN.registry.menu(bundle.case.tool_menu)]
    assert "campaign.detect_anomalies" not in menu_names
    assert "playbook.get_optimization" not in menu_names
    assert "campaign.get_metrics" in menu_names
    assert DOMAIN.registry.get("campaign.detect_anomalies") is not None


# --------------------------------------------------------------------------
# 6. 域完整性
# --------------------------------------------------------------------------


def test_domain_tool_menu_is_complete():
    """菜单 24 个：12 投放治理 + 4 外部 + 5 记忆 + 1 成熟度 + 1 MMP + 1 runtime。"""
    assert len(DOMAIN.default_tool_menu) == 24
    assert "system.wait" in DOMAIN.default_tool_menu
    assert "metrics.get_freshness" in DOMAIN.default_tool_menu
    assert set(DOMAIN.default_tool_menu) <= set(DOMAIN.registry.names())
    assert set(DOMAIN.registry.write_tools()) == {
        "campaign.update_budget", "creative.upload", "approval.create_case",
        "memory.write_proposal", "memory.invalidate", "memory.conflict_resolve"}


def test_bundle_roundtrip_for_every_seed(tmp_path):
    from syncopate.core.schemas import CaseBundle

    for bundle in build_all():
        bundle.write(tmp_path)
        restored = CaseBundle.read(tmp_path, bundle.case_id)
        assert restored.case.metadata.signal_class == bundle.case.metadata.signal_class
        assert restored.verifier.active_caps == bundle.verifier.active_caps
