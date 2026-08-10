"""生成器测试：数量、多样性、边界行为。"""

from __future__ import annotations

import asyncio
import collections

import pytest

from syncopate.authoring.generate import generate, verify_gold
from syncopate.authoring.axes import ANOMALIES, GENRES, PLATFORMS, params_for
from syncopate.authoring.templates import TEMPLATES
from syncopate.core.runner import PlannedCall, run_plan
from syncopate.core.verifier_engine import score_trajectory
from syncopate.domains.adcampaign import build_domain

DOMAIN = build_domain()
SMALL_SPEC = {"name": "test", "quotas": {name: 6 for name in TEMPLATES}}


@pytest.fixture(scope="module")
def generated():
    original = DOMAIN.registry.latency_scale
    DOMAIN.registry.latency_scale = 0.0
    try:
        yield asyncio.run(generate(SMALL_SPEC))
    finally:
        DOMAIN.registry.latency_scale = original


# --------------------------------------------------------------------------
# 1. 产出正确
# --------------------------------------------------------------------------


def test_generates_requested_quota(generated):
    assert len(generated.accepted) == 6 * len(TEMPLATES)
    assert not generated.rejected, f"有 case 没通过 gold 验证: {generated.rejected[:3]}"


def test_case_ids_are_unique(generated):
    ids = [b.case_id for b in generated.accepted]
    assert len(ids) == len(set(ids))


def test_every_case_has_verified_gold(generated):
    """生成器已经逐条验过，这里抽查确认那个验证不是空转。"""
    original = DOMAIN.registry.latency_scale
    DOMAIN.registry.latency_scale = 0.0
    try:
        for bundle in generated.accepted[::7]:
            ok, reason = asyncio.run(verify_gold(bundle, DOMAIN))
            assert ok, f"{bundle.case_id}: {reason}"
    finally:
        DOMAIN.registry.latency_scale = original


# --------------------------------------------------------------------------
# 2. 多样性 —— 组合出来的不能只是同一条 case 换个 id
# --------------------------------------------------------------------------


def test_axes_are_actually_varied(generated):
    """★ 各个轴要真的错开。

    如果平台和品类同步变化，25 种组合实际只有 5 种——这是参数化生成最常见的
    伪多样性陷阱。params_for 用不同质数步长就是为了避免它。
    """
    graded = [b for b in generated.accepted if b.case_id.startswith("DIA_")]
    platforms = {b.env.table("campaigns")[b.case.entities["campaign_id"]]["platform"] for b in graded}
    genres = {b.env.table("campaigns")[b.case.entities["campaign_id"]]["game_genre"] for b in graded}
    anomalies = {f.value_source for b in graded for f in b.verifier.required_answer_fields
                 if f.key == "anomaly_type"}
    assert len(platforms) >= 4, f"平台不够分散: {platforms}"
    assert len(genres) >= 3, f"品类不够分散: {genres}"
    assert len(anomalies) >= 3, f"异常类型不够分散: {anomalies}"

    # 平台和品类不能一一绑定
    pairs = {(b.env.table("campaigns")[b.case.entities["campaign_id"]]["platform"],
              b.env.table("campaigns")[b.case.entities["campaign_id"]]["game_genre"])
             for b in graded}
    assert len(pairs) > max(len(platforms), len(genres)), "平台和品类是同步变化的，组合是假的"


def test_benchmark_table_covers_all_combinations():
    """基准表必须覆盖全部 平台×品类×指标，否则 benchmark.query 会大面积报错。"""
    from syncopate.domains.adcampaign.world import WorldBuilder

    table = WorldBuilder("X").build().table("benchmarks")
    for platform in PLATFORMS:
        for genre in GENRES:
            for metric in ("cpi", "roas_d7", "ctr", "retention_d1"):
                assert f"{platform}|{genre}|{metric}" in table


def test_high_risk_budget_is_policy_derived_not_user_requested(generated):
    """★ 高风险 case 的正确金额必须由政策算出来，不能等于用户要的数。"""
    from syncopate.domains.adcampaign.policies import compute_decision

    risky = [b for b in generated.accepted if b.case_id.startswith("BUD_")]
    assert risky
    capped = 0
    for bundle in risky:
        decision = compute_decision(bundle)
        assert decision is not None
        if decision["capped"]:
            capped += 1
            assert decision["approved_budget"] < decision["requested_budget"]
    assert capped > 0, "至少要有一部分被政策砍到上限，否则查不查政策没区别"


# --------------------------------------------------------------------------
# 3. clarify / reject 边界
# --------------------------------------------------------------------------


def _score(bundle, calls, final_answer, behavior):
    trajectory, sandbox = asyncio.run(
        run_plan(bundle, DOMAIN.registry, calls, final_answer=final_answer, behavior=behavior))
    return score_trajectory(bundle, trajectory, sandbox, policy_scorer=DOMAIN.policy_scorer,
                            decision_fn=DOMAIN.decision_fn, caps=DOMAIN.caps)


def test_clarify_case_missing_slot_is_not_in_context():
    """★ 该 clarify 的 case，缺失的槽位不能出现在 context 里——否则是自相矛盾的样本。"""
    bundle = TEMPLATES["clarify"](params_for(0))
    assert bundle.verifier.expected_behavior == "clarify"
    assert "campaign_id" not in bundle.case.context
    assert "campaign_id" not in bundle.case.entities


def test_answering_directly_instead_of_clarifying_scores_zero():
    bundle = TEMPLATES["clarify"](params_for(1))
    result = _score(bundle, [], {"missing_field": "campaign_id"}, behavior="tool_call")
    assert result.reward == 0.0


def test_calling_tools_when_should_clarify_hits_cap():
    """★ behavior 蒙对了，但过程里乱调工具——那不是学会了边界，是碰巧。"""
    bundle = TEMPLATES["clarify"](params_for(2))
    result = _score(
        bundle,
        [PlannedCall("campaign.get_metrics", {"campaign_id": "CMP_9999"})],
        {"missing_field": "campaign_id"}, behavior="clarify",
    )
    assert "acted_when_should_not_cap" in result.cap_steps
    assert result.cap_steps["acted_when_should_not_cap"] == [1]
    assert result.reward <= 0.20


def test_reject_cases_cover_both_reasons():
    reasons = collections.Counter()
    for index in range(10):
        bundle = TEMPLATES["reject"](params_for(index))
        assert bundle.verifier.expected_behavior == "reject"
        for f in bundle.verifier.required_answer_fields:
            if f.key == "reject_reason":
                reasons[f.value_source.removeprefix("literal:")] += 1
    assert set(reasons) == {"unauthorized", "out_of_scope"}


def test_reject_with_wrong_reason_loses_outcome():
    bundle = TEMPLATES["reject"](params_for(0))
    expected = next(f.value_source.removeprefix("literal:")
                    for f in bundle.verifier.required_answer_fields if f.key == "reject_reason")
    wrong = "out_of_scope" if expected == "unauthorized" else "unauthorized"
    good = _score(bundle, [], {"reject_reason": expected}, behavior="reject")
    bad = _score(bundle, [], {"reject_reason": wrong}, behavior="reject")
    assert good.reward > bad.reward
    assert bad.subscores["outcome"] == 0.0


# --------------------------------------------------------------------------
# 4. 长尾 case 的平台规格自洽
# --------------------------------------------------------------------------


def test_long_tail_duration_respects_platform_limit():
    """Google 只允许 ≤30 秒。模板必须跟着平台调时长，否则 gold 会被规格校验打回。"""
    for index in range(10):
        bundle = TEMPLATES["long_tail"](params_for(index))
        campaign = bundle.env.table("campaigns")[bundle.case.entities["campaign_id"]]
        duration = bundle.case.context["duration_seconds"]
        if campaign["platform"] == "Google":
            assert duration <= 30, f"{bundle.case_id} Google 平台配了 {duration} 秒"
