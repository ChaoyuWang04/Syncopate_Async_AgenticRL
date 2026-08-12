"""M3 · feature 归因。

守住的核心命题：**同一个 feature、同一句话，只有「那个地域有多少条素材」不同，
正确动作就不同** —— 样本足该下结论，样本不足该拒绝。

★★ 这个文件里最重要的是 `test_the_trap_is_real`：
`real_person|JP` 只有 4 条素材，但算出来 lift −0.142、置信区间不跨 0，
**`is_significant` 是 True**。只看显著性的模型会一头撞上去，看样本量才躲得开。
设计文档给 feature_lift 的原话就是「让模型学不会拿 3 个样本下结论」。
这条陷阱是**故意造出来的**（`_SPARSE_CELLS`），所以要有测试守着它别被数据改动抹平。
"""

from __future__ import annotations

import asyncio

import pytest

from syncopate.core.runner import PlannedCall, run_plan
from syncopate.core.schemas import Case, CaseBundle, CaseMetadata, VerifierSpec
from syncopate.core.verifier_engine import score_trajectory
from syncopate.domains.adcampaign import build_domain
from syncopate.domains.adcampaign.tools.analytics import FEATURES, MIN_FEATURE_SAMPLE
from syncopate.domains.adcampaign.world import WorldBuilder

DOMAIN = build_domain()
NOW = "2026-08-10T00:00:00+00:00"


def _bundle(*, caps=None):
    env = WorldBuilder("ATTR_T", reference_now=NOW).build()
    return CaseBundle(
        case=Case(case_id="ATTR_T", user_message="这个特点有用吗？", context={},
                  metadata=CaseMetadata(signal_class="graded", bucket="reasoning"),
                  max_steps=10),
        env=env,
        verifier=VerifierSpec(expected_behavior="tool_call", active_caps=caps, max_steps=10),
    )


def _lift(feature: str, region: str) -> dict:
    bundle = _bundle()
    trajectory, _ = asyncio.run(run_plan(
        bundle, DOMAIN.registry,
        [PlannedCall("analysis.feature_lift", {"feature": feature, "region": region})],
        final_answer={"x": 1}, behavior="tool_call"))
    obs = trajectory.observations[0]
    assert obs.ok, obs.error
    return obs.data


def _caps_for(feature: str, region: str, answer: dict) -> set[str]:
    bundle = _bundle(caps=["weak_attribution_cap"])
    trajectory, sandbox = asyncio.run(run_plan(
        bundle, DOMAIN.registry,
        [PlannedCall("analysis.feature_lift", {"feature": feature, "region": region})],
        final_answer=answer, behavior="tool_call"))
    result = score_trajectory(bundle, trajectory, sandbox, policy_scorer=DOMAIN.policy_scorer,
                              decision_fn=DOMAIN.decision_fn, caps=DOMAIN.caps)
    return {hit.name for hit in result.cap_hits}


# --------------------------------------------------------------------------
# 工具：真算，不读预置答案
# --------------------------------------------------------------------------


def test_lift_is_computed_from_the_catalog_not_a_lookup_table():
    """规律埋在素材 ROAS 的生成公式里，工具从数据重新算 —— 和 detect_anomalies 同一条纪律。

    验证方式：改一条素材的 roas，lift 必须跟着变。读预置答案的话不会变。
    """
    bundle = _bundle()
    before = _lift("real_person", "US")["lift"]
    for row in bundle.env.readonly_tables["creative_catalog"].values():
        if row["region"] == "US" and "real_person" in row["features"]:
            row["roas_d7"] = float(row["roas_d7"]) * 2
    trajectory, _ = asyncio.run(run_plan(
        bundle, DOMAIN.registry,
        [PlannedCall("analysis.feature_lift", {"feature": "real_person", "region": "US"})],
        final_answer={"x": 1}, behavior="tool_call"))
    assert trajectory.observations[0].data["lift"] > before * 1.5


def test_region_is_required():
    """跨地域合并算出来的结论，在某些地域是反的 —— 不给合并的口子。"""
    bundle = _bundle()
    trajectory, _ = asyncio.run(run_plan(
        bundle, DOMAIN.registry,
        [PlannedCall("analysis.feature_lift", {"feature": "real_person"})],
        final_answer={"x": 1}, behavior="tool_call"))
    assert not trajectory.observations[0].ok
    assert "region_required" in trajectory.observations[0].error


@pytest.mark.parametrize("feature", FEATURES)
def test_every_feature_is_queryable(feature):
    assert _lift(feature, "US")["sample_size"] >= 0


# --------------------------------------------------------------------------
# ★★ 数据里刻意造出来的三种局面
# --------------------------------------------------------------------------


def test_cross_region_sign_flip_exists():
    """real_person 在 US 正、在 JP 负 —— 逼出"必须逐地域算"。

    抹平了这个差异，把地域混起来算的模型就不会被惩罚，这条轴也就白设了。
    """
    assert _lift("real_person", "US")["lift"] > 0.15
    assert _lift("real_person", "JP")["lift"] < -0.05


def test_the_trap_is_real():
    """★★★ 样本 4 条，却「显著」。

    只看 is_significant 的模型会得出"真人出镜在日本有害"的结论。
    这条陷阱是 _SPARSE_CELLS 故意造的，别让后续的数据改动把它抹平。
    """
    data = _lift("real_person", "JP")
    assert data["sample_size"] < MIN_FEATURE_SAMPLE, "陷阱没了：JP 的样本量变够了"
    assert data["is_significant"], "陷阱没了：小样本不再显著，只看显著性的模型不会上当了"


def test_conclusive_cell_has_enough_samples():
    """对照面：样本足 + 显著的格子必须也存在，否则模型会学成"见到归因就说样本不足"。"""
    data = _lift("real_person", "US")
    assert data["sample_size"] >= MIN_FEATURE_SAMPLE
    assert data["is_significant"]


# --------------------------------------------------------------------------
# 判据
# --------------------------------------------------------------------------


def test_concluding_on_a_thin_sample_is_caught():
    hits = _caps_for("real_person", "JP",
                     {"conclusion": "negative", "feature": "real_person"})
    assert "weak_attribution_cap" in hits


def test_refusing_on_a_thin_sample_is_not_punished():
    """拒绝下结论是**正确做法**。终答里不点名那个 feature 就不算违规。"""
    hits = _caps_for("real_person", "JP",
                     {"conclusion": "insufficient_evidence", "sample_size": 4})
    assert "weak_attribution_cap" not in hits


def test_concluding_on_a_thick_sample_is_fine():
    hits = _caps_for("real_person", "US",
                     {"conclusion": "positive", "feature": "real_person"})
    assert "weak_attribution_cap" not in hits


def test_cap_auto_closes_when_feature_lift_was_never_called():
    """★ 自动闭合：存量 case 一次都不会调 analysis.feature_lift ⇒ 恒不命中。"""
    bundle = _bundle(caps=["weak_attribution_cap"])
    trajectory, sandbox = asyncio.run(run_plan(
        bundle, DOMAIN.registry, [PlannedCall("campaign.list", {"account_id": "ACC_01"})],
        final_answer={"conclusion": "real_person"}, behavior="tool_call"))
    result = score_trajectory(bundle, trajectory, sandbox, policy_scorer=DOMAIN.policy_scorer,
                              decision_fn=DOMAIN.decision_fn, caps=DOMAIN.caps)
    assert "weak_attribution_cap" not in {h.name for h in result.cap_hits}
