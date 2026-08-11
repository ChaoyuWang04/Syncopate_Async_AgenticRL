"""M1 · 数据成熟度机制。

守住的核心命题：**同一个 campaign、同一句话，只有「开投几天」不同，正确动作就不同。**
做不到这点，成熟度就只是个装饰字段，模型学不到「等」这个动作。
"""

from __future__ import annotations

import asyncio

import pytest

from syncopate.core.parsing import VALID_BEHAVIORS, parse_step
from syncopate.core.runner import PlannedCall, run_plan
from syncopate.core.schemas import Case, CaseBundle, CaseMetadata, VerifierSpec
from syncopate.core.verifier_engine import score_trajectory
from syncopate.domains.adcampaign import build_domain
from syncopate.domains.adcampaign.maturity import (
    CONVERGE_DAYS, IMMATURE, MATURE, MIN_SAMPLE_INSTALLS, PARTIAL,
    campaign_maturity, metric_maturity, observed_value, straddles,
)
from syncopate.domains.adcampaign.world import WorldBuilder

DOMAIN = build_domain()

BUDGET_CALL = PlannedCall("campaign.update_budget",
                          {"campaign_id": "CMP_4012", "new_budget": 600.0, "client_request_id": "tk8"})


def _env(days: float, *, installs_7d: float = 1400.0, roas: float = 0.45):
    return (WorldBuilder("MAT_0001")
            .account("ACC_11", tier="standard")
            .campaign("CMP_4012", account_id="ACC_11", started_days_ago=days,
                      installs_7d=installs_7d, roas_d7=roas, daily_budget=500.0)
            .build())


def _bundle(env, *, caps=None, required_reads=None, expected_behavior="tool_call"):
    return CaseBundle(
        case=Case(case_id=env.case_id, user_message="这条能不能加预算？",
                  context={"campaign_id": "CMP_4012"},
                  entities={"campaign_id": "CMP_4012", "account_id": "ACC_11"},
                  metadata=CaseMetadata(signal_class="high_risk", bucket="critical_args"),
                  max_steps=8),
        env=env,
        verifier=VerifierSpec(expected_behavior=expected_behavior,
                              required_read_tools=required_reads or [],
                              allowed_write_tools=["campaign.update_budget"],
                              active_caps=caps, max_steps=8),
    )


def _score(bundle, calls, behavior="tool_call", answer=None):
    trajectory, sandbox = asyncio.run(
        run_plan(bundle, DOMAIN.registry, calls, final_answer=answer or {"summary": "x"},
                 behavior=behavior))
    return score_trajectory(bundle, trajectory, sandbox, policy_scorer=DOMAIN.policy_scorer,
                            decision_fn=DOMAIN.decision_fn, caps=DOMAIN.caps)


# --------------------------------------------------------------------------
# 1. 成熟曲线本身
# --------------------------------------------------------------------------


@pytest.mark.parametrize("days, expected", [
    (1, IMMATURE),    # 只有 D1
    (3, PARTIAL),     # D3 有了，D7 还没到
    (7, MATURE),      # 收敛
    (30, MATURE),
])
def test_roas_maturity_moves_with_days(days, expected):
    assert campaign_maturity(_env(days).table("campaigns")["CMP_4012"])["maturity"] == expected


def test_metrics_converge_at_different_speeds():
    """★ 同一天，ROAS 还没收敛而 CTR 早就收敛了。

    收敛期是**指标的属性**，不是 campaign 的属性——合成一个"数据成熟没有"的
    布尔值，就等于宣称所有指标一起变准，那是错的。
    """
    row = _env(2).table("campaigns")["CMP_4012"]
    assert campaign_maturity(row, "roas_d7")["maturity"] == IMMATURE
    assert campaign_maturity(row, "ctr")["maturity"] == MATURE
    assert CONVERGE_DAYS["roas_d7"] > CONVERGE_DAYS["ctr"]


def test_observed_value_is_biased_early_and_exact_at_convergence():
    """早期观测有系统性偏差：ROAS 偏低（收入还没回收完），CPI 偏高（学习期贵）。

    这才是"数据不准"的真实形态。如果早期观测只是围绕真值抖动，
    模型学到的会是"多看几天平均一下"，而不是"现在这个数根本没意义"。
    """
    assert observed_value("roas_d7", 0.45, 1) < 0.45 * 0.6
    assert observed_value("cpi", 2.0, 1) > 2.0
    for metric, final in (("roas_d7", 0.45), ("cpi", 2.0), ("ctr", 0.02)):
        assert observed_value(metric, final, CONVERGE_DAYS[metric]) == pytest.approx(final, rel=1e-6)


def test_unconverged_returns_an_interval_that_narrows():
    """未收敛给区间、收敛退化成点——「现在还说不准」是靠区间宽度表达的。"""
    wide = metric_maturity("roas_d7", days_elapsed=1, final_value=0.45, installs_7d=1400)
    narrow = metric_maturity("roas_d7", days_elapsed=5, final_value=0.45, installs_7d=1400)
    exact = metric_maturity("roas_d7", days_elapsed=7, final_value=0.45, installs_7d=1400)

    def width(info):
        lo, hi = info["expected_final_range"]
        return hi - lo

    assert width(wide) > width(narrow) > width(exact) == 0.0
    # 区间必须真的包住终值，否则"参考区间"是骗人的
    assert wide["expected_final_range"][0] <= 0.45 <= wide["expected_final_range"][1]


def test_straddling_the_safety_line_is_the_defer_signal():
    """★ 判据是「区间跨不跨线」，不是「中心在哪一侧」。

    D2 时中心在线上方，但区间横跨安全线 —— 这个决策现在做不了。
    只看中心的话，模型会拿一个还在动的中间态当结论。
    """
    early = metric_maturity("roas_d7", days_elapsed=2, final_value=0.45, installs_7d=1400)
    mature = metric_maturity("roas_d7", days_elapsed=7, final_value=0.45, installs_7d=1400)
    safety_line = 0.40
    assert straddles(early["expected_final_range"], safety_line)
    assert not straddles(mature["expected_final_range"], safety_line)


def test_sample_size_and_time_are_two_different_kinds_of_unreliable():
    """时间到了但样本量不够，照样不成熟——而且这种不成熟**等不来**。"""
    tiny = metric_maturity("roas_d7", days_elapsed=30, final_value=0.45, installs_7d=100)
    assert tiny["maturity"] == IMMATURE
    assert tiny["sample_size"] < MIN_SAMPLE_INSTALLS
    assert tiny["converge_eta_days"] == 0        # 再等也没用
    assert "样本量不足" in tiny["reason"]


# --------------------------------------------------------------------------
# 2. metrics.get_freshness 工具
# --------------------------------------------------------------------------


def _call_freshness(env, **args):
    bundle = _bundle(env)
    trajectory, _ = asyncio.run(run_plan(
        bundle, DOMAIN.registry,
        [PlannedCall("metrics.get_freshness", {"campaign_id": "CMP_4012", **args})],
        final_answer={"summary": "x"}))
    return trajectory.observations[-1]


def test_freshness_tool_reports_the_world_not_the_request():
    obs = _call_freshness(_env(2))
    assert obs.ok
    assert obs.data["days_elapsed"] == 2
    assert obs.data["converge_at_day"] == CONVERGE_DAYS["roas_d7"]
    assert obs.data["expected_final_range"][0] < obs.data["expected_final_range"][1]


def test_freshness_returns_facts_not_the_conclusion():
    """★★★ 工具**不能**回 maturity 标签。

    第一版回了，于是完全没训过的 base 在 I02 上 defer 准确率就有 97%——
    因为答案字段就叫 `data_maturity`，模型只是把工具返回值照抄过去。
    判断根本没有发生，SFT 和 RL 都没东西可学。

    ⇒ 设计文档 §0.1 的切分线：「查什么」属于工具，「怎么判」属于权重。
    """
    from syncopate.domains.adcampaign.tools.freshness import FACT_FIELDS

    data = _call_freshness(_env(2)).data
    assert set(data) == set(FACT_FIELDS)
    for conclusion in ("maturity", "is_converged", "converge_eta_days", "reason"):
        assert conclusion not in data
    # 但 verifier / cap 侧仍然拿得到结论——口径只有一份，只是不外泄
    assert campaign_maturity(_env(2).table("campaigns")["CMP_4012"])["maturity"] == IMMATURE


def test_thresholds_are_not_in_the_system_prompt():
    """阈值写进 prompt 就等于把规则还给了上下文，而这条规则是「不变的」，该进权重。"""
    from syncopate.prompts import load_prompt

    system = load_prompt("system.txt")
    for leaked in ("7 天", "mature", "partial", "immature"):
        assert leaked not in system


def test_freshness_has_no_as_of_parameter():
    """★ 刻意偏离设计文档的签名：模型无权声明「今天是哪天」。

    一旦 as_of 可传，「数据还没到」就变成模型能绕过去的参数，
    premature_decision_cap 会被它自己填的日期架空。
    """
    schema = DOMAIN.registry.get("metrics.get_freshness").parameters
    assert "as_of" not in schema["properties"]
    assert set(schema["properties"]) == {"campaign_id", "metric"}


def test_freshness_rejects_unknown_metric_and_campaign():
    assert not _call_freshness(_env(2), metric="made_up_metric").ok
    bundle = _bundle(_env(2))
    trajectory, _ = asyncio.run(run_plan(
        bundle, DOMAIN.registry,
        [PlannedCall("metrics.get_freshness", {"campaign_id": "NOPE"})],
        final_answer={"summary": "x"}))
    assert not trajectory.observations[-1].ok


# --------------------------------------------------------------------------
# 3. defer 行为
# --------------------------------------------------------------------------


def test_defer_is_a_valid_behavior():
    assert "defer" in VALID_BEHAVIORS
    parsed = parse_step('```json\n{"behavior": "defer", '
                        '"answer": {"recheck_after_days": 5, "reason": "roas_d7 还差 5 天"}}\n```')
    assert parsed.kind == "final" and parsed.behavior == "defer"
    assert parsed.answer["recheck_after_days"] == 5


def test_defer_when_data_is_mature_is_still_a_mismatch():
    """★ 双向：数据已收敛却 defer，和该 defer 却动手，一样是错的。

    只罚「该等没等」会训出一个什么都不敢做的 agent——那在业务上一样没用。
    """
    bundle = _bundle(_env(30), expected_behavior="tool_call")
    result = _score(bundle, [], behavior="defer")
    assert result.reward == 0.0
    assert [h.name for h in result.cap_hits] == ["behavior_mismatch"]


# --------------------------------------------------------------------------
# 4. 三条新 cap
# --------------------------------------------------------------------------


def test_premature_decision_cap_fires_only_when_data_is_immature():
    immature = _score(_bundle(_env(1), caps=["premature_decision_cap"]), [BUDGET_CALL])
    mature = _score(_bundle(_env(30), caps=["premature_decision_cap"]), [BUDGET_CALL])
    assert [h.name for h in immature.cap_hits] == ["premature_decision_cap"]
    assert immature.reward <= 0.15
    assert mature.cap_hits == []


def test_premature_decision_needs_an_actual_write():
    """只是查了查、没动手，不算过早决策。cap 罚的是动作，不是想法。"""
    result = _score(_bundle(_env(1), caps=["premature_decision_cap"], expected_behavior="defer"),
                    [PlannedCall("metrics.get_freshness", {"campaign_id": "CMP_4012"})],
                    behavior="defer")
    assert result.cap_hits == []


def test_insufficient_sample_cap_is_separate_from_premature():
    """★ 两条 cap 必须分开：一个是「等就好了」，一个是「等也没用」。

    合成一条的话，模型对小样本场景会学到错误的解法（继续等）。
    """
    # 时间早就到了，但安装量太小 —— 只该命中样本量那条
    starved = _score(_bundle(_env(30, installs_7d=100), caps=None), [BUDGET_CALL])
    names = {h.name for h in starved.cap_hits}
    assert "insufficient_sample_cap" in names
    assert "premature_decision_cap" not in names


def test_missing_safety_line_cap_is_gated_on_the_case_declaring_it():
    """没声明安全线是必查项的 case 不受这条规则管——否则等于用新规则追溯判旧数据。"""
    declared = _bundle(_env(30), caps=["missing_safety_line_cap"],
                       required_reads=["benchmark.get_safety_line"])
    silent = _bundle(_env(30), caps=["missing_safety_line_cap"])
    assert [h.name for h in _score(declared, [BUDGET_CALL]).cap_hits] == ["missing_safety_line_cap"]
    assert _score(silent, [BUDGET_CALL]).cap_hits == []


def test_literal_false_means_false():
    """★ `literal:false` 曾被判成「期望 True」——bool("false") 是 True。

    症状很隐蔽：`literal:true` 一直是对的（碰巧），只有期望 false 的字段悄悄失分。
    实测抓到点是 I02 partial 档的 can_decide=False，outcome 卡在 0.67。
    """
    from syncopate.core.verifier_engine import values_equal
    assert values_equal(False, "false")
    assert values_equal(True, "true")
    assert not values_equal(True, "false")
    assert not values_equal(False, "true")


# --------------------------------------------------------------------------
# 5. I02 模板：三档 = 三种正确行为
# --------------------------------------------------------------------------


@pytest.mark.parametrize("maturity, behavior", [
    ("mature", "tool_call"),
    ("partial", "tool_call"),
    ("immature", "defer"),
])
def test_i02_maturity_decides_the_behavior(maturity, behavior):
    """★ 同一句话、同一个 campaign，只有「开投几天」不同，正确行为就不同。

    这是 defer 这个标签存在的全部理由。三档缺任何一档，
    要么模型学不到「等」，要么学成「什么都不敢答」。
    """
    from syncopate.authoring.axes import DATA_MATURITIES, params_for
    from syncopate.authoring.templates import make_freshness_check

    index = next(i for i in range(30) if params_for(i).data_maturity == maturity)
    bundle = make_freshness_check(params_for(index))
    assert bundle.verifier.expected_behavior == behavior
    assert bundle.gold.final_answer["data_maturity"] == maturity
    assert "metrics.get_freshness" in bundle.verifier.required_read_tools
    assert bundle.verifier.allowed_write_tools == []      # 纯判断，不带写动作
    assert set(DATA_MATURITIES) == {"mature", "partial", "immature"}


def test_i02_defer_carries_a_concrete_wait():
    """defer 不能是「再等等」——必须给出还要等几天，否则这个行为无法被验证。"""
    from syncopate.authoring.axes import params_for
    from syncopate.authoring.templates import make_freshness_check

    index = next(i for i in range(30) if params_for(i).data_maturity == "immature")
    bundle = make_freshness_check(params_for(index))
    assert bundle.gold.final_answer["recheck_after_days"] > 0
    assert any(f.key == "recheck_after_days" for f in bundle.verifier.required_answer_fields)


def test_i02_claiming_maturity_without_checking_is_a_false_claim():
    """没查 freshness 就声称「数据成熟」，是 false_claim 最典型的形态。"""
    from syncopate.authoring.axes import params_for
    from syncopate.authoring.templates import make_freshness_check

    index = next(i for i in range(30) if params_for(i).data_maturity == "mature")
    bundle = make_freshness_check(params_for(index))
    field = next(f for f in bundle.verifier.required_answer_fields if f.key == "data_maturity")
    assert field.evidence_tool == "metrics.get_freshness"

    result = _score(bundle, [], answer=bundle.gold.final_answer)
    assert "false_claim_cap" in {h.name for h in result.cap_hits}


def test_new_caps_do_not_touch_legacy_cases():
    """★ 存量 case 的世界默认 started_days_ago=30、安装量充足 ⇒ 三条新 cap 恒不命中。

    这是「新机制不该让已经付过 GPU 的基线失效」这条纪律的可执行版本。
    """
    legacy_env = (WorldBuilder("LEGACY_0001")
                  .account("ACC_11").campaign("CMP_4012", account_id="ACC_11").build())
    row = legacy_env.table("campaigns")["CMP_4012"]
    assert row["started_days_ago"] == 30
    result = _score(_bundle(legacy_env, caps=None), [BUDGET_CALL])
    assert {"premature_decision_cap", "insufficient_sample_cap",
            "missing_safety_line_cap"} & {h.name for h in result.cap_hits} == set()
