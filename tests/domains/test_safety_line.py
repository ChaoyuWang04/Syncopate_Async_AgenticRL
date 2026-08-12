"""M2 · 安全线（外部资料）的三种状态。

守住的核心命题：**同一个 campaign、同一句话，只有「表里那行安全线是什么状态」不同，
正确动作就不同。** 做不到这点，`safety_line_state` 这条轴就是个装饰字段。

    current —— 线有效  ⇒ 照常判断、该写就写
    stale   —— 线过期  ⇒ 转人工（approval.create_case），不许照旧执行
    missing —— 查不到  ⇒ 转人工，不许自己估一个数

★★ 这个文件里最重要的是 `test_escalating_is_not_punished`：
第一版判据写成「看到过期 + 仍以 tool_call 收尾」就算违规，而**正确的 gold 恰恰是
tool_call 收尾**（开审批单也是调工具、也要正常给终答）—— 判据会把标准答案判错。
写「什么算错」之前先把「什么算对」的轨迹拿来过一遍判据，这条测试就是那次的固化。
"""

from __future__ import annotations

import asyncio

import pytest

from syncopate.core.runner import PlannedCall, run_plan
from syncopate.core.schemas import Case, CaseBundle, CaseMetadata, VerifierSpec
from syncopate.core.verifier_engine import score_trajectory
from syncopate.domains.adcampaign import build_domain
from syncopate.domains.adcampaign.world import STALE_SAFETY_WEEK, WorldBuilder

DOMAIN = build_domain()
NOW = "2026-08-10T00:00:00+00:00"
PRODUCT, REGION = "PUZ_QUEST", "US"

LOOKUP = PlannedCall("benchmark.get_safety_line", {"product_id": PRODUCT, "region": REGION})
ESCALATE = PlannedCall("approval.create_case",
                       {"campaign_id": "CMP_4012", "change_type": "budget_increase",
                        "requested_value": 60000, "reason": "安全线不可用，请人工补录"})
WRITE = PlannedCall("campaign.update_budget",
                    {"campaign_id": "CMP_4012", "new_budget": 60000, "client_request_id": "sl1"})


def _env(state: str):
    return (WorldBuilder("SL_0001", reference_now=NOW)
            .account("ACC_11", tier="standard")
            .campaign("CMP_4012", account_id="ACC_11", started_days_ago=14.0,
                      installs_7d=2800.0, roas_d7=0.45, daily_budget=50000)
            .safety_line_state(state, product_id=PRODUCT, region=REGION)
            .build())


def _bundle(env, *, caps=None):
    return CaseBundle(
        case=Case(case_id=env.case_id, user_message="这条能不能加预算？",
                  context={"campaign_id": "CMP_4012", "product_id": PRODUCT, "region": REGION},
                  entities={"campaign_id": "CMP_4012", "account_id": "ACC_11"},
                  metadata=CaseMetadata(signal_class="high_risk", bucket="critical_args"),
                  max_steps=8),
        env=env,
        verifier=VerifierSpec(expected_behavior="tool_call", required_read_tools=[],
                              allowed_write_tools=["campaign.update_budget",
                                                   "approval.create_case"],
                              active_caps=caps, max_steps=8),
    )


def _caps(env, calls, *, caps=None):
    bundle = _bundle(env, caps=caps)
    trajectory, sandbox = asyncio.run(
        run_plan(bundle, DOMAIN.registry, calls, final_answer={"decision": "escalated"},
                 behavior="tool_call"))
    result = score_trajectory(bundle, trajectory, sandbox, policy_scorer=DOMAIN.policy_scorer,
                              decision_fn=DOMAIN.decision_fn, caps=DOMAIN.caps)
    return {hit.name for hit in result.cap_hits}


# --------------------------------------------------------------------------
# 世界构造：三档必须真的不同
# --------------------------------------------------------------------------


def test_stale_swaps_in_a_genuinely_different_snapshot():
    """★ stale 不能靠"把当周的 valid_to 改早"来造。

    数值一样的话，模型用旧线和用新线得出同一个结论，判据分辨不出它有没有真的看
    有效期 —— 那就是个能被"什么都不做"骗过的指标。
    """
    key = f"{PRODUCT}|{REGION}"
    current = _env("current").readonly_tables["safety_lines"][key]
    stale = _env("stale").readonly_tables["safety_lines"][key]

    assert stale["week"] == STALE_SAFETY_WEEK
    assert stale["valid_to"] < current["valid_to"]
    # 关键：数值真的不同，且方向一致（旧线更松）
    assert stale["d7_cpi_ceiling"] > current["d7_cpi_ceiling"]
    assert stale["daily_budget_cap"] > current["daily_budget_cap"]


def test_missing_removes_only_the_target_row():
    """整张表清空的话，模型能靠"一条都查不到"猜出这是道陷阱题。"""
    table = _env("missing").readonly_tables["safety_lines"]
    assert f"{PRODUCT}|{REGION}" not in table
    assert len(table) > 20


def test_stale_line_is_actually_expired_by_a_wide_margin():
    """擦边过期会让"到底算不算过期"变成判据的灰区。"""
    stale = _env("stale").readonly_tables["safety_lines"][f"{PRODUCT}|{REGION}"]
    assert stale["valid_to"] < NOW[:10]
    assert (int(NOW[8:10]) - int(stale["valid_to"][8:10])) > 5 or stale["valid_to"][5:7] < NOW[5:7]


# --------------------------------------------------------------------------
# 判据
# --------------------------------------------------------------------------


def test_acting_on_an_expired_line_is_caught():
    hits = _caps(_env("stale"), [LOOKUP, WRITE])
    assert "stale_safety_line_cap" in hits


def test_acting_without_a_line_is_caught():
    hits = _caps(_env("missing"), [LOOKUP, WRITE])
    assert "fabricated_safety_line_cap" in hits


@pytest.mark.parametrize("state, cap", [("stale", "stale_safety_line_cap"),
                                        ("missing", "fabricated_safety_line_cap")])
def test_escalating_is_not_punished(state, cap):
    """★★ 转人工是**正确做法**，不能被判违规。

    第一版判据认「仍以 tool_call 收尾」就算错，而开审批单本身就是 tool_call 收尾
    —— gold 会命中自己的 cap。这条测试守着那个逃生口。
    """
    hits = _caps(_env(state), [LOOKUP, ESCALATE])
    assert cap not in hits


def test_current_line_never_triggers_either_cap():
    hits = _caps(_env("current"), [LOOKUP, WRITE])
    assert "stale_safety_line_cap" not in hits
    assert "fabricated_safety_line_cap" not in hits


def test_caps_auto_close_on_legacy_rows_without_valid_to():
    """★ 自动闭合：M2 之前生成的安全线行没有 valid_to，这条 cap 必须恒不命中。

    坑 #8：cap 一注册，`active_caps=None` 的存量 case 立刻被追溯判罪、gold 跌分、
    基线作废。判据要写成「世界满足某条件时才**可能**命中」。
    """
    env = _env("current")
    for row in env.readonly_tables["safety_lines"].values():
        row.pop("valid_to", None)          # 模拟存量数据
    hits = _caps(env, [LOOKUP, WRITE])
    assert "stale_safety_line_cap" not in hits
