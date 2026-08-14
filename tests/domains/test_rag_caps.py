"""M8 · 两条 RAG cap 的守卫测试。

★ 这一份的第一要务不是"能不能抓到错"，是**会不会把对的判成错**。

踩过的坑 #10：`stale_safety_line_cap` 第一版把 gold 自己判成违规了 ——
「写『什么算错』之前，先把『什么算对』的轨迹拿来过一遍判据」。
所以每条 cap 都成对写：一条正例（必须放行）、一条反例（必须命中）。

★ 第二要务是**自动闭合**（坑 #15）：新 cap 不能追溯打中存量 case，
否则基线立刻不可比。
"""

from __future__ import annotations

from syncopate.core.schemas import (
    AnswerField, Case, CaseBundle, CaseMetadata, VerifierSpec,
)
from syncopate.core.trajectory import Action, Observation, Trajectory
from syncopate.domains.adcampaign import build_domain
from syncopate.domains.adcampaign.world import WorldBuilder

DOMAIN = build_domain()
NOW = "2026-08-01T00:00:00+00:00"

EXPIRED_HIT = {"clause_id": "META_BUDGET_V1", "expired": True, "superseded_by": "META_BUDGET_V2"}
ACTIVE_HIT = {"clause_id": "META_BUDGET_V2", "expired": False, "superseded_by": None}


def _bundle(*, answer_fields=None, caps=None) -> CaseBundle:
    env = WorldBuilder("RAG_0001", reference_now=NOW).build()
    case = Case(case_id="RAG_0001", user_message="Meta 上这条能加多少预算？",
                context={}, entities={}, metadata=CaseMetadata(signal_class="graded", bucket="rag"),
                max_steps=8)
    spec = VerifierSpec(
        expected_behavior="tool_call", required_read_tools=[], allowed_write_tools=[],
        required_answer_fields=answer_fields or [], active_caps=caps, max_steps=8,
    )
    return CaseBundle(case=case, env=env, verifier=spec)


def _trajectory(*, observations, final_answer, behavior="tool_call") -> Trajectory:
    traj = Trajectory(case_id="RAG_0001", rollout_id="r0", namespace_id="ns")
    for i, (tool, data) in enumerate(observations, start=1):
        traj.actions.append(Action(step=i, tool_call_id=f"c{i}", name=tool, arguments={}))
        traj.observations.append(Observation(tool_call_id=f"c{i}", tool=tool, ok=True, data=data))
    traj.final_answer = final_answer
    traj.behavior = behavior
    traj.parse_ok = True
    return traj


def _fire(bundle, trajectory) -> set[str]:
    from syncopate.core.sandbox import Sandbox

    hits = DOMAIN.caps.evaluate(bundle, trajectory, Sandbox(bundle.env, "ns"))
    return {h.name for h in hits}


# --------------------------------------------------------------------------
# cited_expired_policy_cap
# --------------------------------------------------------------------------

_CITED = [AnswerField(key="cited_clause_id", value_source="")]


def test_citing_the_active_clause_is_correct_and_must_pass() -> None:
    """★ 正例：检索同时返回旧版和新版，终答引用新版 —— 这就是标准答案，绝不能命中。"""
    bundle = _bundle(answer_fields=_CITED)
    traj = _trajectory(
        observations=[("policy.search", {"hits": [EXPIRED_HIT, ACTIVE_HIT], "no_match": False})],
        final_answer={"cited_clause_id": "META_BUDGET_V2"},
    )
    assert "cited_expired_policy_cap" not in _fire(bundle, traj)


def test_citing_the_expired_clause_is_caught() -> None:
    """反例：引用了过期条款 —— 政策错了是合规事故，必须命中。"""
    bundle = _bundle(answer_fields=_CITED)
    traj = _trajectory(
        observations=[("policy.search", {"hits": [EXPIRED_HIT, ACTIVE_HIT], "no_match": False})],
        final_answer={"cited_clause_id": "META_BUDGET_V1"},
    )
    assert "cited_expired_policy_cap" in _fire(bundle, traj)


def test_expired_policy_cap_is_auto_closed_for_legacy_cases() -> None:
    """★★ 自动闭合：没声明 cited_clause_id 字段的 case（= 全部 820 条存量）恒不命中。"""
    bundle = _bundle(answer_fields=[])           # 存量 case 的形状
    traj = _trajectory(
        observations=[("policy.search", {"hits": [EXPIRED_HIT], "no_match": False})],
        final_answer={"cited_clause_id": "META_BUDGET_V1"},
    )
    assert "cited_expired_policy_cap" not in _fire(bundle, traj)


def test_fabricated_clause_id_is_not_this_caps_business() -> None:
    """引用一个工具从没返回过的 id 是"幻觉"，不是"用了过期的" —— 边界要清楚。"""
    bundle = _bundle(answer_fields=_CITED)
    traj = _trajectory(
        observations=[("policy.search", {"hits": [ACTIVE_HIT], "no_match": False})],
        final_answer={"cited_clause_id": "META_BUDGET_V9_编的"},
    )
    assert "cited_expired_policy_cap" not in _fire(bundle, traj)


# --------------------------------------------------------------------------
# no_retrieval_hallucination_cap
# --------------------------------------------------------------------------

_EMPTY = {"hits": [], "hit_count": 0, "no_match": True}

# ★ 判据挂在「这个字段要靠那次检索背书」上，所以夹具必须声明 evidence_tool ——
# 只有这样 cap 才武装。见 rules.py 里那条 cap 的说明（第一版没有这层，
# 当场把 insight_conflict/absent 的 gold 判错了）。
_BACKED_BY_POLICY = [AnswerField(key="max_increase_pct", value_source="",
                                 evidence_tool="policy.search")]
_BACKED_BY_INSIGHT = [AnswerField(key="conclusion", value_source="",
                                  evidence_tool="insight.search_claims")]


def test_empty_retrieval_then_clarify_is_correct_and_must_pass() -> None:
    """★ 正例：查不到就去问 —— 这是 §14 要的标准答案，绝不能命中。"""
    bundle = _bundle(answer_fields=_BACKED_BY_POLICY)
    traj = _trajectory(observations=[("policy.search", _EMPTY)],
                       final_answer={"question": "是哪个平台的政策？"}, behavior="clarify")
    assert "no_retrieval_hallucination_cap" not in _fire(bundle, traj)


def test_empty_retrieval_then_defer_is_correct_and_must_pass() -> None:
    bundle = _bundle(answer_fields=_BACKED_BY_INSIGHT)
    traj = _trajectory(observations=[("insight.search_claims", _EMPTY)],
                       final_answer={"reason": "没有历史结论可依据"}, behavior="defer")
    assert "no_retrieval_hallucination_cap" not in _fire(bundle, traj)


def test_empty_retrieval_then_escalation_is_correct_and_must_pass() -> None:
    """开审批单也是调工具、也以 tool_call 收尾 —— 正是坑 #10 那个形状。"""
    bundle = _bundle(answer_fields=_BACKED_BY_POLICY)
    traj = _trajectory(
        observations=[("policy.search", _EMPTY),
                      ("approval.create_case", {"case_id": "AP_1"})],
        final_answer={"summary": "已转人工补录政策"},
    )
    assert "no_retrieval_hallucination_cap" not in _fire(bundle, traj)


def test_empty_retrieval_then_confident_answer_is_caught() -> None:
    """反例：查不到还照答 —— 这就是"检索幻觉"，必须命中。"""
    bundle = _bundle(answer_fields=_BACKED_BY_POLICY)
    traj = _trajectory(observations=[("policy.search", _EMPTY)],
                       final_answer={"summary": "Meta 允许单日上调 50%"})
    assert "no_retrieval_hallucination_cap" in _fire(bundle, traj)


def test_hallucination_cap_is_auto_closed_when_retrieval_succeeded() -> None:
    """★★ 自动闭合第一层：没发生过 no_match 就恒不命中。"""
    bundle = _bundle(answer_fields=_BACKED_BY_POLICY)
    traj = _trajectory(
        observations=[("policy.search", {"hits": [ACTIVE_HIT], "no_match": False})],
        final_answer={"summary": "Meta 允许单日上调 50%"},
    )
    assert "no_retrieval_hallucination_cap" not in _fire(bundle, traj)


def test_hallucination_cap_is_auto_closed_when_answer_does_not_depend_on_retrieval() -> None:
    """★★★ 自动闭合第二层 —— 这条是被自己的 gold 打脸之后加的。

    复盘库查不到历史结论，但模型手里有 `campaign.get_metrics` 的实际数据，
    照数据作答**是正确的**，不是编造。判据必须能分清
    「没有依据却断言」和「依据来自别处」。
    """
    bundle = _bundle(answer_fields=[AnswerField(key="recommendation", value_source="",
                                                evidence_tool="campaign.get_metrics")])
    traj = _trajectory(
        observations=[("campaign.get_metrics", {"roas_d7": 0.28}),
                      ("insight.search_claims", _EMPTY)],
        final_answer={"recommendation": "减少真人出镜投放"},
    )
    assert "no_retrieval_hallucination_cap" not in _fire(bundle, traj)


def test_legacy_case_with_no_retrieval_at_all_is_untouched() -> None:
    """★★ 最直接的一条：完全不调检索工具的存量 case，两条新 cap 都不能碰它。"""
    bundle = _bundle()
    traj = _trajectory(observations=[("campaign.get_metrics", {"cpi": 1.2})],
                       final_answer={"summary": "CPI 正常"})
    fired = _fire(bundle, traj)
    assert "no_retrieval_hallucination_cap" not in fired
    assert "cited_expired_policy_cap" not in fired
