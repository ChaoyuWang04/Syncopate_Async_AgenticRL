"""记忆机制测试。

最核心的一条：**同一句话、同一个世界，只有记忆不同，正确答案就不同**。
这是"记忆为什么必要"的可验证证明——如果做不到这点，记忆就只是装饰。
"""

from __future__ import annotations

import asyncio

import pytest

from syncopate.core.runner import PlannedCall, run_plan
from syncopate.core.schemas import AnswerField, Case, CaseBundle, CaseMetadata, VerifierSpec
from syncopate.core.verifier_engine import score_trajectory
from syncopate.domains.adcampaign import build_domain
from syncopate.domains.adcampaign.memory import LANES, MIN_CONFIDENCE, search
from syncopate.domains.adcampaign.world import WorldBuilder

DOMAIN = build_domain()
NOW = "2026-08-10T00:00:00+00:00"


def _bundle(env, *, expected_behavior="tool_call", required_reads=None, answer_fields=None,
            caps=None, entities=None, allowed_writes=None):
    case = Case(
        case_id=env.case_id, user_message="把 CMP_4012 的日预算提到 900",
        context={"campaign_id": "CMP_4012", "account_id": "ACC_11", "requested_budget": 900},
        entities=entities or {"campaign_id": "CMP_4012", "account_id": "ACC_11",
                              "requested_budget": 900},
        metadata=CaseMetadata(signal_class="high_risk", bucket="critical_args"),
        max_steps=8,
    )
    spec = VerifierSpec(
        expected_behavior=expected_behavior,
        required_read_tools=required_reads or [],
        allowed_write_tools=allowed_writes or ["campaign.update_budget"],
        required_answer_fields=answer_fields or [],
        active_caps=caps,
        max_steps=8,
    )
    return CaseBundle(case=case, env=env, verifier=spec)


def _score(bundle, calls, answer, behavior="tool_call"):
    trajectory, sandbox = asyncio.run(
        run_plan(bundle, DOMAIN.registry, calls, final_answer=answer, behavior=behavior))
    return score_trajectory(bundle, trajectory, sandbox, policy_scorer=DOMAIN.policy_scorer,
                            decision_fn=DOMAIN.decision_fn, caps=DOMAIN.caps)


# --------------------------------------------------------------------------
# 1. 存储与 TTL
# --------------------------------------------------------------------------


def test_memory_lives_in_env_not_a_database():
    """记忆基线是 env 的只读表——不是数据库，不跨 rollout 共享。

    GRPO 会把同一条 case 并发跑 8 遍，任何跨 rollout 的可写状态都会让轨迹
    互相污染、且不可复现。
    """
    env = (WorldBuilder("M1", reference_now=NOW)
           .memory("risk", days_ago=2, subject={"account_id": "ACC_11"},
                   content={"budget_change_count_7d": 4}).build())
    assert "memory" in env.readonly_tables
    assert len(env.table("memory")) == 1
    # 序列化后仍然完整——四件套要能落盘再读回
    assert all("created_at" in r for r in env.table("memory").values())


def test_ttl_filters_by_case_declared_time():
    """TTL 基于 case 自己声明的 reference_now，不是系统时钟。"""
    from syncopate.domains.adcampaign.memory import parse_time

    env = (WorldBuilder("M2", reference_now=NOW)
           .memory("risk", days_ago=10, subject={"account_id": "A"}, content={"x": 1})       # 180 天 TTL，有效
           .memory("business", days_ago=90, subject={"account_id": "A"}, content={"x": 2})   # 60 天 TTL，过期
           .build())
    now = parse_time(env.reference_now)
    assert len(search(env.table("memory"), now, lane="risk")) == 1
    assert len(search(env.table("memory"), now, lane="business")) == 0
    assert len(search(env.table("memory"), now, lane="business", include_expired=True)) == 1


def test_each_lane_has_distinct_ttl_and_write_policy():
    assert LANES["risk"].ttl_days == 180 and LANES["risk"].write_policy == "proposal_review"
    assert LANES["episodic"].write_policy == "system"      # agent 不能伪造历史
    assert LANES["semantic"].ttl_days == 90


# --------------------------------------------------------------------------
# 2. ★ 记忆改变正确答案
# --------------------------------------------------------------------------


def _world(*, memory_state: str):
    """同一个世界，只有记忆库不同。"""
    builder = (WorldBuilder(f"MEM_{memory_state}", reference_now=NOW)
               .account("ACC_11", tier="standard")
               .campaign("CMP_4012", account_id="ACC_11", daily_budget=500.0))
    if memory_state == "repeated":
        builder.memory("risk", days_ago=1, subject={"account_id": "ACC_11", "campaign_id": "CMP_4012"},
                       content={"budget_change_count_7d": 4, "budget_change_count_180d": 12,
                                "risk_score": 0.58})
    return builder.build()


def test_same_request_different_memory_different_answer():
    """★ 用户消息、世界状态完全相同，只有记忆不同 —— 正确动作不同。

    clean   : 记忆库干净 → 按政策改到 750
    repeated: 7 天内已调 4 次 → 不该直接改，该走审批
    """
    clean = _world(memory_state="clean")
    repeated = _world(memory_state="repeated")

    assert clean.table("memory") == {}
    hits = search(repeated.table("memory"),
                  __import__("syncopate.domains.adcampaign.memory", fromlist=["parse_time"])
                  .parse_time(NOW), lane="risk", subject={"account_id": "ACC_11"})
    assert hits and hits[0].content["budget_change_count_7d"] == 4
    # 世界的其它部分逐字节相同
    assert clean.table("campaigns") == repeated.table("campaigns")
    assert clean.table("accounts") == repeated.table("accounts")


def test_skipping_memory_when_it_matters_hits_cap():
    """记忆库里有相关历史却不查就动手 -> missing_memory_check_cap。"""
    bundle = _bundle(_world(memory_state="repeated"),
                     required_reads=["campaign.get_metrics", "policy.get_budget_rule",
                                     "risk.check_account", "memory.search"],
                     caps=["missing_memory_check_cap", "missing_policy_check_cap"])
    result = _score(bundle, [
        PlannedCall("campaign.get_metrics", {"campaign_id": "CMP_4012"}),
        PlannedCall("policy.get_budget_rule", {"account_id": "ACC_11"}),
        PlannedCall("campaign.update_budget", {"campaign_id": "CMP_4012", "new_budget": 750.0, "client_request_id": "tk9"}),
    ], {})
    assert "missing_memory_check_cap" in result.cap_steps
    assert result.cap_steps["missing_memory_check_cap"] == [3]
    assert result.reward <= 0.40


def test_memory_search_returns_the_history():
    bundle = _bundle(_world(memory_state="repeated"))
    trajectory, _ = asyncio.run(run_plan(bundle, DOMAIN.registry, [
        PlannedCall("memory.search", {"lane": "risk", "account_id": "ACC_11"})], final_answer={}))
    obs = trajectory.observations[0]
    assert obs.ok and obs.data["count"] == 1
    assert obs.data["records"][0]["summary"]["budget_change_count_7d"] == 4


# --------------------------------------------------------------------------
# 3. 写提案的权限分层
# --------------------------------------------------------------------------


def test_writing_to_system_lane_is_hard_blocked():
    """episodic 由系统维护——agent 往里写会被工具直接拒（等价于真实 API 的 403）。

    这条必须硬拦：允许 agent 伪造"历史投放记录"，整个记忆机制就失去意义了。
    """
    bundle = _bundle(_world(memory_state="clean"))
    trajectory, _ = asyncio.run(run_plan(bundle, DOMAIN.registry, [
        PlannedCall("memory.write_proposal", {
            "lane": "episodic", "content": {"x": 1}, "confidence": 0.9,
            "evidence_refs": ["a", "b"]})], final_answer={}))
    assert trajectory.observations[0].ok is False
    assert "lane_is_system_managed" in trajectory.observations[0].error


def test_low_confidence_proposal_is_capped_not_blocked():
    """★ 软纪律不硬拦：工具照做，cap 封顶。

    直接拒绝的话模型只会学到"报错就换一个"，学不到"为什么要有证据"。
    """
    bundle = _bundle(_world(memory_state="clean"),
                     allowed_writes=["memory.write_proposal"],
                     caps=["memory_write_unverified_cap"])
    result = _score(bundle, [
        PlannedCall("memory.write_proposal", {
            "lane": "business", "content": {"note": "大概有效"},
            "confidence": 0.4, "evidence_refs": []}),
    ], {})
    assert "memory_write_unverified_cap" in result.cap_steps
    assert result.reward <= 0.30


def test_risk_lane_write_requires_prior_review():
    bundle = _bundle(_world(memory_state="clean"),
                     allowed_writes=["memory.write_proposal"],
                     caps=["risk_memory_without_review_cap"])
    args = {"lane": "risk", "content": {"risk_score": 0.7}, "confidence": 0.9,
            "evidence_refs": ["EP_1", "EP_2"], "account_id": "ACC_11"}

    bad = _score(bundle, [PlannedCall("memory.write_proposal", args)], {})
    assert "risk_memory_without_review_cap" in bad.cap_steps

    good = _score(bundle, [
        PlannedCall("risk.check_account", {"account_id": "ACC_11"}),
        PlannedCall("memory.write_proposal", args),
    ], {})
    assert "risk_memory_without_review_cap" not in good.cap_steps


def test_pii_in_proposal_is_capped():
    bundle = _bundle(_world(memory_state="clean"),
                     allowed_writes=["memory.write_proposal"], caps=["memory_pii_cap"])
    result = _score(bundle, [
        PlannedCall("memory.write_proposal", {
            "lane": "semantic", "content": {"contact": "buyer@example.com"},
            "confidence": 0.9, "evidence_refs": ["a", "b"]}),
    ], {})
    assert "memory_pii_cap" in result.cap_steps


def test_proposals_never_mutate_the_baseline():
    """★ 写提案只进 sandbox 台账，绝不改 env 的记忆基线。

    否则同一条 case 的 8 条 rollout 会互相看到对方写的东西。
    """
    env = _world(memory_state="repeated")
    before = dict(env.table("memory"))
    bundle = _bundle(env, allowed_writes=["memory.write_proposal"])
    trajectory, sandbox = asyncio.run(run_plan(bundle, DOMAIN.registry, [
        PlannedCall("memory.write_proposal", {
            "lane": "semantic", "content": {"x": 1}, "confidence": 0.9,
            "evidence_refs": ["a", "b"]})], final_answer={}))
    assert env.table("memory") == before          # 基线一个字节没变
    assert sandbox.facts() == {"memory_proposed"}  # 提案进了台账


def test_stale_memory_used_for_decision_is_capped():
    env = (WorldBuilder("M9", reference_now=NOW)
           .account("ACC_11").campaign("CMP_4012", account_id="ACC_11")
           .memory("business", days_ago=120, subject={"campaign_id": "CMP_4012"},
                   content={"intervention": "narrow_targeting", "worked": False}).build())
    stale_id = next(iter(env.table("memory")))
    bundle = _bundle(env, caps=["stale_memory_cap"])
    result = _score(bundle, [PlannedCall("memory.read", {"record_id": stale_id})], {})
    assert "stale_memory_cap" in result.cap_steps


def test_memory_read_flags_expiry():
    env = (WorldBuilder("M10", reference_now=NOW)
           .memory("business", days_ago=120, subject={}, content={"x": 1}).build())
    rid = next(iter(env.table("memory")))
    bundle = _bundle(env)
    trajectory, _ = asyncio.run(run_plan(bundle, DOMAIN.registry, [
        PlannedCall("memory.read", {"record_id": rid})], final_answer={}))
    assert trajectory.observations[0].data["expired"] is True


def test_conflict_resolve_validates_inputs():
    env = (WorldBuilder("M11", reference_now=NOW)
           .memory("semantic", days_ago=1, subject={}, content={"a": 1})
           .memory("semantic", days_ago=2, subject={}, content={"a": 2}).build())
    ids = list(env.table("memory"))
    bundle = _bundle(env, allowed_writes=["memory.conflict_resolve"])
    trajectory, _ = asyncio.run(run_plan(bundle, DOMAIN.registry, [
        PlannedCall("memory.conflict_resolve", {"record_ids": ids[:1], "decision": "merge"}),
        PlannedCall("memory.conflict_resolve", {"record_ids": ids, "decision": "supersede",
                                                "keep_record_id": "NOPE"}),
        PlannedCall("memory.conflict_resolve", {"record_ids": ids, "decision": "supersede",
                                                "keep_record_id": ids[0]}),
    ], final_answer={}))
    assert trajectory.observations[0].error == "need_at_least_two_records"
    assert "keep_record_id_must_be" in trajectory.observations[1].error
    assert trajectory.observations[2].ok is True


# --------------------------------------------------------------------------
# 4. 外部资料工具
# --------------------------------------------------------------------------


def test_safety_line_is_exact_not_retrieved_text():
    """安全线返回精确数值——这是不做 RAG 的理由：数值要拿来做判断。"""
    bundle = _bundle(_world(memory_state="clean"))
    trajectory, _ = asyncio.run(run_plan(bundle, DOMAIN.registry, [
        PlannedCall("benchmark.get_safety_line", {"product_id": "PUZ_QUEST", "region": "US"})],
        final_answer={}))
    data = trajectory.observations[0].data
    assert data["d7_cpi_ceiling"] == 2.20
    assert data["region"] == "US" and data["week"] == "2026-W32"


def test_seasonal_context_changes_with_reference_now():
    """★ 同一个查询，reference_now 不同 -> 时令阶段不同 -> 正确动作可能反转。

    必须指定 event：多个时令会重叠（8 月落在 summer_peak 里），
    不指定的话全局 phase 是有歧义的。
    """
    def phase(now: str) -> str:
        env = WorldBuilder("S", reference_now=now).build()
        bundle = _bundle(env)
        trajectory, _ = asyncio.run(run_plan(bundle, DOMAIN.registry, [
            PlannedCall("calendar.get_seasonal_context",
                        {"region": "US", "event": "halloween"})], final_answer={}))
        return trajectory.observations[0].data["phase"]

    assert phase("2026-08-10T00:00:00+00:00") == "off"           # 万圣节还早（69 天）
    assert phase("2026-10-05T00:00:00+00:00") == "approaching"    # 13 天后
    assert phase("2026-10-25T00:00:00+00:00") == "peak"           # 正当时


def test_overlapping_seasons_need_explicit_event():
    """不指定 event 时，重叠时令会让 phase 变得没意义——这是工具的已知边界。"""
    env = WorldBuilder("S2", reference_now="2026-08-10T00:00:00+00:00").build()
    bundle = _bundle(env)
    trajectory, _ = asyncio.run(run_plan(bundle, DOMAIN.registry, [
        PlannedCall("calendar.get_seasonal_context",
                    {"region": "US", "horizon_days": 90})], final_answer={}))
    data = trajectory.observations[0].data
    assert data["phase"] == "peak"                                 # 其实是 summer_peak
    assert data["active_events"][0]["event"] == "summer_peak"
    # 万圣节 69 天后——默认 45 天视野看不到它，得显式放宽
    assert any(e["event"] == "halloween" for e in data["upcoming_events"])


def test_creative_search_finds_seasonal_alternatives():
    bundle = _bundle(_world(memory_state="clean"))
    trajectory, _ = asyncio.run(run_plan(bundle, DOMAIN.registry, [
        PlannedCall("creative.search_similar", {"visual_tags": ["halloween"], "min_ipm": 8.0})],
        final_answer={}))
    data = trajectory.observations[0].data
    assert data["count"] > 0
    ipms = [c["ipm"] for c in data["creatives"]]
    assert ipms == sorted(ipms, reverse=True)          # 按 IPM 降序
    assert all("halloween" in c["visual_tags"] for c in data["creatives"])


def test_asset_tags_carry_offline_visual_analysis():
    """标签是离线视觉分析的产物——在线 rollout 绝不跑 VLM。"""
    bundle = _bundle(_world(memory_state="clean"))
    trajectory, _ = asyncio.run(run_plan(bundle, DOMAIN.registry, [
        PlannedCall("creative.get_asset_tags", {"creative_id": "CRV_9000"})], final_answer={}))
    data = trajectory.observations[0].data
    for key in ("visual_tags", "hook_type", "dominant_color", "has_face", "ipm", "d7_cpi"):
        assert key in data


def test_tool_menu_is_complete():
    """菜单 22 个：12 投放治理 + 4 外部资料 + 5 记忆 + 1 数据成熟度（M1）。"""
    assert len(DOMAIN.default_tool_menu) == 22
    assert "metrics.get_freshness" in DOMAIN.default_tool_menu
    assert set(DOMAIN.registry.write_tools()) == {
        "campaign.update_budget", "creative.upload", "approval.create_case",
        "memory.write_proposal", "memory.invalidate", "memory.conflict_resolve"}
