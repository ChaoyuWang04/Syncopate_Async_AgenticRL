"""M8 · `insight_conflict` 模板的验收测试。

★ 这个模板补的是遗留清单里挂了很久的缺口：`memory.conflict_resolve` 完全没题。
所以第一要务是证明**那道题真的出得来** —— 矛盾双方可见、gold 真的用到了那个工具。
"""

from __future__ import annotations

import asyncio

import pytest

from syncopate.authoring.axes import params_for
from syncopate.authoring.templates import TEMPLATES, _INSIGHT_QUERY
from syncopate.core.runner import PlannedCall, run_plan
from syncopate.core.sandbox import Sandbox
from syncopate.core.tool_registry import ToolContext
from syncopate.core.verifier_engine import score_trajectory
from syncopate.domains.adcampaign import build_domain

DOMAIN = build_domain()
DOMAIN.registry.latency_scale = 0.0
MAKE = TEMPLATES["insight_conflict"]
N = 90


def _bundles():
    return [(params_for(i), MAKE(params_for(i))) for i in range(N)]


def _by_state(state: str):
    out = [(p, b) for p, b in _bundles() if p.insight_state == state]
    assert out, f"{state} 档一条都没有，轴的取模可能被削平了"
    return out


def _search_claims(bundle, **kw):
    ctx = ToolContext(case=bundle.case, env=bundle.env, sandbox=Sandbox(bundle.env, "ns"),
                      step=1, tool_call_id="c1")
    args = {"query": _INSIGHT_QUERY, "region": bundle.case.context["region"], **kw}
    return DOMAIN.registry.get("insight.search_claims").handler(args, ctx)


def _run(bundle, *, final_answer=None, behavior=None, actions=None):
    calls = [PlannedCall(tool=a["tool"], arguments=a.get("arguments", {}))
             for a in (actions if actions is not None else bundle.gold.actions)]
    traj, sandbox = asyncio.run(run_plan(
        bundle, DOMAIN.registry, calls,
        final_answer=final_answer if final_answer is not None else bundle.gold.final_answer,
        behavior=behavior or bundle.verifier.expected_behavior))
    result = score_trajectory(bundle, traj, sandbox, policy_scorer=DOMAIN.policy_scorer,
                              decision_fn=DOMAIN.decision_fn, caps=DOMAIN.caps)
    return result, {h.name for h in result.cap_hits}


# --------------------------------------------------------------------------
# 矛盾这道题真的出得来
# --------------------------------------------------------------------------


def test_conflicting_arm_actually_has_two_contradicting_memories() -> None:
    """`conflict_resolve` 吃的是 ≥2 条记忆 id —— 世界里必须真有两条互相矛盾的。"""
    for p, b in _by_state("conflicting"):
        rows = b.env.readonly_tables["memory"]
        business = [r for r in rows.values() if r["lane"] == "business"]
        assert len(business) >= 2, f"{b.case.case_id}: 只有 {len(business)} 条，凑不出冲突"
        conclusions = {r["content"]["conclusion"] for r in business}
        assert len(conclusions) == 2, "两条结论必须真的不同，否则不成其为矛盾"


def test_conflict_is_visible_in_the_insight_corpus_too() -> None:
    """★ 复盘语料里也要能看见冲突（团队层面的佐证）。

    这条依赖 `insight.search_claims` **默认返回 superseded** —— 那个默认值
    2026-08-14 被翻转过一次，翻回去这道题就瞎了。
    """
    for p, b in _by_state("conflicting"):
        hits = _search_claims(b).data["hits"]
        statuses = {h["status"] for h in hits}
        assert {"active", "superseded"} <= statuses, \
            f"{b.case.case_id}: 只看到 {statuses}，矛盾在语料侧不可见"


def test_gold_actually_calls_conflict_resolve() -> None:
    """★★ 缺口填上的证据：这个工具第一次真的进 gold。"""
    for p, b in _by_state("conflicting"):
        tools = [a["tool"] for a in b.gold.actions]
        assert "memory.conflict_resolve" in tools, f"{b.case.case_id}: 又没用上"


def test_control_arms_do_not_call_conflict_resolve() -> None:
    """★ 对照档不能调 —— 否则模型学成"看到历史结论就报冲突"。"""
    for state in ("aligned", "absent"):
        for p, b in _by_state(state):
            assert "memory.conflict_resolve" not in [a["tool"] for a in b.gold.actions]


def test_absent_arm_has_no_insights_and_no_business_memory() -> None:
    for p, b in _by_state("absent"):
        assert b.env.readonly_tables["insights"] == {}
        assert _search_claims(b).data["no_match"] is True


# --------------------------------------------------------------------------
# 结构纪律（和 policy_drill 同源）
# --------------------------------------------------------------------------


def test_prompt_is_identical_across_arms() -> None:
    """★★ 「同一句话、不同世界、不同正确动作」——**问法不能泄露这是哪一档**。

    ⚠️ **判据在 2026-08-17 改过一次**：原来断言三档的问法**逐字相同**，
    那是"每个模板只有一句话"时代的写法。上了题面改写之后逐字相同不再成立，
    但要守的东西没变 —— **句式不许携带任何关于档位的信息**。

    ⇒ 新判据：**每一档用到的句式集合必须完全相同**（不是"差不多"）。
      某个句式只在 empty 档出现，模型看到它就知道这题会查空，
      根本不用等检索结果 —— 那正是这条测试要挡的。

    ★ 而这是**构造保证**的，不是碰巧：`_phrase` 按 (档 × entry_mode) 轮转选变体。
      第一版用哈希选，实测就漏了 —— empty 独有 2 条、superseded 独有 2 条，
      这条测试当场把它判红了。
    """
    import re as _re

    N = 150
    for mode in ("id_given", "must_discover"):
        by_arm: dict[str, set[str]] = {}
        for i in range(N):
            p = params_for(i)
            if p.entry_mode != mode:
                continue
            bundle = MAKE(p)
            msg = bundle.case.user_message
            for value in (p.campaign_id, p.product, p.region, p.account_id):
                msg = msg.replace(str(value), "§")
            by_arm.setdefault(getattr(p, "insight_state"), set()).add(_re.sub(r"\d+", "#", msg))
        union = set().union(*by_arm.values())
        for arm, phrasings in sorted(by_arm.items()):
            missing = union - phrasings
            assert not missing, (
                f"{mode} 下 {arm} 档缺了 {len(missing)} 种句式 —— "
                f"句式泄露了档位：看到这些说法就知道是哪一档。缺的：{sorted(missing)[:2]}")


def test_outcomes_do_not_collapse() -> None:
    outcomes = {next(t for t in b.case.metadata.tags if t.startswith("outcome:"))
                for p, b in _bundles()}
    assert len(outcomes) == 3, f"outcome 塌成了 {outcomes}"


# --------------------------------------------------------------------------
# 端到端
# --------------------------------------------------------------------------


@pytest.mark.parametrize("state", ["aligned", "conflicting", "absent"])
def test_gold_runs_clean_end_to_end(state: str) -> None:
    p, b = _by_state(state)[0]
    result, caps = _run(b)
    assert caps == set(), f"{state} 档 gold 命中 cap: {caps}"
    assert result.reward >= b.gold.expected_reward_min


def test_answering_one_side_without_reporting_the_conflict_loses_points() -> None:
    """★★★ 「只答一边也算没做这道题」—— 冲突本身就是要报告的信息。

    模型跳过 conflict_resolve、直接给"以数据为准"这个**正确的结论**：
    结论对，但没有记录冲突 ⇒ 下游无法去复核那条历史结论（M12 飞轮的入口断了）。
    ⇒ 分数必须低于 gold。
    """
    p, b = _by_state("conflicting")[0]
    gold_result, _ = _run(b)
    without = [a for a in b.gold.actions if a["tool"] != "memory.conflict_resolve"]
    silent_result, _ = _run(
        b, actions=without,
        final_answer={"decision": "answered", "recommendation": "follow_current_data"})
    assert silent_result.reward < gold_result.reward, \
        "不报冲突和报了冲突同分 ⇒ 这道题没有在考冲突消解"


@pytest.mark.parametrize("state", ["aligned", "conflicting"])
def test_skipping_the_corpus_search_is_expensive(state: str) -> None:
    """★★★ 非结构化检索必须**承重**，不能是装饰。

    自检时量到：修之前 `conflicting` 档不查复盘语料只掉 0.045 分 ——
    因为冲突双方在 memory 里、数据在 metrics 里，**终答没有任何字段依赖语料**。
    那样 `insight.search_claims` 就是个可以跳过的步骤，而它正是 M8 的主体。

    ⇒ 修法：让终答报告一个**只有语料才知道**的事实（团队结论的现行口径），
    跳过检索却填这个字段 ⇒ `false_claim_cap`（终答字段无观测背书）命中。
    """
    p, b = _by_state(state)[0]
    gold_result, _ = _run(b)
    without = [a for a in b.gold.actions if a["tool"] != "insight.search_claims"]
    result, caps = _run(b, actions=without)
    assert "false_claim_cap" in caps, "跳过语料检索却照填字段，没被抓"
    drop = gold_result.reward - result.reward
    assert drop > 0.3, f"跳过检索只掉 {drop:.3f} 分 ⇒ 检索是装饰"


def test_absent_arm_does_not_require_corpus_backed_field() -> None:
    """★ 但 `absent` 档不该要求那个字段 —— 语料里本来就没东西可报。

    要求了就等于逼模型编一个"团队口径"出来，那正是我们要消灭的行为。
    """
    for p, b in _by_state("absent"):
        keys = {f.key for f in b.verifier.required_answer_fields}
        assert "team_conclusion_status" not in keys


def test_absent_arm_answering_from_data_is_not_hallucination() -> None:
    """★★ 这条是那次"cap 判错 gold"的回归守卫。

    复盘库查不到，但手里有实际数据 —— 照数据作答**是正确的**。
    判据必须分清「没有依据却断言」和「依据来自别处」。
    """
    p, b = _by_state("absent")[0]
    _, caps = _run(b)
    assert "no_retrieval_hallucination_cap" not in caps
