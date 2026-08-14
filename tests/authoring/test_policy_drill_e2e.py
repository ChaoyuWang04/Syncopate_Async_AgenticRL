"""M8 · `policy_drill` 的**端到端**验收：真 runner + 真 verifier。

★★★ 为什么必须单独有这一份

`tests/domains/test_rag_caps.py` 里的 10 条是手搓 `Trajectory` 喂给 cap 的单测 ——
它们证明**判据逻辑**对，但**证明不了机制接上了**。要让
`cited_expired_policy_cap` 真的能命中，需要四件事**同时**成立：

    ① 模板的 VerifierSpec 里声明 cited_clause_id 字段
    ② 语料里真的有过期条款 + 现行版本成对出现
    ③ 检索工具真的把两条都返回、且旧的标了 expired
    ④ 终答里真的填了 clause_id

缺任何一件，cap 就永远是 0，而且**看起来一切正常** —— rl_report 里那一列是 0，
你会以为"模型没犯这个错"，实际是"这个错不可能被观测到"。
这是本项目反复栽的那个形状，只有走完整条链路才能证伪。
"""

from __future__ import annotations

import asyncio

import pytest

from syncopate.authoring.axes import params_for
from syncopate.authoring.templates import TEMPLATES
from syncopate.core.runner import PlannedCall, run_plan
from syncopate.core.verifier_engine import score_trajectory
from syncopate.domains.adcampaign import build_domain

DOMAIN = build_domain()
DOMAIN.registry.latency_scale = 0.0
MAKE = TEMPLATES["policy_drill"]


def _first(state: str):
    for i in range(200):
        p = params_for(i)
        if p.rag_state == state:
            return p, MAKE(p)
    raise AssertionError(f"没有 {state} 档的 case")


def _run(bundle, *, final_answer, behavior=None, actions=None):
    calls = [PlannedCall(tool=a["tool"], arguments=a.get("arguments", {}))
             for a in (actions if actions is not None else bundle.gold.actions)]
    traj, sandbox = asyncio.run(run_plan(
        bundle, DOMAIN.registry, calls, final_answer=final_answer,
        behavior=behavior or bundle.verifier.expected_behavior))
    result = score_trajectory(bundle, traj, sandbox,
                              policy_scorer=DOMAIN.policy_scorer,
                              decision_fn=DOMAIN.decision_fn, caps=DOMAIN.caps)
    return result, {h.name for h in result.cap_hits}


# --------------------------------------------------------------------------
# 过期检出
# --------------------------------------------------------------------------


def test_gold_cites_the_active_version_and_stays_clean() -> None:
    """★ 正例先跑：标准答案不能被自己的判据判错（坑 #10）。"""
    _, bundle = _first("superseded")
    result, caps = _run(bundle, final_answer=bundle.gold.final_answer)
    assert caps == set(), f"gold 命中了 cap: {caps}"
    assert result.reward >= bundle.gold.expected_reward_min


def test_citing_the_expired_version_fires_the_cap_end_to_end() -> None:
    """★★★ 这条是整个 M8 「过期检出率」验收的**存在性证明**。

    只把终答里的 clause_id 换成过期那条，其余（工具调用、其它字段）一字不改。
    """
    _, bundle = _first("superseded")
    expired_id = next(k for k in bundle.env.readonly_tables["policy_clauses"]
                      if k.endswith("_V1"))
    wrong = {**bundle.gold.final_answer, "cited_clause_id": expired_id}

    result, caps = _run(bundle, final_answer=wrong)
    assert "cited_expired_policy_cap" in caps, \
        "引用过期条款没被抓到 —— 那四件事里断了一件，这项验收是死的"
    assert result.reward <= 0.10 + 1e-6, f"cap 没封顶，reward={result.reward}"


def test_present_arm_cannot_fire_the_expiry_cap() -> None:
    """对照档里没有过期条款，这条 cap 不该有任何办法命中。"""
    _, bundle = _first("present")
    result, caps = _run(bundle, final_answer=bundle.gold.final_answer)
    assert "cited_expired_policy_cap" not in caps


# --------------------------------------------------------------------------
# 无检索幻觉
# --------------------------------------------------------------------------


def test_empty_arm_gold_escalates_and_stays_clean() -> None:
    _, bundle = _first("empty")
    result, caps = _run(bundle, final_answer=bundle.gold.final_answer)
    assert caps == set(), f"empty 档的 gold 命中了 cap: {caps}"
    assert result.reward >= bundle.gold.expected_reward_min


def test_answering_confidently_after_empty_retrieval_fires_the_cap() -> None:
    """★★★ 「无检索幻觉率」验收的存在性证明：查不到还照答，必须被抓。

    只保留检索那一步（不开审批单），终答编一个限额出来 —— 正是要消灭的行为。
    """
    _, bundle = _first("empty")
    search_only = [a for a in bundle.gold.actions if a["tool"] == "policy.search"]
    fabricated = {"decision": "answered", "max_increase_pct": 50,
                  "cited_clause_id": "META_BUDGET_不存在"}

    result, caps = _run(bundle, final_answer=fabricated, actions=search_only)
    assert "no_retrieval_hallucination_cap" in caps, \
        "检索为空仍硬答没被抓到 —— 这项验收是死的"


@pytest.mark.parametrize("behavior", ["clarify", "defer"])
def test_not_answering_after_empty_retrieval_is_allowed(behavior: str) -> None:
    """★ 逃生口：查不到就去问 / 明说等数据，都是 §14 要的标准答案，不能被判错。"""
    _, bundle = _first("empty")
    search_only = [a for a in bundle.gold.actions if a["tool"] == "policy.search"]
    result, caps = _run(bundle, final_answer={"question": "指的是哪个平台的政策？"},
                        behavior=behavior, actions=search_only)
    assert "no_retrieval_hallucination_cap" not in caps


def test_hallucination_cap_silent_when_retrieval_succeeded() -> None:
    """检索成功的两档不该被这条 cap 碰到 —— 自动闭合在真链路上也要成立。"""
    for state in ("present", "superseded"):
        _, bundle = _first(state)
        _, caps = _run(bundle, final_answer=bundle.gold.final_answer)
        assert "no_retrieval_hallucination_cap" not in caps, state
