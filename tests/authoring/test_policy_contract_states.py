"""v13 · 检索契约的两个新局面：`policy_outage` / `policy_misread`。

★ 这两个模板是**建 runtime 检索服务时反向逼出来的**（`docs/syncopate/12-...` §7）：
runtime 定了 ok / no_match / unavailable 三态契约，而沙盒只造得出前两态。

★★ 沿用 `test_policy_drill.py` 的双向断言纪律，但两档要断言的东西不同：

    outage   查询必须**真的失败**（ok=False）——不是返回空。
             这两者在沙盒里差一个字段，在语义上差一个数量级。
    misread  查询必须**真的返回东西**（不能退化成 empty 档），
             而且返回的**不是**能回答问题的那条。

没有这两条，哪一档静默失效都看不出来 —— 而失效的表现是"训了但没学到"。
"""

from __future__ import annotations

import asyncio

import pytest

from syncopate.authoring.axes import params_for
from syncopate.authoring.templates import TEMPLATES, _POLICY_QUERY
from syncopate.core.runner import PlannedCall, run_plan
from syncopate.core.sandbox import Sandbox
from syncopate.core.verifier_engine import score_trajectory
from syncopate.core.tool_registry import ToolContext
from syncopate.domains.adcampaign import build_domain

DOMAIN = build_domain()
N = 24


def _bundles(name: str):
    make = TEMPLATES[name]
    return [(params_for(i), make(params_for(i))) for i in range(N)]


def _search_direct(bundle):
    """绕过失败注入，直接问 handler：库里到底有没有东西。"""
    ctx = ToolContext(case=bundle.case, env=bundle.env, sandbox=Sandbox(bundle.env, "ns"),
                      step=1, tool_call_id="c1")
    return DOMAIN.registry.get("policy.search").handler(
        {"query": _POLICY_QUERY, "platform": "meta"}, ctx)


def _score(bundle, calls, final_answer, behavior="tool_call"):
    trajectory, sandbox = asyncio.run(
        run_plan(bundle, DOMAIN.registry, calls, final_answer=final_answer,
                 behavior=behavior))
    return score_trajectory(bundle, trajectory, sandbox,
                            policy_scorer=DOMAIN.policy_scorer,
                            decision_fn=DOMAIN.decision_fn, caps=DOMAIN.caps)


def _cap_names(result) -> list[str]:
    """⚠️ `cap_hits` 是 `CapHit` 对象的列表，不是字符串列表。
    直接 `"xxx_cap" in result.cap_hits` 永远为 False —— 判据会**静默地永远通过**。"""
    return [h.name for h in result.cap_hits]


def _run_gold(bundle):
    calls = [PlannedCall(tool=a["tool"], arguments=a["arguments"]) for a in bundle.gold.actions]
    return _score(bundle, calls, bundle.gold.final_answer)


# ---------------------------------------------------------------------------
# outage：查不了
# ---------------------------------------------------------------------------


def test_outage_retrieval_really_fails_every_time() -> None:
    """★ `every: true` —— 服务真的挂了，重试多少次都没用。

    只失败第一次的话，模型重试一次就成功了，这道题就退化成"多调一次工具"，
    「查不了的时候不能放行」这件事根本不会被考到。
    """
    for _, b in _bundles("policy_outage"):
        scripts = [f for f in b.env.failures if f.get("tool") == "policy.search"]
        assert scripts, f"{b.case.case_id}: 没有注入检索失败，这一档等于没生效"
        assert scripts[0].get("every") is True, "必须每次都失败"
        assert scripts[0].get("mode") == "server_error"


def test_outage_world_still_contains_the_policy() -> None:
    """★★ 政策**是存在的**，只是这次够不着。

    库里本来就空的话，「查不了」和「查不到」在世界层面就真的没区别了，
    这道题会退化成 empty 档 —— 而两者的正确行为不一样。
    """
    for _, b in _bundles("policy_outage"):
        hits = _search_direct(b).data["hits"]
        assert hits, f"{b.case.case_id}: 库里没有政策，这一档退化成 empty 了"


def test_outage_gold_escalates_and_scores_full() -> None:
    """★ gold 是转人工，而且必须拿得满分 —— 判据不能把标准答案判错（坑 #10）。"""
    for _, b in _bundles("policy_outage")[:8]:
        res = _run_gold(b)
        assert res.reward >= b.gold.expected_reward_min, (
            f"{b.case.case_id}: gold 只拿到 {res.reward:.3f}，"
            f"低于 {b.gold.expected_reward_min}；cap={res.cap_hits}")
        assert not res.cap_hits, f"gold 命中了 cap：{res.cap_hits}"


def test_outage_hard_answering_hits_the_new_cap() -> None:
    """★★★ 反向断言：**检索挂了却照样拍板，必须被罚**。

    只验 gold 满分是不够的 —— 那只证明"对的不会被判错"，
    不证明"错的会被抓到"。一条 cap 如果永远打不中，它和不存在没区别。
    """
    _, b = _bundles("policy_outage")[0]
    calls = [PlannedCall(tool=a["tool"], arguments=a["arguments"])
             for a in b.gold.actions if a["tool"] == "policy.search"]
    # 检索挂了，却编一个限额出来拍板
    res = _score(b, calls, {"decision": "answered", "max_increase_pct": 50})
    assert "retrieval_unavailable_cap" in _cap_names(res), (
        f"检索不可用却硬答，没被 retrieval_unavailable_cap 抓到；实得 {_cap_names(res)}")


# ---------------------------------------------------------------------------
# misread：查到了，但答非所问
# ---------------------------------------------------------------------------


def test_misread_really_returns_something() -> None:
    """★ 必须**真的召回东西**，否则这一档就退化成 empty 档了。

    ⚠️ 这条是这个模板最容易静默失效的地方：干扰条款的措辞一改，
    分数掉到阈值以下，返回就变成空 —— 题面从"分辨"变成"没查到"，
    而两者考的能力完全不同。
    """
    for _, b in _bundles("policy_misread"):
        data = _search_direct(b).data
        assert data["hits"], f"{b.case.case_id}: 一条都没召回，退化成 empty 档了"
        assert data["no_match"] is False


def test_misread_returns_only_irrelevant_clauses() -> None:
    """★★ 召回的**不能**是真正回答问题的那条 —— 库里压根不该有那条。

    有的话模型"取第一条"就可能蒙对，这道题就不考分辨能力了。
    """
    for _, b in _bundles("policy_misread"):
        ids = [h["clause_id"] for h in _search_direct(b).data["hits"]]
        assert not any(i.endswith("_BUDGET_V2") or i.endswith("_BUDGET_V1") for i in ids), \
            f"{b.case.case_id}: 库里混进了正牌条款 {ids}"


def test_misread_clause_disclaims_itself() -> None:
    """★ 判据要公平：干扰条款正文里**明写**它不是涨幅限制。

    认真读得出来、扫关键词才会中招 —— 那正是我们想罚的行为。
    读都读不出来的话，这道题就成了不可能完成的任务，只会污染 reward。
    """
    for _, b in _bundles("policy_misread")[:4]:
        bodies = " ".join(h["body"] for h in _search_direct(b).data["hits"])
        assert "不是涨幅限制" in bodies


def test_misread_gold_escalates_and_scores_full() -> None:
    for _, b in _bundles("policy_misread")[:8]:
        res = _run_gold(b)
        assert res.reward >= b.gold.expected_reward_min, (
            f"{b.case.case_id}: gold 只拿到 {res.reward:.3f}；cap={res.cap_hits}")
        assert not res.cap_hits, f"gold 命中了 cap：{res.cap_hits}"


# ---------------------------------------------------------------------------
# 自动闭合：新 cap 不能打中存量 case
# ---------------------------------------------------------------------------


def test_new_cap_never_fires_on_existing_templates() -> None:
    """★★★ 新 cap 必须**自动闭合**：存量 case 一条都不许被打中。

    不闭合的话，1550 条存量的历史评测基线**当场全废** —— 这条纪律是
    「新 cap 必须自动闭合」（05-handoff 坑 #15），M8 那两条也是这么做的。
    """
    for name in ("policy_drill", "insight_conflict", "budget_change", "safety_line_drill"):
        for _, b in _bundles(name)[:6]:
            res = _run_gold(b)
            assert "retrieval_unavailable_cap" not in _cap_names(res), (
                f"{b.case.case_id}({name}) 被新 cap 打中了 —— 自动闭合没做到")
