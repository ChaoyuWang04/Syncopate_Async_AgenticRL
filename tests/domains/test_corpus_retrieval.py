"""M8 · RAG v1 语料层与检索的验收测试。

★ 这一份测的不是"检索准不准"，而是**三条轴的可判定性**：
如果"检索为空"在构造上不可能发生、"过期"分不出来、"矛盾"没有载体，
那么 §14 的两项验收（过期检出率 / 无检索幻觉率，都要求趋近 0）就是空话 ——
**机制建好了但没接上**的又一个入口。

⚠️ 还有一条比准确率更重要：**确定性**。GRPO 并发跑同一条 case 8 遍，
检索只要有一点不稳定，组内 reward 差异就分不清是模型还是运气。
"""

from __future__ import annotations

import pytest

from syncopate.core.sandbox import Sandbox
from syncopate.core.schemas import Case, CaseMetadata
from syncopate.core.tool_registry import ToolContext
from syncopate.domains.adcampaign import build_domain
from syncopate.domains.adcampaign.corpus import MATCH_THRESHOLD, overlap_score, tokenize
from syncopate.domains.adcampaign.world import WorldBuilder

DOMAIN = build_domain()
NOW = "2026-08-01T00:00:00+00:00"


def _world() -> WorldBuilder:
    return WorldBuilder("T_0001", reference_now=NOW)


def _call(env, name: str, args: dict):
    """直接调 handler：这两个是纯读工具，不碰台账，不需要走完整 runner。"""
    case = Case(case_id=env.case_id, user_message="-", context={}, entities={},
                metadata=CaseMetadata(signal_class="graded", bucket="rag"), max_steps=8)
    ctx = ToolContext(case=case, env=env, sandbox=Sandbox(env, "ns_test"), step=1,
                      tool_call_id="t1")
    return DOMAIN.registry.get(name).handler(args, ctx)


# --------------------------------------------------------------------------
# 轴一 · 版本对 ⇒ 过期检出
# --------------------------------------------------------------------------


def _versioned_world() -> WorldBuilder:
    return (
        _world()
        .policy_clause(
            "META_BUDGET_V1", title="单日预算涨幅上限",
            body="单日预算上调不得超过前一日的 20%，超过需提交审批。",
            section_path="Meta 广告政策 / 4. 预算与竞价 / 4.2 单日涨幅",
            platform="meta", valid_from_days_ago=400, valid_to_days_ago=30, version="v1",
        )
        .policy_clause(
            "META_BUDGET_V2", title="单日预算涨幅上限",
            body="单日预算上调不得超过前一日的 50%，超过需提交审批。",
            section_path="Meta 广告政策 / 4. 预算与竞价 / 4.2 单日涨幅",
            platform="meta", valid_from_days_ago=29, version="v2", supersedes="META_BUDGET_V1",
        )
    )


def test_expired_clause_is_returned_but_flagged() -> None:
    """过期条款**不能被隐藏** —— 隐藏了就没法考"会不会误用旧版本"。"""
    env = _versioned_world().build()
    res = _call(env, "policy.search", {"query": "单日预算涨幅上限", "platform": "meta"})
    assert res.ok
    by_id = {h["clause_id"]: h for h in res.data["hits"]}

    assert "META_BUDGET_V1" in by_id, "过期条款必须照常返回，不能替模型过滤掉"
    assert by_id["META_BUDGET_V1"]["expired"] is True
    assert by_id["META_BUDGET_V2"]["expired"] is False


def test_expired_clause_points_at_its_successor() -> None:
    """光标 expired 不够：必须给出现行版本，否则"改引用哪条"无从判断。"""
    env = _versioned_world().build()
    res = _call(env, "policy.search", {"query": "单日预算涨幅上限", "platform": "meta"})
    by_id = {h["clause_id"]: h for h in res.data["hits"]}
    assert by_id["META_BUDGET_V1"]["superseded_by"] == "META_BUDGET_V2"
    assert by_id["META_BUDGET_V2"]["superseded_by"] is None


def test_expiry_is_relative_to_reference_now_not_wall_clock() -> None:
    """★ 同一份语料，只改 case 声明的"今天"，过期与否必须翻转。

    「跨轴共用同一条时间线」是踩过的坑：凡是"相对今天"的语义都要相对
    reference_now 算。这条守着它。
    """
    early = WorldBuilder("T_0002", reference_now="2026-06-01T00:00:00+00:00")
    early.policy_clause(
        "C1", title="单日预算涨幅上限", body="上调不得超过 20%。",
        section_path="Meta 广告政策 / 4.2", platform="meta", valid_to_days_ago=-40,
    )  # valid_to 在 reference_now 之后 40 天 ⇒ 还没过期
    res = _call(early.build(), "policy.search", {"query": "单日预算涨幅上限"})
    assert res.data["hits"][0]["expired"] is False


# --------------------------------------------------------------------------
# 轴二 · 空洞 ⇒ 无检索幻觉
# --------------------------------------------------------------------------


def test_no_match_returns_empty_not_the_closest_row() -> None:
    """★ 这条是「无检索幻觉率」这项验收成立的前提。

    真检索系统总能返回 top-k 哪怕全是噪声；我们**必须**保留"确实查不到"这个状态，
    否则「检索为空时不编答案」在构造上就不可能被触发。
    """
    env = _versioned_world().build()
    res = _call(env, "policy.search", {"query": "素材审核时长 iOS 隐私标签"})
    assert res.ok, "查不到是正常状态，不是报错"
    assert res.data["hits"] == []
    assert res.data["no_match"] is True


def test_empty_corpus_is_the_default() -> None:
    """没声明语料 = 检索不到。默认值本身就是「空洞」那条轴。"""
    res = _call(_world().build(), "policy.search", {"query": "任何东西"})
    assert res.data["no_match"] is True
    res2 = _call(_world().build(), "insight.search_claims", {"query": "任何东西"})
    assert res2.data["no_match"] is True


def test_threshold_actually_rejects_partial_overlap() -> None:
    """阈值必须真的能拒绝"只沾一个词"的文档，否则空洞永远不会发生。"""
    assert overlap_score("单日预算涨幅上限", "单日预算上调不得超过 20%") >= MATCH_THRESHOLD
    assert overlap_score("素材审核时长 iOS 隐私标签", "单日预算上调不得超过 20%") < MATCH_THRESHOLD


def test_stopwords_do_not_manufacture_matches() -> None:
    """停用词不去掉的话，任意两条中文文本都会有相似度 ⇒ 空洞消失。"""
    assert overlap_score("这个是在的", "那个也是在的") < MATCH_THRESHOLD
    assert "的" not in tokenize("预算的上限")


# --------------------------------------------------------------------------
# 轴三 · 矛盾对 ⇒ conflict_resolve 的题面
# --------------------------------------------------------------------------


def _contradiction_world() -> WorldBuilder:
    return (
        _world()
        .insight(
            "CLAIM_0001", claim="东南亚地区真人出镜素材的 D7 ROAS 显著高于纯 CG",
            scope={"region": "SEA", "product": "PUZ_QUEST", "period": "2026Q2"},
            evidence="复盘会议 2026-05-15，样本 N=42 campaigns",
            confidence="medium", days_ago=80, status="superseded",
            superseded_by="CLAIM_0002",
        )
        .insight(
            "CLAIM_0002", claim="东南亚地区真人出镜素材的 D7 ROAS 已回落至与纯 CG 持平",
            scope={"region": "SEA", "product": "PUZ_QUEST", "period": "2026Q3"},
            evidence="复盘会议 2026-07-20，样本 N=61 campaigns",
            confidence="high", days_ago=12, status="active",
        )
    )


def test_contradiction_is_visible_without_asking_for_it() -> None:
    """★★ 这条是 M8 里最重要的一条测试：**矛盾必须默认可见**。

    2026-08-14 自审翻转了默认值。原来默认隐藏 superseded/refuted，理由是"降噪"——
    但那样模型永远看不到矛盾，除非它主动想到去翻旧账，**而它没有任何理由想到**。
    ⇒「查到的历史结论和现在的数据矛盾了怎么办」这道题在构造上就出不来，
    `memory.conflict_resolve` 的题面也就永远造不出来（遗留清单里挂着的那个缺口）。
    设计文档 §13：「你需要知道"我们曾经这么以为"」。
    """
    env = _contradiction_world().build()
    res = _call(env, "insight.search_claims", {"query": "东南亚 真人出镜 素材 ROAS"})
    by_id = {h["claim_id"]: h for h in res.data["hits"]}
    assert set(by_id) == {"CLAIM_0001", "CLAIM_0002"}, "默认就该看见矛盾双方"
    assert by_id["CLAIM_0001"]["status"] == "superseded"
    assert by_id["CLAIM_0001"]["superseded_by"] == "CLAIM_0002"
    assert by_id["CLAIM_0002"]["status"] == "active"


def test_active_only_narrows_to_current_conclusions() -> None:
    """确认只需要当前口径时才收窄 —— 是显式选项，不是默认行为。"""
    env = _contradiction_world().build()
    res = _call(env, "insight.search_claims",
                {"query": "东南亚 真人出镜 素材 ROAS", "active_only": True})
    assert [h["claim_id"] for h in res.data["hits"]] == ["CLAIM_0002"]


def test_scope_filter_does_not_leak_across_regions() -> None:
    """把 D7 安全线塞进向量库是灾难 —— 同样的道理，scope 过滤必须是精确的。"""
    env = _contradiction_world().insight(
        "CLAIM_0003", claim="美国地区真人出镜素材的 D7 ROAS 显著高于纯 CG",
        scope={"region": "US", "product": "PUZ_QUEST"}, days_ago=10,
    ).build()
    res = _call(env, "insight.search_claims",
                {"query": "真人出镜 素材 ROAS", "region": "SEA"})
    ids = {h["claim_id"] for h in res.data["hits"]}
    assert "CLAIM_0003" not in ids, "US 的结论泄漏进了 SEA 的查询"
    assert ids == {"CLAIM_0001", "CLAIM_0002"}, "SEA 的结论（含被取代的）都该在"


# --------------------------------------------------------------------------
# 横切 · 确定性
# --------------------------------------------------------------------------


@pytest.mark.parametrize("tool,args", [
    ("policy.search", {"query": "单日预算涨幅上限", "platform": "meta"}),
    ("insight.search_claims", {"query": "东南亚 真人出镜 素材 ROAS"}),
])
def test_retrieval_is_deterministic_across_repeats(tool: str, args: dict) -> None:
    """★★ 同一份输入跑 8 遍（= GRPO 的组大小）必须逐字节相同。

    检索只要有一点不稳定，组内 reward 差异就分不清是"模型做得不同"还是"运气不同"
    —— advantage 被污染。这是 RL 里任何跨 rollout 随机性都不能有的原因。
    """
    env = _versioned_world().insight(
        "CLAIM_0002", claim="东南亚地区真人出镜素材的 D7 ROAS 已回落",
        scope={"region": "SEA"}, days_ago=12,
    ).build()
    results = [_call(env, tool, args).data for _ in range(8)]
    assert all(r == results[0] for r in results)


def test_ranking_does_not_depend_on_dict_insertion_order() -> None:
    """同分时按 id 排 —— 栽过一次「prompt 内容取决于 dict 插入顺序」。"""
    def build(order):
        w = _world()
        for cid in order:
            w.policy_clause(cid, title="素材审核时长", body="素材审核通常在 24 小时内完成。",
                            section_path="SOP / 审核")
        return w.build()

    a = _call(build(["A_CLAUSE", "B_CLAUSE"]), "policy.search", {"query": "素材审核时长"})
    b = _call(build(["B_CLAUSE", "A_CLAUSE"]), "policy.search", {"query": "素材审核时长"})
    assert [h["clause_id"] for h in a.data["hits"]] == [h["clause_id"] for h in b.data["hits"]]
