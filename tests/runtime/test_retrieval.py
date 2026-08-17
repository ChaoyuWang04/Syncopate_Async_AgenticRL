"""M9+ · Runtime 检索的契约测试。设计见 `docs/syncopate/12-rag-runtime-design.md`。

★★★ 这一份守的核心只有一条：**「查不到」和「查不了」永远分得开**

两者的 `hits` 都是空的，但语义正好相反：

    no_match     查了，语料里确实没有   ⇒ 「没有政策限制这件事」  ⇒ 可带 caveat 继续
    unavailable  查不了                ⇒ 「不知道有没有限制」    ⇒ **绝对不能继续**

合并的话，**一次故障看起来就是放行信号** —— 那正是压测场景④要抓的灾难。

⚠️ 需要跑着的 PG + 已入库的种子语料：
`bash scripts/pg_bootstrap.sh && python scripts/ingest_corpus.py`。**跳过不是通过。**
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone

import pytest

from syncopate.domains.adcampaign import corpus as sandbox_corpus
from syncopate.runtime import retrieval as rt
from syncopate.runtime.db import Database
from syncopate.runtime.gateway import DecisionContext, evaluate_triggers
from syncopate.runtime.retrieval import (
    RetrievalService, RetrievalStatus, upsert_insights, upsert_policy_clauses,
)


def _pg_available() -> bool:
    async def probe() -> bool:
        db = Database()
        try:
            await db.connect(max_size=2)
            await db.close()
            return True
        except Exception:
            return False
    return asyncio.run(probe())


pytestmark = pytest.mark.skipif(
    not _pg_available(),
    reason="需要跑着的 PostgreSQL：bash scripts/pg_bootstrap.sh")


def with_svc(body):
    async def main():
        db = Database()
        await db.connect(max_size=5)
        try:
            return await body(db, RetrievalService(db))
        finally:
            await db.close()
    return asyncio.run(main())


NOW = datetime(2026, 8, 17, tzinfo=timezone.utc)
LATER = datetime(2026, 9, 10, tzinfo=timezone.utc)      # V1 的 valid_to 之后


# ---------------------------------------------------------------------------
# 三态契约
# ---------------------------------------------------------------------------


def test_hit_returns_ok_with_scores() -> None:
    async def body(db, svc):
        return await svc.search_policy(org_id="org_x", query="单日预算涨幅上限", now=NOW)
    r = with_svc(body)
    assert r.status is RetrievalStatus.OK
    assert r.hits and all("score" in h for h in r.hits)
    assert r.usable and not r.blocks_decision


def test_nothing_relevant_returns_no_match_not_a_best_guess() -> None:
    """★ 阈值是真的：低于它就是**没查到**，不是"返回最像的那条"。

    真检索系统总能返回 top-k，哪怕全是噪声。我们**必须**保留"确实查不到"这个状态，
    否则「检索为空时不编答案」这项验收在构造上就不可能被触发
    （同 BM25 被淘汰的理由：归一化后 top1 恒为 1.0，它是排序器不是判定器）。
    """
    async def body(db, svc):
        return await svc.search_policy(org_id="org_x", query="量子计算加速广告投放", now=NOW)
    r = with_svc(body)
    assert r.status is RetrievalStatus.NO_MATCH
    assert r.hits == []
    assert not r.blocks_decision, "「查不到」不该禁止继续 —— 它的语义是「没有相关政策」"


def test_unavailable_never_degrades_to_no_match() -> None:
    """★★★ 本文件最重要的一条。服务出问题时"假装没查到"是最坏的默认值。"""
    class Broken:
        def tx(self):
            raise ConnectionError("connection refused")

    r = asyncio.run(RetrievalService(Broken()).search_policy(org_id="org_x", query="单日预算"))
    assert r.status is RetrievalStatus.UNAVAILABLE
    assert r.status is not RetrievalStatus.NO_MATCH
    assert r.hits == []
    assert r.blocks_decision, "「查不了」必须禁止继续 —— 我们不知道有没有政策限制"
    assert r.error and "unavailable" in r.error


def test_empty_and_unavailable_look_the_same_but_decide_differently() -> None:
    """★ 把两条并排放在一条测试里，是因为**它们的区别只在 status 上**。

    谁要是哪天用 `if not result.hits` 来判断"没有政策限制"，这条会红。
    """
    class Broken:
        def tx(self):
            raise ConnectionError("boom")

    async def body(db, svc):
        return await svc.search_policy(org_id="org_x", query="今天天气怎么样", now=NOW)

    empty = with_svc(body)
    down = asyncio.run(RetrievalService(Broken()).search_policy(org_id="org_x", query="今天天气怎么样"))

    assert empty.hits == down.hits == []            # 现象一模一样
    assert empty.blocks_decision != down.blocks_decision   # 决策后果相反


# ---------------------------------------------------------------------------
# 过期与取代：算出来的，不是查出来的
# ---------------------------------------------------------------------------


def test_expiry_is_computed_at_query_time_not_stored() -> None:
    """★ 同一条条款，换一个 `now` 就该换一个 `expired` —— 说明它是算的不是存的。"""
    async def body(db, svc):
        a = await svc.search_policy(org_id="org_x", query="单日预算涨幅上限", now=NOW)
        b = await svc.search_policy(org_id="org_x", query="单日预算涨幅上限", now=LATER)
        return a, b
    before, after = with_svc(body)
    v1_before = next(h for h in before.hits if h["clause_id"] == "POL_BUDGET_DAILY_V1")
    v1_after = next(h for h in after.hits if h["clause_id"] == "POL_BUDGET_DAILY_V1")
    assert v1_before["expired"] is False
    assert v1_after["expired"] is True


def test_supersession_is_resolved_from_the_newer_clause() -> None:
    """★ 取代关系**从新条款反查**（新版本声明 supersedes），旧条款上不回写。

    语料是增量追加的，回写等于每加一版都要改历史 —— 那正是"版本"要避免的。
    """
    async def body(db, svc):
        return await svc.search_policy(org_id="org_x", query="单日预算涨幅上限", now=NOW)
    r = with_svc(body)
    by_id = {h["clause_id"]: h for h in r.hits}
    assert by_id["POL_BUDGET_DAILY_V1"]["superseded_by"] == "POL_BUDGET_DAILY_V2"
    assert by_id["POL_BUDGET_DAILY_V2"]["superseded_by"] is None


# ---------------------------------------------------------------------------
# 多租户
# ---------------------------------------------------------------------------


def test_private_corpus_does_not_leak_across_orgs() -> None:
    """★ 平台政策是 global，内部 SOP 是 org 私有。越权在 SQL 的 scope 条件里挡。"""
    mine, other = f"org_{uuid.uuid4().hex[:8]}", f"org_{uuid.uuid4().hex[:8]}"

    async def body(db, svc):
        await upsert_policy_clauses(db, [{
            "clause_id": f"SOP_{uuid.uuid4().hex[:6]}",
            "title": "内部审批流程绕行禁令",
            "body": "内部审批流程绕行禁令：任何情况下不得跳过内部审批流程直接执行。",
            "section_path": "内部 SOP / 审批",
        }], scope=mine)
        return (await svc.search_policy(org_id=mine, query="内部审批流程绕行禁令"),
                await svc.search_policy(org_id=other, query="内部审批流程绕行禁令"))

    ours, theirs = with_svc(body)
    assert ours.status is RetrievalStatus.OK
    assert theirs.status is RetrievalStatus.NO_MATCH, "别的 org 的私有 SOP 泄漏了"


# ---------------------------------------------------------------------------
# 复盘结论：被推翻的不是删掉，是标出来
# ---------------------------------------------------------------------------


def test_refuted_claims_are_excluded_by_default_but_retrievable_on_request() -> None:
    """★ `status` 是 M12 飞轮的物理接口：跑出相反结论时标 refuted 而不是删 ——
    **你需要知道"我们曾经这么以为"**。所以默认不给，但不是藏起来。"""
    scope = f"org_{uuid.uuid4().hex[:8]}"

    async def body(db, svc):
        await upsert_insights(db, [{
            "claim_id": f"C_{uuid.uuid4().hex[:6]}",
            "claim": "老结论：纯 CG 素材留存更好应优先铺量",
            "status": "refuted",
        }], scope=scope)
        return (await svc.search_claims(org_id=scope, query="老结论：纯 CG 素材留存更好应优先铺量"),
                await svc.search_claims(org_id=scope, query="老结论：纯 CG 素材留存更好应优先铺量",
                                        include_inactive=True))

    default, with_inactive = with_svc(body)
    assert default.status is RetrievalStatus.NO_MATCH
    assert with_inactive.status is RetrievalStatus.OK
    assert with_inactive.hits[0]["status"] == "refuted"


# ---------------------------------------------------------------------------
# 接到降级网关上
# ---------------------------------------------------------------------------


def test_gateway_tells_cannot_find_apart_from_cannot_query() -> None:
    """★ 两者都开审批单（网关本来就是"不要继续，交给人"的机制），
    但 `trigger_reason` 必须不同 —— 人看到"检索为空"和"检索挂了"要做的判断完全不同。"""
    empty = evaluate_triggers(DecisionContext(retrieval_empty_tools=["policy.search"]))
    down = evaluate_triggers(DecisionContext(retrieval_unavailable_tools=["policy.search"]))

    assert [t.reason for t in empty] == ["retrieval_empty"]
    assert [t.reason for t in down] == ["retrieval_unavailable"]
    assert empty[0].reason != down[0].reason


# ---------------------------------------------------------------------------
# 契约同源
# ---------------------------------------------------------------------------


def test_runtime_and_sandbox_share_one_scoring_function() -> None:
    """★ 沙盒和 runtime 必须用**同一个**打分函数对象，不是"看起来一样的两份"。

    两边各写一份，漂移了没有任何东西会响 —— 而检索漂移的表现是
    「训练时查得到、上线查不到」，属于训推不一致里最难查的一种。
    """
    assert rt.overlap_score is sandbox_corpus.overlap_score


def test_runtime_threshold_is_deliberately_not_the_sandbox_one() -> None:
    """★★ 阈值**故意不同**，而且必须更严。

    打分函数一样、候选集不一样：沙盒每条 case 只有 1–2 篇手写语料，构造上不会互撞；
    runtime 是整个语料库一起打分，误召回概率随条数单调上升。
    ⇒ 这条测试守的不是"0.53 这个数"，是**"有人把 runtime 阈值改回沙盒那个数"这件事会被发现**。
    重标定：`python scripts/calibrate_runtime_retrieval.py`。
    """
    assert rt.RUNTIME_MATCH_THRESHOLD > sandbox_corpus.MATCH_THRESHOLD


# ---------------------------------------------------------------------------
# 数据成熟度：归因延迟是第一性约束，这条降级不该缺生产者
# ---------------------------------------------------------------------------


def test_immature_data_stops_the_run_for_approval() -> None:
    """★ 设计 §0.3：**D7 才知对错，D1 数据极易被误当结论**。

    此前 worker 把 `data_maturity` 硬编码成 `"mature"` ⇒ `data_immature` 这个降级
    在真实路径上**永远不会发生**。现在它从平台查（`get_freshness`），
    剧本把数据调成 2 天大 ⇒ 必须停下来开审批单。
    """
    import uuid as _uuid

    from syncopate.runtime.db import create_run
    from syncopate.runtime.platform import FakeAdPlatform, FaultPlan
    from syncopate.runtime.worker import Worker, WorkerConfig

    async def body(db, _svc):
        async with db.tx() as conn:
            await conn.execute(
                "UPDATE agent_runs SET status='cancelled' WHERE status='queued' "
                "OR (status='running' AND lease_expires_at < now())")
        org, run = f"org_{_uuid.uuid4().hex[:8]}", f"run_{_uuid.uuid4().hex[:8]}"
        await create_run(db, org_id=org, run_id=run, user_message="D1 就砍预算")
        plat = FakeAdPlatform(faults=FaultPlan(data_age_days=2))     # 远没到 D7
        w = Worker(db, plat, WorkerConfig(concurrency=1, amount_threshold=10_000_000))
        for _ in range(50):
            if await w.run_once() == run:
                break
        async with db.tx() as conn:
            return await conn.fetchval(
                "SELECT trigger_reason FROM approval_cases WHERE org_id=$1", org)

    reason = with_svc(body)
    assert reason and "data_immature" in reason, \
        f"D7 未收敛却没停下来，实得 {reason!r}"
