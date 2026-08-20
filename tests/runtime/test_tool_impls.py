"""runtime 工具实现（B-2 第一批）：**外部世界的形状 → 模型认识的形状**。

★ 这一层测的不是"功能对不对"，是**适配对不对** ——
  平台给的是 Meta 的形状（`paging.cursors.after`），
  模型被训练成读沙盒的形状（`next_cursor`）。翻译错了不会报错，只会让模型读不懂。

⚠️ 而"模型读不懂"的表现往往是**它不翻页了** —— 看起来像"账户里只有 3 条"，
   而不是像一个 bug。
"""

from __future__ import annotations

import asyncio

import pytest

from syncopate.runtime import tool_impls as impl
from syncopate.runtime.platform import FakeAdPlatform


def _run(coro):
    return asyncio.run(coro)


def _platform(n: int) -> FakeAdPlatform:
    p = FakeAdPlatform()
    for i in range(n):
        p.campaigns[f"CMP_{i:03d}"] = {"name": f"C{i}", "status": "ACTIVE",
                                       "daily_budget": 1000 + i}
    return p


# ── campaign.list：分页形状的翻译 ────────────────────────────────────────

def test_page_size_follows_the_contract_the_model_was_trained_on():
    """★★ 沙盒描述里写死了「每页最多 3 条」，**那句话在模型的 prompt 里**。

    平台自己的上限是 25（实查 Meta）。两个数不一样是**对的**：
    平台的上限是外部世界的事实，3 是**我们和模型之间的契约**。
    ⇒ 适配层按小的那个走。
    """
    page = _run(impl.campaign_list(_platform(10), account_id="ACC_1"))
    assert len(page["campaigns"]) == impl.SANDBOX_PAGE_SIZE == 3


def test_next_cursor_is_empty_exactly_when_there_is_no_next_page():
    """★ 沙盒描述：「`next_cursor` 非空表示还有下一页」——**含义必须一致**。

    ⚠️ 空**必须**表示"没有下一页"，不能表示"我不知道"。
      模型拿它当判据，含义一歪，要么漏翻页、要么无限翻。
    """
    assert _run(impl.campaign_list(_platform(2), account_id="A"))["next_cursor"] == ""
    assert _run(impl.campaign_list(_platform(10), account_id="A"))["next_cursor"] != ""


def test_the_cursor_actually_walks_the_account():
    seen, cursor = [], None
    p = _platform(10)
    while True:
        page = _run(impl.campaign_list(p, account_id="A", cursor=cursor))
        seen += [c["id"] for c in page["campaigns"]]
        cursor = page["next_cursor"]
        if not cursor:
            break
    assert len(seen) == 10 and len(set(seen)) == 10


def test_filtering_does_not_swallow_the_next_cursor():
    """★ 过滤是在取回**之后**做的 ⇒ 这一页可能被滤空，
    但 `next_cursor` 仍然要给 —— **空页不等于没有下一页**。

    ⚠️ 少了这条，模型会在第一个"全被滤掉"的页停下，
      并以为"账户里没有符合条件的 campaign"。
    """
    p = _platform(10)
    for i in range(3):                       # 第一页全部设成别的状态
        p.campaigns[f"CMP_{i:03d}"]["status"] = "PAUSED"
    page = _run(impl.campaign_list(p, account_id="A", status="ACTIVE"))
    assert page["campaigns"] == []
    assert page["next_cursor"] != "", "★ 页被滤空就不给游标 ⇒ 模型会以为到头了"


# ── metrics.get_freshness：只给事实 ──────────────────────────────────────

def test_freshness_gives_facts_not_a_verdict():
    """★★ 沙盒描述原话：「**只给事实，不给结论**」。

    ⇒ 不许返回"可不可信 / 该不该动"这类字段。
      多给一个结论字段，等于**把决策从模型手里拿走** —— 而那正是我们要训练的东西。
    """
    out = _run(impl.metrics_get_freshness(_platform(1), campaign_id="CMP_000"))
    verdicts = {"trustworthy", "should_act", "recommendation", "advice", "reliable"}
    assert not (verdicts & set(out)), f"给出了结论字段：{out}"
    assert {"days_elapsed", "converge_at_day"} <= set(out)


def test_roas_d7_is_the_slowest_to_converge():
    """默认指标是 `roas_d7`，而它需要 7 天 —— 归因延迟是本项目的第一性约束。"""
    p = _platform(1)
    slow = _run(impl.metrics_get_freshness(p, campaign_id="CMP_000"))
    fast = _run(impl.metrics_get_freshness(p, campaign_id="CMP_000", metric="ctr"))
    assert slow["metric"] == "roas_d7"
    assert slow["converge_at_day"] > fast["converge_at_day"]


# ── 检索类：三态必须传下去 ───────────────────────────────────────────────

class _FakeRetrieval:
    def __init__(self, status, hits=None) -> None:
        self._status, self._hits = status, hits or []

    async def search_policy(self, **kw):      # noqa: ANN003
        return self._make()

    async def search_claims(self, **kw):      # noqa: ANN003
        return self._make()

    def _make(self):
        from syncopate.runtime.retrieval import RetrievalResult
        return RetrievalResult(status=self._status, hits=self._hits,
                               query="q", latency_ms=1)


def test_no_match_and_unavailable_stay_distinguishable():
    """★★★ 「查不到」和「查不了」**不能合并**（`12 §3.1`）。

        查不到  = 「没有政策限制这件事」
        查不了  = 「**我们不知道**有没有限制」

    合并的话，一次服务故障看起来就是**放行信号**。
    ⇒ 所以返回里必须有 `status`，而不是只有一个可能为空的列表。
    """
    from syncopate.runtime.retrieval import RetrievalStatus
    empty = _run(impl.policy_search(_FakeRetrieval(RetrievalStatus.NO_MATCH), "o", query="q"))
    down = _run(impl.policy_search(_FakeRetrieval(RetrievalStatus.UNAVAILABLE), "o", query="q"))
    assert empty["clauses"] == down["clauses"] == []
    assert empty["status"] != down["status"], "★ 两种空长得一样，就是在放行未知风险"


def test_claims_default_to_showing_superseded_ones_too():
    """★ `active_only` 默认 **False** —— 和沙盒一致。

    默认只给现行结论的话，模型就**看不见"这条结论被推翻过"**，
    而「发现历史结论和现在的数据矛盾」正是我们要它学会的一类判断。
    """
    from syncopate.runtime.retrieval import RetrievalStatus
    hits = [{"id": 1, "status": "active"}, {"id": 2, "status": "superseded"}]
    r = _FakeRetrieval(RetrievalStatus.OK, hits)
    both = _run(impl.insight_search_claims(r, "o", query="q"))
    only = _run(impl.insight_search_claims(r, "o", query="q", active_only=True))
    assert len(both["claims"]) == 2
    assert len(only["claims"]) == 1


def test_every_seeded_tool_observation_is_json_serializable() -> None:
    """★★★ 观测必须能被**渲染给模型** —— 这是"工具能跑"之外的第二个必要条件。

    2026-08-20 实测抓到的一整族失败：PG 的 NUMERIC→`Decimal`、DATE→`date`、
    TIMESTAMPTZ→`datetime` **都不能 JSON 序列化**，而 observation 最终要经
    `json.dumps` 变成 tool message 喂回模型。少了归一化的后果**不是报错**，
    是收口按 `tool_crashed` 兜住 ⇒ 模型收到"这个工具暂时不可用" ⇒
    如实回答"查不到" ⇒ **人看到的是"模型能力差"**。

    ⚠️⚠️ 这条判据我第一版写空了：用了个**没有数据**的 org ⇒ 全部返回
      `found: False` ⇒ 压根没碰到 NUMERIC/日期字段 ⇒ 撤掉修复照样绿。
      ⇒ **必须先播真数据再查**（"判据太宽会为错误的理由通过"的又一例）。
    """
    from functools import partial

    from syncopate.runtime.db import Database
    from syncopate.train.rollout_loop import observation_message

    async def body(db):
        import datetime as _dt
        import json as _json
        import uuid as _uuid

        org = f"org_{_uuid.uuid4().hex[:8]}"
        today = _dt.date.today()
        async with db.tx() as conn:
            # ★ 三张表各带一个"危险类型"：NUMERIC + DATE + TIMESTAMPTZ
            await conn.execute(
                "INSERT INTO safety_lines (org_id, product_id, region, cpi_d7_max,"
                " roas_d7_min, retention_d1_min, daily_budget_max, valid_from, valid_to)"
                " VALUES ($1,'P1','R1',2.5,0.5,0.3,100000,$2,$3)",
                org, today - _dt.timedelta(days=1), today + _dt.timedelta(days=30))
            await conn.execute(
                "INSERT INTO memory_records (org_id, record_id, lane, subject, content,"
                " confidence, evidence_refs, expires_at)"
                " VALUES ($1,'m1','episodic',$2,'c',0.85,$3,$4)",
                org, _json.dumps({"campaign_id": "C1"}), _json.dumps(["a", "b"]),
                _dt.datetime.now(_dt.timezone.utc) + _dt.timedelta(days=30))
            await conn.execute(
                "INSERT INTO geo_performance (product_id, region, roas_d7, cpi_d7,"
                " asset_count) VALUES ($1,'R1',0.62,2.1,48)"
                " ON CONFLICT (product_id, region) DO NOTHING", f"P_{org[-6:]}")

        # ★★ 必须走**真实的收口路径**（gate.invoke → _observation），
        #   而不是自己调 `_json_safe` —— 我第一版就是自己调的，
        #   于是"撤掉修复"照样绿（判据没量运行时那条路径 = 没量过）。
        from syncopate.runtime.action_gate import ActionGate, ToolBinding
        from syncopate.runtime.db import create_run
        from syncopate.runtime.gateway import DecisionContext
        from syncopate.runtime.tools import ToolRuntime
        from syncopate.runtime.worker import audit as w_audit, emit as w_emit

        run = f"run_{_uuid.uuid4().hex[:8]}"
        await create_run(db, org_id=org, run_id=run, user_message="x")

        async def _ob() -> bool:
            return False

        calls = [
            ("benchmark.get_safety_line",
             partial(impl.benchmark_get_safety_line, db, org),
             {"product_id": "P1", "region": "R1"}, "cpi_d7_max"),
            ("memory.search", partial(impl.memory_search, db, org),
             {"lane": "episodic"}, "records"),
            ("analysis.geo_breakdown", partial(impl.analysis_geo_breakdown, db),
             {"product_id": f"P_{org[-6:]}"}, "regions"),
        ]
        gate = ActionGate(db, ToolRuntime(db),
                          {name: ToolBinding(fn) for name, fn, _a, _m in calls},
                          org_id=org, run_id=run, over_budget=_ob,
                          emit=w_emit, audit=w_audit, max_steps=99)
        for name, _fn, args, must_have in calls:
            out = await gate.invoke(tool=name, arguments=args,
                                    ctx=DecisionContext(), param_source="model")
            assert out.status == "ok", f"{name} 在真实路径上没跑通：{out.error}"
            assert must_have in out.observation, f"{name} 少了 {must_have} ⇒ 判据量空了"
            observation_message(name, out.observation)   # 渲染不抛才算过
        return True

    async def main():
        db = Database()
        await db.connect(max_size=3)
        try:
            return await body(db)
        finally:
            await db.close()

    assert asyncio.run(main())


def test_status_filter_is_case_insensitive_because_the_spec_says_lowercase() -> None:
    """★ 工具 spec 写「按状态过滤，如 active」，平台却存 `ACTIVE`。

    2026-08-20 实测：模型照 spec 传小写 ⇒ 精确匹配把 6 个 campaign 全滤掉 ⇒
    它如实回答"无可执行 campaign"。**spec 是模型的契约，实现要满足 spec。**
    """
    p = _platform(3)
    assert _run(impl.campaign_list(p, account_id="A", status="active"))["count"] == 3
    assert _run(impl.campaign_list(p, account_id="A", status="ACTIVE"))["count"] == 3
