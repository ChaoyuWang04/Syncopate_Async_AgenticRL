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
