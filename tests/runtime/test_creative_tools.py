"""素材库（B-2 第三批）：`upload → poll_review` 这条**异步链**。

★★ 它是 B-1b 那套异步任务机制的第一个真实使用者：

    上传只把素材放进审核队列，**不返回审核结论**
    审核结果由模型**自己决定何时去查**，而每次查都扣 BUC 积分

⇒ `07 §P1-2` 原话：原来的阻塞实现「**把"什么时候该查"这个决策从模型手里拿走了**」。

⚠️ 另有四个「只做一件事」的边界，合并任意两个都会让"该用哪个工具"这个判断消失。
"""

from __future__ import annotations

import asyncio

import pytest

from syncopate.runtime import tool_impls as impl
from syncopate.runtime.platform import READ_POINTS, FakeAdPlatform


class _Clock:
    def __init__(self) -> None:
        self.now = 1_000.0

    def __call__(self) -> float:
        return self.now

    def tick(self, s: float) -> None:
        self.now += s


def _run(coro):
    return asyncio.run(coro)


def _platform() -> tuple[FakeAdPlatform, _Clock]:
    c = _Clock()
    return FakeAdPlatform(clock=c), c


# ── 异步链 ──────────────────────────────────────────────────────────────

def test_upload_does_not_claim_the_review_passed():
    """★★★ **上传成功 ≠ 审核通过**（沙盒描述原话）。

    ⇒ 返回里**不许**有 approved / review_result 这类字段：
      给了就等于替模型断言"审核过了"，而那时候审核根本还没开始。
    """
    p, _ = _platform()
    out = _run(impl.creative_upload(p, campaign_id="C1", creative_name="A",
                                    asset_type="video"))
    assert out["status"] == "in_review"
    forbidden = {"approved", "review_result", "review", "passed"}
    assert not (forbidden & set(out)), f"上传就断言了审核结果：{out}"


def test_poll_returns_immediately_and_says_how_much_longer():
    """★ 立刻返回，且 pending 时**告诉模型还差多久**。

    不给这个数，模型只能瞎猜什么时候再查 —— 而每次查都扣积分，猜错的代价是真的。
    """
    p, clock = _platform()
    up = _run(impl.creative_upload(p, campaign_id="C1", creative_name="A",
                                   asset_type="video"))
    first = _run(impl.creative_poll_review(p, asset_id=up["asset_id"]))
    assert first["review_status"] == "pending"
    assert first["seconds_remaining"] > 0

    clock.tick(481)
    done = _run(impl.creative_poll_review(p, asset_id=up["asset_id"]))
    assert done["review_status"] == "approved"


def test_polling_in_a_tight_loop_burns_quota():
    """★★ 每次查都扣积分 ⇒ **狂查会自然把配额烧掉**。

    代价内建，不靠"不许频繁轮询"这种规训 ——
    规训要靠模型记得遵守，扣费不用。
    """
    p, _ = _platform()
    up = _run(impl.creative_upload(p, campaign_id="C1", creative_name="A",
                                   asset_type="video"))
    before = p._buc.spent
    for _ in range(5):
        _run(impl.creative_poll_review(p, asset_id=up["asset_id"]))
    assert p._buc.spent == before + 5 * READ_POINTS


def test_unknown_asset_reports_not_found_rather_than_pending():
    """★ 查一个不存在的素材 ⇒ 报"没有"。

    ⚠️ **绝不能返回 pending** —— 那会让模型永远等一个不存在的审核。
    """
    p, _ = _platform()
    out = _run(impl.creative_poll_review(p, asset_id="AST_9999"))
    assert out["found"] is False
    assert out.get("review_status") != "pending"


def test_upload_is_idempotent_under_replay():
    """带同一个幂等键重放 ⇒ 同一个 asset_id，**不会传出第二条素材**。"""
    p, _ = _platform()
    key = "org:run:creative.upload:abc"
    a = _run(impl.creative_upload(p, campaign_id="C1", creative_name="A",
                                  asset_type="video", idempotency_key=key))
    b = _run(impl.creative_upload(p, campaign_id="C1", creative_name="A",
                                  asset_type="video", idempotency_key=key))
    assert a["asset_id"] == b["asset_id"]
    assert len(p.assets) == 1, "★ 重放又传了一条素材"


# ── 四个「只做一件事」的边界 ────────────────────────────────────────────

def test_asset_tags_does_not_return_campaign_level_data():
    """「只给**单条**素材的标签和历史表现，**不返回** campaign 层数据」。"""
    p, _ = _platform()
    up = _run(impl.creative_upload(p, campaign_id="C1", creative_name="A",
                                   asset_type="video"))
    out = _run(impl.creative_get_asset_tags(p, creative_id=up["asset_id"]))
    assert out["found"] is True
    assert "campaign_id" not in out and "daily_budget" not in out


def test_metrics_by_asset_does_not_return_the_campaign_rollup():
    """「不返回 campaign 层的汇总（那在 `campaign.get_metrics`）」。"""
    p, _ = _platform()
    _run(impl.creative_upload(p, campaign_id="C1", creative_name="A", asset_type="video"))
    out = _run(impl.creative_get_metrics_by_asset(p, campaign_id="C1"))
    assert "assets" in out
    assert not ({"total", "summary", "campaign_ctr"} & set(out))


def test_search_similar_does_not_judge_fitness():
    """「只按标签检索现有素材，**不生成**新素材，也**不判断**适不适合当前 campaign」。

    ⇒ 返回里不许有 recommended / suitable 这类字段 ——
      那是模型的判断，替它做了就等于把这个能力训没了。
    """
    p, _ = _platform()
    up = _run(impl.creative_upload(p, campaign_id="C1", creative_name="A",
                                   asset_type="video"))
    p.assets[up["asset_id"]]["tags"] = {"themes": ["real_person"]}
    p.assets[up["asset_id"]]["metrics"] = {"ipm": 12.0}
    out = _run(impl.creative_search_similar(p, visual_tags=["real_person"]))
    assert len(out["assets"]) == 1
    for row in out["assets"]:
        assert not ({"recommended", "suitable", "fit_score"} & set(row))


def test_search_similar_sorts_by_ipm_desc():
    p, _ = _platform()
    for name, ipm in (("low", 3.0), ("high", 20.0), ("mid", 9.0)):
        up = _run(impl.creative_upload(p, campaign_id="C1", creative_name=name,
                                       asset_type="video"))
        p.assets[up["asset_id"]]["tags"] = {"themes": ["t"]}
        p.assets[up["asset_id"]]["metrics"] = {"ipm": ipm}
    out = _run(impl.creative_search_similar(p, visual_tags=["t"]))
    assert [a["creative_name"] for a in out["assets"]] == ["high", "mid", "low"]


def test_min_ipm_filters_out_unknown_ipm():
    """★ IPM 未知的素材，在设了 `min_ipm` 时**必须被滤掉**。

    把"未知"当成"满足条件"，是在拿没有数据的素材冒充达标的 ——
    同「查不到 ≠ 没有限制」那条三态。
    """
    p, _ = _platform()
    up = _run(impl.creative_upload(p, campaign_id="C1", creative_name="unknown",
                                   asset_type="video"))
    p.assets[up["asset_id"]]["tags"] = {"themes": ["t"]}       # metrics 为空
    out = _run(impl.creative_search_similar(p, visual_tags=["t"], min_ipm=5.0))
    assert out["assets"] == []


# ── system.wait：两侧行为刻意不同，而且这个不同必须让模型看得见 ──────────

def test_wait_is_capped_by_the_lease_not_by_the_spec():
    """★★★ spec 说单次上限 600 秒，而 worker 的租约默认只有 60 秒。

    照沙盒那样直接睡 600 秒 ⇒ **租约过期** ⇒ 另一个 worker 抢走这条 run
    ⇒ **重复执行**。沙盒里没有租约这回事，所以它可以随便睡 ——
    这正是「训练侧的实现不能套层薄膜就上生产」的一个具体例子。
    """
    slept = []

    async def _sleep(s):
        slept.append(s)

    out = _run(impl.system_wait(_sleep, 60, seconds=600))
    assert out["waited_seconds"] <= 30, "★ 睡过了租约的安全线"
    assert slept and slept[0] <= 30


def test_wait_says_out_loud_that_it_did_not_wait_long_enough():
    """★★ 没等够必须**明说**，别假装等够了。

    假装的话，模型会以为审核该出结果了，然后把一个 pending **当成"审核失败"**。
    """
    async def _sleep(s):
        ...

    short = _run(impl.system_wait(_sleep, 60, seconds=600))
    enough = _run(impl.system_wait(_sleep, 60, seconds=5))
    assert short["truncated_by_lease"] is True
    assert enough["truncated_by_lease"] is False
    assert enough["waited_seconds"] == 5
