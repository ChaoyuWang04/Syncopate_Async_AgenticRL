"""假平台必须**如实建模真实 API 的坏脾气**，否则 runtime 的降级路径是假的。

★ 实查依据全在 `07 §2.1`（Meta Marketing API 官方文档）：

    M4  平台**没有幂等机制**
    M5  每个 ad set **每小时最多改 4 次**预算；超了报 `613` / 子码 `1487632`，
        并**封禁该 ad set 一小时**
    M6  限流是**积分制（BUC）**：读 1 分、写 3 分；标准档 9000 分、开发档 60 分；
        衰减 300 秒；**按广告账户共享额度**

★★ `07 §2.1` 的原话：**M4 + M5 合起来是最要命的组合** ——
   没有幂等键，而改动次数有硬上限
   ⇒ 一次超时后盲目重试，可能**同时**造成「多改一次预算」和「耗尽当小时配额」。

⇒ 假平台不建这两条的话，runtime 的重试策略在真实世界里**从来没被验过**。

⚠️ 时钟是注入的，不用真实 sleep 测窗口 ——
   **一个会偶发红的测试就是一把不可信的尺子**（这个项目记录在案的一条）。
"""

from __future__ import annotations

import asyncio

import pytest

from syncopate.runtime.platform import (
    BUC_DEVELOPMENT_TIER, BUC_WINDOW_SECONDS, BUDGET_CHANGES_PER_HOUR,
    META_TOO_MANY_CHANGES_CODE, META_TOO_MANY_CHANGES_SUBCODE,
    MAX_PAGE_SIZE, READ_POINTS, WRITE_POINTS, FakeAdPlatform, PlatformError,
)


class _Clock:
    """可控时钟。`tick(秒)` 往前推。"""

    def __init__(self) -> None:
        self.now = 1_000.0

    def __call__(self) -> float:
        return self.now

    def tick(self, seconds: float) -> None:
        self.now += seconds


def _run(coro):
    return asyncio.run(coro)


def _platform(**kw) -> tuple[FakeAdPlatform, _Clock]:
    clock = _Clock()
    return FakeAdPlatform(clock=clock, **kw), clock


# ── M6 · BUC 积分制 ──────────────────────────────────────────────────────

def test_reads_and_writes_cost_different_points():
    """读 1 分、写 3 分 —— 不是同一个价（实查 M6）。

    ⚠️ 如果两者同价，"多读几次没关系、多写几次很贵"这条真实约束就消失了，
       而那正是我们希望模型学会的取舍。
    """
    p, _ = _platform(buc_limit=BUC_DEVELOPMENT_TIER)
    assert READ_POINTS < WRITE_POINTS
    _run(p.get_metrics(campaign_id="C1"))
    assert p._buc.spent == READ_POINTS
    _run(p.update_budget(campaign_id="C1", new_budget=1))
    assert p._buc.spent == READ_POINTS + WRITE_POINTS


def test_exhausting_the_budget_raises_429_with_retry_after():
    """耗尽 ⇒ 429 + `retry_after`，而且**可重试** —— 它是"等一下再来"。"""
    p, _ = _platform(buc_limit=5)                 # 只够一次写(3) + 两次读(2)
    _run(p.update_budget(campaign_id="C1", new_budget=1))
    _run(p.get_metrics(campaign_id="C1"))
    _run(p.get_metrics(campaign_id="C1"))
    with pytest.raises(PlatformError) as exc:
        _run(p.get_metrics(campaign_id="C1"))
    assert exc.value.code == "429"
    assert exc.value.retriable is True
    assert exc.value.retry_after is not None and exc.value.retry_after > 0


def test_the_window_decays():
    """300 秒之后额度回来 —— 否则"等一下再来"就是空话。"""
    p, clock = _platform(buc_limit=1)
    _run(p.get_metrics(campaign_id="C1"))
    with pytest.raises(PlatformError):
        _run(p.get_metrics(campaign_id="C1"))
    clock.tick(BUC_WINDOW_SECONDS)
    _run(p.get_metrics(campaign_id="C1"))         # 不该再炸


def test_the_budget_is_shared_across_campaigns():
    """★ 额度**按账户共享**，不是按 campaign（实查 M6）。

    按 campaign 算的话，"多开几条 campaign 就能绕过限流"——真实世界里不成立。
    """
    p, _ = _platform(buc_limit=2)
    _run(p.get_metrics(campaign_id="C1"))
    _run(p.get_metrics(campaign_id="C2"))
    with pytest.raises(PlatformError):
        _run(p.get_metrics(campaign_id="C3"))     # 换一条 campaign 也没用


# ── M5 · 改动频次上限 ────────────────────────────────────────────────────

def test_five_budget_changes_in_an_hour_is_blocked():
    p, _ = _platform()
    for i in range(BUDGET_CHANGES_PER_HOUR):
        _run(p.update_budget(campaign_id="C1", new_budget=100 + i))
    with pytest.raises(PlatformError) as exc:
        _run(p.update_budget(campaign_id="C1", new_budget=999))
    assert exc.value.code == META_TOO_MANY_CHANGES_CODE
    assert exc.value.subcode == META_TOO_MANY_CHANGES_SUBCODE


def test_too_many_changes_is_NOT_retriable():
    """★★ 这条和限流长得像，但**性质相反**。

        限流       "等一下再来"        ⇒ retriable=True
        改动超限   "你已经被封了"      ⇒ retriable=False

    ⚠️ 把它当限流去重试，是**把一小时的封禁续成两小时**。
    而 `ToolRuntime` 只在平台**明确说可重试**时才重试 ——
    所以这个布尔值必须是对的，它直接决定线上会不会自伤。
    """
    p, _ = _platform()
    for i in range(BUDGET_CHANGES_PER_HOUR):
        _run(p.update_budget(campaign_id="C1", new_budget=100 + i))
    with pytest.raises(PlatformError) as exc:
        _run(p.update_budget(campaign_id="C1", new_budget=999))
    assert exc.value.retriable is False


def test_the_block_is_per_campaign_not_global():
    """封禁是**按 ad set**的（实查 M5）：一条被封，别的还能改。"""
    p, _ = _platform()
    for i in range(BUDGET_CHANGES_PER_HOUR + 1):
        try:
            _run(p.update_budget(campaign_id="C1", new_budget=100 + i))
        except PlatformError:
            pass
    _run(p.update_budget(campaign_id="C2", new_budget=1))      # 不该被牵连


def test_the_block_expires_after_an_hour():
    p, clock = _platform()
    for i in range(BUDGET_CHANGES_PER_HOUR + 1):
        try:
            _run(p.update_budget(campaign_id="C1", new_budget=100 + i))
        except PlatformError:
            pass
    clock.tick(3601)
    _run(p.update_budget(campaign_id="C1", new_budget=777))
    assert p.budgets["C1"] == 777


# ── ★★ M4 + M5 的组合：最要命的那一条 ────────────────────────────────────

def test_idempotent_replay_costs_neither_quota_nor_a_change_slot():
    """★★★ 幂等重放**不该**消耗配额，也不该算作"这一小时又改了一次"。

    `07 §2.1` 记的最要命组合就是这个：没有幂等机制 + 改动次数有硬上限
    ⇒ 一次超时后重试，可能**同时**多改一次预算、又耗掉一格额度。

    ⇒ 我们的假平台**建了**幂等（真 Meta 没有，见模块 docstring），
      那就必须把它建对：重放是"什么都没发生"，不是"又发生了一次"。
      放错顺序的代价很具体 —— **额度只有 4 次，一次重试就白吃掉一格。**
    """
    p, _ = _platform()
    key = "org:run:campaign.update_budget:abc"
    first = _run(p.update_budget(campaign_id="C1", new_budget=500, idempotency_key=key))
    spent_after_first = p._buc.spent
    slots_after_first = len(p._budget_changes["C1"])

    again = _run(p.update_budget(campaign_id="C1", new_budget=500, idempotency_key=key))

    assert again["deduped_by_platform"] is True
    assert again["new_budget"] == first["new_budget"]
    assert p._buc.spent == spent_after_first, "★ 重放又扣了一次 BUC 配额"
    assert len(p._budget_changes["C1"]) == slots_after_first, \
        "★ 重放吃掉了一格改动额度 —— 而一小时只有 4 格"


def test_retrying_without_a_key_does_burn_a_slot():
    """反面：**不带幂等键**的重试就是真的又改了一次 —— 额度照吃。

    这条不是 bug，是真实世界的样子（M4：平台没有幂等机制）。
    钉住它是为了让上面那条的价值看得见：**幂等键是唯一的保护**。
    """
    p, _ = _platform()
    _run(p.update_budget(campaign_id="C1", new_budget=500))
    _run(p.update_budget(campaign_id="C1", new_budget=500))
    assert len(p._budget_changes["C1"]) == 2


# ══════════════════════════════════════════════════════════════════════════
# B-1b · 分页 · 显式字段 · 异步任务
# ══════════════════════════════════════════════════════════════════════════

def _with_campaigns(n: int) -> tuple[FakeAdPlatform, _Clock]:
    p, clock = _platform()
    for i in range(n):
        p.campaigns[f"CMP_{i:03d}"] = {"name": f"C{i}", "daily_budget": 1000 + i,
                                       "status": "ACTIVE"}
    return p, clock


def test_fields_must_be_explicit():
    """★ `fields` 不给就报错，**不给一个"默认全给"**（实查 M3）。

    真实 API 不会替你猜。而"本地能跑、上线拿不到字段"是最难查的一类问题 ——
    因为它不报错，只是少了一个键。
    """
    p, _ = _with_campaigns(3)
    with pytest.raises(PlatformError) as exc:
        _run(p.list_campaigns(account_id="ACC_1", fields=[]))
    assert exc.value.code == "400"


def test_only_requested_fields_come_back():
    p, _ = _with_campaigns(2)
    page = _run(p.list_campaigns(account_id="ACC_1", fields=["id", "daily_budget"]))
    assert set(page["data"][0]) == {"id", "daily_budget"}


def test_unknown_field_is_an_error_not_a_silent_none():
    """要一个不存在的字段 ⇒ 报错。**不要静默返回 None** ——
    那会让"字段名写错了"和"这条真的没值"长得一模一样。"""
    p, _ = _with_campaigns(1)
    with pytest.raises(PlatformError):
        _run(p.list_campaigns(account_id="ACC_1", fields=["daily_budget_minor"]))


def test_asking_for_more_than_the_cap_silently_gives_you_the_cap():
    """★★ 要 1000 条只给 25 条，**而且不报错**（实查 P1-3）。

    这正是"以为拿到全部了"的来源：`len(data)` 是 25，账户里其实有 40 条。
    ⇒ 判据必须是**看 paging.has_next**，不是看 len(data)。
    """
    p, _ = _with_campaigns(40)
    page = _run(p.list_campaigns(account_id="ACC_1", fields=["id"], limit=1000))
    assert len(page["data"]) == MAX_PAGE_SIZE
    assert page["paging"]["has_next"] is True, "★ 不告诉你还有更多，就是在骗人"


def test_cursor_walks_the_whole_account():
    p, _ = _with_campaigns(40)
    seen, after = [], None
    while True:
        page = _run(p.list_campaigns(account_id="ACC_1", fields=["id"], after=after))
        seen += [r["id"] for r in page["data"]]
        if not page["paging"]["has_next"]:
            break
        after = page["paging"]["cursors"]["after"]
    assert len(seen) == 40 and len(set(seen)) == 40


def test_read_after_write_sees_the_new_value():
    """`07 §P0-1` 实测过的坑：改成 900，再读还是 500。"""
    p, _ = _with_campaigns(1)
    _run(p.update_budget(campaign_id="CMP_000", new_budget=900))
    page = _run(p.list_campaigns(account_id="ACC_1", fields=["id", "daily_budget"]))
    assert page["data"][0]["daily_budget"] == 900


# ── 异步任务 ────────────────────────────────────────────────────────────

def test_submit_returns_immediately_with_pending():
    """★★ 提交**立刻**返回 id + pending —— 不阻塞。

    `07 §P1-2` 原话：阻塞等待「把"什么时候该查"这个决策从模型手里拿走了」。
    """
    p, _ = _with_campaigns(1)
    out = _run(p.submit_budget_change(campaign_id="CMP_000", new_budget=900,
                                      settle_after=100.0))
    assert out["status"] == "pending" and out["change_id"]


def test_polling_before_it_settles_says_pending_not_a_lie():
    p, clock = _with_campaigns(1)
    out = _run(p.submit_budget_change(campaign_id="CMP_000", new_budget=900,
                                      settle_after=100.0))
    assert _run(p.get_job(job_id=out["change_id"]))["status"] == "pending"
    clock.tick(101)
    assert _run(p.get_job(job_id=out["change_id"]))["status"] == "succeeded"


def test_each_poll_costs_quota():
    """★★ 每次查都扣积分 ⇒ **死循环狂查会自然把配额烧掉**。

    代价内建，不靠"不许频繁轮询"这种规训 ——
    规训要靠模型记得遵守，扣费不用。
    """
    p, _ = _with_campaigns(1)
    out = _run(p.submit_budget_change(campaign_id="CMP_000", new_budget=900,
                                      settle_after=100.0))
    before = p._buc.spent
    _run(p.get_job(job_id=out["change_id"]))
    _run(p.get_job(job_id=out["change_id"]))
    assert p._buc.spent == before + 2 * READ_POINTS


def test_unknown_job_errors_rather_than_pretending_to_be_pending():
    """认不出 job_id ⇒ 报错。**不要返回一个"看起来在跑"的 pending** ——
    那会让 agent 永远等一个不存在的任务。"""
    p, _ = _with_campaigns(1)
    with pytest.raises(PlatformError):
        _run(p.get_job(job_id="job_budget_999"))


def test_validate_only_is_a_real_dry_run():
    """`validate_only` 只校验不生效（实查 M7）。

    ★ 有了它，"先验证再提交"才成为模型**可以学**的策略；没有它，面对不确定只能赌。
    """
    p, _ = _with_campaigns(1)
    out = _run(p.submit_budget_change(campaign_id="CMP_000", new_budget=900,
                                      validate_only=True))
    assert out["valid"] is True
    assert "CMP_000" not in p.budgets, "★ dry-run 改了世界，那它就不是 dry-run"
    assert p._budget_changes.get("CMP_000") is None, "★ dry-run 不该吃掉改动额度"


def test_invalid_parameter_is_not_retriable():
    """参数不合法 ⇒ **不可重试**：重试一百次还是不合法，只是白扣三次积分。"""
    p, _ = _with_campaigns(1)
    with pytest.raises(PlatformError) as exc:
        _run(p.submit_budget_change(campaign_id="CMP_000", new_budget=-1))
    assert exc.value.retriable is False
