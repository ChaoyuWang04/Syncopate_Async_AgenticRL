"""写工具（B-2 第四批）：两条**跨工具前置条件** + 一个乘错基数的并发问题。

★★★ 这一批的核心不是"能不能写成功"，是**沙盒用扣分教的约束，runtime 用硬闸兑现**：

    campaign.create        「本轮如果还没有一次成功的 approval.create_case，**不要调用本工具**」
    campaign.scale_budget  「幅度 ±20% 以内可直接执行；**超出必须先走 approval.create_case**」

⚠️⚠️ 为什么不能只靠模型记得：`campaign.create` 是**不可逆**的 ——
   建出来就在花钱、删不掉。**「模型多数时候会遵守」对不可逆动作是不够的。**
⇒ 这正是「沙盒是 runtime 的子集，契约由 runtime 定义」的正面兑现。
"""

from __future__ import annotations

import asyncio
import uuid

import pytest

from syncopate.runtime import tool_impls as impl
from syncopate.runtime.db import Database, create_run
from syncopate.runtime.platform import FakeAdPlatform, PlatformError


def _pg() -> bool:
    async def probe() -> bool:
        db = Database()
        try:
            await db.connect(max_size=2); await db.close(); return True
        except Exception:
            return False
    return asyncio.run(probe())


pytestmark = pytest.mark.skipif(not _pg(), reason="需要 PostgreSQL：bash scripts/serving/pg_bootstrap.sh")


def with_db(body):
    async def main():
        db = Database()
        await db.connect(max_size=5)
        try:
            return await body(db)
        finally:
            await db.close()
    return asyncio.run(main())


def _ids() -> tuple[str, str]:
    return f"org_{uuid.uuid4().hex[:8]}", f"run_{uuid.uuid4().hex[:8]}"


async def _seed_run(db, org, run):
    await create_run(db, org_id=org, run_id=run, user_message="扩量")
    # K4：审批单只能在 running 的 run 上开（queued→waiting_for_user 不在白名单）⇒ 先 claim
    from syncopate.runtime.db import claim_run
    assert await claim_run(db, worker_id="t", org_id=org, run_id=run)


# ── 前置条件①：campaign.create 必须先有审批单 ───────────────────────────

def test_create_without_a_prior_approval_is_refused():
    """★★★ 不可逆动作 ⇒ **硬拒**，不进平台。

    ⚠️ 判据不只是"抛了异常"，还要**平台上一条 campaign 都没多** ——
      抛异常但已经建出来了，那这道闸等于没有。
    """
    async def body(db):
        org, run = _ids()
        await _seed_run(db, org, run)
        p = FakeAdPlatform()
        before = len(p.campaigns)
        with pytest.raises(impl.PreconditionNotMet, match="approval_required_first"):
            await impl.campaign_create(p, db, org, run, account_id="ACC",
                                       product_id="P1", region="US",
                                       daily_budget=100_000, client_request_id="r1")
        assert len(p.campaigns) == before, "★ 拒了，但 campaign 已经建出来了"
    with_db(body)


def test_create_is_allowed_after_an_approval_case_exists():
    async def body(db):
        org, run = _ids()
        await _seed_run(db, org, run)
        p = FakeAdPlatform()
        await impl.approval_create_case(db, org, run, campaign_id="-",
                                        change_type="campaign_create",
                                        requested_value=100_000, reason="地域扩展")
        out = await impl.campaign_create(p, db, org, run, account_id="ACC",
                                         product_id="P1", region="US",
                                         daily_budget=100_000, client_request_id="r1")
        assert out["status"] == "submitted"
    with_db(body)


def test_create_does_not_claim_it_is_already_running():
    """「返回**只表示提交成功，不代表已开始跑量**」（沙盒描述原话）。

    给 `running` 就等于替模型断言了一件平台还没做完的事。
    """
    async def body(db):
        org, run = _ids()
        await _seed_run(db, org, run)
        p = FakeAdPlatform()
        await impl.approval_create_case(db, org, run, campaign_id="-",
                                        change_type="c", requested_value=1, reason="r")
        out = await impl.campaign_create(p, db, org, run, account_id="A", product_id="P",
                                         region="US", daily_budget=1, client_request_id="r")
        assert out["status"] == "submitted" and out["status"] != "running"
    with_db(body)


# ── 前置条件②：scale_budget 的 ±20% 区间 ────────────────────────────────

def test_scaling_within_the_band_needs_no_approval():
    async def body(db):
        org, run = _ids()
        await _seed_run(db, org, run)
        p = FakeAdPlatform(budgets={"C1": 1000})
        out = await impl.campaign_scale_budget(p, db, org, run, campaign_id="C1",
                                               factor=1.2, reason="扩量",
                                               client_request_id="r1")
        assert out["new_budget"] == 1200
    with_db(body)


def test_scaling_beyond_the_band_is_refused_without_approval():
    """★ 超出 ±20% ⇒ 硬拒，**且预算一分没动**。"""
    async def body(db):
        org, run = _ids()
        await _seed_run(db, org, run)
        p = FakeAdPlatform(budgets={"C1": 1000})
        with pytest.raises(impl.PreconditionNotMet, match="approval_required_first"):
            await impl.campaign_scale_budget(p, db, org, run, campaign_id="C1",
                                             factor=1.5, reason="猛扩",
                                             client_request_id="r1")
        assert p.budgets["C1"] == 1000, "★ 拒了，但预算已经被改了"
    with_db(body)


def test_the_band_is_symmetric():
    """砍 50% 和提 50% 一样要审批 —— **缩量也是花钱决策**（少花也是决策）。"""
    async def body(db):
        org, run = _ids()
        await _seed_run(db, org, run)
        p = FakeAdPlatform(budgets={"C1": 1000})
        with pytest.raises(impl.PreconditionNotMet):
            await impl.campaign_scale_budget(p, db, org, run, campaign_id="C1",
                                             factor=0.5, reason="砍",
                                             client_request_id="r1")
    with_db(body)


def test_preconditions_are_not_retriable_platform_errors():
    """★ 前置条件用**自己的异常类型**，不复用 `PlatformError`。

    后者是"外部世界拒绝了"（可能可重试）；这个是"你还没做该做的那一步"，
    **重试永远没用**。类型混了，`ToolRuntime` 会去重试一件不可能成功的事。
    """
    assert not issubclass(impl.PreconditionNotMet, PlatformError)


# ── 并发：factor 是相对量，读与写之间基数会变 ───────────────────────────

def test_scaling_on_a_stale_base_is_rejected_not_silently_applied():
    """★★★ 这是 `scale_budget` 特有的问题，而且**不做校验不会报错，只会乘错基数**。

    场景：我读到 1000，打算提 20% → 1200。
         但在我算完之前，别人把它改成了 5000。
    ⇒ 不校验的话，我会在 5000 上提 20% = 6000 —— **悄悄多花了 4 倍**。
    ⇒ 所以把读到的值作为 `expected_current` 传回去；对不上就拒绝。
    """
    async def body(db):
        p = FakeAdPlatform(budgets={"C1": 1000})
        # 我读到的是 1000
        stale_base = p.budgets["C1"]
        # 别人改了
        await p.update_budget(campaign_id="C1", new_budget=5000)
        with pytest.raises(PlatformError) as exc:
            await p.scale_budget(campaign_id="C1", factor=1.2,
                                 expected_current=stale_base)
        assert exc.value.code == "409"
        assert exc.value.retriable is False, (
            "★ 标成可重试是错的：基数已经变了，重试只会拿同一个过期期望值再撞一次，"
            "正确应对是**重新读**")
        assert p.budgets["C1"] == 5000, "★ 冲突了却还是写进去了"
    with_db(body)


# ── approval.create_case 走的是同一条路 ─────────────────────────────────

def test_agent_opened_cases_land_in_the_same_table_as_gateway_ones():
    """★ runtime 侧 `approval.create_case` **就是** `open_approval_case`。

    另起一条路的话，「人在哪儿看这些单子」就会有两个答案 ——
    而审批单的全部价值就在于**有人真的会看到它**。
    """
    async def body(db):
        org, run = _ids()
        await _seed_run(db, org, run)
        out = await impl.approval_create_case(db, org, run, campaign_id="C1",
                                              change_type="budget_increase",
                                              requested_value=5000, reason="ROAS 达标")
        assert out["applied"] is False and out["status"] == "pending"
        async with db.tx() as conn:
            n = await conn.fetchval(
                "SELECT count(*) FROM approval_cases WHERE org_id=$1 AND run_id=$2",
                org, run)
        assert n == 1
    with_db(body)
