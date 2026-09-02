"""M9.2 · 三层幂等的验收测试。

★★★ 这一份必须是**实测重复投递**，不能是代码 review

幂等是整个 runtime 里唯一一个"错了就是真金白银"的东西：其余组件出问题最多是
服务不可用，**重复扣款是不可逆损失**。所以每一层都真的投两次，看第二次有没有
被挡住、以及**挡住之后返回的是不是原结果**。

⚠️ 需要一个跑着的 PG：`bash scripts/pg_bootstrap.sh`。没有就跳过 ——
但**不要把跳过当通过**：CI 上必须有 PG。
"""

from __future__ import annotations

import asyncio
import uuid

import pytest

from syncopate.runtime.db import (
    Database, claim_run, create_run, finish_run, record_tool_call,
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


def with_db(body):
    """★ 每条测试自带一个完整的事件循环 + 连接池。

    ⚠️ **asyncpg 的连接池绑定在创建它的事件循环上。** 第一版用 fixture 在一个
    `asyncio.run()` 里建池、测试在另一个里用，报的是
    `InterfaceError: another operation is in progress` —— 错误信息完全没提循环，
    很容易误诊成并发 bug。
    """
    async def main():
        db = Database()
        await db.connect(max_size=5)
        try:
            return await body(db)
        finally:
            await db.close()
    return asyncio.run(main())


def _org() -> str:
    return f"org_{uuid.uuid4().hex[:8]}"


# --------------------------------------------------------------------------
# 第一层 · 请求级：用户点两次
# --------------------------------------------------------------------------


def test_same_idempotency_key_returns_the_original_run() -> None:
    """★ 第二次提交必须**返回原来那次**，而不是新建、也不是报错。

    重复提交是正常现象（用户手抖、网络重发），不是错误。
    """
    org, key = _org(), "user-click-1"

    async def go(db):
        a = await create_run(db, org_id=org, run_id="run_a", user_message="加预算",
                             idempotency_key=key)
        b = await create_run(db, org_id=org, run_id="run_b", user_message="加预算",
                             idempotency_key=key)
        return a, b

    a, b = with_db(go)
    assert a.created is True
    assert b.created is False, "第二次居然新建了一条 run —— 请求级幂等没生效"
    assert b.run_id == a.run_id == "run_a", "返回的不是原来那次"


def test_different_orgs_may_share_a_key() -> None:
    """★ 多租户隔离：幂等键的作用域是 org，不是全局。

    做成全局唯一的话，A 公司用过的 key 会把 B 公司的请求挡掉 —— 那是数据泄漏级的 bug。
    """
    key = "same-key"

    async def go(db):
        a = await create_run(db, org_id=_org(), run_id="r1", user_message="x",
                             idempotency_key=key)
        b = await create_run(db, org_id=_org(), run_id="r2", user_message="x",
                             idempotency_key=key)
        return a, b

    a, b = with_db(go)
    assert a.created and b.created


def test_runs_without_a_key_never_collide() -> None:
    """不带 Idempotency-Key 的请求之间不该互相挡 —— 靠 PARTIAL INDEX 实现。"""
    org = _org()

    async def go(db):
        return [await create_run(db, org_id=org, run_id=f"r{i}", user_message="x")
                for i in range(3)]

    assert all(h.created for h in with_db(go))


# --------------------------------------------------------------------------
# 第二层 · 任务级：队列重投
# --------------------------------------------------------------------------


def test_two_workers_cannot_claim_the_same_run() -> None:
    """★★ 同一条 run 不可能被两个 worker 同时抢到（FOR UPDATE SKIP LOCKED）。

    ⚠️ 队列是**全局**的（一个 worker 池服务所有 org，这是对的），所以比赛之前
    要先把前面测试遗留的 run 排空 —— 否则 4 个 worker 抢到的是别人的活，
    这条测试会变成"看起来失败其实是没隔离"。
    """
    org = _org()

    async def go(db):
        while await claim_run(db, worker_id="drain", lease_seconds=3600):
            pass                                   # 排空队列
        await create_run(db, org_id=org, run_id="solo", user_message="x")
        got = await asyncio.gather(*[claim_run(db, worker_id=f"w{i}") for i in range(4)])
        return [g for g in got if g], [g for g in got if g and g["org_id"] == org]

    all_claimed, mine = with_db(go)
    assert len(mine) == 1, f"我的 run 被抢到 {len(mine)} 次 —— 会重复执行"
    assert len(all_claimed) == 1, f"队列里只有 1 条，却被抢走 {len(all_claimed)} 条"


def test_expired_lease_can_be_reclaimed() -> None:
    """★ worker 崩了不能让任务永远卡住：lease 过期后可被重抢，attempt 递增。"""
    org = _org()

    async def go(db):
        await create_run(db, org_id=org, run_id="crashy", user_message="x")
        first = await claim_run(db, worker_id="w1", lease_seconds=-1)   # 已过期
        second = await claim_run(db, worker_id="w2", lease_seconds=60)
        return first, second

    first, second = with_db(go)
    assert first and second
    assert second["attempts"] > first["attempts"], "重抢没有累加 attempts，重试次数不可见"


def test_finished_run_is_not_claimable() -> None:
    org = _org()

    async def go(db):
        await create_run(db, org_id=org, run_id="done", user_message="x")
        await claim_run(db, worker_id="w1")
        await finish_run(db, org_id=org, run_id="done", status="succeeded", result={"ok": 1})
        again = await claim_run(db, worker_id="w2")
        return again

    again = with_db(go)
    assert again is None or again["org_id"] != org


# --------------------------------------------------------------------------
# 第三层 · 工具级 —— ★ 唯一被外部系统认的那层
# --------------------------------------------------------------------------


def test_same_external_key_does_not_execute_twice() -> None:
    """★★★ 这条是 M9 里最贵的一条：**重试一次就是多花一次钱**。

    第二次调用必须 **① 不真的执行 ② 返回第一次的结果**。
    """
    org, key = _org(), "budget-change-42"
    calls: list[int] = []

    async def execute():
        calls.append(1)
        return True, {"new_budget": 900}, None

    async def go(db):
        await create_run(db, org_id=org, run_id="r", user_message="x")
        a = await record_tool_call(db, org_id=org, run_id="r", step=1,
                                   tool="campaign.update_budget", arguments={"b": 900},
                                   external_idempotency_key=key, execute=execute)
        b = await record_tool_call(db, org_id=org, run_id="r", step=2,
                                   tool="campaign.update_budget", arguments={"b": 900},
                                   external_idempotency_key=key, execute=execute)
        return a, b

    a, b = with_db(go)
    assert len(calls) == 1, f"外部动作执行了 {len(calls)} 次 —— 这就是重复扣款"
    assert b.replayed is True
    assert b.data == a.data, "幂等命中后返回的不是原结果"


def test_idempotency_spans_runs() -> None:
    """★ 跨 run 重试同一个动作**同样是重复扣款** —— 所以唯一索引不按 run_id 分组。"""
    org, key = _org(), "cross-run-key"
    calls: list[int] = []

    async def execute():
        calls.append(1)
        return True, {"v": 1}, None

    async def go(db):
        for rid in ("run_1", "run_2"):
            await create_run(db, org_id=org, run_id=rid, user_message="x")
            await record_tool_call(db, org_id=org, run_id=rid, step=1, tool="t",
                                   arguments={}, external_idempotency_key=key,
                                   execute=execute)

    with_db(go)
    assert len(calls) == 1, "换个 run 就能再扣一次钱"


def test_calls_without_external_key_are_not_deduped() -> None:
    """没有幂等键的读工具不该被去重 —— 查两次天经地义。"""
    org = _org()
    calls: list[int] = []

    async def execute():
        calls.append(1)
        return True, {"v": len(calls)}, None

    async def go(db):
        await create_run(db, org_id=org, run_id="r", user_message="x")
        for i in range(3):
            await record_tool_call(db, org_id=org, run_id="r", step=i, tool="campaign.get_metrics",
                                   arguments={}, external_idempotency_key=None, execute=execute)

    with_db(go)
    assert len(calls) == 3


def test_replay_leaves_an_audit_trail() -> None:
    """★ 被幂等挡下的那次也要留痕 —— 否则"到底重试了几次"事后完全不可见。"""
    org, key = _org(), "trail-key"

    async def execute():
        return True, {"v": 1}, None

    async def go(db):
        await create_run(db, org_id=org, run_id="r", user_message="x")
        for step in (1, 2, 3):
            await record_tool_call(db, org_id=org, run_id="r", step=step, tool="t",
                                   arguments={}, external_idempotency_key=key, execute=execute)
        async with db.tx() as conn:
            return await conn.fetchval(
                "SELECT count(*) FROM tool_calls WHERE org_id=$1 AND replayed_from IS NOT NULL",
                org)

    assert with_db(go) == 2, "重放没留痕，重试次数不可观测"
