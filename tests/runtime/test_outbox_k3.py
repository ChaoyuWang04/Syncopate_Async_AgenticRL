"""K3 门槛（27 §5）的进程内部分：不丢（Outbox 同事务 + 先 publish 后标记，含负向认证）· 不争（定向
claim 只认 queued，两次只成一次）· 退避 cap 与死信 · 错误分类（transient 回队列/permanent 立即 failed）
· 心跳续租与丢失判定 · 三个 attempts 分账 · 积压指标。真 Celery/Redis 的进程级验收在 test_celery_integration。"""
from __future__ import annotations

import asyncio
import uuid

import httpx
import pytest

from syncopate.runtime.db import (Database, claim_run, create_run, fetch_due_outbox,
                                  mark_outbox_retry, renew_lease)
from syncopate.runtime.dispatcher import Dispatcher
from syncopate.runtime.metrics import OLDEST_JOB_AGE_ALERT_S, queue_backlog
from syncopate.runtime.platform import FakeAdPlatform
from syncopate.runtime.worker import (MAX_RUN_ATTEMPTS, LeaseHeartbeat, Worker, WorkerConfig,
                                      classify_error)
from tests.runtime.test_api import _pg_available

pytestmark = pytest.mark.skipif(not _pg_available(), reason="需要 PostgreSQL：bash scripts/pg_bootstrap.sh")


def with_db(body):
    async def main():
        db = Database()
        await db.connect(max_size=6)
        try:
            return await body(db)
        finally:
            await db.close()
    return asyncio.run(main())


def _org() -> str:
    return f"org_{uuid.uuid4().hex[:8]}"


def _run(tag: str) -> str:
    # 库跨会话持久：固定 run_id 会撞上上次留下的同名行（不同 org），读到旧行（09-02 实测）
    return f"{tag}_{uuid.uuid4().hex[:6]}"


async def _outbox_rows(db, org, run_id):
    async with db.tx() as conn:
        return [dict(r) for r in await conn.fetch(
            "SELECT id, status, attempts, next_attempt_at, queue FROM outbox_jobs "
            "WHERE org_id=$1 AND payload->>'run_id'=$2 ORDER BY id", org, run_id)]


async def _kinds(db, org, run_id):
    async with db.tx() as conn:
        return [r["kind"] for r in await conn.fetch(
            "SELECT kind FROM run_events WHERE org_id=$1 AND run_id=$2 ORDER BY seq", org, run_id)]


# --------------------------------------------------------------------------
# 门槛①：不丢
# --------------------------------------------------------------------------


def test_create_run_writes_outbox_in_the_same_transaction() -> None:
    async def go(db):
        org, run = _org(), _run("run_ob1")
        await create_run(db, org_id=org, run_id=run, user_message="x")
        rows = await _outbox_rows(db, org, run)
        return rows

    rows = with_db(go)
    assert len(rows) == 1 and rows[0]["status"] == "pending" and rows[0]["queue"] == "interactive"


def test_publish_first_then_mark_survives_a_crash_between_them() -> None:
    """dispatcher 在 publish 之后、标记之前死掉 ⇒ 下轮重投（无害），最终恰好标记一次、run.enqueued 恰好一条。"""
    async def go(db):
        org, run = _org(), _run("run_ob2")
        await create_run(db, org_id=org, run_id=run, user_message="x")
        published: list[int] = []
        crash_once = {"armed": True}

        async def publish(job):
            published.append(job["id"])
            if crash_once["armed"]:
                crash_once["armed"] = False
                raise RuntimeError("simulated crash after publish, before mark")

        d = Dispatcher(db, publish, org_id=org)
        await d.dispatch_once()                       # 投了、没记 ⇒ 仍 pending（退避 1s）
        rows1 = await _outbox_rows(db, org, run)
        await asyncio.sleep(1.1)
        await d.dispatch_once()                       # 重投 + 标记
        rows2 = await _outbox_rows(db, org, run)
        return published, rows1, rows2, await _kinds(db, org, run)

    published, rows1, rows2, kinds = with_db(go)
    assert len(published) == 2, "第二轮没有重投"
    assert rows1[0]["status"] == "pending" and rows1[0]["attempts"] == 1
    assert rows2[0]["status"] == "dispatched"
    assert kinds.count("run.enqueued") == 1, "run.enqueued 必须恰好一条（标记是条件更新）"


def test_negative_mark_first_loses_the_job_when_publish_fails() -> None:
    """负向认证（门槛④a）：把顺序反过来 ⇒ 「记了没投」= 任务永久消失，判据必红。"""
    async def go(db):
        org, run = _org(), _run("run_ob3")
        await create_run(db, org_id=org, run_id=run, user_message="x")

        async def publish(_job):
            raise RuntimeError("broker down")

        d = Dispatcher(db, publish, org_id=org, _unsafe_mark_first=True)
        with pytest.raises(RuntimeError):
            await d.dispatch_once()
        rows = await _outbox_rows(db, org, run)
        return rows

    rows = with_db(go)
    # 行已 dispatched（假的），pending=0 ⇒ 没有任何东西会再投它：这就是"任务消失"
    assert rows[0]["status"] == "dispatched" and rows[0]["attempts"] == 0


def test_backoff_is_capped_and_exhaustion_goes_to_dead_letter() -> None:
    async def go(db):
        org, run = _org(), _run("run_ob4")
        await create_run(db, org_id=org, run_id=run, user_message="x")
        job = (await _outbox_rows(db, org, run))[0]["id"]
        outcomes = []
        delays = []
        for _ in range(10):
            outcomes.append(await mark_outbox_retry(db, job_id=job, error="broker down", cap_seconds=300))
            async with db.tx() as conn:
                r = await conn.fetchrow("SELECT extract(epoch FROM (next_attempt_at-now())) AS d, status, attempts "
                                        "FROM outbox_jobs WHERE id=$1", job)
            delays.append((float(r["d"]), r["status"], r["attempts"]))
        async with db.tx() as conn:
            dl = await conn.fetch("SELECT source, original_job_id, payload, attempts, error FROM dead_letter_jobs "
                                  "WHERE org_id=$1", org)
        return outcomes, delays, [dict(x) for x in dl]

    outcomes, delays, dl = with_db(go)
    assert outcomes[:9] == ["retry"] * 9 and outcomes[9] == "dead"
    assert all(d <= 300.5 for d, _, _ in delays), f"退避超过 cap：{delays}"
    assert delays[8][0] > 200, "第 9 次退避应已顶到 cap 附近（2^8=256）"
    assert delays[9][1] == "failed"
    assert len(dl) == 1 and dl[0]["source"] == "outbox" and dl[0]["payload"]["run_id"].startswith("run_ob4_")
    assert dl[0]["attempts"] == 10 and dl[0]["error"]["error"] == "broker down"


# --------------------------------------------------------------------------
# 门槛②：不争 —— 定向 claim 只认 queued；同一 job 投两次恰一个成功
# --------------------------------------------------------------------------


def test_targeted_claim_exactly_one_winner_and_only_queued() -> None:
    async def go(db):
        org, run = _org(), _run("run_cl1")
        await create_run(db, org_id=org, run_id=run, user_message="x")
        a, b = await asyncio.gather(
            claim_run(db, worker_id="w1", org_id=org, run_id=run),
            claim_run(db, worker_id="w2", org_id=org, run_id=run))
        winners = [x for x in (a, b) if x]
        # 让 lease 过期：定向 claim 仍不许接管（回收归 sweeper，S-01）
        async with db.tx() as conn:
            await conn.execute("UPDATE agent_runs SET lease_expires_at = now() - interval '1 hour' "
                               "WHERE org_id=$1 AND run_id=$2", org, run)
        stale = await claim_run(db, worker_id="w3", org_id=org, run_id=run)
        async with db.tx() as conn:
            attempts = await conn.fetchval("SELECT attempts FROM agent_runs WHERE org_id=$1 AND run_id=$2", org, run)
        return len(winners), stale, attempts

    n, stale, attempts = with_db(go)
    assert n == 1, "两个 worker 都抢到了同一条 run"
    assert stale is None, "定向 claim 接管了过期 lease 的 running run —— 和 sweeper 成了两个写入者"
    assert attempts == 1, "attempts 应只 +1（执行次数）"


# --------------------------------------------------------------------------
# 门槛⑤：错误分类 —— transient 回队列 + outbox 延迟；permanent 立即 failed；超限 failed
# --------------------------------------------------------------------------


class _RaisingDecider:
    def __init__(self, exc):
        self.exc = exc

    async def decide(self, **_):
        raise self.exc


def _worker(db, decider):
    return Worker(db, FakeAdPlatform(), WorkerConfig(org_id=None, amount_threshold=10 ** 9,
                                                       daily_cost_cap_micros=10 ** 12), decider=decider)


def test_transient_error_reschedules_with_backoff_then_fails_after_max_attempts() -> None:
    async def go(db):
        org, run = _org(), _run("run_tr1")
        await create_run(db, org_id=org, run_id=run, user_message="x")
        w = _worker(db, _RaisingDecider(httpx.ConnectError("vllm down")))
        trail = []
        for i in range(MAX_RUN_ATTEMPTS):
            claimed = await claim_run(db, worker_id="w", org_id=org, run_id=run)
            assert claimed, f"第 {i + 1} 次没抢到（上一轮没回 queued？）"
            await w.execute_claimed(claimed)
            async with db.tx() as conn:
                st = await conn.fetchrow("SELECT status, attempts FROM agent_runs WHERE org_id=$1 AND run_id=$2", org, run)
            trail.append((st["status"], st["attempts"]))
        return trail, await _outbox_rows(db, org, run), await _kinds(db, org, run)

    trail, rows, kinds = with_db(go)
    assert trail == [("queued", 1), ("queued", 2), ("failed", 3)], trail
    # 每次 transient 重投一行 outbox，延迟递增（1min / 5min），最后一次不再投
    delays = [r for r in rows[1:]]
    assert len(rows) == 3, f"outbox 行数 {len(rows)}（创建 1 + 重投 2）"
    assert kinds.count("run.retry_scheduled") == 2 and kinds[-1] == "run.failed"


def test_permanent_error_fails_immediately_without_retry() -> None:
    async def go(db):
        org, run = _org(), _run("run_pm1")
        await create_run(db, org_id=org, run_id=run, user_message="x")
        w = _worker(db, _RaisingDecider(ValueError("schema 非法")))
        claimed = await claim_run(db, worker_id="w", org_id=org, run_id=run)
        await w.execute_claimed(claimed)
        async with db.tx() as conn:
            st = await conn.fetchrow("SELECT status, attempts FROM agent_runs WHERE org_id=$1 AND run_id=$2", org, run)
        return (st["status"], st["attempts"]), await _outbox_rows(db, org, run)

    (status, attempts), rows = with_db(go)
    assert (status, attempts) == ("failed", 1)
    assert len(rows) == 1, "permanent 错误不该再投"


def test_classify_error_table() -> None:
    from syncopate.runtime.platform import PlatformError
    assert classify_error(httpx.ConnectError("x")) == "transient"
    assert classify_error(ValueError("x")) == "permanent"
    assert classify_error(KeyError("x")) == "permanent"
    assert classify_error(PlatformError("rate limited", code="429", retriable=True)) == "transient"
    assert classify_error(PlatformError("invalid param", code="100", retriable=False)) == "permanent"


# --------------------------------------------------------------------------
# 门槛⑥：心跳 —— 续租推进 lease；收走 lease 后心跳判 LOST（K8 前先断言"停止推进"）
# --------------------------------------------------------------------------


def test_heartbeat_renews_and_detects_loss(capsys) -> None:
    async def go(db):
        org, run = _org(), _run("run_hb1")
        await create_run(db, org_id=org, run_id=run, user_message="x")
        claimed = await claim_run(db, worker_id="me", org_id=org, run_id=run, lease_seconds=2)
        assert claimed

        async def lease():
            async with db.tx() as conn:
                return await conn.fetchval("SELECT lease_expires_at FROM agent_runs WHERE org_id=$1 AND run_id=$2", org, run)

        hb = LeaseHeartbeat(db, org_id=org, run_id=run, worker_id="me", ttl_seconds=2, interval_seconds=0.2)
        t0 = await lease()
        hb.start()
        await asyncio.sleep(0.7)
        t1 = await lease()
        # 别人收走 lease（sweeper 会这么做）⇒ 续租 0 行 ⇒ lost
        async with db.tx() as conn:
            await conn.execute("UPDATE agent_runs SET lease_owner='sweeper' WHERE org_id=$1 AND run_id=$2", org, run)
        await asyncio.sleep(0.5)
        lost, renewals = hb.lost, hb.renewals
        await hb.stop()
        t2 = await lease()
        await asyncio.sleep(0.5)
        t3 = await lease()
        return t0, t1, lost, renewals, t2, t3

    t0, t1, lost, renewals, t2, t3 = with_db(go)
    assert t1 > t0 and renewals >= 2, "续租没有推进 lease_expires_at"
    assert lost, "lease 被收走后心跳没有判 LOST"
    assert t3 == t2, "停掉续租后 lease_expires_at 仍在推进"
    out = capsys.readouterr().out
    assert "[lease-heartbeat]" in out and "LOST" in out, "判据行没打"


def test_renew_lease_is_owner_scoped() -> None:
    async def go(db):
        org, run = _org(), _run("run_hb2")
        await create_run(db, org_id=org, run_id=run, user_message="x")
        await claim_run(db, worker_id="a", org_id=org, run_id=run)
        return (await renew_lease(db, org_id=org, run_id=run, worker_id="b", lease_seconds=60),
                await renew_lease(db, org_id=org, run_id=run, worker_id="a", lease_seconds=60))

    assert with_db(go) == (False, True)


# --------------------------------------------------------------------------
# 门槛⑫：三个 attempts 分账 · 门槛⑧：积压指标
# --------------------------------------------------------------------------


def test_outbox_attempts_do_not_touch_run_attempts() -> None:
    async def go(db):
        org, run = _org(), _run("run_at1")
        await create_run(db, org_id=org, run_id=run, user_message="x")
        job = (await _outbox_rows(db, org, run))[0]["id"]
        for _ in range(3):
            await mark_outbox_retry(db, job_id=job, error="x")
        async with db.tx() as conn:
            return (await conn.fetchval("SELECT attempts FROM outbox_jobs WHERE id=$1", job),
                    await conn.fetchval("SELECT attempts FROM agent_runs WHERE org_id=$1 AND run_id=$2", org, run))

    assert with_db(go) == (3, 0)


def test_queue_backlog_reports_oldest_queued_age_and_alerts() -> None:
    async def go(db):
        org = _org()
        for i in range(5):
            await create_run(db, org_id=org, run_id=f"run_bk{i}", user_message="x")
        async with db.tx() as conn:
            await conn.execute("UPDATE agent_runs SET created_at = now() - interval '90 seconds' "
                               "WHERE org_id=$1 AND run_id='run_bk0'", org)
        return await queue_backlog(db, org_id=org)

    m = with_db(go)
    assert m["queued_runs"] == 5 and m["outbox_pending"] == 5
    assert 81 <= m["oldest_queued_run_age_s"] <= 99, m       # ±10%
    assert m["alert"] is True and OLDEST_JOB_AGE_ALERT_S == 60.0
