"""K8 门槛（27 §10）：① 分支 A（模型调用前 kill ⇒ sweeper 回收 ⇒ 新 worker 续跑；留痕 requeued_by_sweeper +
run.restarted；只读工具不重调）· ③ 分支 C（写工具执行后记录前 kill ⇒ sweeper 判 response_lost ⇒ 对账按幂等键
回填 ⇒ run 跑完；平台侧副作用=1）· ④ 取消兑现（cancel_requested + worker 死 ⇒ cancelled 非 failed；顺序：取消先于
次数）· ⑤ 慢不当死（续租中的 run 3×TTL 零误回收）· ⑥ Replay 零副作用（结构）· ⑦ trace 八表聚合 · 僵尸 queued
告警不自动重投 + 人工 requeue · 账本持久化跨实例可见 · Repair 四样留痕。"""
from __future__ import annotations

import asyncio
import json
import uuid

import pytest

from syncopate.runtime.agent_loop import Proposal, save_transcript
from syncopate.runtime.api import create_app
from syncopate.runtime.db import Database, claim_run, create_run, request_cancel
from syncopate.runtime.platform import FakeAdPlatform, PlatformLedger
from syncopate.runtime.sweeper import reconcile_once, requeue_outbox, sweep_once
from syncopate.runtime.tools import derive_idempotency_key
from syncopate.runtime.worker import LeaseHeartbeat, Worker, WorkerConfig
from tests.runtime.test_api import ACME, Client, _pg_available

pytestmark = pytest.mark.skipif(not _pg_available(), reason="需要 PostgreSQL：bash scripts/serving/pg_bootstrap.sh")


def with_db(body):
    async def main():
        db = Database()
        await db.connect(max_size=6)
        try:
            return await body(db)
        finally:
            await db.close()
    return asyncio.run(main())


def _ids():
    return f"org_{uuid.uuid4().hex[:8]}", f"run_{uuid.uuid4().hex[:8]}"


class _Script:
    def __init__(self, *proposals):
        self._q = list(proposals)
        self.calls = 0

    async def decide(self, *, user_message, history):      # noqa: ANN001
        self.calls += 1
        return self._q.pop(0) if self._q else Proposal(kind="final", final_answer={"done": True})


READ = Proposal(kind="tool_call", tool="campaign.get_metrics", arguments={"campaign_id": "CMP_1"}, param_source="model")
WRITE = Proposal(kind="tool_call", tool="campaign.update_budget",
                 arguments={"campaign_id": "CMP_1", "new_budget": 120_000, "client_request_id": "req-K8"},
                 param_source="user")


def _worker(db, platform, script):
    return Worker(db, platform, WorkerConfig(amount_threshold=10 ** 9, daily_cost_cap_micros=10 ** 12), decider=script)


async def _state(db, org, run):
    async with db.tx() as conn:
        r = await conn.fetchrow("SELECT status, attempts FROM agent_runs WHERE org_id=$1 AND run_id=$2", org, run)
        kinds = [x["kind"] for x in await conn.fetch(
            "SELECT kind FROM run_events WHERE org_id=$1 AND run_id=$2 ORDER BY seq", org, run)]
        tcs = [dict(x) for x in await conn.fetch(
            "SELECT id, tool, status FROM tool_calls WHERE org_id=$1 AND run_id=$2 ORDER BY id", org, run)]
        ob = await conn.fetchval("SELECT count(*) FROM outbox_jobs WHERE org_id=$1 AND status='pending'", org)
    return r["status"], r["attempts"], kinds, tcs, ob


async def _approved_case(db, org, run):
    async with db.tx() as conn:
        await conn.execute(
            "INSERT INTO approval_cases (case_ref, run_id, org_id, action_type, proposed_params, status, reviewer_id, reviewed_at) "
            "VALUES ($1,$2,$3,$4,$5,'approved','t',now())", f"apr_{uuid.uuid4().hex[:8]}", run, org, WRITE.tool, WRITE.arguments)


# --------------------------------------------------------------------------
# 门槛①：分支 A —— 模型调用前 kill
# --------------------------------------------------------------------------


def test_branch_a_dead_before_model_call_is_swept_and_continued_without_recalling_reads() -> None:
    async def go(db):
        org, run = _ids()
        await create_run(db, org_id=org, run_id=run, user_message="先查再改")
        # worker_a：跑完一次只读工具（快照有 tool_result），然后死在第 2 轮模型调用前（lease 过期）
        claimed = await claim_run(db, worker_id="worker_a", org_id=org, run_id=run, lease_seconds=-1)
        assert claimed
        async with db.tx() as conn:
            await conn.execute("INSERT INTO tool_calls (run_id, org_id, step, tool, status, ok, result) "
                               "VALUES ($1,$2,1,'campaign.get_metrics','succeeded',TRUE,$3)", run, org, {"campaign_id": "CMP_1"})
        await save_transcript(db, org_id=org, run_id=run, step=1, history=[
            {"role": "action", "tool": "campaign.get_metrics", "arguments": {"campaign_id": "CMP_1"}},
            {"role": "observation", "observation": {"campaign_id": "CMP_1", "ctr": 0.02}}])
        c = await sweep_once(db, org_id=org)
        st1, at1, kinds1, _, ob1 = await _state(db, org, run)
        # worker_b 定向领取（outbox 已投）→ 从快照续：模型看到已有观测，直接给终答
        platform = FakeAdPlatform()
        s = _Script()
        w = _worker(db, platform, s)
        claimed2 = await claim_run(db, worker_id="worker_b", org_id=org, run_id=run)
        await w.execute_claimed(claimed2)
        st2, at2, kinds2, tcs, _ = await _state(db, org, run)
        return c, st1, at1, ob1, st2, at2, kinds2, tcs, s.calls, platform.calls

    c, st1, at1, ob1, st2, at2, kinds, tcs, model_calls, platform_calls = with_db(go)
    assert c["requeued"] == 1 and st1 == "queued" and ob1 == 2, (c, st1, ob1)   # 创建那条 + sweeper 重投那条
    assert st2 == "succeeded" and at2 == 2
    assert "run.requeued_by_sweeper" in kinds and "run.restarted" in kinds, kinds
    assert [t["tool"] for t in tcs] == ["campaign.get_metrics"], "只读工具被重调了"
    assert model_calls == 1 and platform_calls == 0


# --------------------------------------------------------------------------
# 门槛③：分支 C —— 写工具执行后、记录前 kill ⇒ sweeper 标 response_lost ⇒ 对账回填 ⇒ 跑完
# --------------------------------------------------------------------------


def test_branch_c_response_lost_is_reconciled_from_platform_ledger_and_run_finishes() -> None:
    async def go(db):
        org, run = _ids()
        await create_run(db, org_id=org, run_id=run, user_message="扩量")
        await _approved_case(db, org, run)
        ledger = PlatformLedger(db)
        platform_a = FakeAdPlatform(ledger=ledger)
        key = derive_idempotency_key(org_id=org, tool=WRITE.tool, arguments=WRITE.arguments)
        claimed = await claim_run(db, worker_id="worker_a", org_id=org, run_id=run, lease_seconds=-1)
        assert claimed
        await platform_a.update_budget(**WRITE.arguments, idempotency_key=key)        # 钱动了、账本记了
        async with db.tx() as conn:
            await conn.execute(
                "INSERT INTO tool_calls (run_id, org_id, step, tool, arguments, external_idempotency_key, status, side_effect, created_at) "
                "VALUES ($1,$2,1,$3,$4,$5,'running',TRUE, now() - interval '10 minutes')", run, org, WRITE.tool, WRITE.arguments, key)
        await save_transcript(db, org_id=org, run_id=run, step=1,
                              history=[{"role": "action", "tool": WRITE.tool, "arguments": WRITE.arguments}])
        # worker_a 死了 ⇒ sweeper：run 重投 + 写调用超龄 ⇒ response_lost
        c = await sweep_once(db, org_id=org)
        _, _, kinds1, tcs1, _ = await _state(db, org, run)
        # 对账：**另一个进程**的平台实例（空内存）也能通过账本认出这个键
        rc = await reconcile_once(db, PlatformLedger(db), org_id=org)
        _, _, kinds2, tcs2, _ = await _state(db, org, run)
        # worker_b 续跑：loop 第二路读到 succeeded ⇒ 回填观测 ⇒ 模型给终答
        platform_b = FakeAdPlatform(ledger=ledger)
        w = _worker(db, platform_b, _Script())
        await w.execute_claimed(await claim_run(db, worker_id="worker_b", org_id=org, run_id=run))
        st, at, kinds3, tcs3, _ = await _state(db, org, run)
        return c, kinds1, tcs1, rc, kinds2, tcs2, st, kinds3, platform_a.calls + platform_b.calls

    c, kinds1, tcs1, rc, kinds2, tcs2, st, kinds3, total_calls = with_db(go)
    assert c["requeued"] == 1 and c["response_lost"] == 1
    assert tcs1[-1]["status"] == "response_lost" and "tool.response_lost" in kinds1
    assert rc["repaired_succeeded"] == 1 and tcs2[-1]["status"] == "succeeded" and "tool.repaired" in kinds2
    assert st == "succeeded" and "tool.repaired_from_intent_log" in kinds3, (st, kinds3)
    assert total_calls == 1, f"平台侧副作用 {total_calls} 次"


def test_reconcile_marks_failed_when_ledger_has_no_such_key() -> None:
    async def go(db):
        org, run = _ids()
        await create_run(db, org_id=org, run_id=run, user_message="x")
        await claim_run(db, worker_id="w", org_id=org, run_id=run)
        async with db.tx() as conn:
            await conn.execute(
                "INSERT INTO tool_calls (run_id, org_id, step, tool, external_idempotency_key, status, side_effect) "
                "VALUES ($1,$2,1,'campaign.update_budget','never-sent-key','response_lost',TRUE)", run, org)
        rc = await reconcile_once(db, PlatformLedger(db), org_id=org)
        _, _, kinds, tcs, _ = await _state(db, org, run)
        return rc, tcs, kinds

    rc, tcs, kinds = with_db(go)
    assert rc["repaired_failed"] == 1 and tcs[-1]["status"] == "failed" and "tool.repaired" in kinds


def test_reconcile_goes_manual_review_when_ledger_unavailable() -> None:
    class _Broken:
        async def get(self, key):
            raise ConnectionError("ledger down")

    async def go(db):
        org, run = _ids()
        await create_run(db, org_id=org, run_id=run, user_message="x")
        await claim_run(db, worker_id="w", org_id=org, run_id=run)
        async with db.tx() as conn:
            await conn.execute(
                "INSERT INTO tool_calls (run_id, org_id, step, tool, external_idempotency_key, status, side_effect) "
                "VALUES ($1,$2,1,'campaign.update_budget','k-unavail','response_lost',TRUE)", run, org)
        rc = await reconcile_once(db, _Broken(), org_id=org)
        st, _, kinds, tcs, _ = await _state(db, org, run)
        return rc, tcs, kinds, st

    rc, tcs, kinds, st = with_db(go)
    assert rc["manual_review"] == 1 and tcs[-1]["status"] == "response_lost"      # 不猜、不改
    assert "tool.manual_review" in kinds and st == "running"                        # manual_review 是事件不是状态


# --------------------------------------------------------------------------
# 门槛④：取消兑现 + 分支顺序
# --------------------------------------------------------------------------


def test_cancel_request_with_dead_worker_is_honoured_as_cancelled_even_if_attempts_exhausted() -> None:
    async def go(db):
        org, run = _ids()
        await create_run(db, org_id=org, run_id=run, user_message="x")
        await claim_run(db, worker_id="dead", org_id=org, run_id=run, lease_seconds=-1)
        async with db.tx() as conn:
            await conn.execute("UPDATE agent_runs SET attempts=99 WHERE org_id=$1 AND run_id=$2", org, run)  # 次数也用尽
        assert await request_cancel(db, org_id=org, run_id=run, reason="反悔") == "requested"
        c = await sweep_once(db, org_id=org)
        st, _, kinds, _, _ = await _state(db, org, run)
        return c, st, kinds

    c, st, kinds = with_db(go)
    assert c["cancelled"] == 1 and c["failed"] == 0 and st == "cancelled", (c, st)
    assert kinds[-1] == "run.cancelled"


def test_exhausted_attempts_without_cancel_goes_failed() -> None:
    async def go(db):
        org, run = _ids()
        await create_run(db, org_id=org, run_id=run, user_message="x")
        await claim_run(db, worker_id="dead", org_id=org, run_id=run, lease_seconds=-1)
        async with db.tx() as conn:
            await conn.execute("UPDATE agent_runs SET attempts=3 WHERE org_id=$1 AND run_id=$2", org, run)
        c = await sweep_once(db, org_id=org)
        st, _, kinds, _, _ = await _state(db, org, run)
        return c, st, kinds

    c, st, kinds = with_db(go)
    assert c["failed"] == 1 and st == "failed" and kinds[-1] == "run.failed"


# --------------------------------------------------------------------------
# 门槛⑤：慢不当死
# --------------------------------------------------------------------------


def test_slow_but_heartbeating_run_is_never_reclaimed() -> None:
    async def go(db):
        org, run = _ids()
        await create_run(db, org_id=org, run_id=run, user_message="x")
        assert await claim_run(db, worker_id="me", org_id=org, run_id=run, lease_seconds=1)
        hb = LeaseHeartbeat(db, org_id=org, run_id=run, worker_id="me", ttl_seconds=1, interval_seconds=0.25)
        hb.start()
        reclaimed = 0
        for _ in range(6):                      # 3×TTL 观察窗，期间 sweeper 每 0.5s 扫一次
            await asyncio.sleep(0.5)
            reclaimed += (await sweep_once(db, org_id=org))["requeued"]
        await hb.stop()
        st, _, _, _, _ = await _state(db, org, run)
        return reclaimed, st, hb.lost

    reclaimed, st, lost = with_db(go)
    assert reclaimed == 0 and st == "running" and lost is False, (reclaimed, st, lost)


# --------------------------------------------------------------------------
# 僵尸 queued：告警不自动重投；人工 requeue
# --------------------------------------------------------------------------


def test_stuck_queued_is_alerted_not_auto_requeued_and_manual_requeue_works() -> None:
    async def go(db):
        org, run = _ids()
        await create_run(db, org_id=org, run_id=run, user_message="x")
        async with db.tx() as conn:
            await conn.execute("UPDATE outbox_jobs SET status='dispatched' WHERE org_id=$1", org)     # 投了但消息没了
            # updated_at 有触发器（H14）⇒ 回填要先停触发器（仅测试）
            await conn.execute("ALTER TABLE agent_runs DISABLE TRIGGER trg_agent_runs_touch")
            await conn.execute("UPDATE agent_runs SET updated_at = now() - interval '1 hour' WHERE org_id=$1", org)
            await conn.execute("ALTER TABLE agent_runs ENABLE TRIGGER trg_agent_runs_touch")
        c = await sweep_once(db, org_id=org)
        _, _, kinds1, _, ob1 = await _state(db, org, run)
        job = await requeue_outbox(db, org_id=org, run_id=run, operator="chaoyu")
        _, _, kinds2, _, ob2 = await _state(db, org, run)
        return c, kinds1, ob1, job, kinds2, ob2

    c, kinds1, ob1, job, kinds2, ob2 = with_db(go)
    assert c["stuck_queued"] == 1 and "run.stuck_queued" in kinds1 and ob1 == 0
    assert job > 0 and "run.requeued_manually" in kinds2 and ob2 == 1


# --------------------------------------------------------------------------
# 账本持久化：跨平台实例（跨进程）可见
# --------------------------------------------------------------------------


def test_platform_ledger_dedups_across_instances() -> None:
    async def go(db):
        ledger = PlatformLedger(db)
        a, b = FakeAdPlatform(ledger=ledger), FakeAdPlatform(ledger=ledger)
        key = f"org_x:campaign.update_budget:{uuid.uuid4().hex[:8]}"
        r1 = await a.update_budget(campaign_id="CMP_1", new_budget=100, client_request_id="c", idempotency_key=key)
        r2 = await b.update_budget(campaign_id="CMP_1", new_budget=100, client_request_id="c", idempotency_key=key)
        return r1, r2, b.budgets.get("CMP_1")

    r1, r2, b_budget = with_db(go)
    assert r2.get("deduped_by_platform") is True and r2["new_budget"] == r1["new_budget"]
    assert b_budget is None, "实例 b 不该真的改预算——它只是通过账本认出了重复"


# --------------------------------------------------------------------------
# Repair 四样留痕 + Replay 零副作用（结构）
# --------------------------------------------------------------------------


@pytest.fixture()
def client():
    c = Client(create_app())
    yield c
    c.close()


def test_repair_requires_role_and_leaves_four_traces(client) -> None:
    run_id = client.post("/runs", json={"user_message": "x"}, headers=ACME).json()["run_id"]

    async def seed():
        db = client.app.state.db
        await claim_run(db, worker_id="w", org_id="org_acme", run_id=run_id)
        async with db.tx() as conn:
            return await conn.fetchval(
                "INSERT INTO tool_calls (run_id, org_id, step, tool, external_idempotency_key, status, side_effect) "
                "VALUES ($1,'org_acme',1,'campaign.update_budget',$2,'response_lost',TRUE) RETURNING id",
                run_id, f"k-{uuid.uuid4().hex[:6]}")
    call_id = client.loop.run_until_complete(seed())
    body = {"status": "succeeded", "result": {"campaign_id": "CMP_1", "new_budget": 1}, "reason": "人工核对平台后台", "operator": "chaoyu"}
    assert client.post(f"/runs/{run_id}/tool_calls/{call_id}/repair", json=body, headers=ACME).status_code == 403
    r = client.post(f"/runs/{run_id}/tool_calls/{call_id}/repair", json=body,
                    headers={"Authorization": "Bearer dev-token-acme-trace"})
    assert r.status_code == 200, r.text
    trace = client.get(f"/runs/{run_id}/trace", headers={"Authorization": "Bearer dev-token-acme-trace"}).json()
    audit = [a for a in trace["audit_logs"] if a["action"] == "tool.repair"]
    assert audit and {"operator", "reason", "before", "after"} <= set(audit[0]["detail"])
    assert any(e["kind"] == "tool.repaired" for e in trace["run_events"])
    assert trace["tool_calls"][-1]["status"] == "succeeded"


def test_replay_endpoint_is_read_only_by_construction() -> None:
    """Replay = GET /runs/{id}/events 只读渲染（K7 的 AST 断言守着）；这里再钉一次课件的定性：
    回放不调工具——事件流里没有任何"命令"，只有已发生的事实。"""
    from syncopate.runtime import event_layer
    assert all(not k.endswith((".execute", ".invoke", ".call")) for k in event_layer.registered_kinds())
