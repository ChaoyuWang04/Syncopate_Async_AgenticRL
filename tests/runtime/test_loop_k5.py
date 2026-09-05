"""K5 门槛（27 §7）：② 副作用黑洞（写工具执行后、记录前"崩"⇒ 恢复后平台侧副作用=1、run 不卡死、
转对账路径有事件；负向：抹掉意图日志此判据必红）· ③ 只读恢复（只读工具调用计数不变、模型调用 +1）
· ⑤ 安全点（cancel 后写类工具执行数=0，模型调用前那个安全点）· 存档密度（一轮两档、last 字段）。"""
from __future__ import annotations

import asyncio
import json
import uuid

import pytest

from syncopate.runtime.agent_loop import Proposal
from syncopate.runtime.db import Database, claim_run, create_run, request_cancel
from syncopate.runtime.platform import FakeAdPlatform, FaultPlan
from syncopate.runtime.worker import Worker, WorkerConfig
from tests.runtime.test_api import _pg_available

pytestmark = pytest.mark.skipif(not _pg_available(), reason="需要 PostgreSQL：bash scripts/serving/pg_bootstrap.sh")


def with_db(body):
    async def main():
        db = Database()
        await db.connect(max_size=5)
        try:
            return await body(db)
        finally:
            await db.close()
    return asyncio.run(main())


def _ids():
    return f"org_{uuid.uuid4().hex[:8]}", f"run_{uuid.uuid4().hex[:8]}"


class _Script:
    """假模型：照剧本吐提议；剧本吃完就给终答。记录 decide 次数。"""

    def __init__(self, *proposals):
        self._q = list(proposals)
        self.calls = 0

    async def decide(self, *, user_message, history):      # noqa: ANN001
        self.calls += 1
        return self._q.pop(0) if self._q else Proposal(kind="final", final_answer={"done": True})


WRITE = Proposal(kind="tool_call", tool="campaign.update_budget",
                 arguments={"campaign_id": "CMP_1", "new_budget": 120_000, "client_request_id": "req-A"},
                 param_source="user")
READ = Proposal(kind="tool_call", tool="campaign.get_metrics", arguments={"campaign_id": "CMP_1"},
                param_source="model")


def _worker(db, platform, script):
    return Worker(db, platform, WorkerConfig(amount_threshold=10 ** 9, daily_cost_cap_micros=10 ** 12),
                  decider=script)


async def _rows(db, org, run):
    async with db.tx() as conn:
        r = await conn.fetchrow("SELECT status, attempts FROM agent_runs WHERE org_id=$1 AND run_id=$2", org, run)
        kinds = [x["kind"] for x in await conn.fetch(
            "SELECT kind FROM run_events WHERE org_id=$1 AND run_id=$2 ORDER BY seq", org, run)]
        tcs = [dict(x) for x in await conn.fetch(
            "SELECT tool, status, side_effect FROM tool_calls WHERE org_id=$1 AND run_id=$2 ORDER BY id", org, run)]
        cps = [(json.loads(x["state"]) if isinstance(x["state"], str) else x["state"]) for x in await conn.fetch(
            "SELECT state FROM checkpoints WHERE org_id=$1 AND run_id=$2 ORDER BY step", org, run)]
    return r["status"], r["attempts"], kinds, tcs, cps


# --------------------------------------------------------------------------
# 门槛②：副作用黑洞 —— 平台已扣款、回包丢了（response_lost）
# --------------------------------------------------------------------------


async def _crash_between_side_effect_and_record(db, org, run, platform):
    """构造分支 C 现场（课件 CH8 §11）：模型点名了写工具（快照尾 = action）、意图日志占了坑
    （status=running, side_effect）、平台**已经扣款**（去重账本里有这个键）、但结果没记回来，进程就死了。
    lease 已过期 ⇒ 下一次轮询 claim 会内联 sweeper 把它重投。"""
    from syncopate.runtime.agent_loop import save_transcript
    from syncopate.runtime.tools import derive_idempotency_key
    claimed = await claim_run(db, worker_id="dead", org_id=org, run_id=run, lease_seconds=-1)
    assert claimed
    # 写动作在 loop 路径上一律先过审批（tier C）；崩溃窗口只可能出现在**人已批准后**的那次执行
    # ⇒ 预埋一张已批准的审批单，恢复执行才会 skip_triggers（与生产路径同形）
    async with db.tx() as conn:
        await conn.execute(
            "INSERT INTO approval_cases (case_ref, run_id, org_id, action_type, proposed_params, status, "
            "reviewer_id, reviewed_at) VALUES ($1,$2,$3,$4,$5,'approved','t',now())",
            f"apr_{uuid.uuid4().hex[:8]}", run, org, WRITE.tool, WRITE.arguments)
    key = derive_idempotency_key(org_id=org, tool=WRITE.tool, arguments=WRITE.arguments)
    await platform.update_budget(**WRITE.arguments, idempotency_key=key)        # 钱已经动了
    async with db.tx() as conn:
        await conn.execute(
            "INSERT INTO tool_calls (run_id, org_id, step, tool, arguments, external_idempotency_key, status, side_effect) "
            "VALUES ($1,$2,1,$3,$4,$5,'running',TRUE)", run, org, WRITE.tool, WRITE.arguments, key)
    await save_transcript(db, org_id=org, run_id=run, step=1,
                          history=[{"role": "action", "tool": WRITE.tool, "arguments": WRITE.arguments}])


def test_response_lost_write_is_not_retried_and_recovers_via_intent_log() -> None:
    async def go(db):
        org, run = _ids()
        await create_run(db, org_id=org, run_id=run, user_message="扩量")
        platform = FakeAdPlatform()
        await _crash_between_side_effect_and_record(db, org, run, platform)
        calls_at_crash = platform.calls

        # 恢复第一跑：轮询 claim 撞过期 lease ⇒ 内联 sweeper 重投 ⇒ loop 走第二路：意图日志 running ⇒ 等对账
        w = _worker(db, platform, _Script(WRITE))            # 即使模型剧本还想写，也不该再碰平台
        claimed = await claim_run(db, worker_id="w1", org_id=org)
        assert claimed and claimed["run_id"] == run
        await w.execute_claimed(claimed)
        st1, at1, kinds1, tcs1, _ = await _rows(db, org, run)

        # 对账（K8 的活，这里手工模拟）：按幂等键在平台去重账本里查到 ⇒ 回填 succeeded
        async with db.tx() as conn:
            await conn.execute("UPDATE tool_calls SET status='succeeded', ok=TRUE, result=$3, ended_at=now() "
                               "WHERE org_id=$1 AND run_id=$2 AND status IN ('running','response_lost')",
                               org, run, {"campaign_id": "CMP_1", "new_budget": 120_000, "reconciled": True})
            await conn.execute("UPDATE outbox_jobs SET next_attempt_at=now() WHERE org_id=$1", org)
        w2 = _worker(db, platform, _Script())                 # 回填后模型看到观测，给终答
        await w2.execute_claimed(await claim_run(db, worker_id="w2", org_id=org, run_id=run))
        st2, at2, kinds2, tcs2, cps = await _rows(db, org, run)
        return calls_at_crash, st1, kinds1, tcs1, st2, kinds2, tcs2, platform.calls, cps

    calls_at_crash, st1, kinds1, tcs1, st2, kinds2, tcs2, calls_final, cps = with_db(go)
    assert calls_at_crash == 1
    # 恢复第一跑：不重问模型、不裸重跑工具；run 不失败不卡死——回队列等对账，路径有事件
    assert st1 == "queued", (st1, kinds1)
    assert "run.requeued_by_sweeper" in kinds1 and "run.awaiting_reconciliation" in kinds1
    assert "tool.manual_review" in kinds1 and "run.retry_scheduled" in kinds1
    assert [t["status"] for t in tcs1] == ["running"], tcs1     # 没被改成 failed，也没被重发
    # 回填后：从意图日志补观测续跑到终态；平台侧副作用**全程 = 1**
    assert st2 == "succeeded", (st2, kinds2)
    assert "tool.repaired_from_intent_log" in kinds2
    assert calls_final == 1, f"平台被调了 {calls_final} 次——副作用重复"
    assert any(cp.get("last", "").startswith("tool_use:") for cp in cps)


def test_negative_without_intent_log_the_write_is_repeated() -> None:
    """负向认证：抹掉意图日志那一行 ⇒ 恢复时"没占过坑"被当成安全 ⇒ 平台被打第二次。
    这条测试**证明判据依赖意图日志**（去掉机制它必红）。"""
    async def go(db):
        org, run = _ids()
        await create_run(db, org_id=org, run_id=run, user_message="扩量")
        platform = FakeAdPlatform()
        await _crash_between_side_effect_and_record(db, org, run, platform)
        async with db.tx() as conn:
            await conn.execute("DELETE FROM tool_calls WHERE org_id=$1 AND run_id=$2", org, run)   # 抹掉意图日志
        w = _worker(db, platform, _Script(WRITE))
        await w.execute_claimed(await claim_run(db, worker_id="w1", org_id=org))
        return platform.calls

    # 假平台按幂等键去重（同键 ⇒ 平台认出重复，钱不会真的再扣），但**本地**已经又发了一次请求
    assert with_db(go) == 2, "没有意图日志时恢复没有重发——判据不是在量意图日志"


# --------------------------------------------------------------------------
# 门槛③：只读恢复 —— 读工具结果复用，模型调用 +1
# --------------------------------------------------------------------------


def test_read_only_recovery_reuses_observation_and_recalls_model_once() -> None:
    async def go(db):
        org, run = _ids()
        await create_run(db, org_id=org, run_id=run, user_message="先查再改")
        platform = FakeAdPlatform()
        # 剧本：读 → 写（写会超阈值开审批单 ⇒ halted），恢复后模型只需给终答
        big = Proposal(kind="tool_call", tool="campaign.update_budget",
                       arguments={"campaign_id": "CMP_1", "new_budget": 120_000, "client_request_id": "req-B"},
                       param_source="user")
        s1 = _Script(READ, big)
        w = Worker(db, platform, WorkerConfig(amount_threshold=1_000, daily_cost_cap_micros=10 ** 12), decider=s1)
        await w.execute_claimed(await claim_run(db, worker_id="w1", org_id=org, run_id=run))
        reads_after_first = platform.calls
        st1, *_ = await _rows(db, org, run)
        # 人批准 ⇒ 回队列
        from syncopate.runtime.db import resume_after_approval
        async with db.tx() as conn:
            await conn.execute("UPDATE approval_cases SET status='approved', reviewer_id='t', reviewed_at=now() "
                               "WHERE org_id=$1 AND run_id=$2", org, run)
        await resume_after_approval(db, org_id=org, run_id=run)
        s2 = _Script()
        w2 = Worker(db, platform, WorkerConfig(amount_threshold=1_000, daily_cost_cap_micros=10 ** 12), decider=s2)
        await w2.execute_claimed(await claim_run(db, worker_id="w2", org_id=org, run_id=run))
        st2, _, kinds, tcs, _ = await _rows(db, org, run)
        return st1, reads_after_first, s1.calls, st2, platform.calls, s2.calls, tcs

    st1, reads1, model1, st2, calls2, model2, tcs = with_db(go)
    assert st1 == "waiting_for_user" and model1 == 2
    assert st2 == "succeeded"
    reads = [t for t in tcs if t["tool"] == "campaign.get_metrics"]
    assert len(reads) == 1, "恢复后只读工具被重调了"
    assert model2 == 1, "恢复后应只多付一次 model call"


# --------------------------------------------------------------------------
# 门槛⑤：安全点 —— 模型调用前
# --------------------------------------------------------------------------


def test_cancel_stops_before_next_model_call_and_no_write_happens() -> None:
    async def go(db):
        org, run = _ids()
        await create_run(db, org_id=org, run_id=run, user_message="x")
        platform = FakeAdPlatform()
        claimed = await claim_run(db, worker_id="w1", org_id=org, run_id=run)
        outcome = await request_cancel(db, org_id=org, run_id=run, reason="反悔")     # running ⇒ 只记意图
        script = _Script(WRITE)
        w = _worker(db, platform, script)
        await w.execute_claimed(claimed)
        st, _, kinds, tcs, _ = await _rows(db, org, run)
        return outcome, st, kinds, tcs, script.calls, platform.calls

    outcome, st, kinds, tcs, model_calls, platform_calls = with_db(go)
    assert outcome == "requested" and st == "cancelled" and kinds[-1] == "run.cancelled"
    assert model_calls == 0, "取消意图已在，模型调用前的安全点没拦住"
    assert platform_calls == 0 and tcs == []
