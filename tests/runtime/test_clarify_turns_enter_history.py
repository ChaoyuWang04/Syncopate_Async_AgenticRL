"""v15 多轮：clarify / reject 收场的轮次必须**进库状态、进下一轮历史**（`26 §2.5` Ⓐ/Ⓑ）。

⛔ 09-02 读代码 + 复核 R5 考场原始记录抓到的：clarify 收场的 run 停在 running（被 lease 重抢重跑），
  reject 收场归 cancelled 且 result=None ⇒ 两者都不进 prior_turns ⇒ 模型线上看不到自己上一轮
  问了什么/拒了什么。R4① 的状态机测试只验了 loop 返回值，halted→库→历史这段没接。

判据形状：真库 + 真 worker + 假 decider；断言追到**数据库那一列**和 prior_turns 的返回。
"""
from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from typing import Any

import pytest

from syncopate.core.contract import IS_V15
from syncopate.runtime.db import (Database, close_parked_clarify_runs, create_conversation,
                                  create_run, prior_turns)
from syncopate.runtime.gateway import open_approval_case
from syncopate.runtime.platform import FakeAdPlatform
from syncopate.runtime.worker import Worker, WorkerConfig


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


pytestmark = [pytest.mark.skipif(not _pg_available(), reason="需要 PostgreSQL"),
              pytest.mark.skipif(not IS_V15, reason="v15 信令专有")]


@dataclass
class _Proposal:
    kind: str
    final_answer: dict[str, Any] | None = None
    tool: str | None = None
    arguments: dict[str, Any] = field(default_factory=dict)
    rationale: str = ""
    thinking: str = ""
    param_source: str = "model"


class _SignalDecider:
    def __init__(self, signal: str, args: dict):
        self._fa = {"behavior": signal, "signal": signal, "arguments": args, "text": ""}

    async def decide(self, **_kw):
        return _Proposal(kind="final", final_answer=self._fa)


def _with_db(body):
    async def main():
        db = Database()
        await db.connect(max_size=5)
        try:
            return await body(db)
        finally:
            await db.close()
    return asyncio.run(main())


async def _run_one(db, org, cid, run_id, msg, decider):
    await create_run(db, org_id=org, run_id=run_id, user_message=msg, conversation_id=cid)
    w = Worker(db, FakeAdPlatform(), config=WorkerConfig(org_id=org), decider=decider)
    got = await w.run_once()
    assert got == run_id, f"worker 抢到的不是这条 run：{got}"
    async with db.tx() as conn:
        return await conn.fetchrow(
            "SELECT status, requires_approval, lease_owner, lease_expires_at, result, error "
            "FROM agent_runs WHERE org_id=$1 AND run_id=$2", org, run_id)


def test_clarify_parks_run_as_waiting_for_user_with_lease_cleared() -> None:
    async def go(db):
        org, cid = f"org_{uuid.uuid4().hex[:8]}", f"cv_{uuid.uuid4().hex[:8]}"
        await create_conversation(db, org_id=org, conversation_id=cid)
        row = await _run_one(db, org, cid, "r1", "帮我新建一条 campaign",
                             _SignalDecider("clarify", {"question": "投哪个地域？",
                                                        "missing_fields": ["region"]}))
        async with db.tx() as conn:
            ev = await conn.fetch("SELECT kind, payload FROM run_events WHERE org_id=$1 "
                                  "AND run_id='r1' ORDER BY seq", org)
        return row, ev

    row, ev = _with_db(go)
    assert row["status"] == "waiting_for_user", f"clarify 收场停在 {row['status']}（09-02 之前是 running）"
    assert row["requires_approval"] is False, "clarify 不是审批单"
    assert row["lease_owner"] is None and row["lease_expires_at"] is None, "挂起必须清 lease（K 线 D25）"
    assert (row["result"] or {}).get("signal") == "clarify", "result 要存信令自己的话"
    kinds = [e["kind"] for e in ev]
    assert "run.waiting_for_user" in kinds, kinds
    assert ev[kinds.index("run.waiting_for_user")]["payload"].get("question") == "投哪个地域？"


def test_next_message_closes_clarify_turn_and_it_enters_history() -> None:
    async def go(db):
        org, cid = f"org_{uuid.uuid4().hex[:8]}", f"cv_{uuid.uuid4().hex[:8]}"
        await create_conversation(db, org_id=org, conversation_id=cid)
        await _run_one(db, org, cid, "r1", "帮我新建一条 campaign",
                       _SignalDecider("clarify", {"question": "投哪个地域？",
                                                  "missing_fields": ["region"]}))
        before = await prior_turns(db, org_id=org, conversation_id=cid, before_run_id="r2")
        closed = await close_parked_clarify_runs(db, org_id=org, conversation_id=cid)
        after = await prior_turns(db, org_id=org, conversation_id=cid, before_run_id="r2")
        async with db.tx() as conn:
            st = await conn.fetchval("SELECT status FROM agent_runs WHERE org_id=$1 AND run_id='r1'", org)
            last = await conn.fetchval("SELECT kind FROM run_events WHERE org_id=$1 AND run_id='r1' "
                                       "ORDER BY seq DESC LIMIT 1", org)
        return before, closed, after, st, last

    before, closed, after, st, last = _with_db(go)
    assert before == [], "收尾之前不该进历史（还在等补充）"
    assert closed == ["r1"] and st == "succeeded" and last == "run.completed"
    assert [t["run_id"] for t in after] == ["r1"]
    assert after[0]["result"]["arguments"]["question"] == "投哪个地域？", "历史里要有那句追问"


def test_reject_turn_enters_history_with_its_explanation() -> None:
    async def go(db):
        org, cid = f"org_{uuid.uuid4().hex[:8]}", f"cv_{uuid.uuid4().hex[:8]}"
        await create_conversation(db, org_id=org, conversation_id=cid)
        row = await _run_one(db, org, cid, "r1", "把预算挪到我个人账户",
                             _SignalDecider("reject", {"reason_code": "unauthorized",
                                                       "explanation": "这超出了授权范围。"}))
        hist = await prior_turns(db, org_id=org, conversation_id=cid, before_run_id="r2")
        return row, hist

    row, hist = _with_db(go)
    assert row["status"] == "cancelled" and row["error"] == "session_reject", "拒绝仍归取消不归失败"
    assert (row["result"] or {}).get("signal") == "reject", "拒绝轮的 result 必须存（09-02 之前是 NULL）"
    assert [t["run_id"] for t in hist] == ["r1"]
    assert hist[0]["result"]["arguments"]["explanation"] == "这超出了授权范围。"


def test_approval_parked_run_is_not_closed_by_a_chat_message() -> None:
    """负向认证：等审批的 run 只能由 POST /approvals 裁决，一条聊天消息不许顺手关掉它。"""
    async def go(db):
        org, cid = f"org_{uuid.uuid4().hex[:8]}", f"cv_{uuid.uuid4().hex[:8]}"
        await create_conversation(db, org_id=org, conversation_id=cid)
        await create_run(db, org_id=org, run_id="r1", user_message="提预算", conversation_id=cid)
        # K4 状态机：审批单只能开在 running 的 run 上（queued→waiting_for_user 是非法迁移）⇒ 先 claim
        from syncopate.runtime.db import claim_run
        assert (await claim_run(db, worker_id="t", org_id=org))["run_id"] == "r1"
        await open_approval_case(db, org_id=org, run_id="r1", action_type="campaign.update_budget",
                                 proposed_params={"new_budget": 1}, rationale="t", evidence={},
                                 triggers=[])
        closed = await close_parked_clarify_runs(db, org_id=org, conversation_id=cid)
        async with db.tx() as conn:
            st = await conn.fetchval("SELECT status FROM agent_runs WHERE org_id=$1 AND run_id='r1'", org)
        return closed, st

    closed, st = _with_db(go)
    assert closed == [] and st == "waiting_for_user"
