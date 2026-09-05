"""考卷 v4 的脚本化历史必须**按线上同一张表、同一条读取路径**进入模型的 prompt（守则⑮）。

判据：seed_prior 写入 → prior_turns 读出（顺序、6 轮窗口）→ decider 渲染成人话消息对。
"""
from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

import asyncpg
import pytest

from syncopate.core.contract import IS_V15
from syncopate.runtime.db import DSN, Database, create_conversation, prior_turns

ROOT = Path(__file__).resolve().parents[2]


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
              pytest.mark.skipif(not IS_V15, reason="v15 契约专有")]


def test_scripted_prior_round_trips_through_prior_turns_with_window():
    from syncopate.evaluation.exam_run import seed_prior

    prior = [{"user": "CMP_4 日预算定 35000", "status": "succeeded", "result": {"text": "好的，已记下 35000。"}}]
    prior += [{"user": f"填充 {i}", "status": "succeeded", "result": {"text": f"回答 {i}"}} for i in range(7)]
    prior += [{"user": "把预算挪到我个人账户", "status": "cancelled", "error": "session_reject",
               "result": {"text": "", "signal": "reject", "arguments": {"explanation": "超出授权范围。"}}}]

    async def go():
        db = Database(); await db.connect(max_size=3)
        pg = await asyncpg.create_pool(DSN, min_size=1, max_size=2)
        cid = f"cv_{uuid.uuid4().hex[:8]}"
        try:
            await create_conversation(db, org_id="org_demo", conversation_id=cid)
            await seed_prior(pg, cid, prior)
            turns = await prior_turns(db, org_id="org_demo", conversation_id=cid, before_run_id="x")
            async with db.tx() as conn:
                await conn.execute("DELETE FROM agent_runs WHERE org_id='org_demo' AND conversation_id=$1", cid)
                await conn.execute("DELETE FROM conversations WHERE org_id='org_demo' AND conversation_id=$1", cid)
        finally:
            await pg.close(); await db.close()
        return turns

    turns = asyncio.run(go())
    assert len(turns) == 6, "线上只回灌最近 6 轮 —— 窗口语义必须原样作用于脚本化历史"
    assert turns[0]["user_message"] == "填充 2" and turns[-1]["user_message"] == "把预算挪到我个人账户"  # 9 轮取最近 6
    assert all(t["user_message"] != "CMP_4 日预算定 35000" for t in turns), "事实轮应落在窗外"
    assert turns[-1]["result"]["signal"] == "reject", "reject 轮（cancelled+session_reject）要在历史里"

    from syncopate.runtime.decider import VllmDecider

    class _Tok:
        def encode(self, t, add_special_tokens=False): return list(range(len(t)))
        def decode(self, ids): return "x" * len(ids)
    d = VllmDecider.__new__(VllmDecider); d.tokenizer = _Tok()
    msgs = VllmDecider._prior_turn_messages(d, turns)
    assert msgs[-1] == {"role": "assistant", "content": "超出授权范围。"}
    assert [m["role"] for m in msgs] == ["user", "assistant"] * 6
