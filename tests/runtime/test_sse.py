"""M9.6 · SSE 事件流与观测的验收测试。

★ 主验收是**断线补发**：客户端带 `Last-Event-ID` 回来，必须**只收到之后的**，
不能重复推已经收过的（重复推会让前端把同一条动作画两遍），
也不能跳过中间那段（跳过就是永久丢事件）。

★ 第二条是**终态关流**：收到终态事件就结束。不关的话客户端永远挂着等下一条，
而服务端也永远留着一个连接 —— 压测场景①里**连接数才是先撑爆的东西**。
"""

from __future__ import annotations

import asyncio
import uuid

import httpx
import pytest

from syncopate.runtime.api import create_app
from syncopate.runtime.db import Database, create_run
from syncopate.runtime.worker import emit

ACME = {"Authorization": "Bearer dev-token-acme"}
GLOBEX = {"Authorization": "Bearer dev-token-globex"}


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
    not _pg_available(), reason="需要 PostgreSQL：bash scripts/pg_bootstrap.sh")


def parse_sse(text: str) -> list[dict[str, str]]:
    """把 SSE 原文切成事件列表。注释行（`:` 开头的心跳）跳过。"""
    events = []
    for block in text.split("\n\n"):
        if not block.strip() or block.lstrip().startswith(":"):
            continue
        item: dict[str, str] = {}
        for line in block.splitlines():
            if line.startswith(":"):
                continue
            key, _, value = line.partition(":")
            item[key.strip()] = value.strip()
        if item:
            events.append(item)
    return events


def run_case(body):
    """一个测试 = 一个事件循环 + 一份 lifespan（见 test_api.Client 的说明）。"""
    app = create_app()

    async def main():
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app), base_url="http://t") as client:
                return await body(app.state.db, client)
    return asyncio.run(main())


def _org_run() -> tuple[str, str]:
    return "org_acme", f"run_{uuid.uuid4().hex[:10]}"


# --------------------------------------------------------------------------
# 断线补发
# --------------------------------------------------------------------------


def test_stream_replays_all_events_from_scratch() -> None:
    org, run_id = _org_run()

    async def body(db, client):
        await create_run(db, org_id=org, run_id=run_id, user_message="x")
        for kind in ("run.started", "tool.result", "run.completed"):
            await emit(db, org_id=org, run_id=run_id, kind=kind, payload={"k": kind})
        r = await client.get(f"/runs/{run_id}/events", headers=ACME)
        return r

    r = run_case(body)
    events = parse_sse(r.text)
    # seq 1 = 创建事务写的 run.created（K2-6）；之后才是 worker 侧的事件
    assert [e["event"] for e in events] == ["run.created", "run.started", "tool.result",
                                            "run.completed"]
    assert [e["id"] for e in events] == ["1", "2", "3", "4"]


def test_last_event_id_resumes_without_duplicates() -> None:
    """★★★ 断线补发的核心：只收之后的，**不重复也不跳过**。"""
    org, run_id = _org_run()

    async def body(db, client):
        await create_run(db, org_id=org, run_id=run_id, user_message="x")
        for kind in ("run.started", "tool.result", "tool.result", "run.completed"):
            await emit(db, org_id=org, run_id=run_id, kind=kind)
        return await client.get(f"/runs/{run_id}/events",
                                headers={**ACME, "Last-Event-ID": "3"})

    events = parse_sse(run_case(body).text)
    assert [e["id"] for e in events] == ["4", "5"], "补发的位置不对"


def test_garbage_last_event_id_falls_back_to_full_replay() -> None:
    """客户端给了脏值就从头推 —— 总比 500 好（脏 header 是常态，不是攻击）。

    ⚠️ 脏值用 ASCII：HTTP header 只能是 latin-1，非 ASCII 会被客户端**在发出去之前**
    就拒掉（第一版写了中文，报的是 UnicodeEncodeError，根本没到服务端）。
    """
    org, run_id = _org_run()

    async def body(db, client):
        await create_run(db, org_id=org, run_id=run_id, user_message="x")
        await emit(db, org_id=org, run_id=run_id, kind="run.completed")
        return await client.get(f"/runs/{run_id}/events",
                                headers={**ACME, "Last-Event-ID": "not-a-number"})

    r = run_case(body)
    assert r.status_code == 200
    assert len(parse_sse(r.text)) == 2          # run.created + run.completed


# --------------------------------------------------------------------------
# 终态关流
# --------------------------------------------------------------------------


def test_stream_closes_on_terminal_event() -> None:
    """★★ 收到终态就关流。不关的话客户端永远挂着，连接数会先于 CPU 撑爆。"""
    org, run_id = _org_run()

    async def body(db, client):
        await create_run(db, org_id=org, run_id=run_id, user_message="x")
        await emit(db, org_id=org, run_id=run_id, kind="run.started")
        await emit(db, org_id=org, run_id=run_id, kind="run.failed", payload={"e": "boom"})
        # 终态之后又来一条：不该被推出去（流已经该关了）
        await emit(db, org_id=org, run_id=run_id, kind="tool.result")
        return await asyncio.wait_for(client.get(f"/runs/{run_id}/events", headers=ACME),
                                      timeout=10)

    events = parse_sse(run_case(body).text)
    assert [e["event"] for e in events] == ["run.created", "run.started", "run.failed"]


@pytest.mark.parametrize("terminal", ["run.completed", "run.failed", "run.waiting_for_user"])
def test_every_terminal_kind_closes_the_stream(terminal: str) -> None:
    """★ 三种终态都要关流 —— 漏一种就是一类 run 永远挂着连接。

    `waiting_for_user` 尤其容易漏：它不是"结束"，但**对这条流来说是终点**
    （接下来要等人审批，可能几小时）。
    """
    org, run_id = _org_run()

    async def body(db, client):
        await create_run(db, org_id=org, run_id=run_id, user_message="x")
        await emit(db, org_id=org, run_id=run_id, kind=terminal)
        return await asyncio.wait_for(client.get(f"/runs/{run_id}/events", headers=ACME),
                                      timeout=10)

    assert [e["event"] for e in parse_sse(run_case(body).text)] == ["run.created", terminal]


# --------------------------------------------------------------------------
# 越权 & 观测
# --------------------------------------------------------------------------


def test_cannot_stream_another_orgs_run() -> None:
    """★ 越权在 SQL 里挡，SSE 这条路也不例外。"""
    org, run_id = _org_run()

    async def body(db, client):
        await create_run(db, org_id=org, run_id=run_id, user_message="x")
        await emit(db, org_id=org, run_id=run_id, kind="run.completed")
        return await client.get(f"/runs/{run_id}/events", headers=GLOBEX)

    assert run_case(body).status_code == 404


def test_usage_is_scoped_and_counted() -> None:
    """★ 观测同样按 org 隔离 —— 成本数据泄漏等于把别人的经营情况给出去了。"""
    async def body(db, client):
        async with db.tx() as conn:
            await conn.execute(
                "INSERT INTO usage_records (org_id, run_id, tokens_in, tokens_out, cost_micros) "
                "VALUES ($1,$2,$3,$4,$5)", "org_acme", f"u_{uuid.uuid4().hex[:6]}", 100, 20, 500)
        mine = await client.get("/usage", headers=ACME)
        theirs = await client.get("/usage", headers=GLOBEX)
        return mine.json(), theirs.json()

    mine, theirs = run_case(body)
    assert mine["tokens_in"] >= 100 and mine["cost_micros"] >= 500
    assert theirs["cost_micros"] < mine["cost_micros"], "看到了别人的成本"
