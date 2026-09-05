"""K7 门槛（27 §9）：① 顺序断言（先落库再推：门铃断了事件一条不少、重连全补齐）· ② 断线补发（after 与
Last-Event-ID 各一路，M 条恰 M 条、seq 连续）· ③ 分层过滤（public 流零 internal 字段；trace 全量可见；
代码里每个 emit 的 kind 都已登记——未登记默认不推）· ⑤ 鉴权（同域 Cookie 可用；跨租户 404；前端 URL 无长效凭证）
· ⑥ 只读纪律（SSE 路径 AST 无模型/工具/状态迁移调用）。⑦ 前端时间线实测本机无 node，不当通过。"""
from __future__ import annotations

import ast
import asyncio
import re
import uuid
from pathlib import Path

import pytest

from syncopate.runtime import event_layer
from syncopate.runtime.api import create_app
from syncopate.runtime.db import create_run
from syncopate.runtime.worker import emit
from tests.runtime.test_api import ACME, GLOBEX, _pg_available
from tests.runtime.test_sse import parse_sse, run_case

pytestmark = pytest.mark.skipif(not _pg_available(), reason="需要 PostgreSQL：bash scripts/serving/pg_bootstrap.sh")
REPO = Path(__file__).resolve().parents[2]
TRACE = {"Authorization": "Bearer dev-token-acme-trace"}


def _ids():
    return "org_acme", f"run_{uuid.uuid4().hex[:10]}"


# --------------------------------------------------------------------------
# 门槛②：断线补发两路
# --------------------------------------------------------------------------


@pytest.mark.parametrize("via", ["after", "header"])
def test_reconnect_receives_exactly_the_missing_events(via) -> None:
    org, run_id = _ids()

    async def body(db, client):
        await create_run(db, org_id=org, run_id=run_id, user_message="x")        # seq1 run.created
        for i in range(5):                                                         # seq2..6
            await emit(db, org_id=org, run_id=run_id, kind="tool.result", payload={"tool": f"t{i}", "ok": True})
        first = parse_sse((await client.get(f"/runs/{run_id}/events?after=0", headers=ACME)).text) \
            if False else None
        # 断线在 seq 3 之后；随后又写 M=4 条（seq7..10）+ 终态
        for i in range(4):
            await emit(db, org_id=org, run_id=run_id, kind="tool.result", payload={"tool": f"late{i}", "ok": True})
        await emit(db, org_id=org, run_id=run_id, kind="run.completed", payload={"answer": 1})
        if via == "after":
            r = await client.get(f"/runs/{run_id}/events?after=3", headers=ACME)
        else:
            r = await client.get(f"/runs/{run_id}/events", headers={**ACME, "Last-Event-ID": "3"})
        return r.text

    events = parse_sse(run_case(body))
    ids = [int(e["id"]) for e in events]
    assert ids == list(range(4, 12)), ids                 # 4..11：连续、无重复、无空洞
    assert events[-1]["event"] == "run.completed"


def test_query_after_beats_last_event_id_header() -> None:
    org, run_id = _ids()

    async def body(db, client):
        await create_run(db, org_id=org, run_id=run_id, user_message="x")
        for _ in range(3):
            await emit(db, org_id=org, run_id=run_id, kind="tool.result", payload={"tool": "t", "ok": True})
        await emit(db, org_id=org, run_id=run_id, kind="run.completed")
        r = await client.get(f"/runs/{run_id}/events?after=3", headers={**ACME, "Last-Event-ID": "1"})
        return r.text

    ids = [int(e["id"]) for e in parse_sse(run_case(body))]
    assert ids == [4, 5], ids


# --------------------------------------------------------------------------
# 门槛①：先落库再推 —— 门铃拔掉事件一条不少
# --------------------------------------------------------------------------


def test_events_survive_without_the_bell_and_stream_is_bell_independent() -> None:
    org, run_id = _ids()

    async def body(db, client):
        client_app = client._transport.app
        client_app.state.sse_waiters.clear()                    # "拔铃"：没有任何 waiter 会被叫醒
        await create_run(db, org_id=org, run_id=run_id, user_message="x")
        await emit(db, org_id=org, run_id=run_id, kind="tool.result", payload={"tool": "t", "ok": True})
        await emit(db, org_id=org, run_id=run_id, kind="run.completed")
        async with db.tx() as conn:
            n_db = await conn.fetchval("SELECT count(*) FROM run_events WHERE org_id=$1 AND run_id=$2", org, run_id)
        r = await client.get(f"/runs/{run_id}/events", headers=ACME)
        return n_db, r.text

    n_db, text = run_case(body)
    events = parse_sse(text)
    assert n_db == 3 and [e["event"] for e in events] == ["run.created", "tool.result", "run.completed"]


# --------------------------------------------------------------------------
# 门槛③：分层过滤
# --------------------------------------------------------------------------


def test_public_stream_strips_internal_fields_and_hides_internal_kinds() -> None:
    org, run_id = _ids()

    async def body(db, client):
        await create_run(db, org_id=org, run_id=run_id, user_message="x")
        await emit(db, org_id=org, run_id=run_id, kind="run.started",
                   payload={"worker_id": "celery-x-1", "attempts": 1, "automation_tier": "C"})
        await emit(db, org_id=org, run_id=run_id, kind="run.enqueued", payload={"outbox_job_id": 99})   # internal kind
        await emit(db, org_id=org, run_id=run_id, kind="tool.manual_review", payload={"tool": "x"})     # internal kind
        await emit(db, org_id=org, run_id=run_id, kind="tool.result",
                   payload={"tool": "campaign.get_metrics", "ok": True, "idempotency_key": "SECRET", "prompt": "P"})
        await emit(db, org_id=org, run_id=run_id, kind="some.unregistered_kind", payload={"x": 1})     # 未登记 ⇒ 不推
        await emit(db, org_id=org, run_id=run_id, kind="run.completed", payload={"answer": 42})
        stream = (await client.get(f"/runs/{run_id}/events", headers=ACME)).text
        trace = (await client.get(f"/runs/{run_id}/trace", headers=TRACE)).json()
        return stream, trace

    stream, trace = run_case(body)
    kinds = [e["event"] for e in parse_sse(stream)]
    assert kinds == ["run.created", "run.started", "tool.result", "run.completed"], kinds
    for bad in ("worker_id", "outbox_job_id", "idempotency_key", "SECRET", "prompt", "\"P\"", "attempts"):
        assert bad not in stream, f"public 流泄漏了 {bad}"
    assert re.search(r"\b(prompt|tokens|purpose)\b", stream) is None
    trace_kinds = [e["kind"] for e in trace["run_events"]]
    assert "run.enqueued" in trace_kinds and "some.unregistered_kind" in trace_kinds       # 库里存全的
    assert any(e["payload"].get("idempotency_key") == "SECRET" for e in trace["run_events"])   # trace 全量可见


def test_every_emitted_kind_in_code_is_registered_in_the_layer_table() -> None:
    """fail-closed 的另一半：未登记的 kind 不推 ⇒ 忘了登记等于前端看不见。用 grep 把它变成结构判据。"""
    src_dir = REPO / "syncopate" / "runtime"
    found: set[str] = set()
    for path in src_dir.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        found |= set(re.findall(r'kind=f?"([a-z_]+\.[a-z_]+)"', text))
        found |= set(re.findall(r'return "(run\.[a-z_]+)"', text))
        found |= set(re.findall(r'"(run\.[a-z_]+)" if', text))
    found -= {"tool_call"}                                # Proposal.kind 不是事件
    missing = sorted(found - event_layer.registered_kinds())
    assert not missing, f"代码里发出的事件种类未在 event_layer 登记（默认不外推）：{missing}"


# --------------------------------------------------------------------------
# 门槛⑤：鉴权
# --------------------------------------------------------------------------


def test_same_origin_cookie_authenticates_and_cross_tenant_is_404() -> None:
    org, run_id = _ids()

    async def body(db, client):
        await create_run(db, org_id=org, run_id=run_id, user_message="x")
        await emit(db, org_id=org, run_id=run_id, kind="run.completed")
        ok = await client.get(f"/runs/{run_id}/events", cookies={"syncopate_token": "dev-token-acme"})
        other = await client.get(f"/runs/{run_id}/events", cookies={"syncopate_token": "dev-token-globex"})
        none = await client.get(f"/runs/{run_id}/events")
        return ok.status_code, [e["event"] for e in parse_sse(ok.text)], other.status_code, none.status_code

    st, kinds, other, none = run_case(body)
    assert st == 200 and kinds[-1] == "run.completed"
    assert other == 404 and none == 401


def test_frontend_never_puts_a_long_lived_credential_in_a_url() -> None:
    hits = []
    for path in (REPO / "frontend" / "src").rglob("*.ts*"):
        for i, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
            if re.search(r"[?&](token|api_key|apikey|authorization)=", line, re.I):
                hits.append(f"{path.relative_to(REPO)}:{i}")
    assert not hits, f"前端把凭证拼进了 URL：{hits}"


# --------------------------------------------------------------------------
# 门槛⑥：只读纪律 —— SSE 路径结构断言
# --------------------------------------------------------------------------


def test_sse_endpoint_path_calls_nothing_but_read_push_cursor_heartbeat_close() -> None:
    src = (REPO / "syncopate" / "runtime" / "api.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    forbidden = {"transition_run", "finish_run", "claim_run", "record_tool_call", "decide", "invoke",
                 "create_run", "request_cancel", "resume_run", "open_approval_case", "schedule_run_retry"}
    stream_fns = [n for n in ast.walk(tree) if isinstance(n, (ast.AsyncFunctionDef, ast.FunctionDef))
                  and n.name in ("_event_stream", "stream_events")]
    assert stream_fns, "找不到 SSE 端点函数"
    called = set()
    for fn in stream_fns:
        for node in ast.walk(fn):
            if isinstance(node, ast.Call):
                f = node.func
                called.add(f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", ""))
    assert not (called & forbidden), f"SSE 路径里出现了业务调用：{sorted(called & forbidden)}"
