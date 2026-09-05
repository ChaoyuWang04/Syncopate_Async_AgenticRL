"""K4 门槛（27 §6）：白名单全枚举（6×6=36 组合）· 无第二条路（grep）· 事务原子（状态/事件/审计同生共死）
· 事件名映射（含 actor 分支与 attempts>1）· 取消语义（"钱已动"时 cancelled 与审计并存）· rerun 通道 · 409 信封。"""
from __future__ import annotations

import asyncio
import re
import uuid
from pathlib import Path

import asyncpg
import pytest

from syncopate.runtime.api import create_app
from syncopate.runtime.db import (ALLOWED_ACTORS, ALLOWED_RUN_TRANSITIONS, RUN_STATUSES,
                                  TERMINAL_STATUSES, Database, InvalidRunTransition, claim_run,
                                  create_run, event_type_for_transition, transition_run)
from tests.runtime.test_api import ACME, Client, _pg_available

pytestmark = pytest.mark.skipif(not _pg_available(), reason="需要 PostgreSQL：bash scripts/serving/pg_bootstrap.sh")
REPO = Path(__file__).resolve().parents[2]


def with_db(body):
    async def main():
        db = Database()
        await db.connect(max_size=4)
        try:
            return await body(db)
        finally:
            await db.close()
    return asyncio.run(main())


def _ids() -> tuple[str, str]:
    return f"org_{uuid.uuid4().hex[:8]}", f"run_{uuid.uuid4().hex[:8]}"


async def _force_status(conn, org, run, st):
    # 测试专用：绕过状态机把 run 摆到某个起点（生产代码没有这条路——K4 门槛② grep 守着）
    await conn.execute("UPDATE agent_runs SET status=$3 WHERE org_id=$1 AND run_id=$2", org, run, st)


# --------------------------------------------------------------------------
# 门槛①：36 组合表驱动；三个终态出边为空集
# --------------------------------------------------------------------------


def test_transition_matrix_all_36_combinations() -> None:
    async def go(db):
        results = {}
        for frm in RUN_STATUSES:
            for to in RUN_STATUSES:
                org, run = _ids()
                await create_run(db, org_id=org, run_id=run, user_message="x")
                actor = next(iter(ALLOWED_ACTORS.get((frm, to), frozenset({"system"}))))
                async with db.tx() as conn:
                    await _force_status(conn, org, run, frm)
                try:
                    async with db.tx() as conn:
                        await transition_run(conn, org_id=org, run_id=run, to=to, reason="matrix",
                                             actor_type=actor, actor_id="t")
                    results[(frm, to)] = "ok"
                except InvalidRunTransition:
                    results[(frm, to)] = "409"
                async with db.tx() as conn:          # 卫生：别把活的 run 留给全局 claim 的老测试
                    await _force_status(conn, org, run, "cancelled")
        return results

    results = with_db(go)
    legal = {k for k, v in results.items() if v == "ok"}
    expected = {(f, t) for f, ts in ALLOWED_RUN_TRANSITIONS.items() for t in ts}
    assert legal == expected, f"多放行 {legal - expected} / 误拒 {expected - legal}"
    assert len(expected) == 10 and len(results) == 36
    for terminal in TERMINAL_STATUSES:
        assert ALLOWED_RUN_TRANSITIONS[terminal] == frozenset(), f"{terminal} 有出边"
        assert all(results[(terminal, t)] == "409" for t in RUN_STATUSES)


def test_wrong_actor_is_rejected_even_on_a_legal_edge() -> None:
    async def go(db):
        org, run = _ids()
        await create_run(db, org_id=org, run_id=run, user_message="x")
        async with db.tx() as conn:
            with pytest.raises(InvalidRunTransition, match="actor"):
                await transition_run(conn, org_id=org, run_id=run, to="running", reason="x",
                                     actor_type="api", actor_id="t")          # 只有 worker 能把 queued→running
        async with db.tx() as conn:
            with pytest.raises(ValueError, match="reason"):
                await transition_run(conn, org_id=org, run_id=run, to="cancelled", reason="",
                                     actor_type="api", actor_id="t")           # 签名即制度：reason 必填
    with_db(go)


# --------------------------------------------------------------------------
# 门槛②：无第二条路 —— 全仓 `UPDATE agent_runs … SET … status=` 只在 transition_run 里
# --------------------------------------------------------------------------


def _strip_comments(src: str) -> str:
    # 只剥 # 注释行（保留行数），SQL 字符串里的内容照扫
    return "\n".join(("" if line.lstrip().startswith("#") else line) for line in src.splitlines())


def test_no_status_write_outside_transition_run() -> None:
    """判据形状：每个 `UPDATE agent_runs` 的 **SET 段**（到 WHERE 为止）里不许出现 `status =`，
    除非它在 transition_run 函数体内。WHERE 里的 `status='running'` 是条件不是写入，不算。"""
    db_src = (REPO / "syncopate" / "runtime" / "db.py").read_text(encoding="utf-8")
    start = db_src.index("async def transition_run(")
    end = db_src.index("\nasync def ", start + 10)
    offenders = []
    for path in sorted((REPO / "syncopate" / "runtime").glob("*.py")):
        src = _strip_comments(path.read_text(encoding="utf-8"))
        for m in re.finditer(r"UPDATE\s+agent_runs\b", src):
            seg_end = src.find("WHERE", m.end())
            seg = src[m.end(): seg_end if seg_end != -1 else m.end() + 600]
            if re.search(r"\bstatus\s*=", seg):
                if path.name == "db.py" and start <= m.start() < end:
                    continue
                offenders.append(f"{path.name}:{src[:m.start()].count(chr(10)) + 1}")
    assert not offenders, f"状态裸改点（必须走 transition_run）：{offenders}"


def test_no_run_succeeded_literal_anywhere_in_code() -> None:
    """门槛④负向：手拼 run.succeeded 出现即红（事件名只从映射函数来）。"""
    hits = []
    for root in ("syncopate/runtime", "scripts", "frontend/src"):
        for path in (REPO / root).rglob("*"):
            if path.suffix in (".py", ".ts", ".tsx", ".html") and path.is_file():
                for i, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
                    if "run.succeeded" in line and "禁" not in line and "⛔" not in line:
                        hits.append(f"{path.relative_to(REPO)}:{i}")
    assert not hits, hits


# --------------------------------------------------------------------------
# 门槛③：事务原子 —— 事件写入失败 ⇒ 状态与审计都不许留下
# --------------------------------------------------------------------------


def test_transition_is_atomic_when_event_write_fails() -> None:
    async def go(db):
        org, run = _ids()
        await create_run(db, org_id=org, run_id=run, user_message="x")
        async with db.tx() as conn:
            last = await conn.fetchval("SELECT last_seq FROM agent_runs WHERE org_id=$1 AND run_id=$2", org, run)
            # 预埋孤儿事件占住下一个 seq ⇒ append_event 必撞 UNIQUE ⇒ 整个迁移回滚
            await conn.execute("INSERT INTO run_events (run_id, org_id, seq, kind) VALUES ($1,$2,$3,'orphan')",
                               run, org, last + 1)
        with pytest.raises(asyncpg.UniqueViolationError):
            async with db.tx() as conn:
                await transition_run(conn, org_id=org, run_id=run, to="cancelled", reason="x",
                                     actor_type="api", actor_id="t")
        async with db.tx() as conn:
            st = await conn.fetchval("SELECT status FROM agent_runs WHERE org_id=$1 AND run_id=$2", org, run)
            audits = await conn.fetchval("SELECT count(*) FROM audit_logs WHERE org_id=$1 AND run_id=$2 "
                                         "AND action='run.transition'", org, run)
            # 卫生：孤儿事件 + queued 的 run 留在库里会让后面全局 claim 的测试撞 seq（09-02 实测）
            await conn.execute("DELETE FROM run_events WHERE org_id=$1 AND run_id=$2 AND kind='orphan'", org, run)
            await _force_status(conn, org, run, "cancelled")
        return st, audits

    st, audits = with_db(go)
    assert st == "queued", "事件没写成、状态却改了 —— 部分提交"
    assert audits == 0, "事件没写成、审计却留下了"


# --------------------------------------------------------------------------
# 门槛④：事件名映射
# --------------------------------------------------------------------------


@pytest.mark.parametrize("frm,to,actor,attempts,expect", [
    ("running", "succeeded", "worker", 1, "run.completed"),
    ("running", "failed", "worker", 1, "run.failed"),
    ("running", "cancelled", "worker", 1, "run.cancelled"),
    ("queued", "cancelled", "api", 1, "run.cancelled"),
    ("running", "waiting_for_user", "worker", 1, "run.waiting_for_user"),
    ("queued", "running", "worker", 1, "run.started"),
    ("queued", "running", "worker", 2, "run.restarted"),
    ("running", "queued", "sweeper", 1, "run.requeued_by_sweeper"),
    ("running", "queued", "worker", 1, "run.retry_scheduled"),
    ("waiting_for_user", "queued", "approval", 1, "run.resumed"),
    ("waiting_for_user", "queued", "api", 1, "run.resumed"),
])
def test_event_name_mapping(frm, to, actor, attempts, expect) -> None:
    assert event_type_for_transition(frm, to, actor, attempts=attempts) == expect


def test_restart_and_sweeper_events_land_in_the_stream() -> None:
    """轮询 claim 撞到过期 lease：run.requeued_by_sweeper → run.restarted（恢复留痕，课件 §9.1）。"""
    async def go(db):
        org, run = _ids()
        await create_run(db, org_id=org, run_id=run, user_message="x")
        first = await claim_run(db, worker_id="w1", org_id=org, lease_seconds=-1)
        second = await claim_run(db, worker_id="w2", org_id=org, lease_seconds=60)
        async with db.tx() as conn:
            kinds = [r["kind"] for r in await conn.fetch(
                "SELECT kind FROM run_events WHERE org_id=$1 AND run_id=$2 ORDER BY seq", org, run)]
            audits = [r["detail"] for r in await conn.fetch(
                "SELECT detail FROM audit_logs WHERE org_id=$1 AND run_id=$2 AND action='run.transition' ORDER BY id", org, run)]
        return first, second, kinds, audits

    first, second, kinds, audits = with_db(go)
    assert first and second and second["attempts"] == 2
    assert kinds == ["run.created", "run.started", "run.requeued_by_sweeper", "run.restarted"], kinds
    assert [a["actor_type"] for a in audits] == ["worker", "sweeper", "worker"]


# --------------------------------------------------------------------------
# 门槛⑤：取消语义 —— 钱已动才收到取消：cancelled 与工具审计并存，不删真相
# --------------------------------------------------------------------------


def test_cancel_after_money_moved_keeps_both_facts() -> None:
    async def go(db):
        org, run = _ids()
        await create_run(db, org_id=org, run_id=run, user_message="x")
        await claim_run(db, worker_id="w", org_id=org, run_id=run)
        async with db.tx() as conn:
            await conn.execute("INSERT INTO tool_calls (run_id, org_id, tool, ok, status, external_idempotency_key, side_effect) "
                               "VALUES ($1,$2,'campaign.update_budget',TRUE,'succeeded','k-1',TRUE)", run, org)
            await transition_run(conn, org_id=org, run_id=run, to="cancelled", reason="user_cancel_after_write",
                                 actor_type="worker", actor_id="w", fields={"error": "cancel_requested"})
        async with db.tx() as conn:
            st = await conn.fetchval("SELECT status FROM agent_runs WHERE org_id=$1 AND run_id=$2", org, run)
            money = await conn.fetchval("SELECT count(*) FROM tool_calls WHERE org_id=$1 AND run_id=$2 AND status='succeeded'", org, run)
            audit = await conn.fetchval("SELECT detail->>'reason' FROM audit_logs WHERE org_id=$1 AND run_id=$2 "
                                        "AND action='run.transition' ORDER BY id DESC LIMIT 1", org, run)
        return st, money, audit

    st, money, audit = with_db(go)
    assert st == "cancelled" and money == 1 and audit == "user_cancel_after_write"


# --------------------------------------------------------------------------
# K4-5 rerun 通道 + 409 INVALID_RUN_TRANSITION 信封
# --------------------------------------------------------------------------


@pytest.fixture()
def client():
    c = Client(create_app())
    yield c
    c.close()


def test_rerun_creates_child_run_only_from_terminal(client) -> None:
    parent = client.post("/runs", json={"user_message": "原题"}, headers=ACME).json()["run_id"]
    blocked = client.post(f"/runs/{parent}/rerun", json={"reason": "再来一次"}, headers=ACME)
    assert blocked.status_code == 409 and blocked.json()["error"]["code"] == "RUN_NOT_TERMINAL"

    async def finish():
        async with client.app.state.db.tx() as conn:
            await _force_status(conn, "org_acme", parent, "failed")
    client.loop.run_until_complete(finish())
    child = client.post(f"/runs/{parent}/rerun", json={"reason": "再来一次", "user_message": "改了输入"}, headers=ACME)
    assert child.status_code == 201
    cid = child.json()["run_id"]
    assert cid != parent and child.json()["status"] == "queued"
    trace = client.get(f"/runs/{cid}/trace", headers={"Authorization": "Bearer dev-token-acme-trace"}).json()
    assert trace["run"]["parent_run_id"] == parent and trace["run"]["rerun_reason"] == "再来一次"
    assert trace["run_events"][0]["payload"]["parent_run_id"] == parent
