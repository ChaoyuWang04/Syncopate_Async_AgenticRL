"""K2 门槛（27 §4）：约束就位 · 序号无洞无撞（含负向认证）· 创建事务原子 · 账单防重 ·
updated_at 触发器（含负向认证）· id 不可枚举 · 干净库 upgrade head == 仓库快照（28 P-04/P-05）。

每个测试在**临时数据库**上跑完整迁移链（不是复用开发库）：迁移链是唯一真相，
所以判据也必须从"干净库 + 迁移链"出发，否则测的是开发库里手改过的东西。
"""
from __future__ import annotations

import asyncio
import os
import re
import uuid
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
DSN = os.environ.get("SYNCOPATE_PG_DSN", "postgresql://syncopate:syncopate@127.0.0.1:5432/syncopate")


def _pg_available() -> bool:
    async def probe() -> bool:
        try:
            import asyncpg
            c = await asyncpg.connect(DSN)
            await c.close()
            return True
        except Exception:
            return False
    return asyncio.run(probe())


pytestmark = pytest.mark.skipif(not _pg_available(), reason="需要 PostgreSQL：bash scripts/serving/pg_bootstrap.sh")


def _snapshot_module():
    from syncopate.runtime import schema_snapshot
    return schema_snapshot


async def _admin(sql: str) -> None:
    import asyncpg
    c = await asyncpg.connect(DSN)
    try:
        await c.execute(sql)
    finally:
        await c.close()


@pytest.fixture(scope="module")
def fresh_dsn():
    """临时库 + alembic upgrade head。用完即删。"""
    from alembic import command
    from alembic.config import Config

    name = f"syncopate_migtest_{uuid.uuid4().hex[:8]}"
    asyncio.run(_admin(f'CREATE DATABASE "{name}"'))
    dsn = DSN.rsplit("/", 1)[0] + "/" + name
    cfg = Config(str(REPO / "alembic.ini"))
    cfg.set_main_option("script_location", str(REPO / "syncopate" / "runtime" / "migrations"))
    cfg.set_main_option("syncopate.dsn", dsn)
    command.upgrade(cfg, "head")
    try:
        yield dsn
    finally:
        asyncio.run(_admin(f'DROP DATABASE "{name}" WITH (FORCE)'))


def _run(coro):
    return asyncio.run(coro)


# --------------------------------------------------------------------------
# 快照一致（P-04：迁移链是唯一真相；P-05：无 autogenerate，漂移靠这条判据）
# --------------------------------------------------------------------------


def test_fresh_upgrade_head_matches_committed_snapshot(fresh_dsn) -> None:
    snap = _snapshot_module()
    live = _run(snap.dump(fresh_dsn))
    want = snap.SNAPSHOT.read_text(encoding="utf-8")
    assert live == want, "干净库 upgrade head 与仓库快照不一致：改 schema 只许走迁移链并重生快照"


# --------------------------------------------------------------------------
# 门槛①：约束就位（逐条 INSERT 违例数据必被拒）
# --------------------------------------------------------------------------


def test_constraints_reject_violations(fresh_dsn) -> None:
    import asyncpg

    async def go():
        c = await asyncpg.connect(fresh_dsn)
        try:
            org, run = "org_t", "run_t1"
            await c.execute("INSERT INTO agent_runs (run_id, org_id, user_message) VALUES ($1,$2,'x')", run, org)
            cases = {
                "status 值域": ("UPDATE agent_runs SET status='bogus' WHERE run_id=$1", (run,)),
                "tool_calls 五态值域": ("INSERT INTO tool_calls (run_id, org_id, tool, status) VALUES ($1,$2,'t','weird')", (run, org)),
                "有副作用必带幂等键": ("INSERT INTO tool_calls (run_id, org_id, tool, side_effect) VALUES ($1,$2,'t',TRUE)", (run, org)),
                "run_events seq 唯一": ("INSERT INTO run_events (run_id, org_id, seq, kind) VALUES ($1,$2,1,'a'),($1,$2,1,'b')", (run, org)),
                "usage 账单防重": ("INSERT INTO usage_records (org_id, run_id, call_index) VALUES ($1,$2,0),($1,$2,0)", (org, run)),
                "idempotency_key 唯一": ("INSERT INTO agent_runs (run_id, org_id, idempotency_key) VALUES ('a',$1,'k'),('b',$1,'k')", (org,)),
            }
            rejected = {}
            for name, (sql, args) in cases.items():
                try:
                    await c.execute(sql, *args)
                    rejected[name] = False
                except (asyncpg.CheckViolationError, asyncpg.UniqueViolationError):
                    rejected[name] = True
            return rejected
        finally:
            await c.close()

    rejected = _run(go())
    assert all(rejected.values()), f"有约束没生效：{[k for k, v in rejected.items() if not v]}"


# --------------------------------------------------------------------------
# 门槛②：序号无洞无撞 —— 两个写者各写 100 条；负向认证：MAX+1 必撞
# --------------------------------------------------------------------------


def test_seq_allocator_two_writers_no_gaps_no_collisions(fresh_dsn) -> None:
    from syncopate.runtime.db import Database, append_event

    async def go():
        db = Database(fresh_dsn)
        await db.connect(max_size=4)
        try:
            org, run = "org_seq", "run_seq"
            async with db.tx() as conn:
                await conn.execute("INSERT INTO agent_runs (run_id, org_id, user_message) VALUES ($1,$2,'x')", run, org)

            async def writer(tag: str):
                for i in range(100):
                    async with db.tx() as conn:          # 每条事件一个事务（真实形态）
                        await append_event(conn, org_id=org, run_id=run, kind=f"{tag}.{i}")

            await asyncio.gather(writer("api"), writer("worker"))
            async with db.tx() as conn:
                seqs = [r["seq"] for r in await conn.fetch(
                    "SELECT seq FROM run_events WHERE org_id=$1 AND run_id=$2 ORDER BY seq", org, run)]
                last = await conn.fetchval("SELECT last_seq FROM agent_runs WHERE run_id=$1", run)
            return seqs, last
        finally:
            await db.close()

    seqs, last = _run(go())
    assert seqs == list(range(1, 201)), "seq 有洞或有重复"
    assert last == 200, "领号器计数与事件数不一致"


def test_negative_max_plus_one_collides_under_two_writers(fresh_dsn) -> None:
    """负向认证（守则②）：换回 `SELECT max(seq)+1` 两个写者必撞唯一键。
    确定性构造：T1 读到 max=0 插 seq=1 不提交；T2 读到 max=0 也插 seq=1 ⇒ 阻塞在 T1 上；
    T1 提交 ⇒ T2 UniqueViolation。这就是 09 §4 ⑩ 炸死 worker 的那个窗口。"""
    import asyncpg

    async def go():
        c1 = await asyncpg.connect(fresh_dsn)
        c2 = await asyncpg.connect(fresh_dsn)
        try:
            org, run = "org_neg", "run_neg"
            await c1.execute("INSERT INTO agent_runs (run_id, org_id, user_message) VALUES ($1,$2,'x')", run, org)
            sql = ("INSERT INTO run_events (run_id, org_id, seq, kind) VALUES ($1,$2,"
                   " COALESCE((SELECT max(seq) FROM run_events WHERE run_id=$1 AND org_id=$2),0)+1, 'e')")
            t1 = c1.transaction(); await t1.start()
            await c1.execute(sql, run, org)
            t2 = c2.transaction(); await t2.start()
            blocked = asyncio.create_task(c2.execute(sql, run, org))
            await asyncio.sleep(0.2)
            await t1.commit()
            try:
                await blocked
                await t2.commit()
                return False
            except asyncpg.UniqueViolationError:
                await t2.rollback()
                return True
        finally:
            await c1.close(); await c2.close()

    assert _run(go()), "MAX+1 在两个写者下居然没撞——负向认证失败，判据不可信"


# --------------------------------------------------------------------------
# 门槛③：创建事务原子（run 与 run.created 同生共死）
# --------------------------------------------------------------------------


def test_create_run_and_run_created_live_and_die_together(fresh_dsn) -> None:
    import asyncpg
    from syncopate.runtime.db import Database, create_run

    async def go():
        db = Database(fresh_dsn)
        await db.connect(max_size=2)
        try:
            org, run = "org_atom", "run_atom"
            # 正向：成功创建 ⇒ 两样都在，且 run.created 是 seq 1
            await create_run(db, org_id=org, run_id=run, user_message="x")
            async with db.tx() as conn:
                first = await conn.fetchrow("SELECT seq, kind FROM run_events WHERE org_id=$1 AND run_id=$2 ORDER BY seq LIMIT 1", org, run)
            # 反向：让事件写入必败（预埋一条同 (org,run,seq=1) 的孤儿事件），run 行也不许留下
            run2 = "run_atom2"
            async with db.tx() as conn:
                await conn.execute("INSERT INTO run_events (run_id, org_id, seq, kind) VALUES ($1,$2,1,'orphan')", run2, org)
            failed = False
            try:
                await create_run(db, org_id=org, run_id=run2, user_message="x")
            except asyncpg.UniqueViolationError:
                failed = True
            async with db.tx() as conn:
                leftover = await conn.fetchval("SELECT count(*) FROM agent_runs WHERE org_id=$1 AND run_id=$2", org, run2)
            return first, failed, leftover
        finally:
            await db.close()

    first, failed, leftover = _run(go())
    assert first and (first["seq"], first["kind"]) == (1, "run.created")
    assert failed, "前提不成立：事件写入没有失败"
    assert leftover == 0, "事件没写成、run 行却留下了 —— 状态在、事件缺"


# --------------------------------------------------------------------------
# 门槛⑤：updated_at 触发器（负向：去掉触发器必红）
# --------------------------------------------------------------------------


def test_updated_at_moves_on_update_and_not_without_trigger(fresh_dsn) -> None:
    import asyncpg

    async def go():
        c = await asyncpg.connect(fresh_dsn)
        try:
            org, run = "org_ts", "run_ts"
            await c.execute("INSERT INTO agent_runs (run_id, org_id, user_message) VALUES ($1,$2,'x')", run, org)
            created = await c.fetchval("SELECT updated_at FROM agent_runs WHERE run_id=$1", run)
            await c.execute("UPDATE agent_runs SET user_message='y' WHERE run_id=$1", run)   # 不手写 updated_at
            with_trigger = await c.fetchval("SELECT updated_at FROM agent_runs WHERE run_id=$1", run)
            await c.execute("DROP TRIGGER trg_agent_runs_touch ON agent_runs")
            try:
                await c.execute("UPDATE agent_runs SET user_message='z' WHERE run_id=$1", run)
                without = await c.fetchval("SELECT updated_at FROM agent_runs WHERE run_id=$1", run)
            finally:
                await c.execute("CREATE TRIGGER trg_agent_runs_touch BEFORE UPDATE ON agent_runs "
                                "FOR EACH ROW EXECUTE FUNCTION touch_updated_at()")
            return created, with_trigger, without
        finally:
            await c.close()

    created, with_trigger, without = _run(go())
    assert with_trigger > created, "有触发器时 updated_at 没动（H14）"
    assert without == with_trigger, "负向认证失败：没有触发器 updated_at 也在动，判据量的不是触发器"


# --------------------------------------------------------------------------
# 门槛⑦：id 不可枚举
# --------------------------------------------------------------------------


def test_run_ids_are_random_not_sequential() -> None:
    from syncopate.runtime.db import new_conversation_id, new_run_id

    ids = [new_run_id() for _ in range(100)]
    assert len(set(ids)) == 100
    assert all(re.fullmatch(r"run_[0-9a-f]{12}", i) for i in ids)
    assert sorted(ids) != ids, "创建顺序 == 字典序 ⇒ 可预测递增（H18）"
    assert re.fullmatch(r"conv_[0-9a-f]{12}", new_conversation_id())
