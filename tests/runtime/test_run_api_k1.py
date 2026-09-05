"""K1 门槛（27 §3）：幂等三态（含并发双击）· 终态守卫表驱动 · 五入口跨租户 404 · SQL 无一缺 org_id ·
resume 不变量（回 queued、lease 空、随后被 worker 领取跑完）· 错误信封 · POST /runs P95 < 300ms ·
协作式取消真的被 worker 消费（安全点=工具调用前）· trace 独立角色。"""
from __future__ import annotations

import asyncio
import re
import time
import uuid
from pathlib import Path

import pytest

from syncopate.runtime.api import ERROR_CODES, create_app
from syncopate.runtime.db import claim_run
from syncopate.runtime.platform import FakeAdPlatform
from syncopate.runtime.worker import Worker, WorkerConfig
from tests.runtime.test_api import ACME, GLOBEX, Client, _key, _pg_available

pytestmark = pytest.mark.skipif(not _pg_available(), reason="需要 PostgreSQL：bash scripts/serving/pg_bootstrap.sh")

TRACE = {"Authorization": "Bearer dev-token-acme-trace"}
TERMINAL = ("succeeded", "failed", "cancelled")


@pytest.fixture()
def client():
    c = Client(create_app())
    yield c
    c.close()


def _db(client):
    return client.app.state.db


def _sql(client, sql: str, *args):
    async def go():
        async with _db(client).tx() as conn:
            return await conn.fetch(sql, *args)
    return client.loop.run_until_complete(go())


def _new_run(client, headers=ACME, msg="x") -> str:
    r = client.post("/runs", json={"user_message": msg}, headers=headers)
    assert r.status_code == 201, r.text
    return r.json()["run_id"]


def _drain(client, org: str, keep: str) -> None:
    """测试卫生：把本 org 队列里其他测试留下的活清掉，worker/claim 才能确定拿到 `keep`。
    （与 test_worker._drain 同族；绕状态机只在测试里允许。）"""
    _sql(client, "UPDATE agent_runs SET status='cancelled', lease_owner=NULL, lease_expires_at=NULL "
                 " WHERE org_id=$1 AND run_id<>$2 AND status IN ('queued','running','waiting_for_user')", org, keep)


def _set_status(client, org: str, run_id: str, st: str, token: str | None = None) -> None:
    _sql(client, "UPDATE agent_runs SET status=$3, resume_token=$4 WHERE org_id=$1 AND run_id=$2",
         org, run_id, st, token)


# --------------------------------------------------------------------------
# 门槛①：幂等三态 + 并发双击
# --------------------------------------------------------------------------


def test_idempotency_three_states(client) -> None:
    h = {**ACME, "Idempotency-Key": _key()}
    a = client.post("/runs", json={"user_message": "加预算"}, headers=h)
    b = client.post("/runs", json={"user_message": "加预算"}, headers=h)
    c = client.post("/runs", json={"user_message": "减预算"}, headers=h)   # 同 key 不同 input
    assert (a.status_code, b.status_code, c.status_code) == (201, 200, 409)
    assert c.json()["error"]["code"] == "IDEMPOTENCY_CONFLICT"
    assert a.json()["run_id"] == b.json()["run_id"]


def test_idempotency_concurrent_double_click_exactly_one_created(client) -> None:
    """并发同 key 两请求：恰好一个 201 一个 200，库里恰好一行（H01：救你的是约束不是查询）。"""
    h = {**ACME, "Idempotency-Key": _key()}

    async def go():
        return await asyncio.gather(
            client._http.post("/runs", json={"user_message": "x"}, headers=h),
            client._http.post("/runs", json={"user_message": "x"}, headers=h))

    a, b = client.loop.run_until_complete(go())
    assert sorted([a.status_code, b.status_code]) == [200, 201]
    assert a.json()["run_id"] == b.json()["run_id"]
    rows = _sql(client, "SELECT count(*) AS n FROM agent_runs WHERE org_id='org_acme' AND idempotency_key=$1",
                h["Idempotency-Key"])
    assert rows[0]["n"] == 1


def test_idempotency_key_too_short_is_rejected(client) -> None:
    r = client.post("/runs", json={"user_message": "x"}, headers={**ACME, "Idempotency-Key": "abc"})
    assert r.status_code == 422 and r.json()["error"]["code"] == "IDEMPOTENCY_KEY_TOO_SHORT"


# --------------------------------------------------------------------------
# 门槛②：六状态 × cancel/resume 全组合
# --------------------------------------------------------------------------


@pytest.mark.parametrize("st,expect_code", [
    ("queued", 200), ("waiting_for_user", 200), ("running", 202),
    ("succeeded", 409), ("failed", 409), ("cancelled", 409)])
def test_cancel_matrix(client, st, expect_code) -> None:
    run_id = _new_run(client)
    _set_status(client, "org_acme", run_id, st, "tok-12345678")
    r = client.post(f"/runs/{run_id}/cancel", json={"reason": "test"}, headers=ACME)
    assert r.status_code == expect_code, r.text
    if expect_code == 409:
        assert r.json()["error"]["code"] == "RUN_ALREADY_TERMINAL"
    elif expect_code == 202:
        # running：只登记意图，状态不变（意图 ≠ 状态，H104）
        assert r.json()["status"] == "running" and r.json()["cancel_requested"] is True
        kinds = [x["kind"] for x in _sql(client, "SELECT kind FROM run_events WHERE run_id=$1 ORDER BY seq", run_id)]
        assert "run.cancel_requested" in kinds and "run.cancelled" not in kinds
    else:
        assert r.json()["status"] == "cancelled"
        kinds = [x["kind"] for x in _sql(client, "SELECT kind FROM run_events WHERE run_id=$1 ORDER BY seq", run_id)]
        assert kinds[-1] == "run.cancelled"


@pytest.mark.parametrize("st,expect_code,code", [
    ("waiting_for_user", 200, None),
    ("queued", 409, "RUN_NOT_WAITING_FOR_USER"), ("running", 409, "RUN_NOT_WAITING_FOR_USER"),
    ("succeeded", 409, "RUN_ALREADY_TERMINAL"), ("failed", 409, "RUN_ALREADY_TERMINAL"),
    ("cancelled", 409, "RUN_ALREADY_TERMINAL")])
def test_resume_matrix(client, st, expect_code, code) -> None:
    run_id = _new_run(client)
    _set_status(client, "org_acme", run_id, st, "tok-12345678")
    r = client.post(f"/runs/{run_id}/resume", json={"resume_token": "tok-12345678"}, headers=ACME)
    assert r.status_code == expect_code, r.text
    if code:
        assert r.json()["error"]["code"] == code


def test_resume_with_wrong_token_is_403_and_token_is_single_use(client) -> None:
    run_id = _new_run(client)
    _set_status(client, "org_acme", run_id, "waiting_for_user", "tok-12345678")
    bad = client.post(f"/runs/{run_id}/resume", json={"resume_token": "tok-wrong-00"}, headers=ACME)
    assert bad.status_code == 403 and bad.json()["error"]["code"] == "INVALID_RESUME_TOKEN"
    # GET 在 waiting 时给出 token（org 内可见）
    assert client.get(f"/runs/{run_id}", headers=ACME).json()["resume_token"] == "tok-12345678"
    ok = client.post(f"/runs/{run_id}/resume", json={"resume_token": "tok-12345678"}, headers=ACME)
    assert ok.status_code == 200 and ok.json()["status"] == "queued"
    row = _sql(client, "SELECT resume_token, lease_owner, lease_expires_at FROM agent_runs WHERE run_id=$1", run_id)[0]
    assert row["resume_token"] is None and row["lease_owner"] is None and row["lease_expires_at"] is None


# --------------------------------------------------------------------------
# 门槛③：五入口跨租户 404 + SQL 无一缺 org_id
# --------------------------------------------------------------------------


def test_all_five_entries_are_404_across_tenants(client) -> None:
    run_id = _new_run(client)
    assert client.get(f"/runs/{run_id}", headers=GLOBEX).status_code == 404
    assert client.post(f"/runs/{run_id}/cancel", json={}, headers=GLOBEX).status_code == 404
    assert client.post(f"/runs/{run_id}/resume", json={"resume_token": "tok-12345678"}, headers=GLOBEX).status_code == 404
    assert client.get(f"/runs/{run_id}/events", headers=GLOBEX).status_code == 404
    assert client.get(f"/runs/{run_id}/trace", headers=GLOBEX).status_code == 404


def test_every_agent_runs_query_in_api_carries_org_id() -> None:
    """grep 判据：api.py 里每条读 agent_runs 的 SQL 都带 org_id（隔离进 SQL，不是查出来再判）。"""
    src = Path(__file__).resolve().parents[2] / "syncopate" / "runtime" / "api.py"
    lines = src.read_text(encoding="utf-8").splitlines()
    misses = []
    for i, line in enumerate(lines):
        if "FROM agent_runs" in line:
            window = " ".join(lines[max(0, i - 2): i + 3])
            if not re.search(r"org_id\s*=\s*\$\d", window):
                misses.append(i + 1)
    assert not misses, f"api.py 这些行的 agent_runs 查询没带 org_id: {misses}"


# --------------------------------------------------------------------------
# 门槛④：resume 不变量 —— 回 queued、lease 空、随后被 worker 正常领取跑完
# --------------------------------------------------------------------------


def test_resumed_run_is_claimed_and_finished_by_worker(client) -> None:
    run_id = _new_run(client)
    _drain(client, "org_acme", keep=run_id)
    _set_status(client, "org_acme", run_id, "waiting_for_user", "tok-12345678")
    assert client.post(f"/runs/{run_id}/resume", json={"resume_token": "tok-12345678"}, headers=ACME).status_code == 200

    async def go():
        db = _db(client)
        w = Worker(db, FakeAdPlatform(), WorkerConfig(org_id="org_acme", amount_threshold=10 ** 9,
                                                        daily_cost_cap_micros=10 ** 12))
        for _ in range(20):                       # 队列里可能有别的测试留下的 run，跑到自己这条为止
            got = await w.run_once()
            if got == run_id or got is None:
                break
        async with db.tx() as conn:
            row = await conn.fetchrow("SELECT status, attempts FROM agent_runs WHERE org_id='org_acme' AND run_id=$1", run_id)
            kinds = [x["kind"] for x in await conn.fetch(
                "SELECT kind FROM run_events WHERE org_id='org_acme' AND run_id=$1 ORDER BY seq", run_id)]
        return row["status"], row["attempts"], kinds

    status, attempts, kinds = client.loop.run_until_complete(go())
    # "正常领取跑完"= 被 claim（attempts 0→1）、run.resumed 之后有 run.started、不再停在 queued。
    # 假平台默认剧本会在写动作前再开一张审批单 ⇒ 合法地回到 waiting_for_user，同样算跑完这一轮。
    assert attempts == 1, f"恢复后没被领取，attempts={attempts}"
    assert kinds.index("run.resumed") < kinds.index("run.started", kinds.index("run.resumed"))
    assert status in TERMINAL + ("waiting_for_user",) and status != "queued", f"实得 {status}（事件 {kinds}）"


# --------------------------------------------------------------------------
# 门槛⑤：错误信封
# --------------------------------------------------------------------------


def test_every_non_2xx_uses_the_error_envelope(client) -> None:
    run_id = _new_run(client)
    _set_status(client, "org_acme", run_id, "succeeded")
    samples = [
        client.post("/runs", json={"user_message": "x"}),                                  # 401
        client.get("/runs/run_doesnotexist", headers=ACME),                                # 404
        client.post(f"/runs/{run_id}/cancel", json={}, headers=ACME),                      # 409
        client.post("/runs", json={"user_message": ""}, headers=ACME),                     # 422 校验
        client.post("/runs", json={"user_message": "x"}, headers={**ACME, "Idempotency-Key": "ab"}),  # 422 业务码
        client.get("/no-such-route", headers=ACME),                                        # Starlette 404
    ]
    for r in samples:
        assert r.status_code >= 400
        body = r.json()
        assert set(body) == {"error"}, body
        err = body["error"]
        assert err["code"] in ERROR_CODES, err
        assert err["request_id"], err
        assert isinstance(err["message"], str)


def test_request_id_is_echoed_from_header(client) -> None:
    r = client.get("/runs/run_doesnotexist", headers={**ACME, "X-Request-ID": "req-abc-123"})
    assert r.json()["error"]["request_id"] == "req-abc-123"


# --------------------------------------------------------------------------
# 门槛⑥：POST /runs P95 < 300ms（100 次采样，本机 PG）
# --------------------------------------------------------------------------


def test_post_runs_p95_under_300ms(client) -> None:
    samples = []
    for _ in range(100):
        t0 = time.perf_counter()
        assert client.post("/runs", json={"user_message": "p95"}, headers=ACME).status_code == 201
        samples.append(time.perf_counter() - t0)
    samples.sort()
    p95 = samples[int(0.95 * len(samples)) - 1]
    print(f"[k1-p95] POST /runs p50={samples[49]*1000:.1f}ms p95={p95*1000:.1f}ms")
    assert p95 < 0.3, f"P95 {p95*1000:.0f}ms ≥ 300ms"


# --------------------------------------------------------------------------
# 协作式取消真的被消费：running 的 run 在下一个安全点停下，终态 cancelled + 事件在库
# --------------------------------------------------------------------------


def test_cancel_request_is_honoured_by_worker_at_safety_point(client) -> None:
    run_id = _new_run(client)
    _drain(client, "org_acme", keep=run_id)

    async def go():
        db = _db(client)
        # 让它成为 running（lease 已过期 ⇒ 下面的 worker 能接管；模拟"上一个 worker 正在跑"）
        claimed = None
        for _ in range(20):
            claimed = await claim_run(db, worker_id="stale", lease_seconds=-1, org_id="org_acme")
            if claimed is None or claimed["run_id"] == run_id:
                break
        assert claimed and claimed["run_id"] == run_id, "前提不成立：没抢到自己那条"
        r = await client._http.post(f"/runs/{run_id}/cancel", json={"reason": "用户反悔"}, headers=ACME)
        assert r.status_code == 202, r.text
        w = Worker(db, FakeAdPlatform(), WorkerConfig(org_id="org_acme", amount_threshold=10 ** 9,
                                                        daily_cost_cap_micros=10 ** 12))
        for _ in range(20):
            got = await w.run_once()
            if got == run_id or got is None:
                break
        async with db.tx() as conn:
            st = await conn.fetchval("SELECT status FROM agent_runs WHERE run_id=$1", run_id)
            kinds = [x["kind"] for x in await conn.fetch("SELECT kind FROM run_events WHERE run_id=$1 ORDER BY seq", run_id)]
            writes = await conn.fetchval("SELECT count(*) FROM tool_calls WHERE run_id=$1 AND tool='campaign.update_budget'", run_id)
        return st, kinds, writes

    st, kinds, writes = client.loop.run_until_complete(go())
    assert st == "cancelled", f"worker 没在安全点兑现取消，实得 {st}（事件 {kinds}）"
    assert kinds[-1] == "run.cancelled" and "run.cancel_requested" in kinds
    assert writes == 0, "取消后写类工具还执行了"


# --------------------------------------------------------------------------
# K1-8：trace 独立角色
# --------------------------------------------------------------------------


def test_trace_requires_role_and_aggregates_eight_tables(client) -> None:
    run_id = _new_run(client)
    forbidden = client.get(f"/runs/{run_id}/trace", headers=ACME)
    assert forbidden.status_code == 403 and forbidden.json()["error"]["code"] == "TRACE_FORBIDDEN"
    ok = client.get(f"/runs/{run_id}/trace", headers=TRACE)
    assert ok.status_code == 200
    body = ok.json()
    assert set(body) == {"run", "run_events", "agent_steps", "model_calls", "tool_calls",
                         "checkpoints", "usage_records", "audit_logs", "approval_cases"}
    assert body["run"]["run_id"] == run_id and body["run_events"][0]["kind"] == "run.created"
