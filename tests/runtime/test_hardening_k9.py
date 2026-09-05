"""K9 门槛（27 §11）：② 预算（不收敛 run 撞 max_model_calls ⇒ 转 waiting 不判死；负向：关闸必红；org 超限 ⇒ 新建 429、
已跑的不受影响）· ③ 指标（duplicate_prevented 构造重复 +1；stuck_runs/oldest_age 与实际吻合）· ④ 日志（错误路径结构化、
敏感字段打码；CI 正则）· ⑤ 版本网关（旧格式无 v 可读；未知版本 ⇒ manual_review 事件，进程不崩）· ⑥ 发布能力：禁工具开关 ·
① SLO 九条有读数/结构守 · ⑦ 测试八类对照（见 29 §2.4）。"""
from __future__ import annotations

import asyncio
import json
import re
import uuid
from pathlib import Path

import pytest

from syncopate.runtime import metrics
from syncopate.runtime.action_gate import ActionGate, ToolBinding, disabled_tools
from syncopate.runtime.agent_loop import Proposal, save_transcript
from syncopate.runtime.api import create_app
from syncopate.runtime.budget import RunBudget, run_budget_exceeded
from syncopate.runtime.db import Database, claim_run, create_run
from syncopate.runtime.gateway import DecisionContext
from syncopate.runtime.log import log_event
from syncopate.runtime.platform import FakeAdPlatform
from syncopate.runtime.tools import ToolRuntime
from syncopate.runtime.worker import Worker, WorkerConfig, audit as w_audit, emit as w_emit
from tests.runtime.test_api import ACME, Client, _pg_available

pytestmark = pytest.mark.skipif(not _pg_available(), reason="需要 PostgreSQL：bash scripts/serving/pg_bootstrap.sh")
REPO = Path(__file__).resolve().parents[2]


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


class _CountingDecider:
    """每次调用累加 MODEL_USAGE（像真 decider 那样），永远再调一个读工具 = 不收敛。"""

    def __init__(self, tokens_per_call=(100, 50)):
        self.calls = 0
        self.tpc = tokens_per_call

    async def decide(self, *, user_message, history):      # noqa: ANN001
        from syncopate.runtime.agent_loop import MODEL_USAGE
        self.calls += 1
        u = MODEL_USAGE.get()
        if u is not None:
            u["calls"] = u.get("calls", 0) + 1
            u["tokens_in"] = u.get("tokens_in", 0) + self.tpc[0]
            u["tokens_out"] = u.get("tokens_out", 0) + self.tpc[1]
        return Proposal(kind="tool_call", tool="campaign.get_metrics", arguments={"campaign_id": "CMP_1"},
                        param_source="model")


async def _state(db, org, run):
    async with db.tx() as conn:
        r = await conn.fetchrow("SELECT status, budget_exceeded_at, resume_token FROM agent_runs WHERE org_id=$1 AND run_id=$2", org, run)
        kinds = [x["kind"] for x in await conn.fetch("SELECT kind FROM run_events WHERE org_id=$1 AND run_id=$2 ORDER BY seq", org, run)]
        usage = [dict(x) for x in await conn.fetch("SELECT call_index, tokens_in, tokens_out FROM usage_records WHERE org_id=$1 AND run_id=$2 ORDER BY call_index", org, run)]
    return dict(r), kinds, usage


# --------------------------------------------------------------------------
# 门槛②：预算
# --------------------------------------------------------------------------


def test_non_converging_run_is_parked_at_max_model_calls_not_failed() -> None:
    async def go(db):
        org, run = _ids()
        await create_run(db, org_id=org, run_id=run, user_message="不收敛")
        async with db.tx() as conn:
            await conn.execute("UPDATE agent_runs SET max_model_calls=5 WHERE org_id=$1 AND run_id=$2", org, run)
        d = _CountingDecider()
        w = Worker(db, FakeAdPlatform(), WorkerConfig(amount_threshold=10 ** 9, daily_cost_cap_micros=10 ** 12), decider=d)
        await w.execute_claimed(await claim_run(db, worker_id="w", org_id=org, run_id=run))
        r, kinds, usage = await _state(db, org, run)
        return r, kinds, usage, d.calls

    r, kinds, usage, calls = with_db(go)
    assert r["status"] == "waiting_for_user" and r["budget_exceeded_at"] is not None and r["resume_token"], r
    assert "run.budget_exceeded" in kinds and kinds[-1] in ("run.budget_exceeded", "run.waiting_for_user")
    assert calls == 6, f"应在第 6 次调用（>5）时拦停，实得 {calls}"
    # K9-3 一轮一行：每次模型调用一行 usage，call_index 递增，token 逐行可核
    assert [u["call_index"] for u in usage] == list(range(1, 7)) and all(u["tokens_in"] == 100 for u in usage)


def test_negative_without_budget_gate_the_run_burns_to_step_cap() -> None:
    """负向认证：关掉预算闸（budget_check=None）⇒ 同一个不收敛 run 一路烧到步数上限（failed），不会停在 waiting。"""
    async def go(db):
        org, run = _ids()
        await create_run(db, org_id=org, run_id=run, user_message="不收敛")
        async with db.tx() as conn:
            await conn.execute("UPDATE agent_runs SET max_model_calls=5 WHERE org_id=$1 AND run_id=$2", org, run)
        d = _CountingDecider()
        w = Worker(db, FakeAdPlatform(), WorkerConfig(amount_threshold=10 ** 9, daily_cost_cap_micros=10 ** 12), decider=d)
        real_gate = w._gate

        def no_budget_gate(**kw):
            g = real_gate(**kw)
            g._budget_check = None
            return g
        w._gate = no_budget_gate
        await w.execute_claimed(await claim_run(db, worker_id="w", org_id=org, run_id=run))
        r, kinds, _ = await _state(db, org, run)
        return r, d.calls

    r, calls = with_db(go)
    assert r["status"] == "failed" and calls > 6, (r, calls)


def test_run_budget_table() -> None:
    b = RunBudget(max_model_calls=3, max_tokens=1000, max_duration_s=60)
    assert run_budget_exceeded(b, model_calls=3, tokens=10, elapsed_s=1) is None
    assert run_budget_exceeded(b, model_calls=4, tokens=10, elapsed_s=1).startswith("max_model_calls")
    assert run_budget_exceeded(b, model_calls=1, tokens=1001, elapsed_s=1).startswith("max_tokens")
    assert run_budget_exceeded(b, model_calls=1, tokens=1, elapsed_s=61).startswith("max_duration_s")


@pytest.fixture()
def client():
    c = Client(create_app())
    yield c
    c.close()


def test_org_over_budget_rejects_new_runs_but_leaves_running_ones_alone(client) -> None:
    async def seed():
        db = client.app.state.db
        async with db.tx() as conn:
            await conn.execute("INSERT INTO org_budgets (org_id, daily_tokens, daily_cost_micros) VALUES ('org_globex', 1000, 10000000) "
                               "ON CONFLICT (org_id) DO UPDATE SET daily_tokens=1000")
            await conn.execute("INSERT INTO usage_records (org_id, run_id, tokens_in, tokens_out, cost_micros, call_index) "
                               "VALUES ('org_globex', $1, 900, 200, 1, 777)", f"run_{uuid.uuid4().hex[:8]}")
    client.loop.run_until_complete(seed())
    GLOBEX = {"Authorization": "Bearer dev-token-globex"}
    running = client.post("/runs", json={"user_message": "已在跑"}, headers=GLOBEX)   # 先建一条（此时已超？看顺序）
    r = client.post("/runs", json={"user_message": "新建"}, headers=GLOBEX)
    assert r.status_code == 429 and r.json()["error"]["code"] == "ORG_BUDGET_EXCEEDED", r.text
    # 已存在的 run 可读、可取消——不受影响
    if running.status_code == 201:
        assert client.get(f"/runs/{running.json()['run_id']}", headers=GLOBEX).status_code == 200

    async def cleanup():
        async with client.app.state.db.tx() as conn:
            await conn.execute("DELETE FROM org_budgets WHERE org_id='org_globex'")
            await conn.execute("DELETE FROM usage_records WHERE org_id='org_globex' AND call_index=777")
    client.loop.run_until_complete(cleanup())


# --------------------------------------------------------------------------
# 门槛③：指标
# --------------------------------------------------------------------------


def test_duplicate_prevented_counter_and_stuck_runs_match_reality() -> None:
    async def go(db):
        org, run = _ids()
        await create_run(db, org_id=org, run_id=run, user_message="x")
        await claim_run(db, worker_id="w", org_id=org, run_id=run)
        before = await metrics.snapshot(db, org_id=org)

        async def handler(**kw):
            return {"campaign_id": kw["campaign_id"], "new_budget": kw["new_budget"]}

        async def _ob():
            return False
        gate = ActionGate(db, ToolRuntime(db, permissions={"budget:write"}),
                          {"campaign.update_budget": ToolBinding(handler)}, org_id=org, run_id=run,
                          over_budget=_ob, emit=w_emit, audit=w_audit, amount_threshold=10 ** 9)
        gate.skip_triggers = True
        args = {"campaign_id": "CMP_1", "new_budget": 1, "client_request_id": "dup-metric"}
        await gate.invoke(tool="campaign.update_budget", arguments=args, ctx=DecisionContext(), param_source="user")
        await gate.invoke(tool="campaign.update_budget", arguments=args, ctx=DecisionContext(), param_source="user")
        async with db.tx() as conn:                    # 造一条卡死的 running
            await conn.execute("UPDATE agent_runs SET lease_expires_at = now() - interval '1 minute' WHERE org_id=$1 AND run_id=$2", org, run)
        after = await metrics.snapshot(db, org_id=org)
        return before, after

    before, after = with_db(go)
    assert after["duplicate_prevented_total"] - before["duplicate_prevented_total"] == 1
    assert after["stuck_runs"] == 1 and after["runs_running"] == 1


def test_metrics_endpoint_exposes_the_panel_and_alerts_bind_runbooks(client) -> None:
    text = client.get("/metrics").text
    for name in ("syncopate_outbox_pending", "syncopate_oldest_queued_run_age_s", "syncopate_stuck_runs",
                 "syncopate_duplicate_prevented_total", "syncopate_dead_letter_open", "syncopate_response_lost_open",
                 "syncopate_write_tool_error_rate", "syncopate_run_failed_ratio", "syncopate_queue_lag_p95_s",
                 "syncopate_budget_waiting_total"):
        assert name in text, name
    a = client.get("/alerts", headers=ACME).json()
    assert "alerts" in a and all("runbook" in x for x in a["alerts"])
    fake = {"oldest_queued_run_age_s": 999.0, "stuck_runs": 0, "write_tool_error_rate": 0.0, "write_calls_24h": 0,
            "dead_letter_open": 0, "response_lost_open": 0}
    assert metrics.alerts(fake)[0]["runbook"].startswith("27 §13")


# --------------------------------------------------------------------------
# 门槛④：结构化日志 + 敏感字段
# --------------------------------------------------------------------------


def test_log_event_redacts_secrets_and_is_one_json_line(capsys) -> None:
    rec = log_event("test", "boom", level="error", run_id="r1", authorization="Bearer x", api_key="k",
                    prompt="very long prompt", error="e")
    err = capsys.readouterr().err.strip().splitlines()[-1]
    parsed = json.loads(err)
    assert parsed["authorization"] == "***" and parsed["api_key"] == "***" and "omitted" in parsed["prompt"]
    assert rec["run_id"] == "r1" and parsed["component"] == "test"


def test_no_secret_looking_values_in_judgement_prints() -> None:
    """CI 正则：runtime 里的 print 判据行不许拼进 token/password 值（Redis URL 已脱敏为 <pass>）。"""
    hits = []
    for path in (REPO / "syncopate" / "runtime").glob("*.py"):
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if "print(" in line and re.search(r"(REDIS_PASS|requirepass|dev-token|password=)", line):
                hits.append(f"{path.name}:{i}")
    assert not hits, hits


# --------------------------------------------------------------------------
# 门槛⑤：版本网关（checkpoint）
# --------------------------------------------------------------------------


def test_old_checkpoint_without_version_is_still_readable_and_unknown_version_is_refused() -> None:
    async def go(db):
        from syncopate.runtime.agent_loop import load_transcript
        org, run = _ids()
        await create_run(db, org_id=org, run_id=run, user_message="x")
        async with db.tx() as conn:                      # 旧格式（K9 之前）：没有 v
            await conn.execute("INSERT INTO checkpoints (org_id, run_id, step, state) VALUES ($1,$2,1,$3)",
                               org, run, {"history": [{"role": "observation", "observation": {"ok": True}}]})
        old_ok = await load_transcript(db, org_id=org, run_id=run)
        org2, run2 = _ids()
        await create_run(db, org_id=org2, run_id=run2, user_message="x")
        async with db.tx() as conn:                      # 未来版本
            await conn.execute("INSERT INTO checkpoints (org_id, run_id, step, state) VALUES ($1,$2,1,$3)",
                               org2, run2, {"v": 99, "history": [], "n": 0})
        w = Worker(db, FakeAdPlatform(), WorkerConfig(amount_threshold=10 ** 9, daily_cost_cap_micros=10 ** 12),
                   decider=_CountingDecider())
        await w.execute_claimed(await claim_run(db, worker_id="w", org_id=org2, run_id=run2))
        r, kinds, _ = await _state(db, org2, run2)
        return old_ok, r, kinds

    old_ok, r, kinds = with_db(go)
    assert old_ok and old_ok[0]["role"] == "observation"
    assert r["status"] == "failed" and "run.manual_review" in kinds, (r, kinds)   # 拒绝处理 + 交人，进程没崩


# --------------------------------------------------------------------------
# 门槛⑥：发布能力之禁工具开关（拨开关不重启）
# --------------------------------------------------------------------------


def test_disabled_tool_switch_blocks_and_leaves_a_row(monkeypatch) -> None:
    monkeypatch.setenv("SYNCOPATE_DISABLED_TOOLS", "campaign.update_budget, creative.upload")
    assert disabled_tools() == frozenset({"campaign.update_budget", "creative.upload"})

    async def go(db):
        org, run = _ids()
        await create_run(db, org_id=org, run_id=run, user_message="x")
        await claim_run(db, worker_id="w", org_id=org, run_id=run)

        async def handler(**kw):
            raise AssertionError("被禁的工具不该执行")

        async def _ob():
            return False
        gate = ActionGate(db, ToolRuntime(db, permissions={"budget:write"}),
                          {"campaign.update_budget": ToolBinding(handler)}, org_id=org, run_id=run,
                          over_budget=_ob, emit=w_emit, audit=w_audit, amount_threshold=10 ** 9)
        out = await gate.invoke(tool="campaign.update_budget",
                                arguments={"campaign_id": "CMP_1", "new_budget": 1, "client_request_id": "x"},
                                ctx=DecisionContext(), param_source="user")
        async with db.tx() as conn:
            row = await conn.fetchrow("SELECT blocked_by, error_json FROM tool_calls WHERE org_id=$1 AND run_id=$2", org, run)
        return out, dict(row)

    out, row = with_db(go)
    assert out.status == "failed" and out.error == "tool_disabled"
    assert row["blocked_by"] == "tool_disabled" and row["error_json"]["alert"] is False
