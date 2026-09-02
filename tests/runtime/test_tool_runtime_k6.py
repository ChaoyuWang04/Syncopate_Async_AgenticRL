"""K6 门槛（27 §8）：② 注册断言（负向：写工具缺 permission/timeout/输出键必被拒；30 工具全量通过；幽灵条目必红）
· ③ 拦下也落库（四道闸各造一次违例 ⇒ tool_calls(failed, blocked_by, error_json 三字段)）· ④ 幂等闸
（同键二次调用 handler 零执行 + skipped_duplicate 行 + 返回首次结果；负向：绕过代码直插第二条 succeeded 被 UNIQUE 拒）
· ⑤ 反向污染（脏返回不进 context）· ⑥ 分诊表驱动七类 · ⑦ 账本 30/30（test_tool_parity 守）。"""
from __future__ import annotations

import asyncio
import uuid

import asyncpg
import pytest

from syncopate.runtime.action_gate import ActionGate, ToolBinding
from syncopate.runtime.db import Database, claim_run, create_run
from syncopate.runtime.gateway import DecisionContext
from syncopate.runtime.platform import PlatformError
from syncopate.runtime.tool_governance import (GOVERNANCE, NON_ALERTING_CODES, ToolGovernance,
                                               assert_governance_complete, check_output)
from syncopate.runtime.tools import WRITE_TOOLS, PermissionDenied, ToolRuntime
from syncopate.runtime.worker import audit as w_audit, emit as w_emit
from tests.runtime.test_api import _pg_available

pytestmark = pytest.mark.skipif(not _pg_available(), reason="需要 PostgreSQL：bash scripts/pg_bootstrap.sh")


def with_db(body):
    async def main():
        db = Database()
        await db.connect(max_size=4)
        try:
            return await body(db)
        finally:
            await db.close()
    return asyncio.run(main())


async def _seed(db):
    org, run = f"org_{uuid.uuid4().hex[:8]}", f"run_{uuid.uuid4().hex[:8]}"
    await create_run(db, org_id=org, run_id=run, user_message="x")
    assert await claim_run(db, worker_id="t", org_id=org, run_id=run)
    return org, run


async def _tool_calls(db, org, run):
    async with db.tx() as conn:
        return [dict(r) for r in await conn.fetch(
            "SELECT tool, status, ok, blocked_by, error_json, duplicate_of, result FROM tool_calls "
            "WHERE org_id=$1 AND run_id=$2 ORDER BY id", org, run)]


def _gate(db, org, run, bindings, *, permissions=None, amount_threshold=10 ** 9):
    async def _ob():
        return False
    return ActionGate(db, ToolRuntime(db, permissions=permissions), bindings, org_id=org, run_id=run,
                      over_budget=_ob, emit=w_emit, audit=w_audit, amount_threshold=amount_threshold)


# --------------------------------------------------------------------------
# 门槛②：注册断言
# --------------------------------------------------------------------------


def test_governance_covers_the_whole_registry_and_matches_kind() -> None:
    assert_governance_complete()
    assert len(GOVERNANCE) == 34 and len(WRITE_TOOLS) == 8       # 30 业务 + 4 个 v15 信令（契约可选）


@pytest.mark.parametrize("contract", ["v14", "v15"])
def test_runtime_imports_under_both_contracts(contract) -> None:
    """09-02 verl-22 通报的红：v15 下 REGISTRY 多四个 session.* 工具，治理表没登记 ⇒ 生产进程导入即炸。
    用子进程按两种契约各导入一次 decider（链到 tools.py 的导入时断言）。"""
    import os
    import subprocess
    import sys
    env = {**os.environ, "SYNCOPATE_CONTRACT": contract}
    r = subprocess.run([sys.executable, "-c", "import syncopate.runtime.decider, syncopate.runtime.worker; print('ok')"],
                       env=env, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0 and "ok" in r.stdout, r.stderr[-800:]


@pytest.mark.parametrize("bad", [
    dict(side_effect=True, timeout_seconds=30, expected_max_ms=5000, permission=None,
         output_required_keys=("x",), audit_required=True),                       # 无权限
    dict(side_effect=True, timeout_seconds=0, expected_max_ms=5000, permission="p",
         output_required_keys=("x",), audit_required=True),                       # 无 timeout
    dict(side_effect=True, timeout_seconds=30, expected_max_ms=5000, permission="p",
         output_required_keys=(), audit_required=True),                           # 无输出键也无 todo
    dict(side_effect=True, timeout_seconds=30, expected_max_ms=5000, permission="p",
         output_required_keys=("x",), audit_required=False),                      # 不审计
    dict(side_effect=True, timeout_seconds=30, expected_max_ms=5000, permission="p",
         output_required_keys=("x",), audit_required=True,
         retryable_errors=frozenset({"500"})),                                    # 写工具乱设可重试
])
def test_negative_registering_a_side_effect_tool_without_governance_is_rejected(bad) -> None:
    with pytest.raises(ValueError):
        ToolGovernance(**bad)


def test_negative_ghost_or_missing_entry_fails_completeness() -> None:
    ghost = dict(GOVERNANCE)
    ghost["tool.that_does_not_exist"] = GOVERNANCE["campaign.list"]
    with pytest.raises(AssertionError, match="登记了但沙盒没有"):
        assert_governance_complete(ghost)
    missing = dict(GOVERNANCE)
    missing.pop("campaign.update_budget")
    with pytest.raises(AssertionError, match="沙盒有而未登记"):
        assert_governance_complete(missing)
    flipped = dict(GOVERNANCE)
    flipped["campaign.list"] = GOVERNANCE["campaign.update_budget"]      # 把读工具标成写
    with pytest.raises(AssertionError, match="side_effect"):
        assert_governance_complete(flipped)


# --------------------------------------------------------------------------
# 门槛③：拦下也落库
# --------------------------------------------------------------------------


def test_every_gate_refusal_leaves_a_tool_calls_row_with_error_json() -> None:
    async def go(db):
        org, run = await _seed(db)

        async def ok(**kw):
            return {"campaign_id": "CMP_1", "new_budget": kw.get("new_budget", 0)}

        gate = _gate(db, org, run, {"campaign.update_budget": ToolBinding(ok),
                                    "campaign.get_metrics": ToolBinding(ok)}, permissions=set())
        ctx = DecisionContext()
        await gate.invoke(tool="no.such_tool", arguments={}, ctx=ctx, param_source="model")           # ② 找定义
        await gate.invoke(tool="campaign.get_metrics", arguments={}, ctx=ctx, param_source="model")   # ② schema 缺必填
        gate.skip_triggers = True                        # 网关触发器（审批）在权限之前；这里只考权限与 D 档
        await gate.invoke(tool="campaign.update_budget",                                             # ③ 权限（permissions 空）
                          arguments={"campaign_id": "CMP_1", "new_budget": 1, "client_request_id": "r1"},
                          ctx=ctx, param_source="user")
        await gate.invoke(tool="campaign.update_budget",                                             # D 档：金额 ≥ 永不自动线
                          arguments={"campaign_id": "CMP_1", "new_budget": 10 ** 9, "client_request_id": "r2"},
                          ctx=ctx, param_source="user")
        return await _tool_calls(db, org, run)

    rows = with_db(go)
    by = {r["blocked_by"]: r for r in rows if r["blocked_by"]}
    assert set(by) >= {"unknown_tool", "validation_failed", "permission_denied", "tier_d_refused"}, sorted(by)
    for name, r in by.items():
        assert r["status"] == "failed" and r["ok"] is False
        ej = r["error_json"]
        assert set(ej) >= {"code", "message", "retryable"} and ej["code"] == name, ej
        assert ej["alert"] is False, f"{name} 是防线生效，不该告警"


# --------------------------------------------------------------------------
# 门槛④：幂等闸
# --------------------------------------------------------------------------


def test_same_key_second_call_skips_handler_and_returns_first_result() -> None:
    async def go(db):
        org, run = await _seed(db)
        executed = []

        async def handler(**kw):
            executed.append(1)
            return {"campaign_id": kw["campaign_id"], "new_budget": kw["new_budget"], "n": len(executed)}

        gate = _gate(db, org, run, {"campaign.update_budget": ToolBinding(handler)},
                     permissions={"budget:write"})
        gate.skip_triggers = True
        args = {"campaign_id": "CMP_1", "new_budget": 500, "client_request_id": "same-key"}
        a = await gate.invoke(tool="campaign.update_budget", arguments=args, ctx=DecisionContext(), param_source="user")
        b = await gate.invoke(tool="campaign.update_budget", arguments=args, ctx=DecisionContext(), param_source="user")
        rows = await _tool_calls(db, org, run)
        # 负向：绕过代码判重直接 INSERT 第二条 succeeded（同键、非 skipped）必被 UNIQUE 拒
        async with db.tx() as conn:
            key = await conn.fetchval("SELECT external_idempotency_key FROM tool_calls WHERE org_id=$1 AND run_id=$2 "
                                      "AND status='succeeded' LIMIT 1", org, run)
            try:
                await conn.execute("INSERT INTO tool_calls (run_id, org_id, tool, external_idempotency_key, status, side_effect, ok) "
                                   "VALUES ($1,$2,'campaign.update_budget',$3,'succeeded',TRUE,TRUE)", run, org, key)
                dup_rejected = False
            except asyncpg.UniqueViolationError:
                dup_rejected = True
        return len(executed), a, b, rows, dup_rejected

    n, a, b, rows, dup_rejected = with_db(go)
    assert n == 1, "同键第二次调用 handler 又执行了"
    assert a.status == "ok" and b.status == "ok" and b.replayed and b.observation == a.observation
    st = [r["status"] for r in rows]
    assert st == ["succeeded", "skipped_duplicate"], st
    assert rows[1]["duplicate_of"] is not None
    assert dup_rejected, "负向认证失败：绕过代码判重的第二条 succeeded 没被 UNIQUE 拒"


# --------------------------------------------------------------------------
# 门槛⑤：反向污染
# --------------------------------------------------------------------------


def test_dirty_output_does_not_enter_context() -> None:
    async def go(db):
        org, run = await _seed(db)

        async def dirty(**kw):
            return {"unexpected": "shape"}          # 缺 campaign_id

        async def garbage(**kw):
            return "not-a-dict"

        gate = _gate(db, org, run, {"campaign.get_metrics": ToolBinding(dirty),
                                    "metrics.get_freshness": ToolBinding(garbage)})
        a = await gate.invoke(tool="campaign.get_metrics", arguments={"campaign_id": "CMP_1"},
                              ctx=DecisionContext(), param_source="model")
        b = await gate.invoke(tool="metrics.get_freshness", arguments={"campaign_id": "CMP_1"},
                              ctx=DecisionContext(), param_source="model")
        return a, b, await _tool_calls(db, org, run)

    a, b, rows = with_db(go)
    assert a.status == "failed" and "output_schema_violation" in (a.error or "")
    assert "unexpected" not in str(a.observation), "脏返回进了 context"
    assert b.status == "failed" and "not-a-dict" not in str(b.observation)
    assert all(r["error_json"]["code"] == "output_schema_violation" for r in rows)


def test_check_output_table() -> None:
    assert check_output("campaign.get_metrics", {"campaign_id": "x"}) is None
    assert check_output("campaign.get_metrics", {"foo": 1})
    assert check_output("campaign.list", "x")


# --------------------------------------------------------------------------
# 门槛⑥：分诊表驱动七类
# --------------------------------------------------------------------------


def _raiser(exc):
    async def f(**kw):
        raise exc
    return f


@pytest.mark.parametrize("tool,binding,expect_code,expect_status,expect_retryable,attempts_min", [
    ("campaign.get_metrics", _raiser(PlatformError("rl", code="429", retriable=True)), "429", "failed", True, 3),
    ("campaign.get_metrics", _raiser(PlatformError("bad", code="400", retriable=False)), "400", "failed", False, 1),
    ("campaign.update_budget", _raiser(PlatformError("bad", code="100", retriable=False)), "100", "failed", False, 1),
    ("campaign.update_budget", _raiser(RuntimeError("sql boom")), "tool_crashed", "failed", False, 1),
])
def test_triage_table(tool, binding, expect_code, expect_status, expect_retryable, attempts_min) -> None:
    async def go(db):
        org, run = await _seed(db)
        gate = _gate(db, org, run, {tool: ToolBinding(binding)}, permissions={"budget:write"})
        gate.skip_triggers = True
        args = {"campaign_id": "CMP_1"} if tool == "campaign.get_metrics" else \
            {"campaign_id": "CMP_1", "new_budget": 1, "client_request_id": uuid.uuid4().hex}
        out = await gate.invoke(tool=tool, arguments=args, ctx=DecisionContext(), param_source="user")
        return out, await _tool_calls(db, org, run)

    out, rows = with_db(go)
    assert out.status == "failed"
    assert rows and rows[-1]["status"] == expect_status
    ej = rows[-1]["error_json"]
    assert ej["code"] == expect_code and ej["retryable"] is expect_retryable, ej


def test_timeout_on_write_is_response_lost_not_failed() -> None:
    async def go(db):
        org, run = await _seed(db)

        async def slow(**kw):
            await asyncio.sleep(5)
            return {}

        gate = _gate(db, org, run, {"campaign.update_budget": ToolBinding(slow)}, permissions={"budget:write"})
        gate.skip_triggers = True
        gate.tools.timeout_seconds = 0.05          # 兼容旧字段；真正生效的是治理表 ⇒ 用 monkeypatch 改治理
        from syncopate.runtime import tool_governance as tg
        orig = tg.GOVERNANCE["campaign.update_budget"]
        tg.GOVERNANCE["campaign.update_budget"] = ToolGovernance(
            side_effect=True, timeout_seconds=0.05, expected_max_ms=100, permission="budget:write",
            retryable_errors=orig.retryable_errors, output_required_keys=orig.output_required_keys, audit_required=True)
        try:
            out = await gate.invoke(tool="campaign.update_budget",
                                    arguments={"campaign_id": "CMP_1", "new_budget": 1, "client_request_id": "t-1"},
                                    ctx=DecisionContext(), param_source="user")
        finally:
            tg.GOVERNANCE["campaign.update_budget"] = orig
        return out, await _tool_calls(db, org, run)

    out, rows = with_db(go)
    assert rows[-1]["status"] == "response_lost", rows
    assert rows[-1]["error_json"]["code"] == "client_timeout" and rows[-1]["error_json"]["retryable"] is False
    assert "response_lost" in (out.error or "")


def test_permission_denial_is_failed_but_not_alerting() -> None:
    assert "permission_denied" in NON_ALERTING_CODES and "tool_crashed" not in NON_ALERTING_CODES
