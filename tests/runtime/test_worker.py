"""M9.4 + M9.5 · Worker / Tool Runtime / 审批网关的验收测试。

★ 重点在三处，都是"错了不会报错、只会悄悄错"的地方：

  ① **超时的两种形态**：现象一模一样，重试安全性完全相反 —— 只能靠幂等键分辨
  ② **降级触发器全是外部信号**：一个都不许读模型置信度（§39：编造时往往最自信）
  ③ **审批单和 run 状态在同一个事务里**：分开会留下"单开了但 run 还在跑"
"""

from __future__ import annotations

import asyncio
import uuid

import pytest

from syncopate.runtime.db import Database, create_run, resume_after_approval
from syncopate.runtime.gateway import DecisionContext, evaluate_triggers
from syncopate.runtime.platform import TIMEOUT_MESSAGE, FakeAdPlatform, FaultPlan
from syncopate.runtime.tools import derive_idempotency_key
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


pytestmark = pytest.mark.skipif(
    not _pg_available(), reason="需要 PostgreSQL：bash scripts/serving/pg_bootstrap.sh")


def with_db(body):
    async def main():
        db = Database()
        await db.connect(max_size=5)
        try:
            return await body(db)
        finally:
            await db.close()
    return asyncio.run(main())


def _org() -> str:
    return f"org_{uuid.uuid4().hex[:8]}"


async def _drain(db: Database) -> None:
    """排空队列 —— 队列是全局的，前面测试遗留的活会被抢走（见 test_idempotency）。"""
    from syncopate.runtime.db import claim_run
    while await claim_run(db, worker_id="drain", lease_seconds=3600):
        pass


# --------------------------------------------------------------------------
# ① 超时的两种形态 —— M9 里最贵的一条
# --------------------------------------------------------------------------


def test_two_kinds_of_timeout_look_identical() -> None:
    """★★★ 「没发出去」和「回包丢了」现象必须**逐字相同**。

    区分得开的话，runtime 就会去读错误文本做决策 —— 而那个信号在真平台上不存在，
    学到的东西上线即失效。沙盒里 `side_effect_applied` 那条纪律的 runtime 版本。
    """
    async def go(_db):
        errors = []
        for applied in (False, True):
            p = FakeAdPlatform(faults=FaultPlan(timeout_at={1}, side_effect_applied=applied))
            try:
                await p.update_budget(campaign_id="C", new_budget=900, idempotency_key="k")
            except Exception as exc:
                errors.append(str(exc))
        return errors

    a, b = with_db(go)
    assert a == b == TIMEOUT_MESSAGE, "两种超时的错误文本不同 ⇒ 可以据此区分，那是假信号"


def test_retry_after_lost_response_does_not_double_charge() -> None:
    """★★★ 「回包丢了」那种超时，重试**不能重复扣款**。

    平台侧真的执行了（budgets 被改了），但我们没收到回包。重试带同一个幂等键 ⇒
    平台认出来 ⇒ 返回原结果而不是再执行一次。
    """
    async def go(db):
        org = _org()
        await create_run(db, org_id=org, run_id="r", user_message="x")
        p = FakeAdPlatform(faults=FaultPlan(timeout_at={1}, side_effect_applied=True))
        w = Worker(db, p)
        out = await w.tools.call(org_id=org, run_id="r", step=1,
                                 tool="campaign.update_budget",
                                 arguments={"campaign_id": "C", "new_budget": 900},
                                 invoke=p.update_budget)
        return out, p

    out, platform = with_db(go)
    assert out.ok is True, "重试应该靠幂等键拿回原结果"
    assert platform.budgets["C"] == 900
    assert out.attempts >= 2, "第一次超时后没有重试？"
    # 平台侧只认了一个键 ⇒ 只有一次真正的副作用
    assert len(platform._seen_keys) == 1


def test_idempotency_key_is_deterministic() -> None:
    """★ 重试必须推出**同一个键**，否则幂等等于没有。所以不能用 uuid4/时间戳。"""
    args = {"campaign_id": "C", "new_budget": 900}
    a = derive_idempotency_key(org_id="o", run_id="r", tool="t", arguments=args)
    b = derive_idempotency_key(org_id="o", run_id="r", tool="t", arguments=dict(reversed(list(args.items()))))
    assert a == b, "参数顺序变了键就变 ⇒ 重试会绕过幂等"


def test_different_amounts_get_different_keys() -> None:
    """同一个 run 里改成 900 和改成 1200 是两个动作，不能共用一个键。"""
    k1 = derive_idempotency_key(org_id="o", run_id="r", tool="t",
                                arguments={"new_budget": 900})
    k2 = derive_idempotency_key(org_id="o", run_id="r", tool="t",
                                arguments={"new_budget": 1200})
    assert k1 != k2


def test_retries_are_bounded_and_failure_is_reported() -> None:
    """★ 用尽重试要**如实上报失败**，不是重试到成功为止。

    "重试到成功"会让沙盒里教的"失败之后怎么办"变成死代码。
    """
    async def go(db):
        org = _org()
        await create_run(db, org_id=org, run_id="r", user_message="x")
        p = FakeAdPlatform(faults=FaultPlan(rate_limit_at={1, 2, 3, 4, 5}))
        w = Worker(db, p)
        return await w.tools.call(org_id=org, run_id="r", step=1,
                                  tool="campaign.update_budget",
                                  arguments={"campaign_id": "C", "new_budget": 900},
                                  invoke=p.update_budget)

    out = with_db(go)
    assert out.ok is False
    assert out.attempts == 3, f"重试次数没有上限？attempts={out.attempts}"
    assert "rate_limited" in (out.error or "")


def test_non_retriable_error_is_not_retried() -> None:
    """平台说不可重试就别重试 —— 重试只会浪费配额并放大故障。"""
    from syncopate.runtime.platform import PlatformError

    async def go(db):
        org = _org()
        await create_run(db, org_id=org, run_id="r", user_message="x")
        calls = []

        async def invoke(**kw):
            calls.append(1)
            raise PlatformError("invalid_parameter: 预算为负", code="invalid", retriable=False)

        w = Worker(db, FakeAdPlatform())
        out = await w.tools.call(org_id=org, run_id="r", step=1,
                                 tool="campaign.update_budget",
                                 arguments={"campaign_id": "C", "new_budget": -1},
                                 invoke=invoke)
        return out, calls

    out, calls = with_db(go)
    assert out.ok is False
    assert len(calls) == 1, f"不可重试的错误被重试了 {len(calls)} 次"


# --------------------------------------------------------------------------
# ② 降级触发器：全是外部信号
# --------------------------------------------------------------------------


def test_all_six_triggers_fire_on_external_signals_only() -> None:
    """★★ 六个触发器逐个验。**没有一个读模型置信度** —— §39：编造时往往最自信。"""
    cases = [
        (DecisionContext(tool_failed="campaign.update_budget"), "tool_failed"),
        (DecisionContext(validation_errors=["预算为负"]), "validation_failed"),
        (DecisionContext(data_maturity="immature"), "data_immature"),
        (DecisionContext(cap_hits=["unauthorized_write_cap"]), "cap_hit"),
        (DecisionContext(write_amount=999_999), "amount_over_threshold"),
        (DecisionContext(retrieval_empty_tools=["policy.search"]), "retrieval_empty"),
    ]
    for ctx, expected in cases:
        reasons = [t.reason for t in evaluate_triggers(ctx)]
        assert expected in reasons, f"{expected} 没触发，实得 {reasons}"


def test_clean_context_does_not_trigger() -> None:
    """★ 对照：一切正常时**不能**停下来。全都停的网关等于没有网关。"""
    assert evaluate_triggers(DecisionContext(data_maturity="mature",
                                             write_amount=1_000)) == []


def test_all_matching_triggers_are_reported_not_just_the_first() -> None:
    """★ "既超金额又命中 cap" 和 "只是超金额" 风险不同，审批单上要分得出来。"""
    ctx = DecisionContext(write_amount=999_999, cap_hits=["unauthorized_write_cap"],
                          data_maturity="immature")
    assert len(evaluate_triggers(ctx)) == 3


def test_retrieval_empty_comes_from_m8_no_match() -> None:
    """★ ⑥ 是 M8 接上来的：`no_match` 是明确的信号位，不是"结果长度为 0"这种推断。"""
    reasons = [t.reason for t in evaluate_triggers(
        DecisionContext(retrieval_empty_tools=["policy.search", "insight.search_claims"]))]
    assert reasons == ["retrieval_empty"]


# --------------------------------------------------------------------------
# ③ 端到端：worker 跑完一条 run
# --------------------------------------------------------------------------


def test_worker_opens_approval_and_parks_the_run_atomically() -> None:
    """★★ 审批单和 run 状态必须**同时**变。

    分开做的话，中间崩掉会留下"单开了但 run 还在跑"（重复执行）
    或"run 停了但没有单"（永久卡死）。
    """
    async def go(db):
        org = _org()
        await _drain(db)
        await create_run(db, org_id=org, run_id="r", user_message="把预算提到 1200")
        # 默认 new_budget=120_000 > 阈值 100_000 ⇒ 必然触发审批
        await Worker(db, FakeAdPlatform()).run_once()
        async with db.tx() as conn:
            run = await conn.fetchrow(
                "SELECT status, requires_approval FROM agent_runs WHERE org_id=$1", org)
            case = await conn.fetchrow(
                "SELECT status, trigger_reason, evidence FROM approval_cases WHERE org_id=$1", org)
        return run, case

    run, case = with_db(go)
    assert run["status"] == "waiting_for_user"
    assert run["requires_approval"] is True
    assert case is not None and case["status"] == "pending"
    assert "amount_over_threshold" in case["trigger_reason"]
    assert "triggers" in case["evidence"], "审批单没带证据 —— 人只能看到结论"


def test_worker_writes_an_ordered_event_stream() -> None:
    """★ SSE 断线补发靠 seq 定位 ⇒ seq 必须**由数据库分配且连续**。"""
    async def go(db):
        org = _org()
        await _drain(db)
        await create_run(db, org_id=org, run_id="r", user_message="x")
        await Worker(db, FakeAdPlatform()).run_once()
        async with db.tx() as conn:
            return await conn.fetch(
                "SELECT seq, kind FROM run_events WHERE org_id=$1 ORDER BY seq", org)

    rows = with_db(go)
    assert [r["seq"] for r in rows] == list(range(1, len(rows) + 1)), "seq 不连续"
    # K2-6：创建事务写 run.created（seq 1），抢到执行权才 run.started（课件 CH2 §7 口径）
    assert [r["kind"] for r in rows[:2]] == ["run.created", "run.started"]


def test_daily_cost_cap_degrades_before_touching_the_platform() -> None:
    """★ 压测场景⑤「单 org 刷爆预算」：超预算要**在打平台之前**降级。

    先跑再拦的话，钱已经花出去了。
    """
    async def go(db):
        org = _org()
        await _drain(db)
        async with db.tx() as conn:
            await conn.execute(
                "INSERT INTO usage_records (org_id, cost_micros) VALUES ($1, $2)",
                org, 99_999_999)
        await create_run(db, org_id=org, run_id="r", user_message="x")
        platform = FakeAdPlatform()
        await Worker(db, platform, WorkerConfig(worker_id="w-cap")).run_once()
        async with db.tx() as conn:
            run = await conn.fetchrow("SELECT status, error FROM agent_runs WHERE org_id=$1", org)
        return run, platform

    run, platform = with_db(go)
    assert run["status"] == "failed"
    assert run["error"] == "daily_cost_cap_exceeded"
    assert platform.calls == 0, "超预算了还去打了平台 —— 钱已经花出去了"


def test_audit_records_param_source() -> None:
    """★ `param_source` 是防注入的证据：这个金额是用户要的还是工具返回里读来的。

    ⚠️ 2026-08-20：档位改由动作推导后，写动作先停审批 ⇒ 这条要走**批准后**那一跑
    （第一跑开单停下，裁决完 run 回队列，第二跑 `skip_triggers` 才到得了写）。
    """
    async def go(db):
        org = _org()
        await _drain(db)
        await create_run(db, org_id=org, run_id="r", user_message="x")
        w = Worker(db, FakeAdPlatform(),
                   WorkerConfig(daily_cost_cap_micros=10 ** 9, amount_threshold=10 ** 9))
        await w.run_once()                       # 第一跑：停在审批
        async with db.tx() as conn:
            ref = await conn.fetchval(
                "SELECT case_ref FROM approval_cases WHERE org_id=$1 AND run_id='r'", org)
            await conn.execute(
                "UPDATE approval_cases SET status='approved', reviewer_id='t' "
                "WHERE org_id=$1 AND case_ref=$2", org, ref)
        await resume_after_approval(db, org_id=org, run_id="r")
        await w.run_once()                       # 第二跑：已裁决 ⇒ 执行写
        async with db.tx() as conn:
            return await conn.fetch(
                "SELECT action, param_source FROM audit_logs WHERE org_id=$1", org)

    rows = with_db(go)
    assert any(r["action"] == "campaign.update_budget" and r["param_source"] == "user"
               for r in rows), f"没记 param_source，实得 {[dict(r) for r in rows]}"
