"""M9 · 设计符合性：runtime 代码 vs 主设计文档的逐条核对（2026-08-17 审计）。

★★★ 这一份和 test_api / test_worker 的分工不同

那几份验的是「**写出来的东西自己对不对**」；这一份验的是
「**它是不是主设计文档要求的那个东西**」—— 两件事，测试抓到的是不同的东西。
2026-08-17 的审计发现：45 条既有测试全绿，但设计文档 §3 的核心机制**一次都没被接上**，
因为没有任何一条测试问过「C 档动作会不会走审批」。

⚠️ **`xfail(strict=True)` 是刻意的。** 这些条目现在**不满足**，
标 xfail 是为了让缺口**在测试输出里长期可见**而不是被忘掉；
`strict=True` 意味着**哪天修好了会变成 XPASS = 失败**，逼下一个人回来把标记翻过去。
⇒ 判据表在 `docs/syncopate/11-runtime-acceptance.md`，每条测试的名字与那份文档的任务标题一一对应。

⚠️ 需要跑着的 PG（`bash scripts/pg_bootstrap.sh`）。**跳过不是通过。**
"""

from __future__ import annotations

import asyncio
import inspect
import uuid

import pytest

from syncopate.domains.adcampaign import build_domain
from syncopate.runtime import tools as rt_tools
from syncopate.runtime import worker as rt_worker
from syncopate.runtime.db import Database, claim_run, create_run
from syncopate.runtime.gateway import DecisionContext
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


pg_only = pytest.mark.skipif(
    not _pg_available(),
    reason="需要跑着的 PostgreSQL：bash scripts/pg_bootstrap.sh")


def with_db(body):
    async def main():
        db = Database()
        await db.connect(max_size=5)
        try:
            return await body(db)
        finally:
            await db.close()
    return asyncio.run(main())


async def _drain(db: Database) -> None:
    """★ `claim_run` 是**全局 FIFO**（不按 org 隔离），库里任何遗留的可抢 run
    都会被先抢走。不清就会抢到别人的 run —— 审计时第一版探针就栽在这里。"""
    async with db.tx() as conn:
        await conn.execute(
            "UPDATE agent_runs SET status='cancelled' "
            "WHERE status='queued' OR (status='running' AND lease_expires_at < now())")


async def _run_until_mine(w: Worker, run_id: str, limit: int = 50) -> bool:
    for _ in range(limit):
        got = await w.run_once()
        if got == run_id:
            return True
        if got is None:
            return False
    return False


def _ids() -> tuple[str, str]:
    return f"org_{uuid.uuid4().hex[:8]}", f"run_{uuid.uuid4().hex[:8]}"


# ---------------------------------------------------------------------------
# 任务一 · §3「C 档动作全部走审批」
# ---------------------------------------------------------------------------


@pg_only
@pytest.mark.xfail(strict=True, reason="【自动化档位没有消费者】claim_run 不返回 automation_tier，"
                                       "worker 硬编码 DecisionContext(automation_tier=None) "
                                       "⇒ tier_c 触发器在真实路径上永远不可达")
def test_tier_c_action_must_go_through_approval() -> None:
    """★ 设计文档 §3：C 档 = 不可逆或代价高（建 campaign / 大幅扩量 / 关停）
    ⇒ **一律提议 + 人工点确认**。

    判据：一条 `automation_tier='C'` 的 run 跑完之后，必须有一张审批单。
    金额阈值刻意调到极高，把「金额超阈值」那个触发器排除掉 —— **只考 C 档本身**。
    """
    async def body(db: Database) -> int:
        await _drain(db)
        org, run = _ids()
        await create_run(db, org_id=org, run_id=run, user_message="把 CMP_1 预算提到 1200",
                         intent="I09", automation_tier="C")
        w = Worker(db, FakeAdPlatform(), WorkerConfig(amount_threshold=10_000_000))
        assert await _run_until_mine(w, run), "没抢到自己那条 run，测试无效"
        async with db.tx() as conn:
            return await conn.fetchval(
                "SELECT trigger_reason FROM approval_cases WHERE org_id=$1", org)

    # ⚠️ **必须验"为什么停"，不能只验"停了"。** 第一版只数了审批单条数，
    # 后来给 worker 接上检索之后，这条测试因为 `retrieval_empty` 开了单而变成 XPASS ——
    # **它开始为错误的理由通过了**。xfail 会把这种事一起吞掉，所以判据要指名道姓。
    reason = with_db(body)
    assert reason and "tier_c" in reason, f"停下来的理由里没有 tier_c，实得 {reason!r}"


# ---------------------------------------------------------------------------
# 任务二 · 审批闭环：人点了同意之后，run 要能继续
# ---------------------------------------------------------------------------


@pg_only
@pytest.mark.xfail(strict=True, reason="【审批通过后没有恢复路径】没有任何代码把 run 从 waiting_for_user "
                                       "放回 queued，claim_run 也不认这个状态 "
                                       "⇒ 审批通过后 run 永久停住")
def test_approved_case_lets_the_run_continue() -> None:
    """★ 09 §3：网关的输出不是"拒绝"，是"暂停"。**暂停就必须能恢复。**

    判据：开单 → 人点同意 → worker 应该能重新抢到这条 run。
    ⚠️ 开单那半边是对的（审批单与 run 状态同事务翻转，见 gateway.open_approval_case），
    断的是**回来**那半边 —— 所以这条测的是闭环，不是开单。
    """
    async def body(db: Database) -> str | None:
        await _drain(db)
        org, run = _ids()
        await create_run(db, org_id=org, run_id=run, user_message="大额扩量")
        w = Worker(db, FakeAdPlatform(), WorkerConfig())   # 默认阈值 ⇒ 金额必然超
        assert await _run_until_mine(w, run), "没抢到自己那条 run，测试无效"
        async with db.tx() as conn:
            case = await conn.fetchval(
                "SELECT case_ref FROM approval_cases WHERE org_id=$1", org)
            assert case is not None, "前提不成立：这一跑没开出审批单"
            # 人点「同意」——和 API 的 POST /approvals/{case_ref} 走同一条 UPDATE
            await conn.execute(
                "UPDATE approval_cases SET status='approved', reviewer_id='u1', "
                "reviewed_at=now() WHERE org_id=$1 AND case_ref=$2", org, case)
        got = await claim_run(db, worker_id="w-resume")
        return got["run_id"] if got else None

    org_run = with_db(body)
    assert org_run is not None, "审批通过后 run 没有回到可抢队列"


# ---------------------------------------------------------------------------
# 任务三 · 并发同键：命中幂等必须返回**原结果**，不能返回空
# ---------------------------------------------------------------------------


@pg_only
@pytest.mark.xfail(strict=True, reason="【并发重复调用返回空结果】record_tool_call 命中 prior 时不看它是否"
                                       "还在执行中（ok/result 仍为 NULL）"
                                       "⇒ 并发第二次拿到 ok=None/data=None")
def test_concurrent_same_key_returns_the_original_result() -> None:
    """★★★ §38 第三层 + db.py 自己的承诺：「命中 ⇒ **返回原结果**而不是重放」。

    既有的 10 条幂等测试都是**顺序**投两次 —— 第一次早就写完了 ok/result。
    并发投两次时，第二次会命中一条**还在执行中**的占坑行（ok/result 都是 NULL），
    代码把它当"原结果"返回。

    ⚠️ 后果不是崩，是**静默假失败**：worker 判 `not written.ok` ⇒
    run 记成 failed（error 还是 None），**而钱其实已经花出去了**。
    同 flash-attn 那条教训：**返回空比报错更毒。**
    """
    async def body(db: Database) -> tuple[int, list]:
        from syncopate.runtime.db import record_tool_call
        org, run = _ids()
        await create_run(db, org_id=org, run_id=run, user_message="probe")
        key = f"{org}:{run}:campaign.update_budget:deadbeef"
        executed: list[int] = []

        async def execute():
            executed.append(1)
            await asyncio.sleep(0.10)          # 模拟真的打了平台
            return True, {"new_budget": 120_000}, None

        async def one():
            return await record_tool_call(
                db, org_id=org, run_id=run, step=1, tool="campaign.update_budget",
                arguments={"campaign_id": "CMP_1"},
                external_idempotency_key=key, execute=execute)

        results = await asyncio.gather(one(), one())
        return len(executed), results

    n_exec, results = with_db(body)
    assert n_exec == 1, "副作用执行了不止一次 —— 这才是最贵的那种错"
    replayed = [r for r in results if r.replayed]
    assert replayed, "并发两次里应该有一次是被幂等挡下的"
    assert replayed[0].data is not None, (
        "命中幂等却返回了空结果 —— 调用方会把它读成失败，而钱已经花了")


# ---------------------------------------------------------------------------
# 任务四 · 六个降级触发器，每一个都要在真实路径上有生产者
# ---------------------------------------------------------------------------


@pytest.mark.xfail(strict=True, reason="【降级信号缺生产者】worker 只赋 tool_failed / write_amount，"
                                       "另外 5 个信号没有任何生产者")
def test_every_decision_signal_has_a_producer() -> None:
    """★ 这个项目最反复的失效形状是「机制建好了，但没接上」。

    `evaluate_triggers` 的六个分支单元测试全绿 —— 但那验的是**机制本身对不对**，
    不是**机制有没有被接上**。判据必须是：`DecisionContext` 的每一个信号字段，
    在 worker 的真实编排里**都有人给它赋值**。

    ⚠️ 用源码扫描而不是行为测试，是因为「不可达」这件事**没有行为可测** ——
    你测不出一个永远不会发生的事件。
    """
    src = inspect.getsource(rt_worker)
    signals = [f for f in DecisionContext.__dataclass_fields__]
    missing = [f for f in signals if f"ctx.{f}" not in src]
    assert not missing, f"这些降级信号在 worker 里没有任何生产者：{missing}"


# ---------------------------------------------------------------------------
# 任务五 · 沙盒 ⊆ runtime：沙盒的写工具，runtime 必须都认识
# ---------------------------------------------------------------------------


@pytest.mark.xfail(strict=True, reason="【写工具登记不全】沙盒 8 个写工具，runtime WRITE_TOOLS 只有 4 个 "
                                       "⇒ 另外 4 个在 runtime 侧会被当读工具"
                                       "（不校验权限、不生成幂等键）")
def test_every_sandbox_write_tool_is_known_to_runtime() -> None:
    """★★ 09 §1-③：「沙盒可以简化实现，但**不能有 runtime 没有的行为**。」

    这条纪律此前**没有物理载体** —— 谁在沙盒里加一个写工具，runtime 这边
    不会有任何东西响。这条测试就是那个载体。

    ⚠️ runtime 把不在 `WRITE_TOOLS` 里的工具**一律当读工具**：
    既不校验权限，也不生成外部幂等键（§11-①「每个写工具必须带外部幂等键」）。
    所以"漏登记"的代价不是报错，是**一个写动作悄悄没有幂等保护**。
    """
    registry = build_domain().registry
    sandbox_writes = {n for n, s in registry._tools.items() if s.kind == "write"}
    missing = sorted(sandbox_writes - set(rt_tools.WRITE_TOOLS))
    assert not missing, f"沙盒有、runtime 不认识的写工具：{missing}"
