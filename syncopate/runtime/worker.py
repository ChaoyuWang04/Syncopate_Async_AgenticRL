"""M9.4 · Worker：抢 run → 跑编排 → 落事件 → 收尾。

★ 事件流是**追加式**的（`run_events`，永不 UPDATE）

SSE 断线补发靠 `seq` 定位：客户端带 `Last-Event-ID` 回来，我们从那之后接着推。
⇒ **seq 必须由数据库分配且连续**，不能由客户端或内存计数器给 ——
worker 崩了重启，内存计数器会从头开始，客户端就永远收不到中间那段。

★★ 编排本身在这一版是**最小可用**：一次 metrics 读 + 一次预算写。
真正的 Agent Loop（多轮、模型决策）是 M9.4 的下半场，接口留在 `run_step` 上。
现在这一版的价值是把**幂等 / 审批 / 事件 / 计费**四条横切先接通并测住 ——
先接通再变复杂，反过来做的话每条横切都要在一个动态的编排里debug。
"""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass
from typing import Any

from syncopate.runtime.db import Database, claim_run, finish_run
from syncopate.runtime.gateway import DecisionContext, evaluate_triggers, open_approval_case
from syncopate.runtime.platform import FakeAdPlatform
from syncopate.runtime.tools import PermissionDenied, ToolRuntime


async def emit(db: Database, *, org_id: str, run_id: str, kind: str,
               payload: dict[str, Any] | None = None) -> int:
    """写一条事件。**seq 由数据库分配**（见模块 docstring）。"""
    async with db.tx() as conn:
        return await conn.fetchval(
            """
            INSERT INTO run_events (run_id, org_id, seq, kind, payload)
            VALUES ($1, $2,
                    COALESCE((SELECT max(seq) FROM run_events
                               WHERE run_id=$1 AND org_id=$2), 0) + 1,
                    $3, $4)
            RETURNING seq
            """, run_id, org_id, kind, payload or {})


async def record_step(db: Database, *, org_id: str, run_id: str, step: int,
                      phase: str, data_maturity: str | None = None) -> None:
    async with db.tx() as conn:
        await conn.execute(
            """
            INSERT INTO agent_steps (run_id, org_id, step, phase, data_maturity_at_step)
            VALUES ($1,$2,$3,$4,$5) ON CONFLICT (org_id, run_id, step) DO NOTHING
            """, run_id, org_id, step, phase, data_maturity)


async def record_usage(db: Database, *, org_id: str, run_id: str,
                       tokens_in: int = 0, tokens_out: int = 0,
                       cost_micros: int = 0) -> None:
    async with db.tx() as conn:
        await conn.execute(
            """
            INSERT INTO usage_records (org_id, run_id, tokens_in, tokens_out, cost_micros)
            VALUES ($1,$2,$3,$4,$5)
            """, org_id, run_id, tokens_in, tokens_out, cost_micros)


async def audit(db: Database, *, org_id: str, run_id: str, action: str,
                object_key: str | None, param_source: str,
                detail: dict[str, Any] | None = None) -> None:
    """★ `param_source` 是**防注入的证据**（§27.2「假设模型已被策反」）。

    一个从**工具返回**里长出来的写参数，和用户明确要求的写参数，
    事后追责时是完全不同的两件事。不记这一列，注入发生了也查不出来。
    """
    async with db.tx() as conn:
        await conn.execute(
            """
            INSERT INTO audit_logs (run_id, org_id, action, object_key, param_source, detail)
            VALUES ($1,$2,$3,$4,$5,$6)
            """, run_id, org_id, action, object_key, param_source, detail or {})


@dataclass
class WorkerConfig:
    worker_id: str = "worker-1"
    lease_seconds: int = 60
    poll_interval: float = 0.2
    # 单 org 日预算（微单位）。压测场景⑤「单 org 刷爆预算」要靠它。
    daily_cost_cap_micros: int = 10_000_000
    # 审批金额阈值。None = 用 gateway 的默认值。
    # ⚠️ 走配置而不是改模块常量：常量被当默认参数值绑定过一次，改了不生效（见 gateway）。
    amount_threshold: int | None = None


class Worker:
    def __init__(self, db: Database, platform: FakeAdPlatform,
                 config: WorkerConfig | None = None) -> None:
        self.db = db
        self.platform = platform
        self.config = config or WorkerConfig()
        self.tools = ToolRuntime(db)

    # ---- 成本闸：压测场景⑤ ------------------------------------------------

    async def _over_budget(self, org_id: str) -> bool:
        async with self.db.tx() as conn:
            spent = await conn.fetchval(
                "SELECT COALESCE(sum(cost_micros),0) FROM usage_records "
                "WHERE org_id=$1 AND day=CURRENT_DATE", org_id)
        return spent >= self.config.daily_cost_cap_micros

    # ---- 主循环 ------------------------------------------------------------

    async def run_once(self) -> str | None:
        """抢一条并跑完。返回 run_id；没活干返回 None。"""
        claimed = await claim_run(self.db, worker_id=self.config.worker_id,
                                  lease_seconds=self.config.lease_seconds)
        if claimed is None:
            return None
        org_id, run_id = claimed["org_id"], claimed["run_id"]
        await emit(self.db, org_id=org_id, run_id=run_id, kind="run.started",
                   payload={"attempt": claimed["attempt"]})
        try:
            await self._execute(org_id=org_id, run_id=run_id,
                                user_message=claimed["user_message"] or "")
        except Exception as exc:                      # noqa: BLE001
            # ★ 兜底：worker 不能因为一条 run 挂掉而停摆（压测场景②）。
            await emit(self.db, org_id=org_id, run_id=run_id, kind="run.failed",
                       payload={"error": str(exc)[:500]})
            await finish_run(self.db, org_id=org_id, run_id=run_id,
                             status="failed", error=str(exc)[:500])
        return run_id

    async def _execute(self, *, org_id: str, run_id: str, user_message: str) -> None:
        # ---- 成本闸先于一切：超预算直接降级，不烧模型也不打平台 ----
        if await self._over_budget(org_id):
            await emit(self.db, org_id=org_id, run_id=run_id, kind="run.degraded",
                       payload={"reason": "daily_cost_cap"})
            await finish_run(self.db, org_id=org_id, run_id=run_id, status="failed",
                             error="daily_cost_cap_exceeded")
            return

        ctx = DecisionContext(automation_tier=None)

        # ---- step 1：读 ----
        await record_step(self.db, org_id=org_id, run_id=run_id, step=1,
                          phase="investigate", data_maturity="mature")
        metrics = await self.tools.call(
            org_id=org_id, run_id=run_id, step=1, tool="campaign.get_metrics",
            arguments={"campaign_id": "CMP_1"},
            invoke=self.platform.get_metrics)
        await emit(self.db, org_id=org_id, run_id=run_id, kind="tool.result",
                   payload={"tool": "campaign.get_metrics", "ok": metrics.ok})
        await record_usage(self.db, org_id=org_id, run_id=run_id,
                           tokens_in=800, tokens_out=120, cost_micros=1_200)
        if not metrics.ok:
            ctx.tool_failed = "campaign.get_metrics"

        # ---- step 2：决策 + 写 ----
        new_budget = 120_000
        ctx.write_amount = new_budget
        triggers = evaluate_triggers(ctx, amount_threshold=self.config.amount_threshold)
        if triggers:
            # ★ 停下来 ≠ 拒绝：开一张**带证据**的审批单，人看的是证据不是结论。
            case_ref = await open_approval_case(
                self.db, org_id=org_id, run_id=run_id, action_type="campaign.update_budget",
                proposed_params={"campaign_id": "CMP_1", "new_budget": new_budget},
                rationale=f"用户请求：{user_message[:200]}",
                evidence={"metrics": metrics.data, "triggers": [t.reason for t in triggers]},
                triggers=triggers)
            await emit(self.db, org_id=org_id, run_id=run_id, kind="run.waiting_for_user",
                       payload={"case_ref": case_ref,
                                "triggers": [t.reason for t in triggers]})
            return

        await record_step(self.db, org_id=org_id, run_id=run_id, step=2, phase="act")
        try:
            written = await self.tools.call(
                org_id=org_id, run_id=run_id, step=2, tool="campaign.update_budget",
                arguments={"campaign_id": "CMP_1", "new_budget": new_budget},
                invoke=self.platform.update_budget)
        except PermissionDenied as exc:
            await audit(self.db, org_id=org_id, run_id=run_id, action="permission_denied",
                        object_key="CMP_1", param_source="system", detail={"error": str(exc)})
            await finish_run(self.db, org_id=org_id, run_id=run_id, status="failed",
                             error=str(exc))
            return

        # ★ param_source="user"：这个金额是用户要求的，不是从工具返回里读出来的。
        await audit(self.db, org_id=org_id, run_id=run_id, action="campaign.update_budget",
                    object_key="CMP_1", param_source="user",
                    detail={"new_budget": new_budget, "replayed": written.replayed})
        await emit(self.db, org_id=org_id, run_id=run_id, kind="tool.result",
                   payload={"tool": "campaign.update_budget", "ok": written.ok,
                            "replayed": written.replayed})

        if not written.ok:
            await finish_run(self.db, org_id=org_id, run_id=run_id, status="failed",
                             error=written.error)
            await emit(self.db, org_id=org_id, run_id=run_id, kind="run.failed",
                       payload={"error": written.error})
            return

        await finish_run(self.db, org_id=org_id, run_id=run_id, status="succeeded",
                         result={"new_budget": new_budget})
        await emit(self.db, org_id=org_id, run_id=run_id, kind="run.succeeded",
                   payload={"new_budget": new_budget})

    async def serve(self, *, stop: asyncio.Event) -> None:
        """长跑循环。★ 没活干时**睡一下再看**，不是空转 —— 空转会把 CPU 吃满，
        而压测场景①（突发 10× 流量）需要 CPU 留给真正的活。"""
        while not stop.is_set():
            if await self.run_once() is None:
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(stop.wait(), timeout=self.config.poll_interval)
