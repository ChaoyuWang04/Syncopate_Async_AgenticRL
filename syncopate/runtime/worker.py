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

from syncopate.runtime.db import Database, approved_action, claim_run, finish_run
from syncopate.runtime.gateway import DecisionContext, evaluate_triggers, open_approval_case
from syncopate.runtime.platform import FakeAdPlatform
from syncopate.runtime.retrieval import RetrievalService, RetrievalStatus
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
    # ★ 同时在跑几条。1 = 串行（老行为）。
    # §19 要求「长任务（480s 审核）**不阻塞其他任务**」+「并发 run 数 ≥ 8」——
    # 串行实现下这两条**结构上不可能通过**，压测场景①⑤ 也就无从谈起。
    concurrency: int = 8
    # 审批金额阈值。None = 用 gateway 的默认值。
    # ⚠️ 走配置而不是改模块常量：常量被当默认参数值绑定过一次，改了不生效（见 gateway）。
    amount_threshold: int | None = None
    # ★★ 这次政策检索是不是**本次决策的证据来源**。
    #
    # doc 10 §5 用一次实测换来的纪律：「有过空检索 + 给了确定结论 = 幻觉」这个判据
    # **当场把自己的 gold 判错了** —— 复盘库里没有历史结论、但模型手里有实际数据时，
    # 照数据作答是**对的**，不是编造。⇒ 判据必须挂在「**这个答案依赖那次检索**」上。
    #
    # 当前的最小编排**表达不了"答案依赖哪次检索"**（那要等真 Agent Loop），
    # 所以默认 False：检索照查、信号照产、但不拿它阻断。
    # ⇒ 这是一笔明写的欠债，不是忘了接。
    policy_lookup_is_evidence: bool = False


class Worker:
    def __init__(self, db: Database, platform: FakeAdPlatform,
                 config: WorkerConfig | None = None,
                 retrieval: RetrievalService | None = None) -> None:
        self.db = db
        self.platform = platform
        self.config = config or WorkerConfig()
        self.tools = ToolRuntime(db)
        self.retrieval = retrieval or RetrievalService(db)

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
                   payload={"attempt": claimed["attempt"],
                            "automation_tier": claimed.get("automation_tier")})
        try:
            await self._execute(org_id=org_id, run_id=run_id,
                                user_message=claimed["user_message"] or "",
                                automation_tier=claimed.get("automation_tier"))
        except Exception as exc:                      # noqa: BLE001
            # ★ 兜底：worker 不能因为一条 run 挂掉而停摆（压测场景②）。
            await emit(self.db, org_id=org_id, run_id=run_id, kind="run.failed",
                       payload={"error": str(exc)[:500]})
            await finish_run(self.db, org_id=org_id, run_id=run_id,
                             status="failed", error=str(exc)[:500])
        return run_id

    async def _execute(self, *, org_id: str, run_id: str, user_message: str,
                       automation_tier: str | None = None) -> None:
        # ---- D 档：永不自动，连审批单都不开 ----
        # ★ §3 的四档里 D 是「不可逆**且**不可验证」（跨账户 / 竞品 / 合规边界 /
        #   账户级预算）。它和 C 档的区别是**性质**不是程度：C 是"要人点头"，
        #   D 是"这件事不该由 agent 发起" ⇒ 开审批单反而是错的，那等于把它降成了 C。
        if automation_tier == "D":
            await audit(self.db, org_id=org_id, run_id=run_id,
                        action="tier_d_refused", object_key=None, param_source="system",
                        detail={"automation_tier": "D"})
            await emit(self.db, org_id=org_id, run_id=run_id, kind="run.failed",
                       payload={"error": "tier_d_never_automated"})
            await finish_run(self.db, org_id=org_id, run_id=run_id, status="cancelled",
                             error="tier_d_never_automated")
            return

        # ---- 已经被人裁决过？那这一跑是**恢复**，不是重新决策 ----
        decided = await approved_action(self.db, org_id=org_id, run_id=run_id)
        if decided is not None and decided["status"] == "rejected":
            await emit(self.db, org_id=org_id, run_id=run_id, kind="run.failed",
                       payload={"error": "approval_rejected",
                                "case_ref": decided["case_ref"]})
            await finish_run(self.db, org_id=org_id, run_id=run_id, status="cancelled",
                             error="approval_rejected")
            return

        # ---- 成本闸先于一切：超预算直接降级，不烧模型也不打平台 ----
        if await self._over_budget(org_id):
            await emit(self.db, org_id=org_id, run_id=run_id, kind="run.degraded",
                       payload={"reason": "daily_cost_cap"})
            await finish_run(self.db, org_id=org_id, run_id=run_id, status="failed",
                             error="daily_cost_cap_exceeded")
            return

        # ★ automation_tier 终于有消费者了。此前它被 API 校验、被 schema 约束、
        #   被落库，然后**没有任何人读它** ⇒ C 档动作一路直接执行到底。
        ctx = DecisionContext(automation_tier=automation_tier)

        # ---- step 0：查政策 ----
        # ★ 这一步存在的意义不是"查得准"，是让**「查不到」和「查不了」两个信号
        #   第一次有了生产者**。在此之前 gateway 的 retrieval_empty 是个孤儿触发器。
        policy = await self.retrieval.search_policy(
            org_id=org_id, query=user_message or "预算调整")
        await emit(self.db, org_id=org_id, run_id=run_id, kind="retrieval.result",
                   payload={"tool": "policy.search", "status": policy.status.value,
                            "hits": len(policy.hits), "latency_ms": policy.latency_ms})
        if policy.status is RetrievalStatus.NO_MATCH:
            # 只有当这次检索是决策的证据来源时，"查不到"才阻断（见 WorkerConfig 那条注释）。
            if self.config.policy_lookup_is_evidence:
                ctx.retrieval_empty_tools.append("policy.search")
        elif policy.status is RetrievalStatus.UNAVAILABLE:
            # ⚠️ **不能和上面那条合并，而且阻断强度刻意不对称**：
            #   查不到 = 「没有政策限制这件事」    ⇒ 依赖它才阻断
            #   查不了 = 「不知道有没有政策限制」  ⇒ **一律阻断**
            # 因为两种误判的代价不对称：把"没有政策"当成"不知道"，最多多问人一次；
            # 把"不知道"当成"没有政策"，是**放行一个未知风险**。
            ctx.retrieval_unavailable_tools.append("policy.search")

        # ---- step 1：读 ----
        # ★ 数据成熟度**从平台查，不再硬编码**。此前这里写死 `"mature"`，
        #   于是 `data_immature` 这个降级在真实路径上永远不会发生 ——
        #   而归因延迟是本项目的第一性约束，这一条是最不该缺生产者的。
        fresh = await self.platform.get_freshness(campaign_id="CMP_1")
        ctx.data_maturity = fresh["maturity"]
        await record_step(self.db, org_id=org_id, run_id=run_id, step=1,
                          phase="investigate", data_maturity=fresh["maturity"])
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

        # ---- 成本闸再查一次：一条 run 自己也可能把额度烧穿 ----
        # ★ 只在跑之前查一次是不够的：进来时没超，读完之后可能就超了。
        #   写动作是花钱的那一侧 ⇒ **闸门要放在写之前**，不是只放在门口。
        if await self._over_budget(org_id):
            await emit(self.db, org_id=org_id, run_id=run_id, kind="run.degraded",
                       payload={"reason": "daily_cost_cap", "at": "before_write"})
            await finish_run(self.db, org_id=org_id, run_id=run_id, status="cancelled",
                             error="daily_cost_cap_exceeded")
            return

        # ---- step 2：决策 + 写 ----
        # ★ 恢复执行时用**人裁决过的参数**：`modified_params` 优先于 `proposed_params`。
        #   否则"人工修正"这条飞轮回路只是在记账，改了什么并不影响世界。
        new_budget = 120_000
        if decided is not None:
            new_budget = int(decided["params"].get("new_budget", new_budget))
        ctx.write_amount = new_budget
        # ★ 已裁决的动作**不再过网关** —— 否则刚批准就又被同一个触发器拦下来，
        #   run 在 waiting_for_user 和 queued 之间来回弹，永远跑不完。
        triggers = ([] if decided is not None
                    else evaluate_triggers(ctx, amount_threshold=self.config.amount_threshold))
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

    async def _loop(self, *, stop: asyncio.Event) -> None:
        """一条工作线。★ 没活干时**睡一下再看**，不是空转 —— 空转会把 CPU 吃满，
        而压测场景①（突发 10× 流量）需要 CPU 留给真正的活。"""
        while not stop.is_set():
            if await self.run_once() is None:
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(stop.wait(), timeout=self.config.poll_interval)

    async def serve(self, *, stop: asyncio.Event) -> None:
        """长跑。**并发跑 `config.concurrency` 条**。

        ★ 为什么并发放在这里而不是"起多个进程"：一条 run 的时间绝大部分花在
        **等外部**（打平台、等审核），不是算。等待期让出事件循环就能叠很多条，
        而 `claim_run` 的 `FOR UPDATE SKIP LOCKED` 保证它们不会抢到同一条。

        ⚠️ 每条工作线用**同一个** `stop`：停的时候要一起停，
        否则会留下几条还在跑的把 lease 攥着不放。
        """
        n = max(1, self.config.concurrency)
        await asyncio.gather(*(self._loop(stop=stop) for _ in range(n)))
