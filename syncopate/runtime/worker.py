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

from syncopate.runtime.action_gate import ActionGate, ToolBinding
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
    # ★ 把这个 worker 限定在一个租户上（None = 全局）。
    # 生产上用于 worker 池按租户切分；测试/探针上用于**结构性隔离** ——
    # 队列是全局的，不限定就会抢走别人遗留的活，然后得出一个错误的结论。
    org_id: str | None = None


class Worker:
    def __init__(self, db: Database, platform: FakeAdPlatform,
                 config: WorkerConfig | None = None,
                 retrieval: RetrievalService | None = None) -> None:
        self.db = db
        self.platform = platform
        self.config = config or WorkerConfig()
        self.tools = ToolRuntime(db)
        self.retrieval = retrieval or RetrievalService(db)

    # ---- 工具名 → 真正打外部世界的那个协程 --------------------------------
    #
    # ★★ 这张表交给 `ActionGate` 持有，**循环拿不到 platform** ——
    #   `ToolRuntime.call` 的 `invoke=` 由调用方传实现，那是个洞：
    #   agent loop 完全可以绕过收口直接调 platform。收口自己持有绑定就堵上了。
    # ⚠️ 只登记**这一版真的接了实现**的那几个。缺的 22 个读工具是 B-2 的活 ——
    #   缺的工具会被收口报 `unknown_tool`（**报"没有"，不猜**），不会静默走空。
    def _bindings(self, org_id: str = "", run_id: str = "") -> dict[str, ToolBinding]:
        # ⚠️ `ActionGate` 只把模型给的 `arguments` 展开传进去 ——
        #   platform / retrieval / org_id 这些**模型碰不到的东西**在这里闭包绑定。
        #   这也是收口"模型给不了实现"那条的自然结果。
        from functools import partial

        from syncopate.runtime import tool_impls as impl
        return {
            "campaign.get_metrics": ToolBinding(self.platform.get_metrics),
            "campaign.update_budget": ToolBinding(self.platform.update_budget),
            "campaign.list": ToolBinding(partial(impl.campaign_list, self.platform)),
            "metrics.get_freshness": ToolBinding(
                partial(impl.metrics_get_freshness, self.platform)),
            "policy.search": ToolBinding(
                partial(impl.policy_search, self.retrieval, org_id)),
            "insight.search_claims": ToolBinding(
                partial(impl.insight_search_claims, self.retrieval, org_id)),
            # 记忆库 + 安全线（B-2 第二批）
            "memory.read": ToolBinding(partial(impl.memory_read, self.db, org_id)),
            "memory.search": ToolBinding(partial(impl.memory_search, self.db, org_id)),
            "benchmark.get_safety_line": ToolBinding(
                partial(impl.benchmark_get_safety_line, self.db, org_id)),
            # ⚠️ 写类要 run_id（提案要记在哪一条 run 上）⇒ 由 _gate 传进来
            "memory.write_proposal": ToolBinding(
                partial(impl.memory_write_proposal, self.db, org_id, run_id)),
            "memory.invalidate": ToolBinding(
                partial(impl.memory_invalidate, self.db, org_id, run_id)),
            "memory.conflict_resolve": ToolBinding(
                partial(impl.memory_conflict_resolve, self.db, org_id, run_id)),
            # 素材库（B-2 第三批）—— upload → poll_review 是一条**异步链**
            "creative.upload": ToolBinding(partial(impl.creative_upload, self.platform)),
            "creative.poll_review": ToolBinding(
                partial(impl.creative_poll_review, self.platform)),
            "creative.get_asset_tags": ToolBinding(
                partial(impl.creative_get_asset_tags, self.platform)),
            "creative.get_metrics_by_asset": ToolBinding(
                partial(impl.creative_get_metrics_by_asset, self.platform)),
            "creative.search_similar": ToolBinding(
                partial(impl.creative_search_similar, self.platform)),
            # ⚠️ 等待上限受**租约**约束，不是 spec 里那个 600 秒（见 tool_impls 的长注释）
            "system.wait": ToolBinding(
                partial(impl.system_wait, asyncio.sleep, self.config.lease_seconds)),
            # 写工具（B-2 第四批）—— 两条**跨工具前置条件**在实现里硬执行
            "approval.create_case": ToolBinding(
                partial(impl.approval_create_case, self.db, org_id, run_id)),
            "campaign.create": ToolBinding(
                partial(impl.campaign_create, self.platform, self.db, org_id, run_id)),
            "campaign.scale_budget": ToolBinding(
                partial(impl.campaign_scale_budget, self.platform, self.db,
                        org_id, run_id)),
            # 数据源类（B-2 第五批）
            "analysis.feature_lift": ToolBinding(partial(impl.analysis_feature_lift, self.db)),
            "analysis.geo_breakdown": ToolBinding(partial(impl.analysis_geo_breakdown, self.db)),
            "benchmark.get_industry_baseline": ToolBinding(
                partial(impl.benchmark_get_industry_baseline, self.db)),
            "calendar.get_seasonal_context": ToolBinding(
                partial(impl.calendar_get_seasonal_context, self.db)),
            "campaign.detect_anomalies": ToolBinding(
                partial(impl.campaign_detect_anomalies, self.platform)),
            "mmp.get_attribution": ToolBinding(
                partial(impl.mmp_get_attribution, self.platform)),
            "playbook.get_optimization": ToolBinding(
                partial(impl.playbook_get_optimization, self.db)),
            "policy.get_budget_rule": ToolBinding(
                partial(impl.policy_get_budget_rule, self.db, org_id)),
            "risk.check_account": ToolBinding(
                partial(impl.risk_check_account, self.db, org_id)),
        }

    def _gate(self, *, org_id: str, run_id: str) -> ActionGate:
        return ActionGate(
            self.db, self.tools, self._bindings(org_id, run_id),
            org_id=org_id, run_id=run_id,
            over_budget=lambda: self._over_budget(org_id),
            emit=emit, audit=audit,
            amount_threshold=self.config.amount_threshold)

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
                                  lease_seconds=self.config.lease_seconds,
                                  org_id=self.config.org_id)
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
            #   终态事件由 finish_run 在同一事务里发，这里不再补发（发了就是重复）。
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
            # 终态事件（run.cancelled）由 finish_run 发 —— D 档被拒是「取消」不是「失败」，
            # 此前这里发 run.failed 而库里记 cancelled，事件和状态说的是两回事。
            await finish_run(self.db, org_id=org_id, run_id=run_id, status="cancelled",
                             error="tier_d_never_automated")
            return

        # ---- 已经被人裁决过？那这一跑是**恢复**，不是重新决策 ----
        decided = await approved_action(self.db, org_id=org_id, run_id=run_id)
        if decided is not None and decided["status"] == "rejected":
            await audit(self.db, org_id=org_id, run_id=run_id,
                        action="approval_rejected", object_key=decided["case_ref"],
                        param_source="system", detail=None)
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
        # ★ 2026-08-19：改走 `ActionGate` —— 横切从"代码顺序保证"变成"绕不过去"。
        #   这一版行为不变，是为了先证明收口能承载**现有全部横切**，
        #   再把写死的计划换成模型驱动的循环（B-3 的下半段）。
        gate = self._gate(org_id=org_id, run_id=run_id)
        gate.skip_triggers = decided is not None
        metrics = await gate.invoke(tool="campaign.get_metrics",
                                    arguments={"campaign_id": "CMP_1"},
                                    ctx=ctx, param_source="system")
        await record_usage(self.db, org_id=org_id, run_id=run_id,
                           tokens_in=800, tokens_out=120, cost_micros=1_200)
        if metrics.status != "ok":
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

        await record_step(self.db, org_id=org_id, run_id=run_id, step=2, phase="act")
        # ★ 网关触发 / 成本闸 / 权限 / 幂等 / 审计 / 事件 **全部在收口里**，
        #   这里只提出一次动作。`skip_triggers` 在上面按"是否已被人裁决"设过了。
        # ★ param_source="user"：这个金额是用户要求的，不是从工具返回里读出来的。
        written = await gate.invoke(
            tool="campaign.update_budget",
            # ⚠️ `client_request_id` 是沙盒 spec 的**必填参数** —— 此前这条编排没传，
            #   而 B-6 加上参数校验之后当场炸了 12 条测试。判据是对的，该补的是这里。
            #   ★ 取值必须**确定性**（从 run_id 推）：重试要推出同一个键，
            #     否则"有意的第二次"和"重放"就分不开（见 platform.update_budget 的注释）。
            arguments={"campaign_id": "CMP_1", "new_budget": new_budget,
                       "client_request_id": f"{run_id}:budget"},
            ctx=ctx, param_source="user",
            rationale=f"用户请求：{user_message[:200]}")

        if written.status == "halted":
            return                                   # 已开审批单、已发 waiting_for_user
        if written.status == "refused":
            await finish_run(self.db, org_id=org_id, run_id=run_id, status="cancelled",
                             error=written.error)
            return
        if written.status != "ok":
            await finish_run(self.db, org_id=org_id, run_id=run_id, status="failed",
                             error=written.error)
            return

        await finish_run(self.db, org_id=org_id, run_id=run_id, status="succeeded",
                         result={"new_budget": new_budget})

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


# --------------------------------------------------------------------------
# 进程入口：`python -m syncopate.runtime.worker`
#
# ★ 此前 Worker 只在测试里被实例化过 —— 服务的"起法"里只写了 uvicorn（API），
#   worker 没有进程入口，等于队列永远没有消费者（机制在但没接上的又一形态）。
# ⚠️ 平台是 FakeAdPlatform：按 D-2 的决定不对接真实广告平台，这就是生产形状。
# --------------------------------------------------------------------------


async def _serve(config: WorkerConfig) -> None:
    import signal

    db = Database()
    await db.connect()
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        # 收到信号只置位，不 sys.exit —— 让在飞的 run 走完当前动作再放 lease。
        loop.add_signal_handler(sig, stop.set)
    worker = Worker(db, FakeAdPlatform(), config)
    try:
        await worker.serve(stop=stop)
    finally:
        await db.close()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Syncopate runtime worker")
    parser.add_argument("--worker-id", default="worker-1")
    parser.add_argument("--concurrency", type=int, default=WorkerConfig.concurrency)
    parser.add_argument("--org-id", default=None,
                        help="限定只消费一个租户的队列（默认全局）")
    args = parser.parse_args()
    asyncio.run(_serve(WorkerConfig(worker_id=args.worker_id,
                                    concurrency=args.concurrency,
                                    org_id=args.org_id)))


if __name__ == "__main__":
    main()
