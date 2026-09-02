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
import time
from dataclasses import dataclass
from typing import Any

from syncopate.core.contract import IS_V15, REPORT_TOOL
from syncopate.core.session_signals import ack_payload
from syncopate.runtime.action_gate import ActionGate, ToolBinding
from syncopate.runtime.db import (MAX_RUN_ATTEMPTS, Database, InvalidRunTransition, append_event,
                                  approved_action, cancel_requested,
                                  claim_run, finish_run, renew_lease, schedule_run_retry,
                                  prior_turns, park_run_for_user)
from syncopate.runtime.gateway import DecisionContext, evaluate_triggers, open_approval_case
from syncopate.runtime.platform import FakeAdPlatform, PlatformError
from syncopate.runtime.retrieval import RetrievalService, RetrievalStatus
from syncopate.runtime import stage_timing as _st
from syncopate.runtime.tools import PermissionDenied, ToolRuntime


def _timed_binding(b: ToolBinding) -> ToolBinding:
    """B-5 分账：给工具计时（含工具内 DB 的嵌套标记）。只在 _st.ENABLED 时被用。"""
    async def invoke(**kw: Any) -> dict[str, Any]:
        _st.tool_enter()
        t0 = time.perf_counter()
        try:
            return await b.invoke(**kw)
        finally:
            _st.add("tool", time.perf_counter() - t0)
            _st.tool_exit()
    return ToolBinding(invoke=invoke)


class _TimedDecider:
    """B-5 分账：decider.decide 的墙钟。属性透传，不改任何语义。"""

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    async def decide(self, **kw: Any) -> Any:
        t0 = time.perf_counter()
        try:
            return await self._inner.decide(**kw)
        finally:
            _st.add("llm", time.perf_counter() - t0)


async def emit(db: Database, *, org_id: str, run_id: str, kind: str,
               payload: dict[str, Any] | None = None) -> int:
    """写一条事件。**seq 由 `db.append_event` 的领号器分配**（K2-2，课件 H13）。

    历史：这里曾是 `max(seq)+1` + 有界重试——lease 交接窗口（旧 worker 收尾 × 新 worker
    已抢到）真的撞过唯一键并炸死过 worker（2026-08-20）。重试只是掩盖，领号器才是修：
    `UPDATE agent_runs SET last_seq=last_seq+1 … RETURNING` 行锁到 COMMIT，两个写者
    结构上不可能拿到同一个号。
    """
    async with db.tx() as conn:
        return await append_event(conn, org_id=org_id, run_id=run_id, kind=kind,
                                  payload=payload)


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
            INSERT INTO usage_records (org_id, run_id, tokens_in, tokens_out, cost_micros,
                                       call_index)
            VALUES ($1,$2,$3,$4,$5,
                    -- K2（H15）：粒度 = 每次执行（attempts）一行。同一次执行重放第二次
                    -- 会撞 usage_records_once —— 那正是"账单翻倍"要被拒的形状；
                    -- 审批恢复是**新的一次执行**（attempts+1），合法地多一行。
                    -- K9-3 改成"每次模型调用一行"时 call_index 换成调用序号。
                    (SELECT attempts FROM agent_runs WHERE org_id=$1 AND run_id=$2))
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


async def _session_report_invoke(**arguments: Any) -> dict[str, Any]:
    """`session.report` 的 runtime 实现：零副作用 ack，轨迹继续（`25 §3.3`）。

    载荷来自 `session_signals.ack_payload` —— 与沙盒 handler **同一份**，
    这样模型在训练里学会读的那个观测形状，线上逐字节一样。
    """
    return ack_payload(REPORT_TOOL, dict(arguments))


# --------------------------------------------------------------------------
# K3-7 · lease 心跳（课件 H30 跨六章未给，这里定死：TTL = 3×心跳；续租失败 = 立即停）
# --------------------------------------------------------------------------


class LeaseHeartbeat:
    """worker 正常运行时定期续租。`lost=True` = 续租返回 0 行（lease 被收走 / run 离开 running），
    ActionGate 的安全点会读它并拒绝再执行（同取消意图一条路）。判据行 `[lease-heartbeat]` 每次必打。"""

    def __init__(self, db: Database, *, org_id: str, run_id: str, worker_id: str,
                 ttl_seconds: int, interval_seconds: float | None = None) -> None:
        self.db, self.org_id, self.run_id, self.worker_id = db, org_id, run_id, worker_id
        self.ttl = ttl_seconds
        self.interval = interval_seconds if interval_seconds is not None else max(1, ttl_seconds // 3)
        self.lost = False
        self.renewals = 0
        self._task: asyncio.Task | None = None

    async def _loop(self) -> None:
        while True:
            await asyncio.sleep(self.interval)
            try:
                ok = await renew_lease(self.db, org_id=self.org_id, run_id=self.run_id,
                                       worker_id=self.worker_id, lease_seconds=self.ttl)
            except Exception as exc:                # noqa: BLE001 —— 续不上 = 视同丢失
                ok = False
                print(f"[lease-heartbeat] run={self.run_id} 续租异常 {exc!r}", flush=True)
            if ok:
                self.renewals += 1
                print(f"[lease-heartbeat] run={self.run_id} owner={self.worker_id} "
                      f"ttl={self.ttl}s renewed#{self.renewals}", flush=True)
                continue
            self.lost = True
            print(f"[lease-heartbeat] 🔴 run={self.run_id} owner={self.worker_id} lease LOST ⇒ 停止执行",
                  flush=True)
            return

    def start(self) -> None:
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None


# 课件 §9.4 急诊分诊表："拿到错误先问再试会不会好"。
RUN_RETRY_BACKOFF_S = (60, 300, 900)       # 1 / 5 / 15 分钟


def classify_error(exc: BaseException) -> str:
    """'transient'（退避重试）| 'permanent'（立即 failed）。⚠️ 超时后能否安全重试还要看副作用
    （K5/K6 意图日志）；本阶段保守：只有**明确未发出/连接层**的错误算 transient。"""
    try:
        import httpx
        if isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout, httpx.RemoteProtocolError,
                            httpx.PoolTimeout)):
            return "transient"
    except ImportError:                            # pragma: no cover
        pass
    if isinstance(exc, PlatformError) and getattr(exc, "retriable", False):
        return "transient"
    if isinstance(exc, (ConnectionError, asyncio.TimeoutError)):
        return "transient"
    return "permanent"


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
                 retrieval: RetrievalService | None = None,
                 decider=None) -> None:
        self.db = db
        self.platform = platform
        self.config = config or WorkerConfig()
        self.tools = ToolRuntime(db)
        self.retrieval = retrieval or RetrievalService(db)
        # B-4：真模型 Decider。None = 写死三步计划（老行为，测试都在它上面）。
        # ⚠️ 显式传入才生效，且入口会打判据行 —— 静默切换是第一失效形状。
        self.decider = decider
        self.alt_deciders: dict = {}   # dev mode：{'sft':…, 'base':…}
        self._heartbeats: dict[str, LeaseHeartbeat] = {}   # run_id → 心跳（Celery 路径才有）

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
        bindings: dict[str, ToolBinding] = {
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
        if IS_V15:
            # ★ v15 信令族里**只有 session.report 会走到收口** —— 三条终止性信令
            #   在 decider 就被解析成 final（agent_loop 特判各自的终止语义，R4①）。
            # ⛔ 2026-08-30 考场实测：report 在菜单里可见、模型也照训练调了，
            #   但收口这边没有 binding ⇒ 判 `unknown_tool`，模型连试 6 次后改调
            #   session.reject，50 道 L1 里 43 道被取消。**又是"机制在但没接上"**：
            #   训练侧我修过同形的 allowlist，运行时这条通道是**另一份名单**。
            # ⛔ 只绑 report：终止性信令若真的走到这里，`unknown_tool` 是**响的**失败，
            #   而 ack 成功会让"要拒绝"被静默吞掉 —— 后者贵得多。
            bindings[REPORT_TOOL] = ToolBinding(_session_report_invoke)
        return bindings

    def _gate(self, *, org_id: str, run_id: str) -> ActionGate:
        bindings = self._bindings(org_id, run_id)
        if _st.ENABLED:                      # B-5 分账：工具计时（默认关）
            bindings = {k: _timed_binding(v) for k, v in bindings.items()}
        return ActionGate(
            self.db, self.tools, bindings,
            org_id=org_id, run_id=run_id,
            over_budget=lambda: self._over_budget(org_id),
            emit=emit, audit=audit,
            amount_threshold=self.config.amount_threshold,
            # K1-4：安全点「工具调用前」读取消意图；命中 ⇒ refused(cancel_requested) ⇒ 下面映射成 cancelled
            cancel_check=lambda: self._should_stop(org_id, run_id))

    # ---- 成本闸：压测场景⑤ ------------------------------------------------

    async def _should_stop(self, org_id: str, run_id: str) -> bool:
        """安全点判定：用户取消意图 ∨ lease 已丢（K3-7：续租失败 = 立即停止执行不再写库）。"""
        hb = self._heartbeats.get(run_id)
        if hb is not None and hb.lost:
            return True
        return await cancel_requested(self.db, org_id=org_id, run_id=run_id)

    async def _over_budget(self, org_id: str) -> bool:
        async with self.db.tx() as conn:
            spent = await conn.fetchval(
                "SELECT COALESCE(sum(cost_micros),0) FROM usage_records "
                "WHERE org_id=$1 AND day=CURRENT_DATE", org_id)
        return spent >= self.config.daily_cost_cap_micros

    # ---- 主循环 ------------------------------------------------------------

    async def run_once(self) -> str | None:
        """轮询模式：抢**任意**一条并跑完。返回 run_id；没活干返回 None。
        ⚠️ 生产投递走 Celery（celery_app._execute → execute_claimed）；这条轮询路只给
        测试、探针与 26 线的考场链（它们直接起 `python -m syncopate.runtime.worker`）。
        两条入口共用 `execute_claimed` 这**一条**执行路径。"""
        claimed = await claim_run(self.db, worker_id=self.config.worker_id,
                                  lease_seconds=self.config.lease_seconds,
                                  org_id=self.config.org_id)
        if claimed is None:
            return None
        return await self.execute_claimed(claimed)

    async def execute_claimed(self, claimed: dict, *, heartbeat: LeaseHeartbeat | None = None) -> str:
        """已经 claim 到手的 run 跑完（唯一的执行路径）。

        错误分层（课件 §14.3）：业务错误在这里被消化并走状态迁移——transient ⇒ 回 queued +
        outbox 延迟重投（`schedule_run_retry`），permanent ⇒ failed；⛔ 不向调用方抛业务异常。
        """
        org_id, run_id = claimed["org_id"], claimed["run_id"]
        if heartbeat is not None:
            self._heartbeats[run_id] = heartbeat
        _tok = _st.begin_run(run_id)         # B-5 分账（默认 no-op）
        try:
            # run.started / run.restarted 由 claim 时的 transition_run 写（K4），这里不再补发
            await self._execute(org_id=org_id, run_id=run_id,
                                user_message=claimed["user_message"] or "",
                                automation_tier=claimed.get("automation_tier"),
                                intent=claimed.get("intent"),
                                conversation_id=claimed.get("conversation_id"))
        except Exception as exc:                      # noqa: BLE001
            # ★ 兜底：worker 不能因为一条 run 挂掉而停摆（压测场景②）。
            #   终态事件由 finish_run 在同一事务里发，这里不再补发（发了就是重复）。
            kind = classify_error(exc)
            attempts = int(claimed.get("attempts") or 1)
            try:
                if kind == "transient" and attempts < MAX_RUN_ATTEMPTS:
                    delay = RUN_RETRY_BACKOFF_S[min(attempts - 1, len(RUN_RETRY_BACKOFF_S) - 1)]
                    await schedule_run_retry(self.db, org_id=org_id, run_id=run_id,
                                             error=str(exc)[:500], delay_seconds=delay)
                    print(f"[run-retry] run={run_id} attempt={attempts} transient {exc!r} ⇒ {delay}s 后重投",
                          flush=True)
                else:
                    await finish_run(self.db, org_id=org_id, run_id=run_id,
                                     status="failed", error=str(exc)[:500])
            except InvalidRunTransition as it:
                # run 已不在 running（被 sweeper/取消收走）：状态机说不许，就不许——只留一行日志
                print(f"[worker] 🔴 run={run_id} 兜底写终态被状态机拒绝：{it}", flush=True)
        finally:
            _st.end_run(_tok)
            self._heartbeats.pop(run_id, None)
        return run_id

    async def _execute(self, *, org_id: str, run_id: str, user_message: str,
                       automation_tier: str | None = None,
                       intent: str | None = None,
                       conversation_id: str | None = None) -> None:
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

        # ---- B-4：模型驱动的循环（decider 显式接入时）----------------------
        # ★ agent_loop 写好带测试却从没接进 worker（09 §0 记的那条缺口）——这里接上。
        #   D 档拒绝 / 审批已拒 / 进门成本闸 三段共享前置在上面，两条路一致。
        if self.decider is not None:
            await self._execute_with_loop(org_id=org_id, run_id=run_id,
                                          user_message=user_message, ctx=ctx,
                                          resumed=decided is not None, intent=intent,
                                          conversation_id=conversation_id)
            return

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

    async def _execute_with_loop(self, *, org_id: str, run_id: str,
                                 user_message: str, ctx, resumed: bool,
                                 intent: str | None = None,
                                 conversation_id: str | None = None) -> None:
        """B-4：模型驱动的执行路径。横切全在 ActionGate 里，这里只做状态映射。"""
        from syncopate.runtime.agent_loop import (MODEL_USAGE, PRIOR_TURNS, RUN_INTENT,
                                                  run_agent_loop)

        gate = self._gate(org_id=org_id, run_id=run_id)
        gate.skip_triggers = resumed          # 只有"人已裁决"才跳过网关；恢复本身不跳
        usage: dict[str, int] = {}
        token = MODEL_USAGE.set(usage)
        intent_token = RUN_INTENT.set(intent)
        # ★ 多轮壳层：同会话之前几轮的问答进 prompt（会话外的 run 没有历史 = 单轮，
        #   行为与之前逐字节相同）。
        turns = ([] if not conversation_id else
                 await prior_turns(self.db, org_id=org_id,
                                   conversation_id=conversation_id,
                                   before_run_id=run_id))
        prior_token = PRIOR_TURNS.set(turns)
        # dev mode 模型路由（Chaoyu 08-29）：会话建时锁定的 model 标签选 decider；
        # 无标签/无对应端点 ⇒ 默认 decider（rl）。判据行必打——静默回退是记录在案的失效家族。
        chosen = self.decider
        if conversation_id and self.alt_deciders:
            async with self.db.tx() as _c:
                _row = await _c.fetchrow(
                    "SELECT model FROM conversations WHERE org_id=$1 AND conversation_id=$2",
                    org_id, conversation_id)
            _tag = (_row and _row["model"]) or "rl"
            if _tag != "rl":
                if _tag in self.alt_deciders:
                    chosen = self.alt_deciders[_tag]
                    print(f"[decider-route] run={run_id} model={_tag}", flush=True)
                else:
                    print(f"[decider-route] 🔴 run={run_id} 请求 model={_tag} 但无端点，回退 rl",
                          flush=True)
        try:
            decider = _TimedDecider(chosen) if _st.ENABLED else chosen
            # K5-2：**任何**一次执行都从最新快照接着走（没有快照 = 空 = 从头）。此前只有审批裁决后
            # 才 resume ⇒ 崩溃/重投后的执行从头重跑（29 D5）。resumed（审批已裁决）只控制 skip_triggers。
            result = await run_agent_loop(gate, decider, db=self.db,
                                          org_id=org_id, run_id=run_id,
                                          user_message=user_message, ctx=ctx,
                                          resume=True)
        finally:
            PRIOR_TURNS.reset(prior_token)
            RUN_INTENT.reset(intent_token)
            MODEL_USAGE.reset(token)
            if usage.get("calls"):
                # 计价是工程值（in×1 + out×4 micros）：成本闸要的是"有单调的账"，
                # 真实单价接真平台时再定；token 数本身是 §19 成本指标的口径。
                await record_usage(
                    self.db, org_id=org_id, run_id=run_id,
                    tokens_in=usage.get("tokens_in", 0),
                    tokens_out=usage.get("tokens_out", 0),
                    cost_micros=usage.get("tokens_in", 0) + 4 * usage.get("tokens_out", 0))

        if result.status == "finished":
            await finish_run(self.db, org_id=org_id, run_id=run_id,
                             status="succeeded", result=result.final_answer)
        elif result.status == "awaiting_reconciliation":
            # K5-3 分支 C：写工具结果未知（response_lost）。⛔ 禁止自动重试；
            # 回队列延迟重投，等对账（K8）按幂等键回填后，loop 从意图日志接着走。
            await emit(self.db, org_id=org_id, run_id=run_id, kind="tool.manual_review",
                       payload={"tool": result.error, "reason": "response_lost"})
            await schedule_run_retry(self.db, org_id=org_id, run_id=run_id,
                                     error=f"response_lost:{result.error}", delay_seconds=300)
        elif result.status == "halted":
            if result.case_ref is None:
                # ★ 09-02（`26 §2.5` Ⓐ）：没有审批单的挂起 = session.clarify。
                #   此前这里一律 return ⇒ run 停在 running 被 lease 重抢；现在置 waiting_for_user。
                fa = result.final_answer or {}
                await park_run_for_user(
                    self.db, org_id=org_id, run_id=run_id, result=fa,
                    payload={"signal": fa.get("signal"),
                             "question": (fa.get("arguments") or {}).get("question", "")})
            return                              # 审批单已开（或已挂起等补充），等人
        elif result.status == "exhausted":
            # 收口的 refused 有两族：政策性拒绝（灰测/成本）= 取消；步数上限 = 失败
            # ★ session_reject 是模型**做对了**（越权/离题该拒），归"取消"不归"失败" ——
            #   归失败会让线上尺子（人工修正率、失败率）把正确的拒绝算成事故。
            status = ("cancelled" if result.error in
                      ("release_gate", "daily_cost_cap_exceeded", "session_reject",
                       "cancel_requested")           # K1-4：用户取消 = 取消，不是失败
                      else "failed")
            # ★ 09-02（Ⓑ）：拒绝轮要进历史 ⇒ result 存信令自己的话（prior_turns 认它）
            await finish_run(self.db, org_id=org_id, run_id=run_id,
                             status=status, error=result.error,
                             result=(result.final_answer
                                     if result.error == "session_reject" else None))
        else:                                   # failed（连续解析失败等）
            await finish_run(self.db, org_id=org_id, run_id=run_id,
                             status="failed", error=result.error)

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


async def build_worker(config: WorkerConfig, *, pool_size: int | None = None) -> tuple[Database, "Worker"]:
    """轮询入口（_serve）与 Celery 入口（celery_app.worker_process_init）共用的构造：
    连接池 + decider（env 显式接入，判据行必打）+ 假平台 fixture。"""
    import os

    db = Database()
    # B-5 S1：池容量 env 可配（默认 10 不变）。S0 实测 C=96 借连接等待占 24-29%。
    await db.connect(max_size=pool_size or int(os.environ.get("SYNCOPATE_WORKER_DB_POOL", "10")))
    # B-4：SYNCOPATE_DECIDER_URL 显式设了才接真模型；判据行必打（没有 = 没接上）。
    from syncopate.runtime.decider import build_decider_from_env, build_alt_deciders_from_env
    decider = build_decider_from_env()
    alt_deciders = build_alt_deciders_from_env()   # dev mode：{"sft":…, "base":…}（可空）
    if decider is not None:
        print(f"[decider] vllm model={decider.model} tools={len(decider.tools)} "
              f"—— agent_loop 驱动", flush=True)
    else:
        print("[decider] 未配置（SYNCOPATE_DECIDER_URL 空）⇒ 写死三步计划", flush=True)
    platform = FakeAdPlatform.from_fixture()
    slow = float(os.environ.get("SYNCOPATE_TEST_SLOW_SECONDS", "0") or 0)
    if slow > 0:                                    # 测试钩子：给 kill 注入留出窗口
        platform.faults.latency_seconds = slow
        print(f"[worker] TEST 钩子：平台每次调用慢 {slow}s", flush=True)
    worker = Worker(db, platform, config, decider=decider)
    worker.alt_deciders = alt_deciders
    if alt_deciders:
        print(f"[decider] dev mode 多模型：{sorted(alt_deciders)}（会话级锁定）", flush=True)
    return db, worker


async def _serve(config: WorkerConfig) -> None:
    import signal

    db, worker = await build_worker(config)
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        # 收到信号只置位，不 sys.exit —— 让在飞的 run 走完当前动作再放 lease。
        loop.add_signal_handler(sig, stop.set)
    print(f"[worker] mode=poll worker_id={config.worker_id} org={config.org_id or '*'} "
          f"concurrency={config.concurrency}（生产投递 = Celery，见 celery_app）", flush=True)
    try:
        await worker.serve(stop=stop)
    finally:
        if worker.decider is not None:
            await worker.decider.aclose()
        await db.close()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Syncopate runtime worker")
    parser.add_argument("--worker-id", default="worker-1")
    parser.add_argument("--concurrency", type=int, default=WorkerConfig.concurrency)
    parser.add_argument("--org-id", default=None,
                        help="限定只消费一个租户的队列（默认全局）")
    parser.add_argument("--daily-cost-cap-micros", type=int,
                        default=WorkerConfig.daily_cost_cap_micros,
                        help="单 org 日预算（默认不变；压测 org 的阶梯会在 ~300 run 处刷爆"
                             "默认值并让其后所有 run 秒失败——B-4 实录，E32 §7）")
    args = parser.parse_args()
    asyncio.run(_serve(WorkerConfig(worker_id=args.worker_id,
                                    concurrency=args.concurrency,
                                    org_id=args.org_id,
                                    daily_cost_cap_micros=args.daily_cost_cap_micros)))


if __name__ == "__main__":
    main()
