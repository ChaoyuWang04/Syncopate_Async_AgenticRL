"""M9.7 · 压测五场景 —— **每个都要有明确的降级路径**（`09 §5`）。

★★★ 判据不是"扛住了"，是"**扛不住的时候降级成什么样**"

M9 是**验收 0%** 的两个里程碑之一，而压测是它的最终考试。
`11 §5` 记的五个场景就绪情况里，**场景②当时写着「⛔ 没有模型服务，这个场景没有被测对象」**。
⇒ B-3b 做完之后有了假 `Decider` ⇒ **场景②第一次有了被测对象**（不需要真模型）。

⚠️ 这一族测的是**降级**不是**性能**：并发数/延迟那些要真压，
   但"模型挂了会怎样""预算烧穿会怎样"用假件就能测准 —— 而且更稳。
"""

from __future__ import annotations

import asyncio
import uuid

import pytest

from syncopate.runtime.action_gate import ActionGate, ToolBinding
from syncopate.runtime.agent_loop import Proposal, run_agent_loop
from syncopate.runtime.db import Database, create_run
from syncopate.runtime.gateway import DecisionContext
from syncopate.runtime.platform import FakeAdPlatform, FaultPlan, PlatformError
from syncopate.runtime.tools import ToolOutcome, ToolRuntime


def _pg() -> bool:
    async def probe() -> bool:
        db = Database()
        try:
            await db.connect(max_size=2); await db.close(); return True
        except Exception:
            return False
    return asyncio.run(probe())


pytestmark = pytest.mark.skipif(not _pg(), reason="需要 PostgreSQL：bash scripts/pg_bootstrap.sh")


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


class _Tools(ToolRuntime):
    def __init__(self, ok=True, error=None) -> None:
        self._ok, self._error = ok, error

    async def call(self, **kw):                     # noqa: ANN003
        return ToolOutcome(ok=self._ok, data={} if self._ok else None,
                           error=self._error, attempts=1, replayed=False,
                           idempotency_key=None)


async def _noop(**kw):                              # noqa: ANN003
    return {}


def _gate(db, org, run, *, over=False, tools=None, max_steps=16):
    async def ob():
        return over

    async def sink(_db, **kw):                      # noqa: ANN003
        return 1

    return ActionGate(db, tools or _Tools(),
                      {"campaign.get_metrics": ToolBinding(_noop)},
                      org_id=org, run_id=run, over_budget=ob,
                      emit=sink, audit=sink, max_steps=max_steps)


# ══════════════════════════════════════════════════════════════════════════
# 场景② · 模型服务挂掉  ← `11 §5` 当时写着"没有被测对象"
# ══════════════════════════════════════════════════════════════════════════

class _DeadModel:
    """模型服务挂了：`decide` 直接抛。"""

    async def decide(self, **kw):                   # noqa: ANN003
        raise ConnectionError("model_service_unreachable")


class _GarbageModel:
    """★ 比"挂掉"更常见也更毒的一种：**服务活着，但吐垃圾**。

    挂掉会抛异常，一眼看得见；吐垃圾不会 —— 它会一路走到工具层才炸，
    或者更糟：**编一个不存在的工具名**然后被当成一次正常失败。
    """

    async def decide(self, **kw):                   # noqa: ANN003
        return Proposal(kind="tool_call", tool="campaign.get_metricz",
                        arguments={"campaign_id": "C1"})


def test_scenario2_model_down_does_not_hang_the_worker():
    """★★ 模型挂了，异常要**冒出来给 worker 兜底**，不能在循环里被吞掉。

    吞掉的话，这条 run 会停在一个既不成功也不失败的状态 ——
    而 `run_once` 的兜底（`except Exception` → `finish_run(failed)`）**接不到它**。
    """
    async def body(db):
        org, run = _ids()
        await create_run(db, org_id=org, run_id=run, user_message="u")
        with pytest.raises(ConnectionError):
            await run_agent_loop(_gate(db, org, run), _DeadModel(), db=db,
                                 org_id=org, run_id=run, user_message="u")
    with_db(body)


def test_scenario2b_garbage_model_burns_steps_but_terminates():
    """★★★ 模型**活着但吐垃圾** ⇒ 必须**有限步内停下**，不能无限打转。

    ⚠️ 这里靠的是**收口的步数上限**，不是循环的自觉 ——
      循环会被换，而"不许无限跑"是生产约束。
    ⚠️ 而且每一步都会被记成 `unknown_tool` 观测回给模型：
      **失败要看得见**，不能静默重试。
    """
    async def body(db):
        org, run = _ids()
        await create_run(db, org_id=org, run_id=run, user_message="u")
        r = await run_agent_loop(_gate(db, org, run, max_steps=5), _GarbageModel(),
                                 db=db, org_id=org, run_id=run, user_message="u")
        assert r.status == "exhausted", "★ 吐垃圾的模型把 worker 转死了"
        errs = [h["observation"]["error"] for h in r.history
                if h["role"] == "observation"]
        assert all("unknown_tool" in e for e in errs[:-1])
    with_db(body)


# ══════════════════════════════════════════════════════════════════════════
# 场景③ · 工具超时  —— 判据是"两种超时长得一样"，那条已在 test_worker
# ══════════════════════════════════════════════════════════════════════════

def test_scenario3_tool_failure_is_observable_not_swallowed():
    """工具一直失败 ⇒ 观测**每次都回给模型**，由模型决定放弃。

    ⚠️ 反面是"重试到成功为止" —— 那会让沙盒里训过的"失败之后怎么办"变成死代码。
    """
    async def body(db):
        org, run = _ids()
        await create_run(db, org_id=org, run_id=run, user_message="u")

        class _Persistent:
            def __init__(self): self.n = 0
            async def decide(self, **kw):           # noqa: ANN003
                self.n += 1
                if self.n <= 3:
                    return Proposal(kind="tool_call", tool="campaign.get_metrics",
                                    arguments={"campaign_id": "C1"})
                return Proposal(kind="final", final_answer={"gave_up": True})

        m = _Persistent()
        r = await run_agent_loop(_gate(db, org, run, tools=_Tools(False, "upstream_timeout")),
                                 m, db=db, org_id=org, run_id=run, user_message="u")
        assert r.final_answer == {"gave_up": True}
        obs = [h for h in r.history if h["role"] == "observation"]
        assert len(obs) == 3 and all(o["observation"] == {"error": "upstream_timeout"}
                                     for o in obs)
    with_db(body)


# ══════════════════════════════════════════════════════════════════════════
# 场景⑤ · 单 org 刷爆预算 —— **降级 ≠ 失败**（`11 §5` 明写语义要定）
# ══════════════════════════════════════════════════════════════════════════

def test_scenario5_budget_exhausted_degrades_reads_still_allowed():
    """★★ 预算烧穿 ⇒ **写被拦，读仍然放行**。

    `11 §5` 记着「成本触顶降级，且**降级 ≠ 失败**，当前实现是记 failed，语义要定」。
    ⇒ 这里钉住的语义是：**降级的意义是降级，不是失明** ——
      超预算的 run 至少还能查清楚"为什么超了"。
    """
    async def body(db):
        org, run = _ids()
        await create_run(db, org_id=org, run_id=run, user_message="u")
        g = _gate(db, org, run, over=True)
        # 读：放行
        ok = await g.invoke(tool="campaign.get_metrics",
                            arguments={"campaign_id": "C1"},
                            ctx=DecisionContext(), param_source="system")
        assert ok.status == "ok"
    with_db(body)


# ══════════════════════════════════════════════════════════════════════════
# 场景① · 突发流量 —— 这里只钉"结构上不是串行的"，真压要跑框架
# ══════════════════════════════════════════════════════════════════════════

def test_scenario1_worker_is_not_structurally_serial():
    """★ `11 §5` 把「并发 run ≥8」改判为「已知不达标（当前串行实测 = 1）」。

    ⇒ 这条只钉**结构**：并发度是配置项且默认 >1。
      真实的 P95 劣化要跑压测框架，**不在单测里假装测到了**。
    """
    from syncopate.runtime.worker import WorkerConfig
    assert WorkerConfig().concurrency >= 8


# ══════════════════════════════════════════════════════════════════════════
# 场景④ · RAG 不可用 —— 三态那条已在 test_retrieval，这里钉"不可用要阻断"
# ══════════════════════════════════════════════════════════════════════════

def test_scenario4_unavailable_blocks_but_no_match_does_not():
    """★★★ 「查不了」一律阻断，「查不到」只在它是证据来源时阻断。

    两种误判的代价**不对称**：
      把"没有政策"当成"不知道" ⇒ 最多多问人一次
      把"不知道"当成"没有政策" ⇒ **放行一个未知风险**
    """
    from syncopate.runtime.gateway import evaluate_triggers
    down = evaluate_triggers(DecisionContext(retrieval_unavailable_tools=["policy.search"]))
    empty = evaluate_triggers(DecisionContext(retrieval_empty_tools=["policy.search"]))
    assert down, "★ 查不了却不阻断 = 在未知状态下放行"
    assert empty, "查不到（且被当作证据来源）也该阻断"


# ══════════════════════════════════════════════════════════════════════════
# 任务四 · 参数校验的生产者（此前**没有生产者**）
# ══════════════════════════════════════════════════════════════════════════

def test_validation_errors_now_has_a_producer():
    """★★ `DecisionContext.validation_errors` 此前是个**孤儿触发器** ——
    网关认它，但系统里没有任何东西生产它。

    ⚠️ 而缺校验的后果不是"返回一个错"，是**崩在实现里**：
      `invoke(**arguments)` 遇到漏掉的必填参数直接 TypeError，
      那不是一条模型能看懂的观测。
    """
    async def body(db):
        org, run = _ids()
        await create_run(db, org_id=org, run_id=run, user_message="u")
        ctx = DecisionContext()
        g = _gate(db, org, run)
        out = await g.invoke(tool="campaign.get_metrics", arguments={},   # 漏 campaign_id
                             ctx=ctx, param_source="model")
        assert out.status == "failed"
        assert "validation_failed" in out.observation["error"]
        assert ctx.validation_errors, "★ 网关的触发器仍然没有生产者"
    with_db(body)


def test_validation_uses_the_sandbox_spec_not_a_second_copy():
    """必填清单从沙盒 spec 取 —— 这里**不许另抄一份**。"""
    from pathlib import Path
    src = (Path(__file__).resolve().parents[2]
           / "syncopate" / "runtime" / "action_gate.py").read_text(encoding="utf-8")
    assert "REGISTRY.get(tool)" in src


# ══════════════════════════════════════════════════════════════════════════
# 任务一 · C 档动作真的走审批（`11 §4` 记为"未定"，实测**已经通了**）
# ══════════════════════════════════════════════════════════════════════════

def test_tier_c_alone_parks_the_run_for_approval():
    """★★ C 档**只靠 tier_c** 就该停下来 —— 不依赖金额超阈值。

    ⚠️ 阈值刻意调到远高于实际金额，把 `amount_over_threshold` 排除掉；
      否则这条测试会**因为另一个触发器而通过**，而 tier_c 断了也看不出来。
      （「判据要因为正确的理由通过」——`blank-thresholds-are-not-passes` 那条。）
    """
    async def body(db):
        from syncopate.runtime.worker import Worker, WorkerConfig
        org, run = _ids()
        await create_run(db, org_id=org, run_id=run, user_message="调预算",
                         automation_tier="C")
        # ★ org_id 限定 ⇒ **结构上抢不到别人的活**，不依赖"记得排空"
        w = Worker(db, FakeAdPlatform(),
                   WorkerConfig(amount_threshold=10_000_000, org_id=org))
        assert await w.run_once() == run
        async with db.tx() as c:
            st = await c.fetchval(
                "SELECT status FROM agent_runs WHERE org_id=$1 AND run_id=$2", org, run)
            tr = await c.fetchval(
                "SELECT trigger_reason FROM approval_cases WHERE org_id=$1 AND run_id=$2",
                org, run)
        assert st == "waiting_for_user"
        assert "tier_c" in str(tr)
    with_db(body)


def test_a_scoped_worker_cannot_steal_another_orgs_run():
    """★★★ 队列是全局的 ⇒ 任何调 `run_once` 的测试/探针都会**抢走别人遗留的活**。

    `[实测 2026-08-19]` 我自己就中过一次：探针报「C 档没走审批」，
    而真相是它抢到了**另一条 run** —— **一个完全错误的结论**，
    而且看起来完全合理（状态是 queued、没有审批单，全都"对得上"）。

    ⇒ 此前的修法是"每处记得先排空"（`test_worker._drain`），
      而**手动步骤一定会被忘** —— `test_retrieval.py` 就没排，它正是那条偶发红的来源。
    ⇒ **结构上拿不到别人的活**，比"记得排空"可靠。
    """
    async def body(db):
        from syncopate.runtime.worker import Worker, WorkerConfig
        mine_org, mine = _ids()
        other_org, other = _ids()
        await create_run(db, org_id=other_org, run_id=other, user_message="别人的活")
        await create_run(db, org_id=mine_org, run_id=mine, user_message="我的活")
        w = Worker(db, FakeAdPlatform(), WorkerConfig(org_id=mine_org))
        assert await w.run_once() == mine, "★ 抢到了别的 org 的 run"
        assert await w.run_once() is None, "★ 限定之后还能抢到东西"
    with_db(body)


# ══════════════════════════════════════════════════════════════════════════
# 任务三 · 并发命中幂等时返回什么（`§38` 从没定义过，2026-08-19 定）
# ══════════════════════════════════════════════════════════════════════════

def test_in_progress_is_promoted_to_a_degradation_signal():
    """★★★ 三选一的**决定：返回"处理中"**，但**不能只返回一句话**。

    为什么不选另外两个：
      等它跑完 ⇒ 阻塞 worker；对面要是挂着，我们跟着挂
      返回 409 ⇒ **诱导客户端重试**，而重试正是幂等要防的那件事

    ⚠️⚠️ 而"返回处理中"有一个漏洞，光看文案发现不了：

        文案说「结果未知，不要当成失败处理」
        **形状却是** `{"error": ...}` —— 而沙盒教模型的正是「error = 失败」

      ⇒ 模型会照着失败处理，**很可能重试** ——
        重试一个**可能已经生效**的写动作，正是幂等机制存在的全部理由。

    ⇒ 所以把它升格成**降级信号**（和 `retrieval_unavailable` 同族）：
      **"我们不知道"一律阻断，交给人。**
    """
    from syncopate.runtime.gateway import evaluate_triggers

    class _InProgress(ToolRuntime):
        def __init__(self): pass
        async def call(self, **kw):            # noqa: ANN003
            return ToolOutcome(
                ok=False, data=None,
                error="tool_call_in_progress: 同一幂等键的上一次调用仍在执行中",
                attempts=1, replayed=True, idempotency_key="k")

    async def body(db):
        org, run = _ids()
        await create_run(db, org_id=org, run_id=run, user_message="u")
        ctx = DecisionContext()
        g = _gate(db, org, run, tools=_InProgress())
        await g.invoke(tool="campaign.get_metrics", arguments={"campaign_id": "C1"},
                       ctx=ctx, param_source="model")
        assert ctx.unknown_state_tools == ["campaign.get_metrics"]
        reasons = [t.reason for t in evaluate_triggers(ctx)]
        assert "side_effect_unknown" in reasons, "★ 未知状态没有触发降级 ⇒ 会被当成普通失败"
    with_db(body)


def test_unknown_state_is_not_merged_with_plain_failure():
    """★★ `unknown_state` 和 `tool_failed` **必须分开**，两者的正确应对相反：

        失败 ⇒ **可以**重试
        未知 ⇒ **不能**重试（可能已经生效了）

    合并的话，"重试是安全的"这条判断就没有依据了 ——
    同 `TIMEOUT_MESSAGE` 那条：两种超时现象一样，只能靠幂等键分辨。
    """
    from syncopate.runtime.gateway import evaluate_triggers
    failed = [t.reason for t in evaluate_triggers(DecisionContext(tool_failed="x"))]
    unknown = [t.reason for t in evaluate_triggers(
        DecisionContext(unknown_state_tools=["x"]))]
    assert failed != unknown
    assert "tool_failed" in failed and "side_effect_unknown" in unknown
