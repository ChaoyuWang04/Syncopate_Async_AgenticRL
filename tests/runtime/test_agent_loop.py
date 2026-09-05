"""模型驱动的循环（B-3b）：用**假模型**测循环本身。

★ 为什么用假模型：循环的正确性判据（观测怎么回到模型、失败谁来决定、
  审批中断后怎么恢复、步数谁来管）**和模型好不好无关**。
  用真模型测这些，等于把"循环对不对"和"模型聪不聪明"绑在一起 ——
  而那正是这个项目一路在拆的那种耦合。
⇒ 接真模型只是换一个 `decide` 实现（B-4）。
"""

from __future__ import annotations

import asyncio
import uuid

import pytest

from syncopate.runtime.action_gate import ActionGate, ToolBinding
from syncopate.runtime.agent_loop import (
    Proposal, load_transcript, run_agent_loop,
)
from syncopate.runtime.db import Database, create_run
from syncopate.runtime.gateway import DecisionContext
from syncopate.runtime.tools import ToolOutcome, ToolRuntime


def _pg() -> bool:
    async def probe() -> bool:
        db = Database()
        try:
            await db.connect(max_size=2); await db.close(); return True
        except Exception:
            return False
    return asyncio.run(probe())


pytestmark = pytest.mark.skipif(not _pg(), reason="需要 PostgreSQL：bash scripts/serving/pg_bootstrap.sh")


def with_db(body):
    async def main():
        db = Database()
        await db.connect(max_size=5)
        try:
            return await body(db)
        finally:
            await db.close()
    return asyncio.run(main())


class _Script:
    """假模型：照剧本逐条吐提议，并把每次看到的 history 长度记下来。"""

    def __init__(self, *proposals: Proposal) -> None:
        self._q = list(proposals)
        self.seen_history: list[list] = []

    async def decide(self, *, user_message, history):      # noqa: ANN001
        self.seen_history.append(list(history))
        return self._q.pop(0) if self._q else Proposal(kind="final",
                                                       final_answer={"done": True})


class _Tools(ToolRuntime):
    def __init__(self, ok: bool = True, error: str | None = None) -> None:
        self._ok, self._error = ok, error

    async def call(self, **kw):                            # noqa: ANN003
        return ToolOutcome(ok=self._ok, data={"v": 1} if self._ok else None,
                           error=self._error, attempts=1, replayed=False,
                           idempotency_key=None)


async def _noop(**kw):                                     # noqa: ANN003
    return {}


def _gate(db, org, run, *, tools=None, max_steps=16, over=False, bindings=None):
    async def _ob():
        return over

    async def _emit(_db, **kw):                            # noqa: ANN003
        return 1

    async def _audit(_db, **kw):                           # noqa: ANN003
        return None

    return ActionGate(db, tools or _Tools(), bindings or {"t": ToolBinding(_noop)},
                      org_id=org, run_id=run, over_budget=_ob,
                      emit=_emit, audit=_audit, max_steps=max_steps)


def _ids():
    return f"org_{uuid.uuid4().hex[:8]}", f"run_{uuid.uuid4().hex[:8]}"


# ── 观测一律回到模型 ────────────────────────────────────────────────────

def test_a_failed_observation_goes_back_to_the_model():
    """★★★ 失败的观测**必须回到模型**，循环不许自己吞掉。

    吞掉就等于把「失败之后怎么办」那段策略变成死代码 ——
    而沙盒里专门训过这一段（`tools.py` 也记着同一条：不许重试到成功为止）。
    """
    async def body(db):
        org, run = _ids()
        await create_run(db, org_id=org, run_id=run, user_message="u")
        model = _Script(Proposal(kind="tool_call", tool="t"),
                        Proposal(kind="final", final_answer={"ok": False}))
        r = await run_agent_loop(_gate(db, org, run, tools=_Tools(False, "boom")),
                                 model, db=db, org_id=org, run_id=run, user_message="u")
        assert r.status == "finished"
        obs = [h for h in r.history if h["role"] == "observation"]
        # ★ 形状同沙盒：失败 = 只含 error 的字典
        assert obs[0]["observation"] == {"error": "boom"}
        # ★ 而且模型**第二轮真的看到了**它
        assert any(h["role"] == "observation" for h in model.seen_history[1])
    with_db(body)


def test_the_loop_does_not_decide_to_stop_on_failure_the_model_does():
    """★ 动作失败**不终止循环** —— 由模型决定重试 / 换工具 / 说做不了。"""
    async def body(db):
        org, run = _ids()
        await create_run(db, org_id=org, run_id=run, user_message="u")
        model = _Script(Proposal(kind="tool_call", tool="t"),
                        Proposal(kind="tool_call", tool="t"),
                        Proposal(kind="final", final_answer={"tried_twice": True}))
        r = await run_agent_loop(_gate(db, org, run, tools=_Tools(False, "boom")),
                                 model, db=db, org_id=org, run_id=run, user_message="u")
        assert r.final_answer == {"tried_twice": True}
        assert len([h for h in r.history if h["role"] == "action"]) == 2
    with_db(body)


def test_missing_tool_name_is_an_observation_not_a_guess():
    """模型说要调工具却没给名字 ⇒ 回一条失败观测，**不猜一个工具**。"""
    async def body(db):
        org, run = _ids()
        await create_run(db, org_id=org, run_id=run, user_message="u")
        model = _Script(Proposal(kind="tool_call", tool=None),
                        Proposal(kind="final", final_answer={"x": 1}))
        r = await run_agent_loop(_gate(db, org, run), model, db=db, org_id=org,
                                 run_id=run, user_message="u")
        assert r.status == "finished"
        assert any(h.get("observation", {}).get("error") == "missing_tool_name"
                   for h in r.history if h["role"] == "observation")
    with_db(body)


# ── 收口的权力，循环拿不到 ──────────────────────────────────────────────

def test_step_cap_is_enforced_by_the_gate_and_the_loop_stops():
    """★★ 步数上限是**收口**判的；循环只能接受结果。

    ⚠️ 判据同时钉两件事：循环停了 **且** 状态是 `exhausted`（不是 `finished`）——
      报成 finished 会让"跑完了"和"被截断了"长得一样。
    """
    async def body(db):
        org, run = _ids()
        await create_run(db, org_id=org, run_id=run, user_message="u")
        model = _Script(*[Proposal(kind="tool_call", tool="t") for _ in range(10)])
        r = await run_agent_loop(_gate(db, org, run, max_steps=3), model, db=db,
                                 org_id=org, run_id=run, user_message="u")
        assert r.status == "exhausted"
    with_db(body)


def test_the_loop_has_no_crosscutting_code():
    """★★★ 循环里**不许出现**横切逻辑。

    判据是源码扫描：横切一旦出现在循环里，就会随着"换模型/换 prompt/换编排"
    一起被改掉，而**「机制在，但没接上」是本项目的第一失效形状**。
    """
    from pathlib import Path
    src = (Path(__file__).resolve().parents[2]
           / "syncopate" / "runtime" / "agent_loop.py").read_text(encoding="utf-8")
    body = "\n".join(l for l in src.splitlines()
                     if not l.lstrip().startswith("#") and '"""' not in l)
    for forbidden in ("_over_budget", "open_approval_case", "evaluate_triggers",
                      "WRITE_TOOLS", "derive_idempotency_key", "PermissionDenied"):
        assert forbidden not in body, (
            f"循环里出现了横切逻辑：{forbidden} ⇒ 它该在 ActionGate 里")


# ── 审批中断与恢复 ──────────────────────────────────────────────────────

class _HaltingGate:
    """一调用就报 halted 的收口替身。"""

    def __init__(self) -> None:
        self.step = 1

    async def stop_requested(self) -> bool:               # K5-5 安全点契约（假 gate 显式实现）
        return False

    async def budget_exceeded(self, *, model_calls, tokens):  # noqa: ANN001 —— K9-2 预算闸契约
        return None

    async def record_model_usage(self, *, call_no, tokens_in, tokens_out):  # noqa: ANN001 —— K9-3 记账契约
        return None

    def observation_for(self, tool, *, ok, data, error):    # noqa: ANN001
        return {"tool": tool, "ok": ok, "data": data, "error": error}

    async def invoke(self, **kw):                          # noqa: ANN003
        from syncopate.runtime.action_gate import GateOutcome
        return GateOutcome(status="halted", case_ref="CASE_9",
                           observation={"tool": kw["tool"], "ok": False,
                                        "error": "waiting_for_approval"})


def test_halted_is_not_a_failure_and_the_transcript_is_persisted():
    """★★ 审批中断**不是失败**，而且 transcript 必须**已经存好**。

    存不好的话，恢复就只能从头重跑 —— 而重跑现在是要花钱的（见下一条）。
    """
    async def body(db):
        org, run = _ids()
        await create_run(db, org_id=org, run_id=run, user_message="u")
        model = _Script(Proposal(kind="tool_call", tool="t"))
        r = await run_agent_loop(_HaltingGate(), model, db=db, org_id=org,
                                 run_id=run, user_message="u")
        assert r.status == "halted" and r.case_ref == "CASE_9"
        saved = await load_transcript(db, org_id=org, run_id=run)
        assert saved, "★ transcript 没存下来 ⇒ 恢复只能从头重跑"
    with_db(body)


def test_resume_continues_from_the_transcript_instead_of_replaying():
    """★★★ 恢复从 transcript 续，**不重跑已经做过的**。

    ⚠️ `db.resume_after_approval` 原本记的是「从头重跑，读是便宜的那一侧」。
      **那个前提 2026-08-19 已经不成立**：
        ① 平台加了 BUC 积分制 ⇒ **读也扣配额**
        ② 循环由模型驱动 ⇒ 重跑要**重新花模型调用的钱**
        ③ 重跑还会重新踩改动频次上限（一小时只有 4 格）
      ⇒ 所以给 `checkpoints` 补了它一直缺的写入路径。
    ⚠️ 写动作的安全性仍由**幂等键**兜底 —— transcript 只是省重复劳动，
      不是正确性的唯一依赖。
    """
    async def body(db):
        org, run = _ids()
        await create_run(db, org_id=org, run_id=run, user_message="u")
        await run_agent_loop(_HaltingGate(), _Script(Proposal(kind="tool_call", tool="t")),
                             db=db, org_id=org, run_id=run, user_message="u")

        resumed = _Script(Proposal(kind="final", final_answer={"resumed": True}))
        r = await run_agent_loop(_gate(db, org, run), resumed, db=db, org_id=org,
                                 run_id=run, user_message="u", resume=True)
        assert r.status == "finished"
        # ★ 恢复后模型**第一眼**就该看到之前的 transcript
        assert resumed.seen_history[0], "★ 恢复时 history 是空的 ⇒ 等于从头重跑"
        assert any(h["role"] == "action" for h in resumed.seen_history[0])
    with_db(body)


def test_fresh_run_does_not_inherit_an_old_transcript():
    """★ 不带 `resume` 时必须是**空历史** —— 否则一条新 run 会继承上一条的上下文。"""
    async def body(db):
        org, run = _ids()
        await create_run(db, org_id=org, run_id=run, user_message="u")
        await run_agent_loop(_HaltingGate(), _Script(Proposal(kind="tool_call", tool="t")),
                             db=db, org_id=org, run_id=run, user_message="u")
        fresh = _Script(Proposal(kind="final", final_answer={"x": 1}))
        await run_agent_loop(_gate(db, org, run), fresh, db=db, org_id=org,
                             run_id=run, user_message="u")     # resume=False
        assert fresh.seen_history[0] == []
    with_db(body)
