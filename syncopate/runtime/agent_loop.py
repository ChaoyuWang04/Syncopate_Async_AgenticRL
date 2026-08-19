"""M9.4 下半场 · **模型驱动的循环**（B-3b）。

★★★ 这个循环**只做一件事：把模型的提议交给收口**。

    模型          看 transcript，提出**一次**动作（或给终答）
    ActionGate    权限 · 幂等 · 重试 · 成本闸 · 审批触发 · 审计 · 事件 · 步数上限
    本模块        只负责：把观测喂回模型 · 持久化 transcript · 决定何时停

⚠️ 循环里**没有一行**横切代码 —— 那是刻意的。
  横切一旦出现在循环里，就会随着"换模型/换 prompt/换编排"一起被改掉，
  而**「机制在，但没接上」是本项目记了十几次的第一失效形状**。

★★ 恢复语义：**从 transcript 续**，不是从头重跑（2026-08-19 改）

`db.resume_after_approval` 原本记的是「从头重跑，重跑一遍读操作 ——
读是便宜的那一侧，这个代价可以接受」。⚠️ **那个前提现在不成立了**：

    ① 平台加了 BUC 积分制（B-1a）⇒ **读也扣配额**，不再免费
    ② 循环由模型驱动 ⇒ 重跑要**重新花模型调用的钱**，而那是真金白银
    ③ 重跑还会重新踩一遍改动频次上限（一小时只有 4 格）

⇒ 所以这里给 `checkpoints` 补上了它一直缺的写入路径（`db.py:184` 明写"那张表现在没人写"），
  审批中断后从**上次的 transcript** 接着走。
⚠️ 写动作的安全性仍然由**幂等键**兜底 —— transcript 只是省掉重复劳动，
  不是正确性的唯一依赖。两条都在，才敢在生产上恢复。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from syncopate.runtime.action_gate import ActionGate, GateOutcome
from syncopate.runtime.gateway import DecisionContext

Kind = Literal["tool_call", "final"]


@dataclass(frozen=True)
class Proposal:
    """模型的一次输出。**只有两种**：再调一个工具，或者给终答。

    ⚠️ 刻意**不提供**"直接执行"这一档 —— 任何碰外部世界的动作都必须过收口。
    """

    kind: Kind
    tool: str | None = None
    arguments: dict[str, Any] = field(default_factory=dict)
    # ★ 没有默认值：这个参数是谁定的，模型自己最清楚（见 ActionGate 的 PARAM_SOURCES）
    param_source: str = "model"
    rationale: str = ""
    final_answer: dict[str, Any] | None = None


class Decider(Protocol):
    """模型的接口。**只有这一个方法** —— 换真模型时只换它的实现。"""

    async def decide(self, *, user_message: str,
                     history: list[dict[str, Any]]) -> Proposal: ...


@dataclass
class LoopResult:
    status: Literal["finished", "halted", "exhausted", "failed"]
    final_answer: dict[str, Any] | None = None
    history: list[dict[str, Any]] = field(default_factory=list)
    case_ref: str | None = None
    error: str | None = None


async def save_transcript(db, *, org_id: str, run_id: str, step: int,
                          history: list[dict[str, Any]]) -> None:
    """把 transcript 存进 `checkpoints`。

    ★ 这是 `checkpoints` 表**第一个写入者** —— 它建了很久但一直没人写
      （`db.py:184` 明写这一点，并因此选了"从头重跑"）。
    """
    async with db.tx() as conn:
        await conn.execute(
            "INSERT INTO checkpoints (org_id, run_id, step, state) VALUES ($1,$2,$3,$4) "
            "ON CONFLICT (org_id, run_id, step) DO UPDATE SET state=EXCLUDED.state",
            org_id, run_id, step, json.dumps({"history": history}, ensure_ascii=False,
                                             default=str))


async def load_transcript(db, *, org_id: str, run_id: str) -> list[dict[str, Any]]:
    """读回最后一次 transcript。没有就返回空 —— **报"没有"，不猜**。"""
    async with db.tx() as conn:
        row = await conn.fetchrow(
            "SELECT state FROM checkpoints WHERE org_id=$1 AND run_id=$2 "
            "ORDER BY step DESC LIMIT 1", org_id, run_id)
    if row is None:
        return []
    state = row["state"]
    if isinstance(state, str):
        state = json.loads(state)
    return list(state.get("history") or [])


async def run_agent_loop(gate: ActionGate, decider: Decider, *, db,
                         org_id: str, run_id: str, user_message: str,
                         ctx: DecisionContext | None = None,
                         resume: bool = False) -> LoopResult:
    """跑一条 run。

    停下来的四种方式，**每一种都要能分辨**：

        finished   模型给了终答
        halted     收口开了审批单（**不是失败** —— 等人裁决后会恢复）
        exhausted  撞到步数上限（收口判的，循环改不了）
        failed     动作失败且模型选择不再继续

    ⚠️ `failed` 的观测**必须先回到模型**再由模型决定 ——
      循环自己吞掉失败，就等于把"失败之后怎么办"这段策略变成死代码
      （沙盒里专门训过这一段）。
    """
    ctx = ctx or DecisionContext()
    history = await load_transcript(db, org_id=org_id, run_id=run_id) if resume else []

    while True:
        proposal = await decider.decide(user_message=user_message, history=history)

        if proposal.kind == "final":
            history.append({"role": "final", "answer": proposal.final_answer})
            await save_transcript(db, org_id=org_id, run_id=run_id,
                                  step=gate.step, history=history)
            return LoopResult(status="finished", final_answer=proposal.final_answer,
                              history=history)

        if not proposal.tool:
            # ⚠️ 模型说要调工具却没给工具名 ⇒ 当成一次失败的观测回给它，**不要猜**
            history.append({"role": "observation",
                            "observation": {"ok": False, "error": "missing_tool_name"}})
            continue

        outcome: GateOutcome = await gate.invoke(
            tool=proposal.tool, arguments=dict(proposal.arguments), ctx=ctx,
            param_source=proposal.param_source, rationale=proposal.rationale)

        history.append({"role": "action", "tool": proposal.tool,
                        "arguments": proposal.arguments})
        # ★★ 观测**一律**回到模型 —— 成功、失败、被拒，都回。
        #   只回成功的，模型就永远学不到"失败之后怎么办"。
        history.append({"role": "observation", "observation": outcome.observation})
        await save_transcript(db, org_id=org_id, run_id=run_id,
                              step=gate.step, history=history)

        if outcome.status == "halted":
            # 审批中断：**不是失败**。transcript 已经存好，恢复时从这里接着走。
            return LoopResult(status="halted", case_ref=outcome.case_ref,
                              history=history)
        if outcome.status == "refused":
            # 收口拒绝（步数上限 / 成本闸）⇒ 循环无权继续
            return LoopResult(status="exhausted", error=outcome.error, history=history)
        # ⚠️ status == "failed" **不在这里返回** —— 观测已经回给模型了，
        #   下一轮由**模型**决定是重试、换个工具、还是给终答说做不了。
