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
★ K5-2（2026-09-02，27 §16-4 裁定 = 课件快照式）：**每 append 一条消息就存一次**——一轮两档
  （模型点名工具后 / 工具结果回灌后），快照带 `last` 与 `completed_tool_calls`。存档密度决定
  恢复分辨率：只有"已点名、结果未回"那一档存在，恢复时才能走第二路（读意图日志判断钱动没动，
  而不是重问模型或裸重跑工具）。
⚠️ 写动作的安全性仍然由**幂等键**兜底 —— transcript 只是省掉重复劳动，
  不是正确性的唯一依赖。两条都在，才敢在生产上恢复。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from contextvars import ContextVar

from syncopate.runtime.action_gate import ActionGate, GateOutcome
from syncopate.runtime.gateway import DecisionContext

Kind = Literal["tool_call", "final"]

# ★ 模型 token 用量的回传通道：worker 在跑一条 run 之前 set 一个 dict，
#   Decider 每次调用往里累加（tokens_in/tokens_out/calls）。
#   用 contextvar 而不是 Decider 实例属性：一个 Decider 服务并发 8 条 run，
#   实例属性会串账（每条 run 一个 asyncio task = 一个 context，天然隔离）。
MODEL_USAGE: ContextVar[dict[str, int] | None] = ContextVar("MODEL_USAGE", default=None)

# ★ 当前 run 的 intent（I01/I07/I09/I11…）：Decider 按它选**训练同形的工具子菜单**
#   （训练 case 的菜单是 12–16 个工具，从来没有全量 30 —— 全量塞进 prompt 直接
#   超 max_model_len，且是模型没见过的分布）。同 MODEL_USAGE 的隔离理由。
RUN_INTENT: ContextVar[str | None] = ContextVar("RUN_INTENT", default=None)

# ★ 同一会话里之前几轮的 (user_message, result)：Decider 把它们渲染进 prompt，
#   让"第二句话"能指代第一句（2026-08-20 多轮壳层）。同上的隔离理由。
# ⚠️ 这是**壳层**多轮：拼上下文而已，模型没训过第二轮 user 消息 ⇒ 探针先量格式保持率。
PRIOR_TURNS: ContextVar[list[dict] | None] = ContextVar("PRIOR_TURNS", default=None)


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
    # CoT 观察（Chaoyu 08-29）：SYNCOPATE_RUNTIME_THINKING=1 时 decider 保留 <think> 段
    # 供前端折叠展示；空串=无思考。只作展示，不进任何判定/持久化契约。
    thinking: str = ""


class Decider(Protocol):
    """模型的接口。**只有这一个方法** —— 换真模型时只换它的实现。"""

    async def decide(self, *, user_message: str,
                     history: list[dict[str, Any]]) -> Proposal: ...


@dataclass
class LoopResult:
    status: Literal["finished", "halted", "exhausted", "failed", "awaiting_reconciliation"]
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
    state = {"history": history, "last": _last_of(history),
             "completed_tool_calls": [i for i, m in enumerate(history) if m.get("role") == "observation"],
             "n": len(history)}
    async with db.tx() as conn:
        # step 只是"第几轮"；同一轮的两档以 n（消息数）区分 ⇒ 用 (step, n) 做键的等价物：
        # 覆盖同 step 的快照，但 state.n 单调，恢复读最新一条即可
        await conn.execute(
            "INSERT INTO checkpoints (org_id, run_id, step, state) VALUES ($1,$2,$3,$4) "
            "ON CONFLICT (org_id, run_id, step) DO UPDATE SET state=EXCLUDED.state",
            org_id, run_id, step, json.dumps(state, ensure_ascii=False, default=str))


def _last_of(history: list[dict[str, Any]]) -> str:
    """快照的 `last`（课件 CH8 §4.2）：决定恢复走哪一路。"""
    if not history:
        return "user:input"
    m = history[-1]
    role = m.get("role")
    if role == "action":
        return f"tool_use:{m.get('tool')}"
    if role == "observation":
        return "tool_result"
    if role == "final":
        return "final"
    return str(role)


async def load_transcript(db, *, org_id: str, run_id: str) -> list[dict[str, Any]]:
    """读回最后一次 transcript。没有就返回空 —— **报"没有"，不猜**。"""
    async with db.tx() as conn:
        row = await conn.fetchrow(
            # 最新 = 消息最多的那档（一轮两档可能落在相邻 step；n 单调，step 不一定）
            "SELECT state FROM checkpoints WHERE org_id=$1 AND run_id=$2 "
            "ORDER BY COALESCE((state->>'n')::int, 0) DESC, id DESC LIMIT 1", org_id, run_id)
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
    # ⚠️ tool=None 的回灌不过收口、不计步数 ⇒ 连续输出解析不了的东西会无限烧模型。
    #   步数上限管不到它（那是 gate.invoke 记的），所以这里单独设一条连续失败上限。
    fumbles = 0

    if resume and history and history[-1].get("role") == "action":
        # ★ 恢复第二路（课件 CH8 §4.3）：last = 已点名工具、结果未回。既不重问模型，也不裸重跑工具：
        #   读意图日志——succeeded ⇒ 回填观测续跑；running/response_lost ⇒ 停下转对账；
        #   没有记录（只读工具或还没占坑）⇒ 重做这一步是安全的。
        from syncopate.runtime.db import last_write_call
        pending = history[-1]
        rec = await last_write_call(db, org_id=org_id, run_id=run_id, tool=pending["tool"])
        if rec is not None and rec["side_effect"] and rec["status"] in ("running", "response_lost"):
            await gate.emit_info(kind="run.awaiting_reconciliation",
                                 payload={"tool": pending["tool"], "tool_call_id": rec["id"],
                                          "status": rec["status"]})
            return LoopResult(status="awaiting_reconciliation", error=pending["tool"], history=history)
        if rec is not None and rec["status"] in ("succeeded", "failed"):
            obs = gate.observation_for(pending["tool"], ok=bool(rec["ok"]), data=rec["result"],
                                       error=rec["error"])
            history.append({"role": "observation", "observation": obs})
            await save_transcript(db, org_id=org_id, run_id=run_id, step=gate.step, history=history)
            await gate.emit_info(kind="tool.repaired_from_intent_log",
                                 payload={"tool": pending["tool"], "tool_call_id": rec["id"]})
        else:
            # 没占过坑 ⇒ 副作用不可能发生过；把这一步交回下面的正常路径重做（去掉悬空的 action）
            history.pop()

    while True:
        # ── 安全点：模型调用前 / 下一轮 loop 前（K5-5；工具调用前那个在收口入口）──
        if await gate.stop_requested():
            await save_transcript(db, org_id=org_id, run_id=run_id, step=gate.step, history=history)
            return LoopResult(status="exhausted", error="cancel_requested", history=history)

        proposal = await decider.decide(user_message=user_message, history=history)

        if getattr(proposal, "thinking", ""):
            # 思考事件：独立 kind，前端折叠渲染；截 6000 字防事件超载
            await gate.emit_info(kind="model.thinking",
                                 payload={"step": gate.step,
                                          "text": proposal.thinking[:6000]})

        if proposal.kind == "final":
            history.append({"role": "final", "answer": proposal.final_answer})
            await save_transcript(db, org_id=org_id, run_id=run_id,
                                  step=gate.step, history=history)
            # ★ v15：三条信令各有**不同的**终止语义（`25 §R4` 门槛①，N4 行为即动作）。
            #   v14 下 signal 恒为 None ⇒ 走原来那条 finished 路径，逐字节不变。
            signal = (proposal.final_answer or {}).get("signal")
            if signal:
                await gate.emit_info(kind=f"session.{signal}",
                                     payload={"step": gate.step,
                                              "arguments": (proposal.final_answer or {})
                                              .get("arguments", {}),
                                              "text": (proposal.final_answer or {})
                                              .get("text", "")})
            if signal == "clarify":
                # 等用户补充 —— 和"开审批单等人"同一族的挂起，不是成功也不是失败
                return LoopResult(status="halted", final_answer=proposal.final_answer,
                                  case_ref=None, history=history)
            if signal == "reject":
                return LoopResult(status="exhausted", final_answer=proposal.final_answer,
                                  error="session_reject", history=history)
            # defer 与纯文本终答都算本轮正常收工（defer 的复查靠 recheck_after_days 另行调度）
            return LoopResult(status="finished", final_answer=proposal.final_answer,
                              history=history)

        if not proposal.tool:
            # ⚠️ 模型说要调工具却没给工具名 ⇒ 当成一次失败的观测回给它，**不要猜**。
            # ★ rationale 可携带更准的纠正文本（Decider 用它回灌 parse_error /
            #   一步多调用的拦截语，P0-2 同法）；没给就落回老文案。
            fumbles += 1
            if fumbles > 3:
                await save_transcript(db, org_id=org_id, run_id=run_id,
                                      step=gate.step, history=history)
                return LoopResult(status="failed", error="unparseable_output",
                                  history=history)
            history.append({"role": "observation",
                            "observation": {"ok": False,
                                            "error": proposal.rationale
                                            or "missing_tool_name"}})
            continue
        fumbles = 0

        # ── 存档①：模型点名工具后、执行前（"已点名、结果未回"这一档是分支 C 的全部依据）──
        history.append({"role": "action", "tool": proposal.tool,
                        "arguments": proposal.arguments})
        await save_transcript(db, org_id=org_id, run_id=run_id,
                              step=gate.step + 1, history=history)

        outcome: GateOutcome = await gate.invoke(
            tool=proposal.tool, arguments=dict(proposal.arguments), ctx=ctx,
            param_source=proposal.param_source, rationale=proposal.rationale)

        # ★★ 观测**一律**回到模型 —— 成功、失败、被拒，都回。
        #   只回成功的，模型就永远学不到"失败之后怎么办"。
        # ── 存档②：结果回灌后 ──
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
