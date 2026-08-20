"""M9.4 下半场 · **Agent Loop 碰外部世界的唯一出口**。

★★★ 这个模块存在的唯一理由：让生产级横切**绕不过去**，而不是"记得调用"

在它之前，`worker._execute` 是一段**写死的两步计划**（查政策 → 读 metrics → 决策+写），
而所有横切是**穿插在这段计划里**的：

    D 档拒绝 · 审批恢复 · 成本闸（进门 + 写之前各一次） · 网关触发器 → 开审批单
    · 权限 · 幂等键 · 重试 · 审计的 param_source · 事件的 seq · step 记录 · 步数上限

只要计划是写死的，"顺序对不对"由代码顺序保证。
⚠️⚠️ **一旦换成模型驱动的循环，这些横切就退化成"但愿那个循环记得调"** ——
而**「机制在，但没接上」是本项目记了十几次的第一失效形状**。

⇒ 所以边界这么划（`09 §4.5.2`）：

    模型能看到的      只有工具菜单 + observation
    模型能做的        提出一次工具调用
    模型**碰不到**的  权限 · 幂等 · 重试 · 成本闸 · 审批触发 · 审计 · 事件 · 步数上限

★ 顺带堵上一个洞：`ToolRuntime.call` 收 `invoke=<callable>` 由**调用方**传实现
  ⇒ 循环完全可以绕过它直接调 platform。
  ⇒ 本模块**自己持有** 工具名→实现 的绑定表，调用方给不了实现。

⚠️ **不要把训练侧的 `run_rollout` 套一层薄膜当成这个**（`22 §D-2`）：
  它假设单进程、无并发、工具是纯函数、没有"人"这一环、不花钱、失败即结束 ——
  这五条在生产上一条都不成立。设计思路可复用，实现必须独立。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Literal

from syncopate.runtime.db import Database
from syncopate.runtime.gateway import DecisionContext, Trigger, evaluate_triggers, open_approval_case
from syncopate.runtime.release import current_state
from syncopate.runtime.tier_policy import derive_tier, more_cautious
from syncopate.runtime.tools import PermissionDenied, ToolRuntime, WRITE_TOOLS

# 一次 run 最多允许模型走多少步。
#
# ⚠️ 为什么这条**必须在收口里**而不是在循环里：循环是会被换的（换模型、换 prompt、
#   换编排策略），而"不许无限跑"是**生产约束**，不能跟着循环一起换。
#   ⇒ 同一条纪律：保护逻辑提成一份，所有路径共用。
# ⚠️ 这个数是工程值，不是推导出来的：沙盒里逐 case 的 max_assistant_turns 是 4–14
#   （`data/rl/v13/*.parquet` 的 extra_info），取上界再留一点余量。
#   **压测（B-6）之后用实测反填。**
MAX_STEPS_PER_RUN = 16

# 审计里 `param_source` 的合法取值 —— **必须显式传**，不给默认值。
#
# ★ 为什么不给默认值：这一列的意义是"这个参数是谁定的"，
#   而"忘了传所以填了个默认值"会让整列失去意义 —— 那正是本项目的第一失效形状。
#   ⇒ 传错会被 ValueError 拦下，比悄悄记成 system 好。
PARAM_SOURCES = ("user", "model", "tool", "system", "human_review")

Status = Literal["ok", "failed", "halted", "refused"]


@dataclass(frozen=True)
class GateOutcome:
    """一次动作的结果。

    ★ `observation` 是**给模型看的那一份**，刻意和内部字段分开：
      模型不该看到 idempotency_key / attempts / 触发器细节 ——
      它看到了就会开始"绕着判据走"，而判据一旦可被优化就不再是判据。
    """

    status: Status
    observation: dict[str, Any]
    error: str | None = None
    triggers: list[Trigger] = field(default_factory=list)
    case_ref: str | None = None
    replayed: bool = False


@dataclass(frozen=True)
class ToolBinding:
    """工具名 → 真正打外部世界的那个协程。

    ⚠️ `is_write` **不在这里声明** —— 它由 `tools.WRITE_TOOLS` 单方面定义，
      这里再写一份就会有两份真相。（本项目为"同一件事两份实现"付过多次钱。）
    """

    invoke: Callable[..., Awaitable[dict[str, Any]]]


class ActionGate:
    """agent loop 碰外部世界的**唯一**出口。

    用法（循环侧只能做这三件事）：

        gate = ActionGate(db, tools, bindings, org_id=..., run_id=...,
                          cost_gate=..., emit=..., audit=..., record_step=...)
        outcome = await gate.invoke(tool="campaign.get_metrics",
                                    arguments={...}, ctx=ctx, param_source="model")
        if outcome.status != "ok": break        # halted / refused / failed 都要停
        history.append(outcome.observation)

    ⚠️ 循环**拿不到** platform、拿不到 ToolRuntime.call 的 `invoke` 参数、
      也不能自己决定横切的顺序。
    """

    def __init__(self, db: Database, tools: ToolRuntime,
                 bindings: dict[str, ToolBinding], *,
                 org_id: str, run_id: str,
                 over_budget: Callable[[], Awaitable[bool]],
                 emit: Callable[..., Awaitable[Any]],
                 audit: Callable[..., Awaitable[Any]],
                 record_step: Callable[..., Awaitable[Any]] | None = None,
                 amount_threshold: int | None = None,
                 max_steps: int = MAX_STEPS_PER_RUN) -> None:
        self.db = db
        self.tools = tools
        self._bindings = dict(bindings)
        self.org_id = org_id
        self.run_id = run_id
        self._over_budget = over_budget
        self._emit = emit
        self._audit = audit
        self._record_step = record_step
        self.amount_threshold = amount_threshold
        self.max_steps = max_steps
        # ★ 步数由**收口**记，不由循环记 —— 循环自己记的话，改循环就会把它改没
        self.step = 0
        # ★ 已裁决的动作不再过网关（否则刚批准就又被同一个触发器拦下来，
        #   run 会在 waiting_for_user 和 queued 之间来回弹，永远跑不完）
        self.skip_triggers = False

    # ── 内部：把"给模型看的那一份"和内部字段切开 ──────────────────────────
    @staticmethod
    def _observation(tool: str, *, ok: bool, data: dict[str, Any] | None,
                     error: str | None) -> dict[str, Any]:
        """★★★ **必须和沙盒 `ToolResult.observation()` 逐字段同形**（B-5b，2026-08-19）。

        沙盒（`core/tool_registry.py:88`）：

            return dict(self.data) if self.ok else {"error": self.error or "tool_failed"}

        ⇒ 模型看到的就是**工具数据本身**，或者一个只含 `error` 的字典。

        ⛔ 第一版这里写成 `{"tool":..., "ok":..., "result": {...}}` —— **多包了一层**：

            沙盒     {"cpi": 2.1, "roas_d7": 0.42}
            runtime  {"tool":"...", "ok":true, "result":{"cpi":2.1, ...}}

        ⇒ **模型被训练成读第一种，生产上会拿到第二种** ——
          它得去解一层从没见过的壳，而这层壳还是冗余的
          （工具名在 tool message 的 `name` 字段里已经有了）。
        ⇒ 这正是 B-5b 要找的东西：**不是"功能没做"，是"两边都能跑但含义不同"。**

        ⚠️ 成败的信息**没有丢**：它在 `GateOutcome.status` 上，那是给**循环**看的；
          `observation` 是给**模型**看的。两个消费者，两份数据，别合并。
        """
        if ok:
            return dict(data or {})
        # ⚠️ 失败也要给模型看见 —— 沙盒里教的"失败之后怎么办"要在线上用得上，
        #   而"重试到成功为止"会让那部分策略变成死代码（`tools.py` 同一条）。
        return {"error": error or "tool_failed"}

    async def invoke(self, *, tool: str, arguments: dict[str, Any],
                     ctx: DecisionContext, param_source: str,
                     rationale: str = "") -> GateOutcome:
        """走一次动作。**横切的顺序在这里固定，调用方改不了。**

        顺序是有理由的，不是随便排的：

            ① 步数上限     —— 最便宜，先挡
            ② 工具存在吗   —— 模型可能编一个不存在的工具名；**报错，不猜**
            ③ 成本闸       —— 写动作之前必查（"进门查一次"不够：进来时没超，读完可能就超了）
            ④ 权限         —— 由 ToolRuntime 统一校验（API 和 worker 两条路共用同一道闸）
            ⑤ 网关触发器   —— 写动作且命中 ⇒ 开**带证据**的审批单并停下（停下 ≠ 拒绝）
            ⑥ 幂等 + 重试  —— ToolRuntime
            ⑦ 审计 + 事件  —— 无论成败都要落
        """
        if param_source not in PARAM_SOURCES:
            raise ValueError(f"param_source 必须显式且合法：{PARAM_SOURCES}，收到 {param_source!r}")

        self.step += 1
        # ── ① 步数上限 ──────────────────────────────────────────────────
        if self.step > self.max_steps:
            await self._emit(self.db, org_id=self.org_id, run_id=self.run_id,
                             kind="run.degraded",
                             payload={"reason": "max_steps", "limit": self.max_steps})
            return GateOutcome(status="refused",
                               observation={"error": "max_steps"},
                               error="max_steps_exceeded")

        # ── ② 工具存在吗（模型会编工具名）────────────────────────────────
        binding = self._bindings.get(tool)
        if binding is None:
            # ⚠️ **报"没有"，不猜** —— 不要模糊匹配到一个名字相近的工具，
            #   那是「量错对象」那一族里最贵的形态：它会成功、会返回数据。
            obs = {"error": f"unknown_tool: {tool}"}   # ★ 和沙盒 execute() 的文案同形
            await self._emit(self.db, org_id=self.org_id, run_id=self.run_id,
                             kind="tool.result", payload={"tool": tool, "ok": False,
                                                          "error": "unknown_tool"})
            return GateOutcome(status="failed", observation=obs, error="unknown_tool")

        # ── ②.5 参数校验（B-6：`validation_errors` 此前**没有生产者**）────
        # ⚠️ 不校验的后果不是"返回一个错"，是**崩在实现里**：
        #   `invoke(**arguments)` 遇到漏掉的必填参数直接 TypeError ——
        #   那不是一条模型能看懂的观测，模型学不到"参数漏了要补"。
        # ★ 契约来源是**沙盒 spec**，不在这里另抄一份必填清单
        #   （`tool_parity` 那条纪律：沙盒是唯一真相来源）。
        missing = _missing_required(tool, arguments)
        if missing:
            ctx.validation_errors.append(f"{tool}: 缺必填参数 {missing}")   # ← 网关的生产者
            obs = {"error": f"validation_failed: 缺必填参数 {missing}"}
            await self._emit(self.db, org_id=self.org_id, run_id=self.run_id,
                             kind="tool.result",
                             payload={"tool": tool, "ok": False,
                                      "error": "validation_failed"})
            return GateOutcome(status="failed", observation=obs,
                               error="validation_failed")

        is_write = tool in WRITE_TOOLS

        # ── ②.6 档位推导（2026-08-20）：**档位是动作的属性** ───────────────
        # ★ 此前档位由调用方在建 run 时声明（用户在 UI 上选 A/B/C/D）——
        #   那既是糟糕的 UX，也是错误的归属：敏感度取决于**动作是什么**，
        #   不取决于谁调的。⇒ `tier_policy.derive_tier` 从工具+参数推导。
        # ★ 声明值仍然有效，但只能**往严了拉**（org 说"这条 run 只许看" = D）：
        #   `more_cautious` 是"能升不能降"的唯一实现处。
        # ⚠️ 模型没有降级接口 —— 它的升级通道是调 `approval.create_case`（训过的），
        #   刻意不给它"声明敏感度"的字段（见 tier_policy 模块 docstring）。
        decision = derive_tier(tool, arguments)
        effective_tier = more_cautious(decision.tier, ctx.automation_tier)

        # ── ②.7 D 档：这件事不该由 agent 发起 ────────────────────────────
        # ★ 拒的是**这个动作**不是整条 run（run 级 D 仍由 worker 在入口拒）——
        #   动作级拒绝会作为观测回到模型，它可以改用 `approval.create_case`
        #   把事情交给人。**把硬性终止变成可恢复的观测**，同「失败要回到模型」那条。
        if is_write and effective_tier == "D":
            reason = (decision.reason if decision.tier == "D"
                      else f"调用方声明 automation_tier=D（{decision.reason}）")
            await self._audit(self.db, org_id=self.org_id, run_id=self.run_id,
                              action="tier_d_refused", object_key=_object_key(arguments),
                              param_source=param_source, detail={"tool": tool,
                                                                 "reason": reason})
            await self._emit(self.db, org_id=self.org_id, run_id=self.run_id,
                             kind="run.degraded",
                             payload={"reason": "tier_d_never_automated",
                                      "tool": tool, "detail": reason})
            return GateOutcome(
                status="failed",
                observation={"error": f"tier_d_never_automated: {reason}；"
                                      "如需推进，请用 approval.create_case 交给人"},
                error="tier_d_never_automated")

        # ── ②.8 灰测闸门（B-7）—— **比审批更外层**────────────────────────
        # ★ 顺序有理由：审批是"停下来问人"，灰测闸门是"这一档现在根本不许自动做"。
        #   后者该先判 —— 已经关掉了还去开审批单，等于给人送一堆没意义的单子。
        # ⚠️ 只管**写动作**：读不改世界，灰测期间照样要能查
        #   （同成本闸那条：降级的意义是降级，不是失明）。
        if is_write and decision.tier is not None:
            rel = current_state()
            if not rel.allows(effective_tier):
                await self._emit(self.db, org_id=self.org_id, run_id=self.run_id,
                                 kind="run.degraded",
                                 payload={"reason": "release_gate",
                                          "halted": rel.halted,
                                          "max_tier": rel.max_tier,
                                          "tier": effective_tier,
                                          "tier_reason": decision.reason,
                                          # ★ 关闭要**可追溯**：重新打开时按它找回受影响的 run
                                          "halt_reason": rel.halt_reason})
                return GateOutcome(
                    status="refused",
                    observation={"error": "release_gate: 当前灰测档位不允许自动执行"},
                    error="release_gate")

        # ── ③ 成本闸（写之前）───────────────────────────────────────────
        if is_write and await self._over_budget():
            await self._emit(self.db, org_id=self.org_id, run_id=self.run_id,
                             kind="run.degraded",
                             payload={"reason": "daily_cost_cap", "at": "before_write"})
            return GateOutcome(status="refused",
                               observation={"error": "daily_cost_cap"},
                               error="daily_cost_cap_exceeded")

        # ── ⑤ 网关触发器（写动作）──────────────────────────────────────
        # ⚠️ 放在真正执行**之前**：审批的意义是"先停下再问人"，
        #   执行完再问就只剩记账了。
        if is_write and not self.skip_triggers:
            # ★ 档位触发器按**推导出来的**档位判，不按声明值
            #   （声明值可能压根没有 —— 这正是这次改动的目的：用户不用选）。
            #   做法是临时把 effective 塞进 ctx：gateway 是纯函数，
            #   判据的输入必须显式，不能让它去猜（"判据要写在两个东西应当相同的地方"）。
            declared = ctx.automation_tier
            ctx.automation_tier = effective_tier
            try:
                triggers = evaluate_triggers(
                    ctx, **({"amount_threshold": self.amount_threshold}
                            if self.amount_threshold is not None else {}))
            finally:
                ctx.automation_tier = declared
            if triggers:
                case_ref = await open_approval_case(
                    self.db, org_id=self.org_id, run_id=self.run_id, action_type=tool,
                    proposed_params=dict(arguments),
                    rationale=rationale or f"agent 提议执行 {tool}",
                    evidence={"triggers": [t.reason for t in triggers],
                              "tier": effective_tier, "tier_reason": decision.reason},
                    triggers=triggers)
                await self._emit(self.db, org_id=self.org_id, run_id=self.run_id,
                                 kind="run.waiting_for_user",
                                 payload={"case_ref": case_ref,
                                          "triggers": [t.reason for t in triggers]})
                return GateOutcome(
                    status="halted",
                    observation={"tool": tool, "ok": False, "error": "waiting_for_approval"},
                    triggers=list(triggers), case_ref=case_ref)

        # ── ④⑥ 权限 + 幂等 + 重试（统一走 ToolRuntime）──────────────────
        try:
            outcome = await self.tools.call(
                org_id=self.org_id, run_id=self.run_id, step=self.step,
                tool=tool, arguments=arguments, invoke=binding.invoke)
        except PermissionDenied as exc:
            await self._audit(self.db, org_id=self.org_id, run_id=self.run_id,
                              action="permission_denied", object_key=None,
                              param_source="system", detail={"tool": tool, "error": str(exc)})
            return GateOutcome(status="failed",
                               observation={"error": "permission_denied"},
                               error=str(exc))
        except Exception as exc:                       # noqa: BLE001
            # ★ 工具**实现**崩了（不是工具返回 error）⇒ 也要变成一次失败观测。
            #   否则异常一路穿到 run_once 的兜底，整条 run 记 failed ——
            #   「动作失败不终止循环，由模型决定下一步」在崩溃这条路上就是空话
            #   （2026-08-20 压测：calendar 的一个 SQL 类型错让 I11 8/8 全灭）。
            #   审计留全文；给模型的观测只说"这个工具坏了"，别把 SQL 栈喂给它。
            await self._audit(self.db, org_id=self.org_id, run_id=self.run_id,
                              action="tool_crashed", object_key=None,
                              param_source="system",
                              detail={"tool": tool, "error": str(exc)[:500]})
            await self._emit(self.db, org_id=self.org_id, run_id=self.run_id,
                             kind="tool.result",
                             payload={"tool": tool, "ok": False, "error": "tool_crashed"})
            return GateOutcome(status="failed",
                               observation={"error": f"tool_crashed: {tool} 暂时不可用"},
                               error=f"tool_crashed: {exc}")

        # ── ⑥.5 副作用状态未知 ⇒ **不是失败，是未知**（任务三）────────────
        # ⚠️ `db.record_tool_call` 命中一条**仍在执行中**的同键记录时返回
        #   `tool_call_in_progress`。它的文案写着"不要当成失败处理" ——
        #   但**形状就是失败**（`{"error": ...}`），而沙盒教模型的正是"error = 失败"。
        #   ⇒ 光靠文案托付给模型是不够的：模型会照着失败处理，**很可能重试**，
        #     而重试一个可能已经生效的写动作正是幂等要防的那件事。
        # ⇒ 所以把它升格成一个**降级信号**，由网关决定停不停 —— 和
        #   `retrieval_unavailable` 同族：**"我们不知道"一律交给人。**
        if outcome.error and str(outcome.error).startswith("tool_call_in_progress"):
            ctx.unknown_state_tools.append(tool)

        # ── ⑦ 审计 + 事件（无论成败）────────────────────────────────────
        if is_write:
            await self._audit(self.db, org_id=self.org_id, run_id=self.run_id,
                              action=tool, object_key=_object_key(arguments),
                              param_source=param_source,
                              detail={**arguments, "replayed": outcome.replayed,
                                      "ok": outcome.ok})
        await self._emit(self.db, org_id=self.org_id, run_id=self.run_id,
                         kind="tool.result",
                         payload={"tool": tool, "ok": outcome.ok,
                                  "replayed": outcome.replayed})

        return GateOutcome(
            status="ok" if outcome.ok else "failed",
            observation=self._observation(tool, ok=outcome.ok, data=outcome.data,
                                          error=outcome.error),
            error=outcome.error, replayed=outcome.replayed)


def _missing_required(tool: str, arguments: dict[str, Any]) -> list[str]:
    """模型给的参数够不够。**必填清单从沙盒 spec 取，不另抄一份。**

    ⚠️ 沙盒里没有这个工具 ⇒ 返回空（不报缺参数）——
      "工具不存在"由上一道闸负责，这里**不越权替它报错**：
      一个错误由两处报，人会以为是两个问题。
    """
    try:
        import syncopate.domains.adcampaign  # noqa: F401
        from syncopate.core.tool_registry import REGISTRY
        spec = REGISTRY.get(tool)
    except Exception:
        return []
    if spec is None:
        return []
    required = spec.parameters.get("required", []) or []
    return [r for r in required if r not in arguments]


def _object_key(arguments: dict[str, Any]) -> str | None:
    """审计的 `object_key`：动作作用在哪个对象上。

    ⚠️ 认不出就返回 None，**不猜** —— 猜错的 object_key 比没有更糟，
    它会让审计看起来完整而实际指向另一个对象。
    """
    for k in ("campaign_id", "asset_id", "case_id", "memory_id", "account_id"):
        v = arguments.get(k)
        if isinstance(v, str):
            return v
    return None
