"""M9.5 · 审批网关：C 档动作在此暂停成 `waiting_for_user`。

★★★ 触发条件**不能靠 token 概率**（设计文档 §39）

原话：「LLM 的 token 概率和"答案对不对"关系很弱（**编造时往往最自信**）。」
所以下面**七个**触发器全部是外部信号，没有一个读模型的置信度：

    ① 工具调用失败（重试用尽）
    ② 参数校验不通过
    ③ 数据未收敛（data_maturity 不足）
    ④ 命中任一 cap
    ⑤ 写动作且金额 > 阈值
    ⑥ RAG 检索为空                  ← ★ 直接接上 M8 的 `no_match`
    ⑦ RAG 检索**不可用**             ← 🆕 和⑥分开：查不到 ≠ 查不了

★ ⑥ 为什么值得单独列：M8 之前"检索为空"这件事在系统里**没有表示**，
所以它不可能成为降级条件。M8 把 `no_match` 做成明确的信号位之后，
runtime 才拿得到它。**机制先要存在，才谈得上被接上。**

★★ 网关的输出不是"拒绝"，是**一张带证据的审批单**：
`proposed_params` + `rationale` + `evidence`。人看的是证据不是结论 ——
而且 `modified_params`（人改了什么）是**飞轮回路 2 的燃料**（§37）。
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from syncopate.runtime.db import Database, transition_run

# 金额阈值（微单位，和 usage_records.cost_micros 同口径）。
# ⚠️ 这个数**该由业务定**，设计文档附录 A 的 A2 还空着（"扩量的常见幅度档位？
# 超过多少必须审批？"）。这里给一个保守默认值，等业务侧定了再改。
DEFAULT_AMOUNT_THRESHOLD = 100_000


@dataclass(frozen=True)
class Trigger:
    """一个降级触发器。`reason` 会落进 `approval_cases.trigger_reason`，
    压测场景要按它分类统计，所以必须是稳定的短标识而不是一句话。"""

    reason: str
    detail: str


@dataclass
class DecisionContext:
    """判断要不要停下来所需要的**全部外部信号**。刻意不含任何模型置信度字段。"""

    tool_failed: str | None = None            # ① 失败的工具名
    validation_errors: list[str] = field(default_factory=list)   # ②
    data_maturity: str | None = None          # ③ mature / partial / immature
    cap_hits: list[str] = field(default_factory=list)            # ④
    write_amount: int | None = None           # ⑤
    retrieval_empty_tools: list[str] = field(default_factory=list)  # ⑥ M8 的 no_match
    # ⑦ 🆕 **查不了 ≠ 查不到**。见 runtime/retrieval.py 与 12-rag-runtime-design.md §3.1：
    #    no_match 的语义是「没有政策限制这件事」，unavailable 的语义是
    #    「**我们不知道有没有限制**」。合并的话一次故障看起来就是放行信号。
    retrieval_unavailable_tools: list[str] = field(default_factory=list)
    automation_tier: str | None = None        # C 档天然要审批（§3）
    # ⑧ 🆕 **副作用状态未知**（2026-08-19，任务三的设计决定）
    #
    # 来源：同一幂等键的上一次调用**仍在执行中** —— 我们既不能说它成功，也不能说它失败。
    # ⚠️ 它和 `tool_failed` **必须分开**：失败可以重试，**未知不能重试** ——
    #   重试一个可能已经生效的写动作，正是幂等机制存在的全部理由。
    # ★ 语义上它和 `retrieval_unavailable` 同族：**"我们不知道" 一律阻断，交给人。**
    unknown_state_tools: list[str] = field(default_factory=list)


def evaluate_triggers(ctx: DecisionContext, *,
                      amount_threshold: int | None = None) -> list[Trigger]:
    """返回**所有**命中的触发器，不是第一个。

    ★ 为什么要全部返回：审批单上要写清楚"为什么停下来"。只报第一个的话，
    "既超金额又命中 cap"和"只是超金额"在人眼里长得一样，而它们的风险不同。
    """
    # ⚠️ 阈值**在函数体里解析**，不能写成默认参数值 —— 默认值绑定在**函数定义时**，
    # 之后改模块属性（配置热更、测试里调阈值）一律不生效，而且悄无声息。
    if amount_threshold is None:
        amount_threshold = DEFAULT_AMOUNT_THRESHOLD

    fired: list[Trigger] = []
    if ctx.tool_failed:
        fired.append(Trigger("tool_failed", f"{ctx.tool_failed} 重试用尽仍失败"))
    if ctx.validation_errors:
        fired.append(Trigger("validation_failed", "；".join(ctx.validation_errors)))
    if ctx.data_maturity in ("immature", "partial"):
        # ★ 归因延迟是第一性约束（§0.3）：D7 才知对错，D1 数据极易被误当结论。
        fired.append(Trigger("data_immature", f"数据成熟度={ctx.data_maturity}，D7 未收敛"))
    if ctx.cap_hits:
        fired.append(Trigger("cap_hit", "命中护栏：" + "、".join(ctx.cap_hits)))
    if ctx.write_amount is not None and ctx.write_amount > amount_threshold:
        fired.append(Trigger("amount_over_threshold",
                             f"金额 {ctx.write_amount} > 阈值 {amount_threshold}"))
    if ctx.retrieval_empty_tools:
        # ★ M8 的接口：no_match 是明确的信号位，不是"结果长度为 0"这种推断。
        fired.append(Trigger("retrieval_empty",
                             "检索为空：" + "、".join(ctx.retrieval_empty_tools)))
    if ctx.retrieval_unavailable_tools:
        # ★★ 和上面那条**必须分开**：人看到"检索为空"和"检索挂了"要做的判断完全不同。
        # 前者是「查过了，没有相关政策」，后者是「查不了，不知道有没有政策」——
        # 后者下**任何**放行都是在赌。⇒ trigger_reason 不同，审批单上写得清清楚楚。
        fired.append(Trigger("retrieval_unavailable",
                             "检索不可用（**不是**查不到）：" +
                             "、".join(ctx.retrieval_unavailable_tools)))
    if ctx.unknown_state_tools:
        # ★★ 「结果未知」**不能靠一句话托付给模型** —— 见 DecisionContext 那段。
        #   两种误判的代价不对称：
        #     当成失败去重试 ⇒ **可能重复扣款**（不可逆）
        #     停下来问人     ⇒ 最多慢一点
        fired.append(Trigger("side_effect_unknown",
                             "副作用状态未知（上一次同键调用仍在执行）：" +
                             "、".join(ctx.unknown_state_tools)))
    if ctx.automation_tier == "C":
        fired.append(Trigger("tier_c", "C 档动作按 §3 一律走审批"))
    return fired


async def open_approval_case(db: Database, *, org_id: str, run_id: str, action_type: str,
                             proposed_params: dict[str, Any], rationale: str,
                             evidence: dict[str, Any],
                             triggers: list[Trigger]) -> str:
    """开一张审批单，并把 run 置成 `waiting_for_user`。**两件事在同一个事务里。**

    分开做的话，中间崩掉会留下"单开了但 run 还在跑"或者"run 停了但没有单" ——
    前者会重复执行，后者是永久卡死。
    """
    case_ref = f"apr_{uuid.uuid4().hex[:12]}"
    async with db.tx() as conn:
        await conn.execute(
            """
            INSERT INTO approval_cases (case_ref, run_id, org_id, action_type,
                                        proposed_params, rationale, evidence,
                                        trigger_reason, status)
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,'pending')
            """,
            case_ref, run_id, org_id, action_type, proposed_params, rationale, evidence,
            ",".join(t.reason for t in triggers))
        # K4：走唯一迁移入口——状态 + run.waiting_for_user 事件 + 审计同事务；resume_token 由入口发
        await transition_run(conn, org_id=org_id, run_id=run_id, to="waiting_for_user",
                             reason="approval:" + ",".join(t.reason for t in triggers),
                             actor_type="worker", actor_id="gateway",
                             fields={"requires_approval": True},
                             event_payload={"case_ref": case_ref, "action_type": action_type,
                                            "triggers": [t.reason for t in triggers]})
    return case_ref
