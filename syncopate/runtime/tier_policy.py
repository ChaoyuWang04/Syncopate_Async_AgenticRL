"""动作敏感度推导：**档位是动作的属性，不是调用方的声明、也不是模型的自选**。

★★★ 这个模块存在的理由（Chaoyu 2026-08-20 的质疑与回答）

质疑很合理：「我们训练时已经用 reward 教过它越权写要罚、安全线要查、
该交给人要交给人了 —— 为什么还要在外面再套一层规定？」

回答分三条，前两条有我们自己的实测垫底：

**① 训练给的是能力（统计），闸门给的是保证（绝对）。**
   六点曲线实测：同一个模型、同一套 reward，只是从 100 步训到 400 步 ——
   `missing_safety_line_cap` 4 → **60**（15 倍），`abandoned_without_escalation`
   2 → **20**，而**任务总分在这期间还是涨的**（0.897 → 0.902）。
   记录在案的原话：**reward 没罚到的地方，RL 就会去住。**
   盲区的定义就是"我们没想到要测的东西"，它永远存在。

**② 写动作不可逆，一次就够。** 99.9% 的正确率 × 每天 1000 个写动作
   = 每天一次真金白银的错误，而且是事后才知道。

**③ 这张表描述的是「世界」不是「模型」。** 哪些动作不可逆、代价多大，
   这个事实与模型好坏无关 —— 人类组织也有 code review、大额转账双人复核。
   ⇒ 而**模型不能自我授权**（设计文档 §27.2「假设模型已被策反」）：
     被注入的模型第一件事就是给自己开 A 档。

★ 但闸门必须**薄**，不重复训练做的事：
    模型的判断力  调查该查的 · 该 defer 就 defer · 该交给人就交给人
    这张表        只回答一件事：这个动作要不要人点头
  读工具完全不过闸；`approval.create_case`（交给人这条路本身）也不过闸 ——
  **给"请求审批"这件事加审批是循环的**，而且正是 C-5 记的那个坑
  （模型交给人反而被罚）。

★ 模型怎么"升级"：它已经有升级通道 —— **调用 `approval.create_case`**（训过的）。
  ⇒ 刻意**不加**"模型声明敏感度"的字段：那是没训过的输出，而且一旦存在，
    降级方向就有了口子。**能升不能降是结构性的：它压根没有降的接口。**
"""

from __future__ import annotations

from dataclasses import dataclass

from syncopate.runtime.gateway import DEFAULT_AMOUNT_THRESHOLD
from syncopate.runtime.tools import WRITE_TOOLS

# 自主度从低到高 —— ⚠️ 必须与 `release.TIER_ORDER` 同序（同一份语义的两处消费者）
TIER_ORDER: tuple[str, ...] = ("D", "C", "B", "A")

# 「永不自动」的金额线：超过它的预算变更按 §3 归 D（账户级预算 = 不可逆且不可验证）。
# ⚠️ 和 `DEFAULT_AMOUNT_THRESHOLD` 一样，**这个数该由业务定**
#   （设计文档附录 A2「超过多少必须审批 / 多少绝不自动」还空着）。
#   这里给保守占位值 = 审批阈值的 10 倍，业务定了就改这一处。
NEVER_AUTOMATE_AMOUNT = 10 * DEFAULT_AMOUNT_THRESHOLD

# 不过闸的工具：**交给人这条路本身**。给它加审批 = 循环 + C-5 那个坑。
ESCALATION_TOOLS = frozenset({"approval.create_case"})

# 带金额语义的写工具 → 从哪个参数读金额
_AMOUNT_ARG = {
    "campaign.update_budget": "new_budget",
    "campaign.scale_budget": "new_budget",
    "campaign.create": "daily_budget",
}


@dataclass(frozen=True)
class TierDecision:
    """`tier=None` ⇒ **这个动作不过档位闸**（读工具 / 升级通道）。

    `reason` 会写进审批单和事件流 —— 人看到的是"为什么判成这一档"，
    不是一个光秃秃的字母（同 `Trigger.reason`：判据要能被追问）。
    """

    tier: str | None
    reason: str


def more_cautious(*tiers: str | None) -> str | None:
    """取最谨慎的那一档（自主度最低）。None 视为"没有意见"，被忽略。

    ★ 这是"能升不能降"的**唯一**实现处：任何来源都只能把档位往严了拉。
    """
    present = [t for t in tiers if t in TIER_ORDER]
    if not present:
        return None
    return min(present, key=TIER_ORDER.index)


def derive_tier(tool: str, arguments: dict | None = None, *,
                never_automate_amount: int | None = None) -> TierDecision:
    """这个动作**本身**要求哪一档。不看谁调的、不看模型怎么说。"""
    if tool in ESCALATION_TOOLS:
        return TierDecision(None, "升级通道本身不过闸（给请求审批加审批是循环的）")
    if tool not in WRITE_TOOLS:
        # ⚠️ 读不改变世界 ⇒ 完全不过闸。灰测期间也必须能查
        #   （降级的意义是降级，不是失明）。
        return TierDecision(None, "读工具：不改变外部世界")

    ceiling = never_automate_amount or NEVER_AUTOMATE_AMOUNT
    amount_arg = _AMOUNT_ARG.get(tool)
    if amount_arg is not None:
        raw = (arguments or {}).get(amount_arg)
        try:
            # ⚠️ 模型给的是 JSON 值，数字常以字符串到达（"30"）——
            #   这里认不出就等于把大额动作判成小额（calendar 那个坑的同族）。
            amount = int(float(raw)) if raw is not None else None
        except (TypeError, ValueError):
            # 认不出金额 ⇒ **按最谨慎处理**，不是按最宽（"我们不知道"一律保守）
            return TierDecision("D", f"{tool}：金额参数无法解析（{raw!r}），按最谨慎处理")
        if amount is not None and amount >= ceiling:
            return TierDecision("D", f"{tool}：金额 {amount} ≥ 永不自动线 {ceiling}")

    return TierDecision("C", f"{tool}：不可逆的写动作，需人点头")
