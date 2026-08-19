"""B-7 · 灰测的**闸门**：放量档位 + 回滚开关。

★★★ 三个设计判断（2026-08-19），每个都换掉了一个"想当然"的做法

**① 放量的维度不是"流量比例"，是 `automation_tier`。**

    流量比例  控制**多少人碰到** agent
    tier      控制 **agent 能自己做多少**

⇒ 出事的**严重度由后者决定**，不由前者。10% 的流量配 A 档（全自动改预算），
  比 100% 的流量配 C 档（每个动作都要人点头）危险得多。
⇒ 而且 `automation_tier` **已经在 schema 里、已经有消费者**
  （D 档拒绝、C 档走审批）——不需要新造一套灰度机制。

    灰度阶梯：**D（只看不做）→ C（都要审批）→ B → A**
    而不是：  10% → 50% → 100%

**② 回滚开关必须 fail-closed。**

读不到开关状态 ⇒ 按**关闭**处理，不是按打开。
理由和 `retrieval_unavailable` 那条一样：**"我们不知道"一律保守**。
⚠️ 反过来（读不到就放行）意味着**配置服务一挂，全量放开** —— 那是最坏的时刻放最大的权。

**③ 闸门放在 `ActionGate` 里，不放在 API 层。**

同一个动作可能来自 API、也可能来自 worker 自己的编排。
放在 API 层的话，worker 那条路**完全绕过**它。
⇒ 和权限闸同一个道理（`tools.py` 的注释：两条路都得过同一道闸）。

---

⚠️⚠️ **一个必须一起想清楚的问题：关掉之后，在飞的 run 怎么办。**

关掉不能只是"新的不许起" —— 已经停在 `waiting_for_user` 的 run
如果没人再去恢复它们，就**永远卡在那里**。
⇒ 所以 `halt_reason` 会写进事件流：**关闭是一个可追溯的动作**，
  重新打开时能按它把受影响的 run 找出来。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

Tier = Literal["A", "B", "C", "D"]

# 档位从严到宽。★ D 最严（只看不做），A 最宽（全自动）。
TIER_ORDER: tuple[Tier, ...] = ("D", "C", "B", "A")

# ★ 默认上限是 **C** —— 灰测第一档不是"全自动"，是"每个写动作都要人点头"。
#   ⚠️ 默认值刻意选保守的那一端：忘了配置的后果应该是**太严**，不是太松。
DEFAULT_MAX_TIER: Tier = "C"

# 没声明档位时按哪一档算。见 `ReleaseState.allows` 的长注释。
UNDECLARED_TIER: Tier = "C"

ENV_KILL = "SYNCOPATE_RELEASE_HALTED"
ENV_MAX_TIER = "SYNCOPATE_RELEASE_MAX_TIER"


@dataclass(frozen=True)
class ReleaseState:
    """当前的灰测闸门状态。"""

    halted: bool
    max_tier: Tier
    halt_reason: str | None = None

    def allows(self, tier: str | None) -> bool:
        """这一档动作**要的自主度**在当前上限之内吗。

        比的是**自主度**：D < C < B < A。上限 C 意味着"最多允许到 C 这么自主"，
        所以声明 A（全自动）的 run 会被拦下，声明 C/D 的放行。

        ⚠️⚠️ `tier=None`（**没声明**）怎么办 —— 这一条我写错过一次：

            第一版映射到 "D" ⇒ D 是自主度最低的一档 ⇒ **恒为 True，全部放行**
            而我的文档串写着"按最严处理" —— **代码和文档说的是反的**，
            是自己的测试当场炸出来的。

        ⇒ 想清楚之后的解：**按 `C`（要人点头）那一档处理**。
          · 不按 `A`：那等于默认全自动，「没写」和「写了 A」是两件事
          · 也不按"直接全挡"：现在**绝大多数 run 都没声明**（API 不强制），
            全挡等于灰测还没开始就把系统关了
        ⬜ **真正该做的是让 `automation_tier` 在建 run 时必填** ——
           那样就没有"没声明"这种状态了。记为缺口，别用默认值把它盖住。
        """
        if self.halted:
            return False
        t = tier if tier in TIER_ORDER else UNDECLARED_TIER
        return TIER_ORDER.index(t) <= TIER_ORDER.index(self.max_tier)


def current_state() -> ReleaseState:
    """读闸门状态。**fail-closed**：读不到 / 读错 ⇒ 当成已关闭。

    ⚠️ 反过来（读不到就放行）意味着**配置服务一挂就全量放开** ——
      那是最坏的时刻放最大的权。同「查不了 ≠ 查不到」那条不对称：
      两种误判的代价差得很远。
    """
    raw = os.environ.get(ENV_KILL, "0").strip().lower()
    halted = raw not in ("0", "", "false", "no")
    tier = os.environ.get(ENV_MAX_TIER, DEFAULT_MAX_TIER).strip().upper()
    if tier not in TIER_ORDER:
        # ★ 配错了档位 ⇒ **当成关闭**，不要退回默认值继续跑。
        #   退回默认值会让"配错了"看起来像"配对了"——而那是最贵的一类错误。
        return ReleaseState(halted=True, max_tier="D",
                            halt_reason=f"invalid_max_tier: {tier!r}")
    return ReleaseState(halted=halted, max_tier=tier,  # type: ignore[arg-type]
                        halt_reason="kill_switch" if halted else None)
