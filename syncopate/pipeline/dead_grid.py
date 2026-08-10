"""从 base 评测审计里提取「死格」，供 `split.py` 的 dead_grid 模式选 SFT 桶。

★ 核心区分：`p=0` 有两种成因，混在一起会把 SFT 引向错误方向

    约定未知   base 没见过我们的输出约定（`behavior: clarify/reject`），
               于是一路调工具撞到 max_steps。**廉价**，SFT 一个 epoch 就解决。
               实测证据：v3_plain 一轮把这批从 0.000 拉到 1.000。
    能力不足   真的不会做——跳过 memory.search、跳过安全线核查就下结论。
               **昂贵**，这才是「死格」的实质，也是 SFT 冷启动的真正目标。

只看 reward 分不出这两者：两类都可能 reward 低、组内 std 为 0。分开的判据是
**cap 命中**——走捷径会打中 missing_memory_check / false_claim / unauthorized_write，
而约定未知只是 behavior_mismatch + 高截断率。

⚠️ 一个必须记住的坑：「全灭」不等于「难」。
   按 reward 从低到高排序去选 SFT 桶，选出来的是「约定未知」那批（最便宜的），
   真正的死格反而卡在 0.3–0.4 的中间分区，排序时被排到后面去了。

---

★ 外推的边界：这份审计只测了 52 条冻结 EVAL

死格是在 EVAL 上测出来的，而 EVAL 是冻结集、永不训练。所以要把死格「翻译」成
SFT 桶，必须经过一次外推：EVAL 里的死 case → 它所在的格子 → 池子里同格的 case。

**这个外推是有损的**，而且我们能量到损失有多大：12 个死格里有 3 个
（`REJ|reject|-|id_given`、`LOW|tool_call|-|id_given`、`BUD|tool_call|denied|must_discover`）
的 2 条 EVAL 样本只死了 1 条 —— 说明 (模板, behavior, 结局, entry_mode) 这个粒度
**分辨不出同格内的死活**。想消掉这个外推，只能直接在池子上跑 base 评测拿 per-case 的 p。

在那之前，配额（见 `DEFAULT_QUOTA`）就是控制外推损失的旋钮：格子越可疑，取得越少。
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---- 分类判据（阈值和 `_audit/M0_base_4b.json` 的分析口径一致，改这里等于改口径）----

STD_FLOOR = 0.01        # 组内 std 高于它就有 advantage，GRPO 能学
WIPEOUT_CEIL = 0.15     # 低于它算「全灭」
SATURATED_FLOOR = 0.9   # 高于它算「饱和」

# 「系统性跳过前置」的指纹：这三条 cap 说的都是同一件事——
# 没做该做的调查就下了结论 / 动了手。它们把 A 类卡死和 B 类卡死分开。
SHORTCUT_CAPS = frozenset({
    "missing_memory_check_cap",
    "false_claim_cap",
    "unauthorized_write_cap",
})

# ---- 五种格子 ----

GRADIENT = "gradient"       # 有梯度，RL 自己能学，SFT 别碰
SATURATED = "saturated"     # 已经会了
CONVENTION = "convention"   # 全灭·约定未知    → 廉价，少量样本即可
SHORTCUT = "shortcut"       # A 类卡死·走捷径  → ★ SFT 冷启动的正主
SUBSCORE = "subscore"       # B 类卡死·子分丢分 → 这是 RL 该管的，别放进 SFT

DEAD_KINDS = (CONVENTION, SHORTCUT)

# ★ 每个死格从池子里取多少条。
#
# 为什么不整格全取：12 个死格对应池子里 288/528 条，SFT 桶会吃掉一多半的 RL 池，
# 而 SFT 冷启动的职责只是把 p 从 0 抬到 5–10%，不需要这么多样本。
#
# 为什么 convention 给得少：它一个 epoch 就学会（实测 0.000 → 1.000），
# 多给的样本换不来任何东西，只是把 RL 池挖空。
#
# 为什么总量卡在 ~105：旧的 difficulty_proxy 桶正好是 105 条。保持规模不变、
# 只换成分，新旧两次 SFT 才是**单变量对比**——否则分数变了也说不清是
# 「选对了对象」还是「样本多了」。
DEFAULT_QUOTA: dict[str, int] = {CONVENTION: 6, SHORTCUT: 10}


def classify(row: dict[str, Any]) -> str:
    """把一条 base 评测结果归到五种格子之一。

    顺序不能换：先判有没有梯度（std），再判死活（reward），最后才用 cap 分死因。
    """
    if row["reward_std"] > STD_FLOOR:
        return GRADIENT
    if row["reward"] > SATURATED_FLOOR:
        return SATURATED
    if row["reward"] < WIPEOUT_CEIL:
        return CONVENTION
    return SHORTCUT if set(row["caps"]) & SHORTCUT_CAPS else SUBSCORE


@dataclass
class DeadGrid:
    """一个死格：格子键 + 死因 + 支持它的 EVAL 证据。"""

    stratum: tuple[str, ...]
    kind: str
    eval_case_ids: list[str] = field(default_factory=list)
    eval_seen: int = 0          # 这个格子在 EVAL 里一共几条（分母，用来看外推有多可靠）

    @property
    def confidence(self) -> float:
        """格子内的死亡率。< 1.0 说明同格里有活的，外推会把活 case 一起选进来。"""
        return len(self.eval_case_ids) / self.eval_seen if self.eval_seen else 0.0


def load_audit(path: Path) -> list[dict[str, Any]]:
    return json.loads(Path(path).read_text(encoding="utf-8"))["rows"]


def analyze(rows: list[dict[str, Any]], strata: dict[str, tuple[str, ...]]) -> tuple[dict[tuple[str, ...], DeadGrid], Counter]:
    """rows（base 评测）+ strata（case_id → 格子键）→ 死格表 + 五种格子的计数。"""
    seen: Counter = Counter()
    for row in rows:
        seen[strata[row["case_id"]]] += 1

    grids: dict[tuple[str, ...], DeadGrid] = {}
    kinds: Counter = Counter()
    for row in rows:
        kind = classify(row)
        kinds[kind] += 1
        if kind not in DEAD_KINDS:
            continue
        key = strata[row["case_id"]]
        grid = grids.get(key)
        if grid is None:
            grids[key] = grid = DeadGrid(stratum=key, kind=kind, eval_seen=seen[key])
        elif grid.kind != kind:
            # 同一格里两种死因都出现了：按更贵的那种算（配额更大，宁可多给样本）。
            grid.kind = SHORTCUT
        grid.eval_case_ids.append(row["case_id"])
    return grids, kinds
