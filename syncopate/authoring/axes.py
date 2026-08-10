"""控制轴：让同一个业务意图长出多条不同的正确路径。

★ 这是 v1 最大的缺陷所在

v1 的 480 条 case 只有 **7 种工具序列**（老师包是 72 种），因为每个模板只写死了
一条 gold。模型只要认出模板就赢了——4B 两个 epoch 就背到 reward 1.000。

真正让任务变难的不是"链更长"，是**"走到第 3 步时，观测结果决定第 4 步走哪边"**。
老师包的 72 种骨架不是硬造出来的，是 5-6 个业务意图 × 若干条件分支自然长出来的：

    entry_mode      给不给关键 id      → 要不要先自己查
    evidence_state  证据清不清楚        → 该动手还是该追问
    amount_band     金额落在阈值哪侧    → 直接做 / 走审批
    exception_flag  有没有政策例外      → 规则本身变了

我们对应到广告域，再加上 v2 新增的两条轴（记忆、时令）。

**关键纪律：轴的取值必须真的改变「正确动作」，不只是改变数字。**
否则骨架数不会涨，只是同一条路上换了参数。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

PLATFORMS = ["Meta", "Google", "TikTok", "AppLovin", "Unity"]
GENRES = ["casual", "puzzle", "hyper_casual", "rpg", "strategy"]
PRODUCTS = ["PUZ_QUEST", "MERGE_FARM", "IDLE_HERO", "TAP_RUSH", "WAR_THRONE"]
REGIONS = ["US", "GB", "DE", "JP", "BR"]
ANOMALIES = ["cpi_spike", "roas_drop", "ctr_decline", "creative_fatigue"]
TIERS = ["standard", "plus"]

# ---- 控制轴 ----
# 关键 id 给不给。must_discover 逼出一步 campaign.list
ENTRY_MODES = ["id_given", "must_discover"]
# 记忆库状态。改变的是**正确动作本身**，不是回答里多说一句
#   clean    —— 没有相关历史，按常规流程走
#   repeated —— 近期已频繁操作，该走审批而不是直接执行
#   risky    —— 有风控标记，该拒绝并说明
MEMORY_STATES = ["clean", "repeated", "risky"]
# 时令阶段。peak 时该主题素材的安全线按 lift_factor 放宽 → 同一条素材从"不能投"变"能投"
SEASON_PHASES = ["off", "approaching", "peak"]
# 金额落在政策阈值的哪一侧
#   below    —— 涨幅 20%，不触发审批，直接执行
#   boundary —— 涨幅刚好卡在上限
#   above    —— 涨幅 80%，既超上限又要审批
AMOUNT_BANDS = ["below", "boundary", "above"]

# 各时令阶段对应的 reference_now（相对万圣节 2026-10-18 ~ 11-02）
SEASON_REFERENCE_NOW = {
    "off": "2026-08-10T00:00:00+00:00",          # 还有 69 天
    "approaching": "2026-10-05T00:00:00+00:00",   # 13 天后
    "peak": "2026-10-25T00:00:00+00:00",          # 正当时
}

# 收尾动作：做完任务要不要把结论沉淀进记忆。
#   none    —— 只回答
#   propose —— 追加一步 memory.write_proposal（需 confidence≥0.7 + 证据≥2 条）
# 这条轴的意义是让"写提案"这套机制真的进 gold ——
# 否则我们建了 write_proposal / invalidate / conflict_resolve，却没有一条 gold 用到它。
MEMORY_ACTIONS = ["none", "propose"]

# 涨幅系数
AMOUNT_FACTOR = {"below": 1.20, "boundary": 1.50, "above": 1.80}


@dataclass(frozen=True)
class Params:
    """一条 case 的全部可变参数。完全由 index 决定，保证生成可复现。"""

    index: int
    platform: str
    genre: str
    product: str
    region: str
    anomaly: str
    tier: str
    entry_mode: str
    memory_state: str
    season_phase: str
    amount_band: str
    memory_action: str

    @property
    def campaign_id(self) -> str:
        return f"CMP_{4000 + self.index}"

    @property
    def account_id(self) -> str:
        return f"ACC_{10 + self.index % 7}"

    @property
    def creative_name(self) -> str:
        return f"halloween_hook_{chr(97 + self.index % 26)}_v{1 + self.index % 3}"

    @property
    def reference_now(self) -> str:
        return SEASON_REFERENCE_NOW[self.season_phase]


def _mix(index: int, size: int, stride: int) -> int:
    """把序号映射到某个轴上，且**和其它轴去相关**。

    ⚠️ 直觉陷阱：`index * k % n` 在模数相同时只是**重排**，和 `index % n`
    仍然一一对应——两个轴会同步变化，25 种组合实际只有 5 种。
    "乘个质数就去相关"只在模数互质时成立。

    正确做法是让轴同时依赖 index 的**高位和低位**（`index // stride`），
    这样它的周期变成 size×stride 而不是 size。
    tests/authoring 里有一条测试专门守着这件事。
    """
    return (index * 2 + index // stride) % size


def params_for(index: int) -> Params:
    return Params(
        index=index,
        platform=PLATFORMS[index % 5],
        genre=GENRES[_mix(index, 5, 5)],
        product=PRODUCTS[_mix(index, 5, 7)],
        region=REGIONS[_mix(index, 5, 3)],
        anomaly=ANOMALIES[_mix(index, 4, 4)],
        tier=TIERS[(index // 3) % 2],
        entry_mode=ENTRY_MODES[(index // 2) % 2],
        memory_state=MEMORY_STATES[(index // 5 + index % 3) % 3],
        season_phase=SEASON_PHASES[(index // 7 + index % 2) % 3],
        amount_band=AMOUNT_BANDS[(index // 11 + index % 3) % 3],
        memory_action=MEMORY_ACTIONS[(index // 4 + index % 2) % 2],
    )


def axis_summary(params: list[Params]) -> dict[str, dict[str, int]]:
    """各轴的取值分布，用来确认组合是真的铺开了。"""
    out: dict[str, dict[str, int]] = {}
    for axis in ("platform", "genre", "region", "entry_mode", "memory_state",
                 "season_phase", "amount_band", "tier", "memory_action"):
        counts: dict[str, int] = {}
        for p in params:
            key = str(getattr(p, axis))
            counts[key] = counts.get(key, 0) + 1
        out[axis] = dict(sorted(counts.items()))
    return out


def as_dict(params: Params) -> dict[str, Any]:
    return {axis: getattr(params, axis) for axis in
            ("entry_mode", "memory_state", "season_phase", "amount_band", "tier",
             "memory_action", "platform", "genre", "region")}
