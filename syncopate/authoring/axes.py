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
# ★ M1 新增：数据成熟度。这条轴改变的是**行为本身**，不是回答里多一句免责声明
#   mature   —— D7 已收敛，正常给结论
#   partial  —— D3 有、D7 未到，可给倾向性结论但必须标不确定性
#   immature —— 只有 D1，正确动作是 defer（"X 天后再看"）
# 它是全项目唯一一条能长出 `defer` 行为的轴。
DATA_MATURITIES = ["mature", "partial", "immature"]

# ★ M2 新增：安全线（外部资料）的状态。这条轴长出 RAG 侧的两个核心失败模式
#   current  —— 表里是当周的线，有效，正常拿来判断
#   stale    —— 表里只有旧版（运营忘更新），已过期 ⇒ **不能拿它当决策依据**
#   missing  —— 表里根本没有这个 产品×地域 ⇒ 查不到，**不许编一个数出来**
#
# ⚠️ 两条设计纪律，缺一条这条轴就是摆设：
#
# 1. **旧线和新线的数值必须真的不同**（`scripts/make_test_external_data.py` 的
#    `_WEEK_DRIFT`）。数值一样的话，用旧线和用新线得出同一个结论，判据分辨不出
#    模型有没有真的看有效期 —— 那就成了"能被什么都不做骗过"的指标。
# 2. **工具不替模型判断过没过期**，只如实返回 `valid_to`。真实世界里没人会在
#    返回里塞一个 `expired: true`。模型必须自己拿它和今天比 ——
#    所以 `reference_now` 必须进 prompt，否则这道题没有比较基准、不公平。
SAFETY_LINE_STATES = ["current", "stale", "missing"]

# M8 · RAG v1：政策条款库的三种状态。
#
#   present     只有现行版本            ⇒ 照常引用它作答       ← ★ 对照档
#   superseded  旧版(已过期)+新版同在    ⇒ **必须引用新版**     ← 过期检出率
#   empty       库里根本没这个主题       ⇒ 转人工/反问，不许编   ← 无检索幻觉率
#
# ★ `present` 不是凑数：只装 superseded/empty 的话，模型会学成"见到检索就别信"。
# `defer` 97%→0% 和 dead_grid 只装难例，这个教训已经吃过两次。
#
# ⚠️ 和 safety_line_state 的区别（两者都关于"资料不可用"，但正确出口不同）：
#   安全线过期  世界**没有**可用数据 ⇒ 转人工补录
#   政策过期    现行版本**就在同一次检索结果里** ⇒ 改引用新版继续办事，转人工反而过度保守
RAG_STATES = ["present", "superseded", "empty"]

# M8 · RAG v1：复盘结论（非结构化侧）与**当前数据**的关系。
#
#   aligned      历史结论和现在的数据一致   ⇒ 照常引用作答          ← ★ 对照档
#   conflicting  历史结论被现在的数据推翻   ⇒ **显式做冲突消解**    ← 本轴的主角
#   absent       根本没有相关历史结论       ⇒ 只按数据判断，不许编经验
#
# ★★ `conflicting` 这一档补的是遗留清单里挂了很久的那个缺口：
# 「记忆写机制三个工具只有一个有题（`invalidate` 只当干扰项、`conflict_resolve`
# 完全没用上）。现在没有任何一道题考『查到的历史结论和现在的数据矛盾了怎么办』」。
#
# ⚠️ 正确动作**不是二选一硬答**，是 `memory.conflict_resolve` 显式记录冲突
# 并在终答里说明"历史经验是 X、本次数据是 Y、建议以数据为准并复核该结论"。
# 只答一边（哪怕答对了那边）都算没做这道题 —— 因为**冲突本身就是要报告的信息**。
INSIGHT_STATES = ["aligned", "conflicting", "absent"]

# 各成熟度对应的开投天数（ROAS 7 天收敛）。
# 安装量统一给足，是为了让这条轴**只由时间驱动**——
# 样本量不足是另一种不可信（insufficient_sample），混进来会让两条 cap 同时命中，归因就糊了。
MATURITY_DAYS = {"mature": 14, "partial": 3, "immature": 1}
MATURITY_INSTALLS_7D = 2800.0

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
    data_maturity: str
    safety_line_state: str
    rag_state: str
    insight_state: str

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
        data_maturity=DATA_MATURITIES[(index // 3 + index % 5) % 3],
        # 用 //13 和 %7 这对互不整除的因子，避免和 data_maturity(//3,%5) 同步变化
        safety_line_state=SAFETY_LINE_STATES[(index // 13 + index % 7) % 3],
        # //17 和 %11 又是一对互不整除的因子：和 safety_line_state(//13,%7)、
        # data_maturity(//3,%5) 都不同步 —— 否则两条轴会被绑成一条，
        # 90 个格子里就会有一半永远填不满（稀疏格子被取模削成 0 条的同源坑）。
        rag_state=RAG_STATES[(index // 17 + index % 11) % 3],
        # ★ 因子是**扫出来的**，不是拍的：第一版用 (//19 + %13)，和 rag_state 的
        # 9 格联合分布出现对角线富集（50 vs 22–29，均匀期望 33）——两条轴被部分绑住。
        # 这里乘 2 打破同余结构，实测 9 格 32–35、卡方 0.3。
        # 「乘个质数就去相关」只在模数互质时成立，这条坑 `_mix` 的注释里记着。
        insight_state=INSIGHT_STATES[(index // 11 + index % 7 * 2) % 3],
    )


def axis_summary(params: list[Params]) -> dict[str, dict[str, int]]:
    """各轴的取值分布，用来确认组合是真的铺开了。"""
    out: dict[str, dict[str, int]] = {}
    for axis in ("platform", "genre", "region", "entry_mode", "memory_state",
                 "season_phase", "amount_band", "tier", "memory_action", "data_maturity",
                 "safety_line_state", "rag_state", "insight_state"):
        counts: dict[str, int] = {}
        for p in params:
            key = str(getattr(p, axis))
            counts[key] = counts.get(key, 0) + 1
        out[axis] = dict(sorted(counts.items()))
    return out


def as_dict(params: Params) -> dict[str, Any]:
    return {axis: getattr(params, axis) for axis in
            ("entry_mode", "memory_state", "season_phase", "amount_band", "tier",
             "memory_action", "data_maturity", "platform", "genre", "region")}
