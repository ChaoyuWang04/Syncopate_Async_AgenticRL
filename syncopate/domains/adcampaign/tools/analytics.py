"""分析类读工具：查指标、查素材、诊断异常、查行业基准。

四个工具里有两组容易混淆的：

  campaign.get_metrics  vs  creative.get_metrics_by_asset      campaign 层 vs 素材层
  benchmark.get_industry_baseline  vs  benchmark.get_safety_line   行业基准 vs 内部安全线

★ 早期这两组的名字分别是 `creative.get_performance` 和 `benchmark.query`，
**光看名字分不出粒度和用途**——这是典型的"名称偏见"，模型选错工具的锅
一半在 schema 设计而不是训练。改名比训模型便宜三个数量级，所以 M0 一并改了。

描述统一采用「先说我做什么，再说**我不做什么**」的写法：
指出边界比罗列功能更能防混淆。
"""

from __future__ import annotations

import math
from typing import Any

from syncopate.core.tool_registry import REGISTRY, ToolContext, ToolResult

_STR = {"type": "string"}


@REGISTRY.tool(
    name="campaign.get_metrics",
    description=(
        "查询单个 campaign 的投放大盘指标：花费、安装、CPI、ROAS、CTR、频次。\n"
        "不含单条素材的明细（那在 creative.get_metrics_by_asset）。\n"
        "不判断数据是否收敛、是否可信（那是另一回事）。\n"
        "不含行业对比（那在 benchmark.get_industry_baseline）。"
    ),
    parameters={
        "type": "object",
        "properties": {"campaign_id": {**_STR, "description": "campaign 主键，如 CMP_1024"}},
        "required": ["campaign_id"],
    },
    kind="read",
)
def get_campaign_metrics(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    row = ctx.row("campaigns", args.get("campaign_id"))
    if row is None:
        return ToolResult(ok=False, error=f"campaign_not_found: {args.get('campaign_id')}")
    # 只回投放指标，不回 account_id 这类内部字段——模型要拿账户信息得另外查。
    keys = ("campaign_id", "name", "platform", "game_genre", "status", "daily_budget",
            "spend_7d", "installs_7d", "cpi", "roas_d7", "ctr", "frequency", "impressions")
    return ToolResult(ok=True, data={k: row[k] for k in keys if k in row})


@REGISTRY.tool(
    name="creative.get_metrics_by_asset",
    description=(
        "按素材粒度查表现：逐条素材的 CTR、IPM、花费、曝光频次、疲劳分。\n"
        "不返回 campaign 层的汇总（那在 campaign.get_metrics）。\n"
        "不返回素材的视觉标签（那在 creative.get_asset_tags）。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "campaign_id": {**_STR, "description": "按 campaign 拉全部素材"},
            "creative_id": {**_STR, "description": "只查单条素材"},
        },
        "required": [],
    },
    kind="read",
)
def get_creative_performance(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    table = ctx.table("creatives")
    creative_id, campaign_id = args.get("creative_id"), args.get("campaign_id")
    if creative_id:
        row = table.get(creative_id)
        return ToolResult(ok=True, data={"creatives": [row]}) if row else ToolResult(
            ok=False, error=f"creative_not_found: {creative_id}")
    if campaign_id:
        rows = [r for r in table.values() if r.get("campaign_id") == campaign_id]
        if not rows:
            return ToolResult(ok=False, error=f"no_creatives_for_campaign: {campaign_id}")
        return ToolResult(ok=True, data={"creatives": rows})
    return ToolResult(ok=False, error="missing_argument: need campaign_id or creative_id")


# 异常判定阈值。detect_anomalies 是**真算**出来的，不是从表里读预置答案——
# 这样 case 只要调指标数值，异常就自然变化，不用手工维护一张异常表。
_ANOMALY_RULES = (
    ("cpi_spike",        lambda r: _ratio(r, "cpi", "cpi_baseline") > 1.15),
    ("roas_drop",        lambda r: _ratio(r, "roas_d7", "roas_d7_baseline") < 0.85),
    ("ctr_decline",      lambda r: _ratio(r, "ctr", "ctr_baseline") < 0.80),
    ("creative_fatigue", lambda r: float(r.get("frequency") or 0) > 4.0),
)


def _ratio(row: dict[str, Any], actual_key: str, baseline_key: str) -> float:
    baseline = float(row.get(baseline_key) or 0.0)
    if baseline <= 0:
        return 1.0
    return float(row.get(actual_key) or 0.0) / baseline


@REGISTRY.tool(
    name="campaign.detect_anomalies",
    description="诊断 campaign 是否存在指标异常，返回异常类型列表（如 cpi_spike / roas_drop / creative_fatigue）。要拿优化方案必须先用它确定异常类型。",
    parameters={
        "type": "object",
        "properties": {"campaign_id": _STR},
        "required": ["campaign_id"],
    },
    kind="read",
)
def detect_anomalies(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    row = ctx.row("campaigns", args.get("campaign_id"))
    if row is None:
        return ToolResult(ok=False, error=f"campaign_not_found: {args.get('campaign_id')}")
    hits = [name for name, predicate in _ANOMALY_RULES if predicate(row)]
    return ToolResult(ok=True, data={
        "campaign_id": row["campaign_id"],
        "anomalies": hits,
        "severity": "high" if len(hits) >= 2 else ("medium" if hits else "none"),
    })


@REGISTRY.tool(
    name="benchmark.get_industry_baseline",
    description=(
        "查询**行业**基准值（不是你自己的投放数据，也不是内部安全线）。\n"
        "按 平台+游戏品类+指标 定位，用于判断自己的数据在行业里是高是低。\n"
        "不是决策依据——决定能不能扩量要用 benchmark.get_safety_line 的内部安全线。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "platform": {**_STR, "description": "Meta / Google / TikTok / AppLovin / Unity"},
            "game_genre": {**_STR, "description": "casual / puzzle / rpg / strategy / hyper_casual"},
            "metric": {**_STR, "description": "cpi / roas_d7 / ctr / retention_d1"},
        },
        "required": ["platform", "game_genre", "metric"],
    },
    kind="read",
)
def query_benchmark(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    key = f"{args.get('platform')}|{args.get('game_genre')}|{args.get('metric')}"
    row = ctx.table("benchmarks").get(key)
    if row is None:
        return ToolResult(ok=False, error=f"benchmark_not_found: {key}")
    return ToolResult(ok=True, data=row)


# ==========================================================================
# ★★★ M3 · feature 归因
# ==========================================================================

# 少于这么多条素材就不该下归因结论。
# 和 maturity.MIN_SAMPLE_INSTALLS 是同一个思路的**另一个维度**：
# 那个管「时间够不够」（D7 要 7 天才收敛），这个管「样本够不够」。
# 设计文档给 feature_lift 的原话是 **"让模型学不会拿 3 个样本下结论"**。
MIN_FEATURE_SAMPLE = 12

# 归因认的 feature 全集。摊平成二值字段而不是从 visual_tags 里挑 ——
# 让模型从一个混着主题词的标签数组里判断"哪些算 feature"，是它不该承担的负担。
FEATURES = ("real_person", "before_after", "dark_palette", "fast_cut", "ugc_style")


def _mean_var(values: list[float]) -> tuple[float, float]:
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    mean = sum(values) / n
    if n < 2:
        return mean, 0.0
    return mean, sum((v - mean) ** 2 for v in values) / (n - 1)


@REGISTRY.tool(
    name="analysis.feature_lift",
    description=(
        "算某个素材 feature 在某个地域对 d7 ROAS 的提升幅度（lift），"
        "带 95% 置信区间、两组样本量和显著性判定。\n"
        f"· feature 取值：{' / '.join(FEATURES)}\n"
        "· **必须逐地域分别算**。同一个 feature 在不同地域可能符号相反，"
        "把地域混在一起算出的结论，在某些地域是反的。\n"
        f"· **样本量少于 {MIN_FEATURE_SAMPLE} 条素材时，不论 lift 多大都不能据此下结论**"
        "——小样本下噪声会淹没信号，算出来的数字看着唬人但不可信。"
        "这种情况应如实说明样本不足、拒绝给出归因结论。\n"
        "· 不返回素材清单（那在 creative.get_metrics_by_asset），"
        "不返回标签（那在 creative.get_asset_tags）。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "feature": {"type": "string", "description": f"{' / '.join(FEATURES)}"},
            "region": {"type": "string", "description": "地域，如 US / JP。必填，不支持跨地域合并"},
            "product_id": {"type": "string", "description": "只看该产品；不给则看该地域全部"},
        },
        "required": ["feature", "region"],
    },
    kind="read",
)
def feature_lift(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    """★ **真算，不读预置答案**。

    和同文件的 `detect_anomalies` 是同一条纪律：规律埋在素材表现的生成方式里
    （`scripts/make_test_external_data.py` 的 `_roas_for`），这里从数据重新算出来。
    于是改一条素材的数值，归因结论就自然跟着变，不用手工维护一张"正确答案表"。

    统计口径刻意用最朴素的一种（两独立样本均值差 + 正态近似的 95% CI）：
    判据必须能被**离线复算**——gold 的结论就是拿这个函数算出来的。
    换成需要查表的 t 分布，gold 和工具就有了两套实现，早晚会漂。
    """
    feature, region = args.get("feature"), args.get("region")
    if feature not in FEATURES:
        return ToolResult(ok=False, error=f"unknown_feature: {feature}（可选 {list(FEATURES)}）")
    if not region:
        return ToolResult(ok=False, error="region_required: 归因必须逐地域算，不支持跨地域合并")

    product_id = args.get("product_id")
    treat: list[float] = []
    control: list[float] = []
    for row in ctx.table("creative_catalog").values():
        if row.get("region") != region:
            continue
        if product_id and row.get("product_id") != product_id:
            continue
        roas = row.get("roas_d7")
        if roas is None:
            continue
        (treat if feature in (row.get("features") or []) else control).append(float(roas))

    if not treat or not control:
        return ToolResult(ok=False,
                          error=f"no_samples: {feature}|{region} 两组之一为空 "
                                f"(有该 feature {len(treat)} 条 / 无 {len(control)} 条)")

    mean_t, var_t = _mean_var(treat)
    mean_c, var_c = _mean_var(control)
    lift = mean_t / mean_c - 1.0 if mean_c else 0.0
    # 均值差的标准误 → 相对提升的标准误（除以对照组均值）
    se_diff = math.sqrt(var_t / len(treat) + var_c / len(control))
    se_lift = se_diff / mean_c if mean_c else 0.0
    lo, hi = lift - 1.96 * se_lift, lift + 1.96 * se_lift
    return ToolResult(ok=True, data={
        "feature": feature,
        "region": region,
        "product_id": product_id,
        "lift": round(lift, 4),
        "confidence_interval": [round(lo, 4), round(hi, 4)],
        "sample_size": len(treat),
        "control_size": len(control),
        # ★ 显著 = 置信区间不跨 0。**不叠加样本量门槛** ——
        # 小样本下 CI 很宽，本来就很难显著；但也存在"样本少却碰巧显著"的格子，
        # 那正是我们要模型自己用 sample_size 拦下来的陷阱。
        # 工具只报事实，判断留给模型（和 get_freshness 同一个分工）。
        "is_significant": lo > 0 or hi < 0,
        "baseline_roas_d7": round(mean_c, 4),
        "min_sample_for_conclusion": MIN_FEATURE_SAMPLE,
    })
