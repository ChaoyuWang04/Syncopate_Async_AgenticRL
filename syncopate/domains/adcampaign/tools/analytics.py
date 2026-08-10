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
    row = ctx.env.row("campaigns", args.get("campaign_id"))
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
    table = ctx.env.table("creatives")
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
    row = ctx.env.row("campaigns", args.get("campaign_id"))
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
    row = ctx.env.table("benchmarks").get(key)
    if row is None:
        return ToolResult(ok=False, error=f"benchmark_not_found: {key}")
    return ToolResult(ok=True, data=row)
