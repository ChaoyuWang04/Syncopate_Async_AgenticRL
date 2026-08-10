"""`metrics.get_freshness` —— 本业务最重要的一个新工具。

它把「这个数字现在可不可信」从模型的猜测变成了**可查询的事实**。

没有它，模型只有两条路：要么相信眼前的数字（于是拿 D1 当 D7），
要么凭感觉怀疑（于是什么都不敢做）。两种都不是我们要的行为。
有了它，「等」才成为一个有依据的动作。

★ 描述里写死了「不给建议」

这个工具只回答「数据成熟到什么程度」，不回答「所以该不该扩量」。
把判断留给模型，是因为判断正是我们要训进权重的东西；
而成熟度是客观事实，属于世界，不该让模型自己编。
"""

from __future__ import annotations

from typing import Any

from syncopate.core.tool_registry import REGISTRY, ToolContext, ToolResult
from syncopate.domains.adcampaign.maturity import CONVERGE_DAYS, campaign_maturity


@REGISTRY.tool(
    name="metrics.get_freshness",
    description=(
        "查询某个指标此刻的数据成熟度：跑了几天、还差几天收敛、样本量够不够、"
        "以及未收敛时该指标的**预期区间**。\n"
        "不返回指标本身的业务解读，也不给「该不该动」的建议。\n"
        "不查行业基准（那在 benchmark.get_industry_baseline），"
        "不查内部安全线（那在 benchmark.get_safety_line）。\n"
        "★ 涉及扩量、砍量、归因结论之前应当先查它——"
        "数据未收敛时给出的结论，对错完全是运气。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "campaign_id": {"type": "string", "description": "campaign 主键，如 CMP_1024"},
            "metric": {
                "type": "string",
                "enum": sorted(CONVERGE_DAYS),
                "description": "要判断成熟度的指标，默认 roas_d7（最慢收敛的那个）",
            },
        },
        "required": ["campaign_id"],
    },
    kind="read",
)
def get_freshness(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    row = ctx.env.row("campaigns", args.get("campaign_id"))
    if row is None:
        return ToolResult(ok=False, error=f"campaign_not_found: {args.get('campaign_id')}")
    metric = args.get("metric") or "roas_d7"
    if metric not in CONVERGE_DAYS:
        return ToolResult(ok=False, error=f"unknown_metric: {metric}")
    if metric not in row:
        return ToolResult(ok=False, error=f"metric_not_tracked: {metric}")
    return ToolResult(ok=True, data=campaign_maturity(row, metric))
