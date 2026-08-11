"""`metrics.get_freshness` —— 本业务最重要的一个新工具。

它把「这个数字现在可不可信」从模型的猜测变成了**可查询的事实**。

没有它，模型只有两条路：要么相信眼前的数字（于是拿 D1 当 D7），
要么凭感觉怀疑（于是什么都不敢做）。两种都不是我们要的行为。
有了它，「等」才成为一个有依据的动作。

★★★ 它只返回**事实**，不返回**判断** —— 这条线是实测校准出来的

第一版把 `maturity: "mature|partial|immature"` 直接返回了。结果 base 模型
（完全没训过）在 I02 上的 `defer` 准确率就有 **97%** —— 因为 I02 要求的答案字段
就叫 `data_maturity`，模型只需要把工具返回值**照抄**过去。判断根本没有发生。

⇒ 设计文档 §0.1 的切分线在这里被我自己越过了：
  **「查什么」属于工具，「怎么判」属于权重。**
  工具一旦把结论算好，这个任务就退化成一次字段拷贝，SFT 和 RL 都没东西可学。

所以现在返回的是原始事实：跑了几天、这个指标几天收敛、样本量多少、
未收敛时的预期区间。**从这些事实推出「现在能不能下结论」，是模型的活。**

同理，阈值（几天算收敛、样本量下限）也不写在 system prompt 里——
写了就等于把规则又还给了模型的上下文，而这条规则是"不变的"，该进权重。
"""

from __future__ import annotations

from typing import Any

from syncopate.core.tool_registry import REGISTRY, ToolContext, ToolResult
from syncopate.domains.adcampaign.maturity import CONVERGE_DAYS, campaign_maturity

# 回给模型的字段：全是**可观测的事实**，没有一个是判断。
FACT_FIELDS = ("metric", "days_elapsed", "converge_at_day", "sample_size",
               "min_sample_size", "current_value", "expected_final_range")


@REGISTRY.tool(
    name="metrics.get_freshness",
    description=(
        "查询某个指标的观测条件：这条 campaign 开投了几天、该指标通常几天收敛、"
        "累计样本量多少、以及该指标的预期区间。\n"
        "**只给事实，不给结论**——它不判断数据可不可信、也不建议该不该动，那是你的判断。\n"
        "不查行业基准（那在 benchmark.get_industry_baseline），"
        "不查内部安全线（那在 benchmark.get_safety_line）。\n"
        "★ 涉及扩量、砍量、归因结论之前应当先查它——"
        "在还没收敛的数字上给出的结论，对错完全是运气。"
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
    row = ctx.row("campaigns", args.get("campaign_id"))
    if row is None:
        return ToolResult(ok=False, error=f"campaign_not_found: {args.get('campaign_id')}")
    metric = args.get("metric") or "roas_d7"
    if metric not in CONVERGE_DAYS:
        return ToolResult(ok=False, error=f"unknown_metric: {metric}")
    if metric not in row:
        return ToolResult(ok=False, error=f"metric_not_tracked: {metric}")
    info = campaign_maturity(row, metric)
    # ★ 只透传事实字段。`maturity` / `is_converged` / `reason` 是**结论**，
    # 留在 campaign_maturity() 里给 verifier 和 cap 用，绝不回给模型。
    return ToolResult(ok=True, data={k: info[k] for k in FACT_FIELDS})
