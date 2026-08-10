"""优化方案工具。

这个工具的存在只为一件事：**制造真实的顺序依赖**。

`anomaly_type` 是必填参数，而它的合法取值只能从 `campaign.detect_anomalies` 的
返回里拿到。模型如果不先诊断就来要方案，只能瞎猜一个类型 —— 猜错就报错。
这是 sequential_dependency bucket 的核心训练信号，而且是**结构性**的，
不靠 prompt 里写「你要先诊断」这种软约束。
"""

from __future__ import annotations

from typing import Any

from syncopate.core.tool_registry import REGISTRY, ToolContext, ToolResult

PLAYBOOK: dict[str, dict[str, Any]] = {
    "cpi_spike": {
        "recommended_action": "narrow_targeting",
        "detail": "收窄定向、暂停高 CPI 的受众包，优先保留历史 CPI 低于基准的组合",
        "expected_impact": "CPI 下降 10-20%",
    },
    "roas_drop": {
        "recommended_action": "rebalance_budget",
        "detail": "把预算从低 ROAS 的 ad group 转移到高 ROAS 的组合，必要时下调总预算",
        "expected_impact": "ROAS 回升 15-25%",
    },
    "ctr_decline": {
        "recommended_action": "refresh_creative",
        "detail": "更换素材开头 3 秒的 hook，测试新的视觉方向",
        "expected_impact": "CTR 提升 20-30%",
    },
    "creative_fatigue": {
        "recommended_action": "rotate_creative",
        "detail": "轮换 top-3 素材、下线曝光频次超过 4 的旧素材",
        "expected_impact": "频次回落至 3 以下，CPI 下降 15-20%",
    },
}


@REGISTRY.tool(
    name="playbook.get_optimization",
    description="根据已确认的异常类型，返回对应的优化方案。anomaly_type 必须是 campaign.detect_anomalies 实际返回过的类型。",
    parameters={
        "type": "object",
        "properties": {
            "anomaly_type": {
                "type": "string",
                "description": "异常类型，取值来自 campaign.detect_anomalies 的返回",
                "enum": sorted(PLAYBOOK),
            }
        },
        "required": ["anomaly_type"],
    },
    kind="read",
    requires=["campaign.detect_anomalies"],
)
def get_optimization(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    anomaly_type = args.get("anomaly_type")
    entry = PLAYBOOK.get(str(anomaly_type))
    if entry is None:
        return ToolResult(
            ok=False,
            error=f"unknown_anomaly_type: {anomaly_type}. Run campaign.detect_anomalies first.",
        )
    return ToolResult(ok=True, data={"anomaly_type": anomaly_type, **entry})
