"""素材类工具：上传 + 轮询审核。

★ `creative.poll_review` 是整个项目的长尾来源。

素材上传后要等平台审核，真实世界里是 2–4 小时。我们把它建模成 **480 秒的真实
`asyncio.sleep`**，不是打个时间戳假装慢。原因见 tool_registry 的模块 docstring：
假的慢暴露不了阻塞问题，异步对照实验就白做了。

调试时把 `REGISTRY.latency_scale` 调成 0.01，480 秒变 4.8 秒。
"""

from __future__ import annotations

from typing import Any

from syncopate.core.tool_registry import REGISTRY, ToolContext, ToolResult
from syncopate.domains.adcampaign.policies import select_policy

_STR = {"type": "string"}

# 审核等待时长（秒）。真实平台是 2-4 小时，取下界的一个保守值。
REVIEW_LATENCY_SECONDS = 480.0


def _asset_id(campaign_id: str, creative_name: str) -> str:
    """确定性 asset_id：同样的输入永远得到同样的 id，重放才可能。"""
    return f"ASSET_{campaign_id}_{creative_name}"


@REGISTRY.tool(
    name="creative.upload",
    description="上传一条素材到指定 campaign。上传后进入平台审核队列，需要用 creative.poll_review 查询审核结果。",
    parameters={
        "type": "object",
        "properties": {
            "campaign_id": _STR,
            "creative_name": {**_STR, "description": "素材名称"},
            "asset_type": {**_STR, "description": "video / image"},
            "duration_seconds": {"type": "number", "description": "视频时长，图片可不填"},
        },
        "required": ["campaign_id", "creative_name", "asset_type"],
    },
    kind="write",
    fact_key="creative_uploaded",
)
def upload_creative(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    campaign = ctx.row("campaigns", args.get("campaign_id"))
    if campaign is None:
        return ToolResult(ok=False, error=f"campaign_not_found: {args.get('campaign_id')}")

    # 平台规格校验：不合规直接退回，这是 critical_args bucket 的信号来源
    policy = select_policy(ctx.env.policies, "creative_spec", {"platform": campaign.get("platform")})
    if policy:
        rules = policy["rules"]
        asset_type = args.get("asset_type")
        if asset_type not in rules.get("allowed_types", []):
            return ToolResult(ok=False, error=f"asset_type_not_allowed: {asset_type} on {campaign.get('platform')}")
        duration = args.get("duration_seconds")
        if duration is not None and float(duration) > rules.get("max_duration_seconds", 1e9):
            return ToolResult(
                ok=False,
                error=f"duration_exceeds_limit: {duration}s > {rules['max_duration_seconds']}s",
            )

    asset_id = _asset_id(campaign["campaign_id"], str(args.get("creative_name")))
    return ToolResult(ok=True, data={
        "asset_id": asset_id,
        "campaign_id": campaign["campaign_id"],
        "review_status": "pending",
        "estimated_review_seconds": int(REVIEW_LATENCY_SECONDS),
    })


@REGISTRY.tool(
    name="creative.poll_review",
    description="查询已上传素材的审核结果。审核需要时间，这个调用会阻塞等待直到出结果。",
    parameters={
        "type": "object",
        "properties": {"asset_id": {**_STR, "description": "creative.upload 返回的 asset_id"}},
        "required": ["asset_id"],
    },
    kind="read",
    latency_seconds=REVIEW_LATENCY_SECONDS,   # ← 真实等待，长尾轨迹的来源
)
def poll_review(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    asset_id = args.get("asset_id")
    # 必须先上传过：查一个没上传的 asset 直接报错，形成真实的顺序依赖
    uploaded = [r for r in ctx.sandbox.records_for("creative.upload") if r.result.get("asset_id") == asset_id]
    if not uploaded:
        return ToolResult(ok=False, error=f"asset_not_found: {asset_id} (upload it first)")

    # 审核结果由 env 决定（按素材名查预设结果），默认通过。
    # 让 case 能构造「上传了但被拒」的分支，而不是永远一路绿灯。
    creative_name = uploaded[-1].arguments.get("creative_name")
    outcome = ctx.table("review_outcomes").get(str(creative_name), {})
    status = outcome.get("review_status", "approved")
    return ToolResult(ok=True, data={
        "asset_id": asset_id,
        "review_status": status,
        "reject_reason": outcome.get("reject_reason") if status == "rejected" else None,
    })
