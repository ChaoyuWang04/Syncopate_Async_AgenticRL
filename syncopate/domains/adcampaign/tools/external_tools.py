"""外部资料工具：安全线（Excel）、时令日历、素材标签与相似检索。

这四个工具查的都是**离线预处理好的结构化数据**，在线一律不跑 Excel 解析、不跑视觉模型：

    运营改 Excel / 素材库进新图
        → scripts/ingest_external.py（离线跑一次，含 VLM 打标）
        → data/external/ingested.json
        → WorldBuilder 注入 env 只读表
        → 这里的工具查

★ 为什么安全线不做 RAG：它是拿来做**判断**的（`cpi > ceiling` 就告警），数值必须精确。
检索文本让模型自己读数一定会读错，而且不可验证——那就又得上 LLM judge。
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from syncopate.core.tool_registry import REGISTRY, ToolContext, ToolResult
from syncopate.domains.adcampaign.memory import parse_time

_STR = {"type": "string"}


@REGISTRY.tool(
    name="benchmark.get_safety_line",
    description=(
        "查询本产品在指定地域的投放安全线（内部每周更新）："
        "d7 CPI 上限、d7 ROAS 下限、d1 留存下限、日预算上限。"
        "判断投放是否超标、能否加预算时必须以这条线为准，不要用行业基准代替。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "product_id": {**_STR, "description": "产品标识，如 PUZ_QUEST"},
            "region": {**_STR, "description": "地域，如 US / GB / DE / JP / BR"},
        },
        "required": ["product_id", "region"],
    },
    kind="read",
)
def get_safety_line(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    key = f"{args.get('product_id')}|{args.get('region')}"
    row = ctx.table("safety_lines").get(key)
    if row is None:
        return ToolResult(ok=False, error=f"safety_line_not_found: {key}")
    return ToolResult(ok=True, data=dict(row))


@REGISTRY.tool(
    name="calendar.get_seasonal_context",
    description=(
        "查询当前时间点附近的时令活动（万圣节、黑五、圣诞等），"
        "返回距离天数、出量放大倍数和对应的素材标签。"
        "判断某类主题素材现在是否适合投放时用。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "region": {**_STR, "description": "只看该地域生效的活动"},
            "event": {**_STR, "description": "只看某个活动，如 halloween；不给则返回全部"},
            "horizon_days": {"type": "integer", "description": "向前看多少天，默认 45"},
        },
        "required": ["region"],
    },
    kind="read",
)
def get_seasonal_context(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    now = parse_time(ctx.env.reference_now)
    horizon = int(args.get("horizon_days", 45))
    region = args.get("region")
    wanted = args.get("event")
    upcoming, active = [], []
    for event in ctx.table("seasonal_events").values():
        if region and region not in event.get("regions", []):
            continue
        if wanted and event["event"] != wanted:
            continue
        start, end = parse_time(event["start"]), parse_time(event["end"])
        if start <= now <= end:
            # phase=peak：时令正当时，该主题素材的安全线可以按 lift_factor 放宽
            active.append({**event, "days_until": 0, "phase": "peak"})
        elif now < start <= now + timedelta(days=horizon):
            days = (start - now).days
            upcoming.append({**event, "days_until": days,
                             "phase": "approaching" if days <= 21 else "off"})
    return ToolResult(ok=True, data={
        "reference_now": ctx.env.reference_now,
        "queried_event": wanted,
        "active_events": active,
        "upcoming_events": sorted(upcoming, key=lambda e: e["days_until"]),
        # ⚠️ 多个时令可能同时生效（比如夏季长档期里又冒出万圣节），
        # 所以这个字段只在指定了 event 时才无歧义。case 要判断某个主题素材
        # 当下适不适合投，必须传 event 参数。
        "phase": (active[0]["phase"] if active else
                  (upcoming[0]["phase"] if upcoming else "off")),
    })


@REGISTRY.tool(
    name="creative.get_asset_tags",
    description=(
        "读取素材的视觉标签与历史表现（标签由离线视觉分析产出）："
        "主题标签、开头钩子类型、主色、是否有人脸、文字占比，以及历史 IPM/CTR/d7 CPI 和投放地域。"
    ),
    parameters={
        "type": "object",
        "properties": {"creative_id": _STR, "creative_name": _STR},
        "required": [],
    },
    kind="read",
)
def get_asset_tags(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    catalog = ctx.table("creative_catalog")
    creative_id, name = args.get("creative_id"), args.get("creative_name")
    row = catalog.get(creative_id) if creative_id else None
    if row is None and name:
        row = next((r for r in catalog.values() if r.get("creative_name") == name), None)
    if row is None:
        return ToolResult(ok=False, error=f"creative_not_found: {creative_id or name}")
    return ToolResult(ok=True, data=dict(row))


@REGISTRY.tool(
    name="creative.search_similar",
    description=(
        "按视觉标签检索素材库，可按地域、平台过滤并设 IPM 下限，"
        "结果按 IPM 从高到低。用于找当前表现好的同主题替代素材。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "visual_tags": {"type": "array", "items": _STR, "description": "主题/元素标签"},
            "region": _STR,
            "platform": _STR,
            "min_ipm": {"type": "number", "description": "IPM 下限"},
            "top_k": {"type": "integer"},
        },
        "required": ["visual_tags"],
    },
    kind="read",
)
def search_similar(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    wanted = {t.lower() for t in (args.get("visual_tags") or [])}
    if not wanted:
        return ToolResult(ok=False, error="visual_tags_required")
    min_ipm = float(args.get("min_ipm", 0.0))
    hits = []
    for row in ctx.table("creative_catalog").values():
        tags = {t.lower() for t in row.get("visual_tags", [])}
        if not (wanted & tags):
            continue
        if args.get("region") and row.get("region") != args["region"]:
            continue
        if args.get("platform") and row.get("platform") != args["platform"]:
            continue
        if float(row.get("ipm", 0)) < min_ipm:
            continue
        hits.append({k: row[k] for k in
                     ("creative_id", "creative_name", "visual_tags", "hook_type",
                      "ipm", "ctr", "d7_cpi", "region", "platform", "week_launched")
                     if k in row})
    hits.sort(key=lambda r: r.get("ipm", 0), reverse=True)
    return ToolResult(ok=True, data={"count": len(hits),
                                     "creatives": hits[: int(args.get("top_k", 5))]})
