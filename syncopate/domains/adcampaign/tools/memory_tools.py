"""记忆工具：2 读 + 3 写（写全部是提案，不直接改基线）。

写权限的分层落在两处：
  - **硬边界**（往系统专属 lane 写）→ 工具直接报错，等价于真实 API 的 403
  - **软纪律**（证据不足、没先过风控）→ 工具照做，由 rules.py 的 cap 封顶

后者不硬拦的理由见 governance.py：工具直接拒绝的话，模型学到的是"报错就换一个"，
而不是"为什么要先查"。让它做成再拿低分，信号才是对的。
"""

from __future__ import annotations

from typing import Any

from syncopate.core.tool_registry import REGISTRY, ToolContext, ToolResult
from syncopate.domains.adcampaign.memory import (
    LANES,
    WRITABLE_LANES,
    build_record,
    check_proposal,
    parse_time,
    search,
)

_STR = {"type": "string"}
_LANE_ENUM = {**_STR, "enum": sorted(LANES), "description": "记忆分区"}


def _now(ctx: ToolContext):
    """时间来自 case 自己声明的 reference_now，不是系统时钟——否则不可复现。"""
    return parse_time(ctx.env.reference_now)


def _subject_from(args: dict[str, Any]) -> dict[str, Any]:
    keys = ("account_id", "campaign_id", "creative_id", "creative_name", "region", "platform")
    return {k: args[k] for k in keys if args.get(k)}


@REGISTRY.tool(
    name="memory.search",
    description=(
        "检索历史记忆。按分区(lane)和主体过滤，自动剔除已过 TTL 的记录。"
        "分区：episodic=历史投放动作 / semantic=素材与受众属性 / "
        "business=优化干预效果 / risk=风控标记。"
        "涉及重复投放、频繁调预算、历史干预是否有效时必须先查。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "lane": _LANE_ENUM,
            "account_id": _STR, "campaign_id": _STR,
            "creative_id": _STR, "creative_name": _STR,
            "region": _STR, "platform": _STR,
            "top_k": {"type": "integer", "description": "返回条数上限，默认 5"},
        },
        "required": ["lane"],
    },
    kind="read",
)
def memory_search(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    lane = args.get("lane")
    if lane not in LANES:
        return ToolResult(ok=False, error=f"unknown_lane: {lane}")
    hits = search(
        ctx.env.table("memory"), _now(ctx),
        lane=lane, subject=_subject_from(args), top_k=int(args.get("top_k", 5)),
    )
    return ToolResult(ok=True, data={
        "lane": lane,
        "count": len(hits),
        # 只回摘要，详情要 memory.read —— 和真实检索系统一致，也控制 prompt 长度
        "records": [{"record_id": r.record_id, "subject": r.subject,
                     "created_at": r.created_at, "summary": r.content} for r in hits],
    })


@REGISTRY.tool(
    name="memory.read",
    description="按 record_id 读取一条记忆的完整内容，含置信度、证据引用和过期时间。",
    parameters={"type": "object", "properties": {"record_id": _STR}, "required": ["record_id"]},
    kind="read",
)
def memory_read(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    raw = ctx.env.table("memory").get(args.get("record_id"))
    if raw is None:
        return ToolResult(ok=False, error=f"memory_record_not_found: {args.get('record_id')}")
    record = build_record(raw)
    payload = record.to_dict()
    # 明确告诉模型这条是不是过期了——用过期信息做决策会被 stale_memory_cap 封顶
    payload["expired"] = record.is_expired(_now(ctx))
    return ToolResult(ok=True, data=payload)


@REGISTRY.tool(
    name="memory.write_proposal",
    description=(
        "提交一条记忆写入提案（不会立即入库，需经审核）。"
        "要求 confidence ≥ 0.7 且 evidence_refs 至少 2 条；"
        "写 risk 分区前必须先调 risk.check_account。episodic 分区由系统维护，不可写入。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "lane": {**_STR, "enum": sorted(WRITABLE_LANES)},
            "content": {"type": "object", "description": "要记住的事实，需已脱敏"},
            "confidence": {"type": "number", "description": "0-1"},
            "evidence_refs": {"type": "array", "items": _STR,
                              "description": "支撑该结论的记录 id 或工具观测"},
            "account_id": _STR, "campaign_id": _STR, "creative_id": _STR,
        },
        "required": ["lane", "content", "confidence", "evidence_refs"],
    },
    kind="write",
    fact_key="memory_proposed",
)
def memory_write_proposal(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    lane = args.get("lane")
    content = args.get("content") or {}
    if not isinstance(content, dict):
        return ToolResult(ok=False, error="content_must_be_object")

    # 本轮有没有真的过风控——risk 分区的强审核要求
    risk_reviewed = any(
        r.tool == "risk.check_account" for r in ctx.sandbox.audit_log
    ) or _called_risk_check(ctx)

    issues = check_proposal(
        lane=str(lane), content=content,
        confidence=float(args.get("confidence", 0.0)),
        evidence_refs=list(args.get("evidence_refs") or []),
        risk_reviewed=risk_reviewed,
    )
    if issues.hard:
        # 硬边界：真实 API 会 403
        return ToolResult(ok=False, error="; ".join(issues.hard))

    return ToolResult(ok=True, data={
        "proposal_id": f"MP_{ctx.step}_{ctx.tool_call_id}",
        "lane": lane, "status": "pending_review",
        "subject": _subject_from(args),
        "content": content,
        "confidence": args.get("confidence"),
        "evidence_refs": args.get("evidence_refs"),
        # 软问题原样回给模型，它有机会在终答里说明——同时 cap 会封顶
        "review_findings": issues.soft,
    })


def _called_risk_check(ctx: ToolContext) -> bool:
    """风控是读工具，不进 sandbox 台账，所以要另外看轨迹。

    ToolContext 拿不到 trajectory，这里用一个折中：由 rules.py 的 cap 做权威判定，
    工具侧只做尽力而为的提示。
    """
    return bool(getattr(ctx, "_risk_checked", False))


@REGISTRY.tool(
    name="memory.invalidate",
    description="提议把一条已失效的记忆标记为作废（例如素材已下线、政策已变更）。同样需要审核。",
    parameters={
        "type": "object",
        "properties": {"record_id": _STR, "reason": _STR},
        "required": ["record_id", "reason"],
    },
    kind="write",
    fact_key="memory_invalidated",
)
def memory_invalidate(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    raw = ctx.env.table("memory").get(args.get("record_id"))
    if raw is None:
        return ToolResult(ok=False, error=f"memory_record_not_found: {args.get('record_id')}")
    if not (args.get("reason") or "").strip():
        return ToolResult(ok=False, error="reason_required")
    return ToolResult(ok=True, data={
        "record_id": args["record_id"], "status": "pending_invalidation",
        "reason": args["reason"], "lane": raw.get("lane"),
    })


@REGISTRY.tool(
    name="memory.conflict_resolve",
    description=(
        "两条记忆互相矛盾时，提议如何处置：supersede=用新的取代旧的，merge=合并。"
        "record_ids 必须至少给两条。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "record_ids": {"type": "array", "items": _STR},
            "decision": {**_STR, "enum": ["supersede", "merge"]},
            "keep_record_id": {**_STR, "description": "supersede 时保留哪一条"},
        },
        "required": ["record_ids", "decision"],
    },
    kind="write",
    fact_key="memory_conflict_resolved",
)
def memory_conflict_resolve(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    ids = list(args.get("record_ids") or [])
    if len(ids) < 2:
        return ToolResult(ok=False, error="need_at_least_two_records")
    table = ctx.env.table("memory")
    missing = [i for i in ids if i not in table]
    if missing:
        return ToolResult(ok=False, error=f"memory_record_not_found: {missing}")
    if args.get("decision") == "supersede" and args.get("keep_record_id") not in ids:
        return ToolResult(ok=False, error="keep_record_id_must_be_one_of_record_ids")
    return ToolResult(ok=True, data={
        "record_ids": ids, "decision": args["decision"],
        "keep_record_id": args.get("keep_record_id"), "status": "pending_review",
    })
