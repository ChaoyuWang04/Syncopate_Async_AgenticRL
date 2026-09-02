"""M8 · RAG v1 的两个检索工具：半结构化政策 + 非结构化复盘结论。

★ 这两个工具的说明书是照 `benchmark.get_safety_line` 写的 —— 那条已经把
「有效期过了怎么办」「查不到怎么办」放进 description 的**前部**并配了 cap，
是现成的正确范式。踩过的坑：**规则写在工具说明第二行会被淹没**，12 条 EVAL 全崩。

★ 两条必须由工具**如实返回**、不能替模型兜底的状态（设计文档 §14 的两项验收）：

    检索为空       → 返回 ok=True 但 hits=[]，**不返回"最像的那条"**
                     正确行为是 clarify / defer，不是硬答
    命中已过期条款 → 照常返回，但标 `expired=true` 并给出取代它的新版本
                     正确行为是引用新版本；引用旧版本 = 合规事故

⚠️ **工具不做决策，只如实描述世界。** 过期的条款不隐藏（隐藏了就没法考"会不会
误用旧版本"），查不到不编（编了就没法考"会不会硬答"）。这和沙盒的总原则一致：
**沙盒不要比真实世界友好。**
"""

from __future__ import annotations

from typing import Any

from syncopate.core.tool_registry import REGISTRY, ToolContext, ToolResult
from syncopate.domains.adcampaign.corpus import Hit, search_rows
from syncopate.domains.adcampaign.memory import parse_time

_STR = {"type": "string"}


def _clause_text(row: dict[str, Any]) -> str:
    return " ".join(filter(None, [row.get("title"), row.get("body"), row.get("section_path")]))


def _insight_text(row: dict[str, Any]) -> str:
    scope = row.get("scope") or {}
    return " ".join([str(row.get("claim") or ""), *(str(v) for v in scope.values())])


@REGISTRY.tool(
    name="policy.search",
    description=(
        "按关键词检索平台广告政策 / 广告法 / 内部 SOP 的条款（半结构化，可按平台和地域过滤）。每条结果带 valid_from / valid_to、expired 标记（true = 已被新版本取代）和 superseded_by；查不到返回空 hits（不是报错）。关键词匹配而非语义理解，换个说法可能查不到。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {**_STR, "description": "检索词，如 '单日预算涨幅上限' / 'creative review'"},
            "platform": {**_STR, "description": "只看某平台，如 meta / google / tiktok；不给则全部"},
            "region": {**_STR, "description": "只看某地域，如 US / SEA；不给则全部"},
            "top_k": {"type": "integer", "description": "最多返回几条，默认 3"},
        },
        "required": ["query"],
    },
    kind="read",
)
def policy_search(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    now = parse_time(ctx.env.reference_now)
    rows = ctx.table("policy_clauses")
    hits: list[Hit] = search_rows(
        rows, str(args.get("query") or ""), text_of=_clause_text,
        top_k=int(args.get("top_k") or 3),
        filters={"platform": args.get("platform"), "region": args.get("region")},
    )

    out = []
    for hit in hits:
        row = dict(hit.row)
        valid_to = row.get("valid_to")
        expired = bool(valid_to) and parse_time(valid_to) < now
        # ★ 取代关系是**从新条款反查**出来的：新版本声明 supersedes=<旧 id>。
        #   不在旧条款上写 superseded_by，是因为语料是增量追加的——
        #   写旧条款等于每加一版都要回改历史，那正是"版本"这个概念要避免的。
        successor = next(
            (r["clause_id"] for r in rows.values() if r.get("supersedes") == row["clause_id"]),
            None,
        )
        out.append({**row, "expired": expired, "superseded_by": successor, "score": hit.score})

    return ToolResult(ok=True, data={
        "query": args.get("query"), "hits": out, "hit_count": len(out),
        # 显式给一个"确实查不到"的信号，而不是让模型从空列表自己悟。
        "no_match": len(out) == 0,
    })


@REGISTRY.tool(
    name="insight.search_claims",
    description=(
        "检索历史复盘沉淀的结论（如「某类素材在某地域表现更好」）。返回的是经验不是实时数据，下决策仍需用实时指标核实。每条带 status（active 现行 / superseded 已被取代，superseded_by 指向新结论 / refuted 已被推翻）、confidence 与 evidence 样本量；非 active 的结论会一并返回，因为「老结论已被推翻」本身就是要报告的信息。查不到返回空 hits（不是报错）。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {**_STR, "description": "检索词，如 '真人出镜素材 东南亚 ROAS'"},
            "region": {**_STR, "description": "只看某地域的结论"},
            "product_id": {**_STR, "description": "只看某产品的结论"},
            "active_only": {
                "type": "boolean",
                "description": "只要 active 结论；默认 false（连已被取代 / 推翻的一起给并标 status）",
            },
            "top_k": {"type": "integer", "description": "最多返回几条，默认 3"},
        },
        "required": ["query"],
    },
    kind="read",
)
def insight_search_claims(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    rows = ctx.table("insights")
    # ★★ 2026-08-14 自审后翻转的默认值：原来默认**隐藏** superseded/refuted。
    #
    # 那样写和 M8 的目的直接冲突：模型永远看不到矛盾，除非它主动想到去翻旧账 ——
    # 而它没有任何理由想到。⇒「查到的历史结论和现在的数据矛盾了怎么办」这道题
    # **在构造上就出不来**，`memory.conflict_resolve` 的题面也就永远造不出来
    # （那正是遗留清单里挂着的缺口）。设计文档 §13 说得很直白：
    # 「你需要知道"我们曾经这么以为"」。
    #
    # ⇒ 默认全给、用 status 标清楚。
    #
    # ⚠️ 试过"active 排前面"，去掉了：`search_rows` 已按分数截断到 top_k，
    # 在**截断之后**重排既纠正不了截断偏差，又会把"高度相关的旧结论"压到
    # "勉强相关的现行结论"下面 —— 而查询如果问的正是老结论，那条才是答案。
    # 纯分数序更诚实也更可预测；矛盾对同主题、分数相近，本来就会一起进 top_k。
    active_only = bool(args.get("active_only"))
    scope_filters = {
        k: v for k, v in (("region", args.get("region")), ("product", args.get("product_id")))
        if v is not None
    }

    candidates = {}
    for key, row in rows.items():
        if active_only and row.get("status") != "active":
            continue
        scope = row.get("scope") or {}
        if any(scope.get(f) not in (None, want) for f, want in scope_filters.items()):
            continue
        candidates[key] = row

    hits = search_rows(candidates, str(args.get("query") or ""), text_of=_insight_text,
                       top_k=int(args.get("top_k") or 3))
    out = [{**hit.row, "score": hit.score} for hit in hits]
    return ToolResult(ok=True, data={
        "query": args.get("query"), "hits": out, "hit_count": len(out),
        "no_match": len(out) == 0,
    })
