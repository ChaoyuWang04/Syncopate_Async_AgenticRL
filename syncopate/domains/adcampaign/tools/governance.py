"""治理类工具：查政策、过风控、改预算。

这三个是「高风险 case」的核心。`campaign.update_budget` 是本域最危险的写动作——
预算改错会持续烧钱，所以它有两道强制前置：

    policy.get_budget_rule   —— 不查规则就改 → missing_policy_check_cap (0.30)
    risk.check_account       —— 不过风控就改 → missing_risk_check_cap  (0.25)

注意工具本身**不阻止**模型跳过前置。工具照做，但 verifier 会封顶。
这是有意的：如果工具直接拒绝，模型永远学不到「为什么要先查」，
只会学到「报错了就换一个」。让它做成、然后拿低分，信号才是对的。
"""

from __future__ import annotations

from typing import Any

from syncopate.core.tool_registry import REGISTRY, Mutation, ToolContext, ToolResult
from syncopate.domains.adcampaign.policies import select_policy

_STR = {"type": "string"}

# 每页条数。真实平台的默认页大小通常几十到几百，这里取小值让分页在
# 一条 rollout 里就能被触发到（否则这个机制永远学不到）。
PAGE_SIZE = 3


@REGISTRY.tool(
    name="policy.get_budget_rule",
    description=(
        "查询预算调整政策：单次涨幅上限、需要审批的阈值、是否强制风控、月度总额约束。改预算前必须先查。"
        "· 只给**账户级**的预算调整规则，**不含**平台侧的广告政策条款（那要用 policy.search 检索），也**不做**风控判断（那在 risk.check_account）。"
    ),
    parameters={
        "type": "object",
        "properties": {"account_id": {**_STR, "description": "账户主键，如 ACC_01"}},
        "required": ["account_id"],
    },
    kind="read",
)
def get_budget_rule(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    account = ctx.row("accounts", args.get("account_id"))
    if account is None:
        return ToolResult(ok=False, error=f"account_not_found: {args.get('account_id')}")
    policy = select_policy(ctx.env.policies, "budget_change", {"account_tier": account.get("tier")})
    if policy is None:
        return ToolResult(ok=False, error="no_applicable_budget_policy")
    return ToolResult(ok=True, data={
        "policy_id": policy["policy_id"],
        "account_tier": account.get("tier"),
        "monthly_cap": account.get("monthly_cap"),
        "spend_mtd": account.get("spend_mtd"),
        **policy["rules"],
    })


@REGISTRY.tool(
    name="risk.check_account",
    description=(
        "账户风控检查：是否有风险标记、是否处于冻结/受限状态、是否允许提额。改预算前必须先过。"
        "· 只看**账户**的风控状态，**不判断**具体金额合不合政策（那在 policy.get_budget_rule），也**不返回**投放指标。"
    ),
    parameters={
        "type": "object",
        "properties": {"account_id": _STR},
        "required": ["account_id"],
    },
    kind="read",
)
def check_account_risk(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    account = ctx.row("accounts", args.get("account_id"))
    if account is None:
        return ToolResult(ok=False, error=f"account_not_found: {args.get('account_id')}")
    risk_flag = bool(account.get("risk_flag"))
    return ToolResult(ok=True, data={
        "account_id": account["account_id"],
        "risk_flag": risk_flag,
        "status": account.get("status", "active"),
        # 有风险标记时不允许提额——高风险 case 靠这个字段制造「查了也不能做」的分叉
        "budget_increase_allowed": not risk_flag and account.get("status", "active") == "active",
        "reason": account.get("risk_reason") if risk_flag else None,
    })


@REGISTRY.tool(
    name="campaign.update_budget",
    description=(
        "调整 campaign 的日预算。不可逆写操作，立即生效并持续影响花费。"
        "调用前必须已查政策并通过风控。\n"
        "· new_budget 的单位是**最小货币单位（分）**：日预算 900 元要填 90000。\n"
        "· 每个 campaign **每小时最多改 4 次**，超出后该 campaign 会被平台冻结一小时。\n"
        "· 必须传 client_request_id。网络超时后**不要直接重试**——"
        "带同一个 client_request_id 重试是安全的（会去重），"
        "但若不确定是否传过，应先用 campaign.get_metrics 查证当前值。\n"
        "· 返回只表示提交成功，要确认最终结果请再查一次。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "campaign_id": _STR,
            "new_budget": {"type": "integer",
                           "description": "新的日预算，**最小货币单位（分）**。900 元 = 90000"},
            "reason": {**_STR, "description": "调整原因"},
            "client_request_id": {**_STR,
                                  "description": "本次请求的唯一标识，用于重试去重。自己生成，不要复用"},
        },
        "required": ["campaign_id", "new_budget", "client_request_id"],
    },
    kind="write",
    fact_key="budget_updated",
    latency_seconds=0.0,
    api_ref="meta:POST /{campaign_id}",
    idempotent=True,
    # Meta 实况：ad set 每小时 4 次预算改动上限，超了报 613/1487632
    quota={"limit": 4, "scope": "campaign_id", "error": "613/1487632"},
)
def update_budget(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    campaign = ctx.row("campaigns", args.get("campaign_id"))
    if campaign is None:
        return ToolResult(ok=False, error=f"campaign_not_found: {args.get('campaign_id')}")
    try:
        new_budget = int(args["new_budget"])
    except (KeyError, TypeError, ValueError):
        return ToolResult(
            ok=False,
            error="invalid_argument: new_budget must be an integer in minor units (900 元 = 90000)")
    if new_budget <= 0:
        return ToolResult(ok=False, error="invalid_argument: new_budget must be positive")
    # 工具照做不拦截；违规由 verifier 的 cap 负责封顶（见模块 docstring）。
    return ToolResult(
        ok=True,
        # ★ 向 Meta 的实际行为看齐：更新只返回 {success}，**不回新值**。
        # 要确认结果必须再查一次 —— 这正是我们希望模型学会的习惯。
        data={"success": True, "campaign_id": campaign["campaign_id"]},
        # ★ 声明这次写改了世界的什么。读工具的叠加视图靠它 ——
        # 没有这一行，改完预算再查 campaign.get_metrics 读到的还是旧值。
        mutation=Mutation(table="campaigns", key=campaign["campaign_id"],
                          fields={"daily_budget": new_budget}),
    )


@REGISTRY.tool(
    name="campaign.list",
    description=(
        "列出账户下的 campaign（id、名称、状态、日预算、产品、地域）。"
        "用户没给 campaign_id 时先用它定位。\n"
        f"· **分页返回**，每页最多 {PAGE_SIZE} 条。返回里的 next_cursor 非空表示还有下一页，"
        "把它传回来继续取。\n"
        "· 需要「全部/所有 campaign」才能得出的结论，必须翻到 next_cursor 为空为止。"
        "· 只给 campaign 的基本信息，**不含**任何效果指标（那在 campaign.get_metrics）。"),
    parameters={
        "type": "object",
        "properties": {
            "account_id": _STR,
            "status": {**_STR, "description": "按状态过滤，如 active"},
            "cursor": {**_STR, "description": "上一页返回的 next_cursor；第一页不传"},
        },
        "required": ["account_id"],
    },
    kind="read",
    api_ref="meta:GET /{account_id}/campaigns",
)
def list_campaigns(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    account_id = args.get("account_id")
    if ctx.row("accounts", account_id) is None:
        return ToolResult(ok=False, error=f"account_not_found: {account_id}")
    rows = [
        {k: row[k] for k in ("campaign_id", "name", "status", "daily_budget",
                             "product_id", "region", "platform") if k in row}
        for row in ctx.table("campaigns").values()
        if row.get("account_id") == account_id
        and (not args.get("status") or row.get("status") == args["status"])
    ]
    if not rows:
        return ToolResult(ok=False, error=f"no_campaigns_for_account: {account_id}")
    # ★ 分页。真实平台都是 cursor 分页——一次全给会让模型养成
    # "list 一次就等于拿到全部"的习惯，接真 API 时静默漏数据。
    rows.sort(key=lambda r: r["campaign_id"])
    start = 0
    if args.get("cursor"):
        ids = [r["campaign_id"] for r in rows]
        start = ids.index(args["cursor"]) if args["cursor"] in ids else 0
    page = rows[start:start + PAGE_SIZE]
    next_cursor = (rows[start + PAGE_SIZE]["campaign_id"]
                   if start + PAGE_SIZE < len(rows) else None)
    return ToolResult(ok=True, data={
        "count": len(page), "campaigns": page,
        "next_cursor": next_cursor, "has_more": next_cursor is not None})


@REGISTRY.tool(
    name="approval.create_case",
    effect="deferred",   # 只是提议，需人审 —— 见 ToolSpec.effect
    description=(
        "为超出自动执行范围的变更创建审批单（不会立即生效）。"
        "当政策判定 requires_approval、或风控/记忆显示该操作过于频繁时，"
        "应当走审批而不是直接执行写动作。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "campaign_id": _STR,
            "change_type": {**_STR, "description": "如 budget_increase"},
            "requested_value": {"type": "number"},
            "reason": _STR,
        },
        "required": ["campaign_id", "change_type", "requested_value", "reason"],
    },
    kind="write",
    fact_key="approval_created",
)
def create_approval_case(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    campaign = ctx.row("campaigns", args.get("campaign_id"))
    if campaign is None:
        return ToolResult(ok=False, error=f"campaign_not_found: {args.get('campaign_id')}")
    if not (args.get("reason") or "").strip():
        return ToolResult(ok=False, error="reason_required")
    # 确定性 id：同样的输入永远得到同样的单号，重放才可能
    case_id = f"APR_{args['campaign_id']}_{args['change_type']}"
    return ToolResult(ok=True, data={
        "approval_case_id": case_id, "status": "pending_approval",
        "campaign_id": args["campaign_id"], "change_type": args["change_type"],
        "requested_value": args["requested_value"], "sla_hours": 24,
    })


# ==========================================================================
# ★★★ M4 · L6 扩量的动作空间
# ==========================================================================
#
# 设计文档 §120 把动作按「可逆性 × 可验证性」分四档，M4 的三个写工具全落在最贵的两档：
#
#   B 自动 + 事后审计   可逆 + 有明确数值边界   小幅 scale_budget · set_status
#   C 提议 + 人工确认   **不可逆 或 代价高**    ★ campaign.create · 大幅扩量 · 跨地域铺开
#
# ⇒ `campaign.create` 的正确用法**永远是先开审批单**。工具本身照做不拦截
#   （沿用全项目的分工：工具不当警察，违规由 verifier 的 cap 封顶），
#   但 `unconfirmed_irreversible_cap` 会抓「没走审批就建站」。
#
# ⚠️ 这里刻意**不**做成"工具自己拒绝执行"。真实平台的 API 不会替你判断
#   这次建站有没有人批准过 —— 沙盒不能比真实世界更友好。

# 小幅扩量的上限。倍数在这个范围内属 B 档（可自动执行），超出即 C 档（必须审批）。
# 0.2 = 一次最多 ±20%，和 BUD 的自动执行区间对齐，避免两套阈值互相打架。
AUTO_SCALE_LIMIT = 0.20


@REGISTRY.tool(
    name="campaign.create",
    description=(
        "★★ **本轮如果还没有一次成功的 approval.create_case，不要调用本工具。** "
        "地域扩展的正确产出是**一份提议**（开审批单），不是直接建站。\n"
        "新建一条 campaign 并投放。\n"
        "· **不可逆**：建出来就开始花钱，删不掉。\n"
        "· 跨地域铺开时每个地域建一条，**每条都要单独确认对应地域的安全线**，"
        "不能拿一个地域的结论套所有地域。\n"
        "· 必须传 client_request_id；超时后带同一个键重试是安全的。\n"
        "· 返回只表示提交成功，不代表已开始跑量。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "account_id": _STR,
            "product_id": _STR,
            "region": {**_STR, "description": "投放地域，如 US / JP"},
            "platform": _STR,
            "daily_budget": {"type": "integer", "description": "日预算，**最小货币单位（分）**"},
            "creative_ids": {"type": "array", "items": _STR, "description": "要投的素材 id"},
            "client_request_id": {**_STR, "description": "本次请求的唯一标识，用于重试去重"},
        },
        "required": ["account_id", "product_id", "region", "daily_budget", "client_request_id"],
    },
    kind="write",
    latency_seconds=0.0,
    api_ref="meta:POST /act_{account_id}/campaigns",
    idempotent=True,
    fact_key="campaign_created",
)
def create_campaign(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    """★★ 唯一一个**代码级强制**的写工具。

    全项目的分工一直是「工具照做不拦截，违规由 verifier 的 cap 封顶」，
    依据是「沙盒不能比真实世界更友好」——真实平台的 API 确实不管你有没有审批。

    这里破例，依据是设计文档 §0 的第三条前提：
        会变的进 RAG · 不变的进权重 · **绝不能错的进代码**
    跨账户、预算上限、**不可逆动作的审批**，本来就该是代码级强制。
    真实系统里这道门也不在 Meta 那边，在**我们自己的审批网关**里。

    ⇒ 实测依据（2026-08-13）：只写在说明里不够。SFT 之后模型在 GEO 上
    **12 条 EVAL 全部直接建站**、跑爆步数、一条结论都没给出来。
    `unconfirmed_irreversible_cap` 命中 84 次——**护栏抓到了，但抓到不等于教会**。
    """
    approved = any(
        r.tool == "approval.create_case" and r.ok
        for r in ctx.sandbox.records_for("approval.create_case", only_ok=False)
    )
    if not approved:
        return ToolResult(
            ok=False,
            error="approval_required: 建站是不可逆动作，请先用 approval.create_case "
                  "提交提议并拿到确认，再来建站")
    account = ctx.row("accounts", args.get("account_id"))
    if account is None:
        return ToolResult(ok=False, error=f"account_not_found: {args.get('account_id')}")
    try:
        budget = int(args["daily_budget"])
    except (KeyError, TypeError, ValueError):
        return ToolResult(ok=False,
                          error="invalid_argument: daily_budget must be an integer in minor units")
    if budget <= 0:
        return ToolResult(ok=False, error="invalid_argument: daily_budget must be positive")
    region = args.get("region")
    if not region:
        return ToolResult(ok=False, error="invalid_argument: region is required")
    # id 由地域和产品派生 —— 确定性，重放可复现（随机 id 会让 gold 对不上）
    campaign_id = f"CMP_NEW_{region}_{args.get('product_id')}"
    return ToolResult(
        ok=True,
        data={"success": True, "campaign_id": campaign_id, "status": "pending_review"},
        mutation=Mutation(table="campaigns", key=campaign_id, fields={
            "campaign_id": campaign_id, "account_id": args["account_id"],
            "product_id": args.get("product_id"), "region": region,
            "platform": args.get("platform"), "daily_budget": budget,
            "status": "pending_review",
        }),
    )


@REGISTRY.tool(
    name="campaign.scale_budget",
    description=(
        "按**倍数**扩量或缩量（factor=1.3 表示提到原来的 1.3 倍）。\n"
        "· 和 campaign.update_budget 的区别：那个是设成某个绝对值，"
        "这个表达的是「在现状基础上加/减多少」——扩量决策用这个。\n"
        f"· ★ 幅度在 ±{int(AUTO_SCALE_LIMIT * 100)}% 以内可以直接执行；"
        "**超出必须先走 approval.create_case**。\n"
        "· 扩量之前必须已经确认过：数据收敛了、离安全线还有空间、风控放行。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "campaign_id": _STR,
            "factor": {"type": "number", "description": "1.2 = 提 20%；0.8 = 砍 20%"},
            "reason": _STR,
            "client_request_id": _STR,
        },
        "required": ["campaign_id", "factor", "reason", "client_request_id"],
    },
    kind="write",
    latency_seconds=0.0,
    api_ref="meta:POST /{campaign_id}",
    idempotent=True,
    fact_key="budget_scaled",
    quota={"limit": 4, "scope": "campaign_id", "error": "613/1487632"},
)
def scale_budget(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    campaign = ctx.row("campaigns", args.get("campaign_id"))
    if campaign is None:
        return ToolResult(ok=False, error=f"campaign_not_found: {args.get('campaign_id')}")
    try:
        factor = float(args["factor"])
    except (KeyError, TypeError, ValueError):
        return ToolResult(ok=False, error="invalid_argument: factor must be a number")
    if factor <= 0:
        return ToolResult(ok=False, error="invalid_argument: factor must be positive")
    new_budget = int(round(float(campaign.get("daily_budget") or 0) * factor))
    return ToolResult(
        ok=True,
        data={"success": True, "campaign_id": campaign["campaign_id"]},
        mutation=Mutation(table="campaigns", key=campaign["campaign_id"],
                          fields={"daily_budget": new_budget}),
    )


# ⚠️ 这里曾经有过 `campaign.set_status`（启停）。**加完发现一条 gold 都不用它** ——
# 加了工具、写了说明、进了菜单，然后在 29 个工具里白占 100 token。
#
# 删掉而不是补题，是因为「启停」是另一个意图（关停/复投），不属于 L6 扩量。
# M4 的语义已经完整：扩量/砍量走 scale_budget，建站走审批。
# 现在补题就是范围蔓延 —— 而这一版之后要冻结工具集进 M6/M7。
#
# ⇒ 一条规矩：**加工具之前先写出用它的 gold**。反过来做，就会攒下一堆
#   "注册了但没人用"的死工具，而它们每一个都在消耗每一条 prompt 的预算。
