"""K6 · 工具治理表（课件 CH6 的"注册时断言"，落在 runtime 侧）。

★ 为什么不改 `syncopate/core/tool_registry.py`：它是训练/serving 共用件（spec 唯一真相来源），
  改它要按铁律走 MAINLINE-INFRA，而且训练侧根本不需要 timeout/权限/告警这些**运行时**属性。
  ⇒ 治理属性放这里，但**完整性对着 REGISTRY 断言**：
        set(GOVERNANCE) == set(REGISTRY 全部工具)          （无遗漏、无幽灵条目）
        GOVERNANCE[t].side_effect == (REGISTRY[t].kind == "write")   （"哪些是写工具"只有一份真相）
  这条断言在 `tools.py` 导入时跑一次、在测试里再跑一次——登记表本身不能再是一张"最像证据的名单"
  （08-19 WRITE_TOOLS 登记 8 实现 2 的老病）。

字段（课件 §3.1 十三条职责的"注册时就该定死"那几条）：
  permission            写工具必填（权限闸）；读工具 None
  timeout_seconds       必填，禁全局常量（课件反例：读 ~480ms vs 写 ~1260ms 差三倍）
  expected_max_ms       sweeper 判"超龄 running ⇒ response_lost"用（K8-1）
  retryable_errors      分诊映射：平台错误码 ∈ 此集合才重试。写工具默认空集——
                        但 `429`（限流）带同一个幂等键重试是安全的（B-1a 实查：幂等命中先于扣分），
                        所以写工具显式登记 {"429"}（**这是登记的决策，不是默认**）
  output_required_keys  output_schema 轻量版：回灌前必须有的顶层键；缺 ⇒ 不进 context（防反向污染）
  output_todo           写工具若还没登记输出键，必须写"为什么"（缺口要写下来才算被承认）
  audit_required        写工具必 True
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# 防线生效不是故障（课件 H56）：这些拒绝记 failed，但**不告警**
NON_ALERTING_CODES = frozenset({"permission_denied", "validation_failed", "tier_d_refused",
                                "release_gate", "daily_cost_cap", "cancel_requested", "max_steps",
                                "unknown_tool", "skipped_duplicate", "tool_disabled"})
READ_RETRYABLE = frozenset({"429", "rate_limited", "server_error", "timeout", "client_timeout"})
# 写工具：课件说 retryable_errors=空集；我们**改造**（准则五）——平台的限流/超时/5xx 在**带同一个幂等键**
# 重试时是安全的（B-1a 实查：幂等命中先于扣分与频次检查），且沙盒（训练侧）的重试契约就是这样，
# runtime 不能比沙盒少一种行为（09 §1-③）。⛔ 唯独本地 `client_timeout`（回包没等到）不重试：
# 那是"结果未知"，记 response_lost 交对账。
WRITE_RETRYABLE = frozenset({"429", "rate_limited", "server_error", "timeout"})


@dataclass(frozen=True)
class ToolGovernance:
    side_effect: bool
    timeout_seconds: float
    expected_max_ms: int
    permission: str | None = None
    retryable_errors: frozenset[str] = field(default_factory=frozenset)
    output_required_keys: tuple[str, ...] = ()
    output_todo: str | None = None
    audit_required: bool = False

    def __post_init__(self) -> None:
        # 注册断言（课件 C④：断言写在注册函数里，不是 checklist 里）
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds 必填且 > 0（禁全局常量）")
        if self.expected_max_ms <= 0:
            raise ValueError("expected_max_ms 必填且 > 0")
        if self.side_effect:
            if not self.permission:
                raise ValueError("有副作用的工具必须声明 permission")
            if not self.audit_required:
                raise ValueError("有副作用的工具必须 audit_required=True")
            if not self.output_required_keys and not self.output_todo:
                raise ValueError("有副作用的工具必须登记 output_required_keys（或写明 output_todo 原因）")
            if not self.retryable_errors <= WRITE_RETRYABLE:
                raise ValueError(f"写工具的 retryable_errors 只能 ⊆ {sorted(WRITE_RETRYABLE)}")


def _read(timeout: float = 10.0, expected_ms: int = 2_000, keys: tuple[str, ...] = ()) -> ToolGovernance:
    return ToolGovernance(side_effect=False, timeout_seconds=timeout, expected_max_ms=expected_ms,
                          retryable_errors=READ_RETRYABLE, output_required_keys=keys)


def _write(permission: str, *, timeout: float = 30.0, expected_ms: int = 5_000,
           keys: tuple[str, ...] = (), todo: str | None = None) -> ToolGovernance:
    return ToolGovernance(side_effect=True, timeout_seconds=timeout, expected_max_ms=expected_ms,
                          permission=permission, retryable_errors=WRITE_RETRYABLE,
                          output_required_keys=keys, output_todo=todo, audit_required=True)


GOVERNANCE: dict[str, ToolGovernance] = {
    # ---- 读（30 业务工具里的 22 个）----
    "analysis.feature_lift":           _read(keys=("feature",)),
    "analysis.geo_breakdown":          _read(keys=("product_id",)),
    "benchmark.get_industry_baseline": _read(),
    "benchmark.get_safety_line":       _read(),
    "calendar.get_seasonal_context":   _read(),
    "campaign.detect_anomalies":       _read(),
    "campaign.get_metrics":            _read(keys=("campaign_id",)),
    "campaign.list":                   _read(),
    "creative.get_asset_tags":         _read(),
    "creative.get_metrics_by_asset":   _read(),
    "creative.poll_review":            _read(keys=("status",)),
    "creative.search_similar":         _read(),
    "insight.search_claims":           _read(),
    "memory.read":                     _read(),
    "memory.search":                   _read(),
    "metrics.get_freshness":           _read(keys=("campaign_id", "maturity")),
    "mmp.get_attribution":             _read(keys=("campaign_id",)),
    "playbook.get_optimization":       _read(),
    "policy.get_budget_rule":          _read(),
    "policy.search":                   _read(),
    "risk.check_account":              _read(),
    # system.wait 最长等到 lease 安全线（09 §4.5.9）：timeout 取 lease/2 以上
    "system.wait":                     _read(timeout=600.0, expected_ms=600_000),
    # ---- 写（8 个）----
    "campaign.update_budget":  _write("budget:write", keys=("campaign_id", "new_budget")),
    "campaign.create":         _write("campaign:write", keys=("campaign_id", "status", "daily_budget")),
    "campaign.scale_budget":   _write("budget:write", keys=("campaign_id", "new_budget", "previous_budget")),
    "creative.upload":         _write("creative:write", keys=("asset_id", "status")),
    "approval.create_case":    _write("approval:write", keys=("case_ref", "status")),
    "memory.write_proposal":   _write("memory:write", keys=("proposal_id",)),
    "memory.invalidate":       _write("memory:write", keys=("proposal_id",)),
    "memory.conflict_resolve": _write("memory:write", keys=("proposal_id",)),
    # ---- v15 信令族（contract.SESSION_TOOL_NAMES；只在 SYNCOPATE_CONTRACT=v15 时进 REGISTRY）----
    # 不是外部调用：clarify/defer/reject 由 loop 当终止信令处理，report 走收口 binding（零副作用 ack）。
    # ⚠️ 09-02 verl-22 通报：漏登记 ⇒ v15 生产进程导入即炸（断言做对了，登记漏了）。
    "session.clarify": _read(timeout=5.0, expected_ms=1_000),
    "session.defer":   _read(timeout=5.0, expected_ms=1_000),
    "session.reject":  _read(timeout=5.0, expected_ms=1_000),
    "session.report":  _read(timeout=5.0, expected_ms=1_000),
}

# 契约相关的可选条目：v14 下 REGISTRY 没有它们，不算"幽灵"；v15 下必须有且必须登记
def _optional_by_contract() -> frozenset[str]:
    from syncopate.core.contract import SESSION_TOOL_NAMES
    return frozenset(SESSION_TOOL_NAMES)


def sandbox_specs() -> dict[str, Any]:
    """真相来源：沙盒 REGISTRY（domain 导入后才有内容）。"""
    from syncopate.domains.adcampaign import build_domain
    reg = build_domain().registry
    return {name: reg.get(name) for name in reg.names()}


def assert_governance_complete(governance: dict[str, ToolGovernance] | None = None) -> None:
    """已登记 == REGISTRY 全集，且"是不是写工具"两边一致。导入时跑一次（tools.py），测试再跑一次。"""
    gov = governance if governance is not None else GOVERNANCE
    specs = sandbox_specs()
    missing = sorted(set(specs) - set(gov))
    ghost = sorted(set(gov) - set(specs) - _optional_by_contract())
    if missing or ghost:
        raise AssertionError(f"治理表与 REGISTRY 不一致：沙盒有而未登记 {missing}；登记了但沙盒没有 {ghost}")
    wrong = sorted(n for n, g in gov.items() if n in specs and g.side_effect != (specs[n].kind == "write"))
    if wrong:
        raise AssertionError(f"side_effect 与 REGISTRY 的 kind 不一致：{wrong}")


def governance(tool: str) -> ToolGovernance:
    try:
        return GOVERNANCE[tool]
    except KeyError:
        raise KeyError(f"工具 {tool!r} 没有治理登记（也就没有 timeout/权限/分诊）") from None


def write_permissions() -> dict[str, str]:
    """`WRITE_TOOLS` 的唯一来源：从治理表派生，不再手抄一份。"""
    return {n: g.permission for n, g in GOVERNANCE.items() if g.side_effect and g.permission}


def classify_platform_error(tool: str, *, code: str, message: str) -> dict[str, Any]:
    g = governance(tool)
    return {"code": code, "message": message[:300], "retryable": code in g.retryable_errors,
            "alert": code not in NON_ALERTING_CODES}


def error_json(code: str, message: str, *, retryable: bool = False) -> dict[str, Any]:
    return {"code": code, "message": message[:300], "retryable": retryable,
            "alert": code not in NON_ALERTING_CODES}


def check_output(tool: str, data: Any) -> str | None:
    """output_schema 轻量版：不是 dict / 缺顶层键 ⇒ 返回原因（调用方据此**不回灌**）。"""
    g = governance(tool)
    if not isinstance(data, dict):
        return f"output_schema_violation: {tool} 返回的不是对象（{type(data).__name__}）"
    missing = [k for k in g.output_required_keys if k not in data]
    if missing:
        return f"output_schema_violation: {tool} 缺顶层键 {missing}"
    return None
