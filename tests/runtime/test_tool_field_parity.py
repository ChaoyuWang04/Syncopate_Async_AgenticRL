"""B-5b · **表驱动的字段对照台**：同名工具两侧真跑，比模型看得见的字段名。

★ 为什么比**字段名**不比值：值本来就该不同（两侧的世界不同）。
  **字段名不同才是契约破了** —— 模型按字段名取数，少一个就是取不到，
  而它不会报错，只会让模型**取不到然后自己编一个**。

`[实测]` 这台对照台在**第一个**工具上就抓到 2 个真分歧
（observation 多包一层 · `metrics.get_freshness` 少 4 个字段）。

⚠️ 覆盖是**逐条登记**的（`EXERCISED` / `NOT_EXERCISED`），不是"扫到哪算哪" ——
   跟 `tool_parity` 的账本同一条纪律：**缺口必须写下来才算被承认。**
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _run(coro):
    return asyncio.run(coro)


# ── 共同的世界（两侧各自建，但语义对齐）────────────────────────────────

def _sandbox_ctx():
    import syncopate.domains.adcampaign  # noqa: F401
    from syncopate.core.sandbox import Sandbox
    from syncopate.core.schemas import Case, CaseMetadata
    from syncopate.core.tool_registry import ToolContext
    from syncopate.domains.adcampaign.world import WorldBuilder
    env = (WorldBuilder("T_0001", reference_now="2026-08-19T00:00:00Z")
           .account("ACC_01").campaign("CMP_1", account_id="ACC_01")
           .safety_line_state("current", product_id="P1", region="US")
           .benchmark("meta", "casual", "cpi", p50=2.0)
           .build())
    case = Case(case_id=env.case_id, user_message="-", context={}, entities={},
                metadata=CaseMetadata(signal_class="graded", bucket="rag"), max_steps=8)
    return ToolContext(case=case, env=env, sandbox=Sandbox(env, "ns"), step=1,
                       tool_call_id="t")


def _runtime_platform():
    from syncopate.runtime.platform import FakeAdPlatform
    p = FakeAdPlatform()
    p.campaigns["CMP_1"] = {"name": "C", "status": "ACTIVE", "daily_budget": 50_000}
    p.budgets["CMP_1"] = 50_000
    return p


# ── 覆盖登记：跑了哪些、没跑哪些**为什么** ──────────────────────────────

EXERCISED: dict[str, dict] = {
    "campaign.get_metrics":   {"campaign_id": "CMP_1"},
    "campaign.list":          {"account_id": "ACC_01"},
    "metrics.get_freshness":  {"campaign_id": "CMP_1"},
    "campaign.detect_anomalies": {"campaign_id": "CMP_1"},
    "mmp.get_attribution":    {"campaign_id": "CMP_1"},
}

# ⚠️ 没跑的**必须写清楚为什么** —— 否则这张表会变成"扫到哪算哪"，
#   而漏掉的工具正是最可能藏着分歧的那些。
NOT_EXERCISED: dict[str, str] = {
    "benchmark.get_safety_line": "runtime 侧要 PG（safety_lines 表）⇒ 已在 test_memory_and_safety 单独测",
    "memory.read": "runtime 侧要 PG（memory_records 表）⇒ 已在 test_memory_and_safety 测",
    "memory.search": "runtime 侧要 PG；且两侧的 TTL 剔除语义已单独测（read 不剔、search 剔）",
    "memory.write_proposal": "写工具：沙盒入台账、runtime 落 memory_proposals —— 形状本就不同",
    "memory.invalidate": "写工具：两侧都只提案不生效，语义已在 test_memory_and_safety 钉住",
    "memory.conflict_resolve": "写工具：需要冲突检出，runtime 侧只落提案",
    "policy.get_budget_rule": "runtime 侧要 PG（budget_rules 表）",
    "risk.check_account": "runtime 侧要 PG（account_risk 表）",
    "playbook.get_optimization": "runtime 侧要 PG（playbooks 表）",
    "analysis.feature_lift": "runtime 侧要 PG（feature_lifts 表）",
    "analysis.geo_breakdown": "runtime 侧要 PG（geo_performance 表）· 沙盒侧还要造 P1 的地域数据",
    "benchmark.get_industry_baseline": "runtime 侧要 PG（industry_baselines 表）",
    "calendar.get_seasonal_context": "runtime 侧要 PG（seasonal_events 表）",
    "creative.upload": "写工具：沙盒侧走台账，runtime 侧走异步任务 —— 形状本就不同，见 09 §4.5.9",
    "creative.poll_review": "runtime 侧走 get_job（不阻塞）· 沙盒侧阻塞等待 —— 形状**刻意不同**",
    "creative.get_asset_tags": "沙盒侧要先建 creative；runtime 侧素材库是内存态",
    "creative.get_metrics_by_asset": "runtime 侧素材库是内存态，要先 upload 才有数据",
    "creative.search_similar": "runtime 侧素材库是内存态 + 标签要手工塞",
    "policy.search": "两侧都走检索服务，三态语义已在 test_tool_impls 单独测",
    "insight.search_claims": "两侧都走检索服务，三态语义已在 test_tool_impls 单独测",
    "campaign.update_budget": "写工具：沙盒侧入台账、runtime 侧打平台 —— 幂等语义已单独测",
    "campaign.scale_budget": "写工具：runtime 侧带乐观并发校验（沙盒没有）—— 形状刻意不同",
    "campaign.create": "写工具：runtime 侧有「必须先开审批单」的硬前置，沙盒侧靠 cap 教",
    "approval.create_case": "runtime 侧是 open_approval_case，不是平台调用",
    "system.wait": "两侧行为**刻意不同**（租约上限），已在 test_creative_tools 单独测",
}


def test_the_coverage_ledger_is_complete():
    """★★ 30 个工具**每个都要有交代**：跑了，或者写清楚为什么没跑。

    ⚠️ 这条和 `tool_parity` 的账本同族 —— **缺口必须写下来才算被承认**。
      不写的话，这张表看起来"都测过了"，而实际只覆盖了五分之一。
    """
    from syncopate.runtime.tool_parity import CONTRACT_SIGNALS, sandbox_tools
    names = set(sandbox_tools())
    # v15 的 session.* 不是业务工具（runtime 侧由 agent_loop 状态机接），
    # 它们的交代写在 tool_parity.CONTRACT_SIGNALS 里 —— 同一份来源，不在这里另抄。
    names -= set(CONTRACT_SIGNALS)
    ledgered = set(EXERCISED) | set(NOT_EXERCISED)
    assert not (names - ledgered), f"这些工具没交代：{sorted(names - ledgered)}"
    assert not (ledgered - names), f"表里有沙盒没有的：{sorted(ledgered - names)}"
    assert not (set(EXERCISED) & set(NOT_EXERCISED))


def test_every_skip_reason_is_specific():
    thin = [k for k, v in NOT_EXERCISED.items() if len(v.strip()) < 10]
    assert not thin, f"这些「没跑」的理由太薄：{thin}"


# ── 真正的对照 ──────────────────────────────────────────────────────────

async def _runtime_call(name: str, args: dict):
    from functools import partial
    from syncopate.runtime import tool_impls as impl
    p = _runtime_platform()
    table = {
        "campaign.get_metrics": p.get_metrics,
        "campaign.list": partial(impl.campaign_list, p),
        "metrics.get_freshness": partial(impl.metrics_get_freshness, p),
        "campaign.detect_anomalies": partial(impl.campaign_detect_anomalies, p),
        "mmp.get_attribution": partial(impl.mmp_get_attribution, p),
    }
    return await table[name](**args)


@pytest.mark.parametrize("tool", sorted(EXERCISED))
def test_field_names_match(tool: str):
    """★★★ 沙盒有的字段，runtime **一个都不能少**。

    ⚠️ 反过来不要求（runtime 多给字段是安全的：模型不认识就不看）——
      **少给才致命**：模型按名字取，取不到就自己编一个。
    """
    from syncopate.core.tool_registry import REGISTRY
    import syncopate.domains.adcampaign  # noqa: F401

    args = EXERCISED[tool]
    sb = REGISTRY.get(tool).handler(args, _sandbox_ctx())
    assert sb.ok, f"沙盒侧没跑成，对照无从谈起：{sb.error}"
    rt = _run(_runtime_call(tool, args))

    missing = sorted(set(sb.data) - set(rt))
    assert not missing, (
        f"\n  {tool} —— runtime 少了沙盒有的字段：{missing}"
        f"\n    沙盒   {sorted(sb.data)}"
        f"\n    runtime {sorted(rt)}"
        f"\n  ⇒ 模型按字段名取数，少一个就是取不到（而且不会报错）")
