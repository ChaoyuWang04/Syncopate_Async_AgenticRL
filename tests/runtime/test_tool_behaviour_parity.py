"""B-5b · **行为对齐**：同名工具在沙盒与 runtime 两侧，模型看到的东西必须同形。

★★★ 这一族和 B-5a（账本）的分工

    B-5a  签名对齐 —— 静态可查：必填参数接不住就 TypeError
    B-5b  **行为对齐** —— 只有真跑起来才看得见：返回字段、失败语义、observation 形状

⚠️ B-5b 找的**不是"功能没做"，是"两边都能跑但含义不同"** —— 那种不会报错。

★★ 第一次跑就抓到一个**结构性**分歧（2026-08-19）：

    沙盒     {"cpi": 2.1, "roas_d7": 0.42}                      ← 数据**直接**给模型
    runtime  {"tool":"...", "ok":true, "result":{"cpi":2.1,…}}  ← **多包了一层**

⇒ **模型被训练成读第一种，生产上会拿到第二种** —— 它得去解一层从没见过的壳，
  而这层壳还是冗余的（工具名在 tool message 的 `name` 字段里已经有了）。
⇒ 已修：`ActionGate._observation` 改成和沙盒逐字段同形。
"""

from __future__ import annotations

import asyncio
import inspect
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _run(coro):
    return asyncio.run(coro)


# ── ① observation 的形状（模型直接读的那个）────────────────────────────

def test_observation_shape_matches_the_sandbox():
    """★★★ 两侧的 observation 必须**逐字段同形**。

    沙盒 `ToolResult.observation()`：
        成功 ⇒ `dict(self.data)`            —— 工具数据本身
        失败 ⇒ `{"error": ...}`             —— 只有 error 一个键
    """
    from syncopate.core.tool_registry import ToolResult
    from syncopate.runtime.action_gate import ActionGate

    data = {"cpi": 2.1, "roas_d7": 0.42}
    sandbox_ok = ToolResult(ok=True, data=data).observation()
    runtime_ok = ActionGate._observation("campaign.get_metrics", ok=True,
                                         data=data, error=None)
    assert runtime_ok == sandbox_ok, (
        f"成功时形状不同：\n  沙盒   {sandbox_ok}\n  runtime {runtime_ok}\n"
        f"⇒ 模型被训练成读沙盒那种，生产上读不懂 runtime 这种")

    sandbox_err = ToolResult(ok=False, error="platform_down").observation()
    runtime_err = ActionGate._observation("campaign.get_metrics", ok=False,
                                          data=None, error="platform_down")
    assert runtime_err == sandbox_err, (
        f"失败时形状不同：\n  沙盒   {sandbox_err}\n  runtime {runtime_err}")


def test_the_failure_fallback_text_is_the_same():
    """连"没给 error 时的兜底文案"都要一样 ——
    模型可能按这个字符串分支（沙盒里训过），两边不同就分支到别处去了。"""
    from syncopate.core.tool_registry import ToolResult
    from syncopate.runtime.action_gate import ActionGate
    assert (ToolResult(ok=False, error=None).observation()
            == ActionGate._observation("t", ok=False, data=None, error=None))


def test_unknown_tool_text_is_the_same_on_both_sides():
    """★ 模型会编工具名，两边的报错**文案前缀**必须一致。

    沙盒 `REGISTRY.execute` 返回 `unknown_tool: <name>`；
    runtime 收口若报成别的词，模型学到的"工具名写错了"这个识别就失效了。
    """
    src = (ROOT / "syncopate" / "core" / "tool_registry.py").read_text(encoding="utf-8")
    gate = (ROOT / "syncopate" / "runtime" / "action_gate.py").read_text(encoding="utf-8")
    assert 'f"unknown_tool: {name}"' in src
    assert 'unknown_tool: {tool}' in gate


# ── ② 真的两边各跑一遍，比返回字段 ──────────────────────────────────────

def _sandbox_call(name: str, args: dict):
    """直接调沙盒 handler（纯读工具不碰台账，不需要完整 runner）。"""
    import syncopate.domains.adcampaign  # noqa: F401
    from syncopate.core.sandbox import Sandbox
    from syncopate.core.schemas import Case, CaseMetadata
    from syncopate.core.tool_registry import REGISTRY, ToolContext
    from syncopate.domains.adcampaign.world import WorldBuilder
    # ⚠️ WorldBuilder 默认是**空世界**，campaign 要显式建 ——
    #   第一版没建，工具返回 campaign_not_found，测试就 skip 了。
    #   而**「跳过不是通过」**：一条静默 skip 的对照测试等于没有对照。
    env = (WorldBuilder("T_0001", reference_now="2026-08-19T00:00:00Z")
           .account("ACC_01")
           .campaign("CMP_1", account_id="ACC_01")
           .build())
    case = Case(case_id=env.case_id, user_message="-", context={}, entities={},
                metadata=CaseMetadata(signal_class="graded", bucket="rag"), max_steps=8)
    ctx = ToolContext(case=case, env=env, sandbox=Sandbox(env, "ns"), step=1,
                      tool_call_id="t1")
    return REGISTRY.get(name).handler(args, ctx)


def test_metrics_get_freshness_returns_the_same_field_names():
    """★★ `metrics.get_freshness` 两侧真跑，比**字段名**。

    ⚠️ 比字段名不比值：值本来就该不同（两侧的世界不同）。
      **字段名不同才是契约破了** —— 模型按字段名取数。
    """
    from syncopate.runtime import tool_impls as impl
    from syncopate.runtime.platform import FakeAdPlatform

    sb = _sandbox_call("metrics.get_freshness", {"campaign_id": "CMP_1"})
    # ⚠️ **不 skip**：取不到数据说明对照台本身坏了，那比"字段不一致"更该红
    assert sb.ok, f"沙盒侧没跑成，对照无从谈起：{sb.error}"
    rt = _run(impl.metrics_get_freshness(FakeAdPlatform(), campaign_id="CMP_1"))

    missing = set(sb.data) - set(rt)
    assert not missing, (
        f"runtime 少了沙盒有的字段：{sorted(missing)}\n"
        f"  沙盒   {sorted(sb.data)}\n  runtime {sorted(rt)}\n"
        f"⇒ 模型按字段名取数，少一个就是取不到")


# ── ③ 失败语义分族（重试安全性完全不同）────────────────────────────────

def test_failure_families_are_not_collapsed():
    """★★★ 三族异常**刻意不合并**，因为它们的重试安全性完全相反：

        PlatformError        外部世界拒绝了      **可能可重试**（带 retriable）
        MemoryWriteRefused   我们的规则不允许    重试永远没用
        PreconditionNotMet   你还没做该做的那步  重试永远没用

    ⚠️ 类型混了，`ToolRuntime` 会去**重试一件不可能成功的事** ——
      而它只在平台明确说可重试时才重试，所以这个分类直接决定线上会不会自伤。
    """
    from syncopate.runtime import tool_impls as impl
    from syncopate.runtime.platform import PlatformError
    assert not issubclass(impl.MemoryWriteRefused, PlatformError)
    assert not issubclass(impl.PreconditionNotMet, PlatformError)
    assert not issubclass(impl.MemoryWriteRefused, impl.PreconditionNotMet)


def test_every_runtime_tool_is_async():
    """★ 全部 runtime 工具必须是协程 —— 收口 `await` 它们。

    同步实现会**静默返回一个 coroutine 之外的东西**，
    而 `await` 一个非 awaitable 是 TypeError，那还算好的；
    真正糟的是有人把它包成 `asyncio.run` —— 在事件循环里会直接炸。
    """
    from syncopate.runtime import tool_impls as impl
    # ⚠️ 只挑**函数**：异常类也是 callable，第一版把它们也算进来了 ——
    #   判据太宽，当场误伤两个（守则③，今天第三次）。
    fns = [v for k, v in vars(impl).items()
           if inspect.isfunction(v) and not k.startswith("_")
           and getattr(v, "__module__", "") == impl.__name__]
    bad = [f.__name__ for f in fns if not inspect.iscoroutinefunction(f)]
    assert not bad, f"这些不是协程：{bad}"
