"""B-7 · 灰测闸门：**fail-closed**，且放量的维度是档位不是流量比例。

★★★ 三个设计判断，每条都换掉了一个"想当然"的做法：

    ① 放量维度   不是"路由 10% 流量"，是 `automation_tier`
                 —— 出事的**严重度由后者决定**：10% 流量配 A 档，
                    比 100% 流量配 C 档危险得多
    ② 开关语义   **fail-closed** —— 读不到就当关闭
                 反过来意味着**配置服务一挂就全量放开**，那是最坏的时刻放最大的权
    ③ 闸门位置   在 `ActionGate` 里，不在 API 层
                 —— 同一个动作可能来自 API 也可能来自 worker，两条路要过同一道闸
"""

from __future__ import annotations

import asyncio

import pytest

from syncopate.runtime import release
from syncopate.runtime.action_gate import ActionGate, ToolBinding
from syncopate.runtime.gateway import DecisionContext
from syncopate.runtime.tools import ToolOutcome, ToolRuntime


def _run(coro):
    return asyncio.run(coro)


class _Tools(ToolRuntime):
    def __init__(self): pass
    async def call(self, **kw):                      # noqa: ANN003
        return ToolOutcome(ok=True, data={}, error=None, attempts=1,
                           replayed=False, idempotency_key=None)


async def _noop(**kw):                               # noqa: ANN003
    return {}


async def _boom(**kw):                               # noqa: ANN003
    raise AssertionError("★ 这个写动作不该被执行到 —— 灰测闸门没拦住")


def _gate(*, write_impl=_noop, events=None):
    async def ob(): return False

    async def emit(_db, **kw):                       # noqa: ANN003
        (events if events is not None else []).append(kw)
        return 1

    async def audit(_db, **kw):                      # noqa: ANN003
        return None

    return ActionGate(None, _Tools(),
                      {"campaign.update_budget": ToolBinding(write_impl),
                       "campaign.get_metrics": ToolBinding(_noop)},
                      org_id="o", run_id="r", over_budget=ob,
                      emit=emit, audit=audit)


WRITE_ARGS = {"campaign_id": "C1", "new_budget": 1, "client_request_id": "r1"}


# ── ② fail-closed ───────────────────────────────────────────────────────

def test_unreadable_switch_means_closed_not_open(monkeypatch):
    """★★★ 配错档位 ⇒ **当成关闭**，不退回默认值继续跑。

    退回默认值会让"配错了"看起来像"配对了" —— 而那是最贵的一类错误。
    """
    monkeypatch.setenv(release.ENV_MAX_TIER, "全自动")     # 非法值
    st = release.current_state()
    assert st.halted is True
    assert "invalid_max_tier" in (st.halt_reason or "")


def test_missing_tier_is_treated_as_needs_approval_not_as_full_auto():
    """★★ `tier=None`（**没声明**）按 `C`（要人点头）算 —— 不是按 `A`。

    ⚠️ 我这条写错过一次：第一版映射到 `D`，而 D 是自主度**最低**的一档
      ⇒ 判断恒为 True ⇒ **全部放行**，而文档串写着"按最严处理"。
      **代码和文档说的是反的**，是这条测试当场炸出来的。

    ⇒ 「没写」和「写了 A」是两件事 —— 把前者当后者就是在默认全自动。
    ⬜ 真正该做的是让 `automation_tier` 在建 run 时**必填**（记为缺口）。
    """
    st = release.ReleaseState(halted=False, max_tier="C")
    assert st.allows(None) is True          # C 这一档是允许的
    strict = release.ReleaseState(halted=False, max_tier="D")
    assert strict.allows(None) is False, "★ 上限收到 D 时，没声明的也该被拦下"
    assert strict.allows("A") is False


def test_default_ceiling_is_conservative(monkeypatch):
    """★ 默认上限是 **C**（每个写动作都要人点头），不是 A。

    ⚠️ 忘了配置的后果应该是**太严**，不是太松。
    """
    monkeypatch.delenv(release.ENV_KILL, raising=False)
    monkeypatch.delenv(release.ENV_MAX_TIER, raising=False)
    assert release.current_state().max_tier == release.DEFAULT_MAX_TIER == "C"


# ── ① 档位是放量的维度 ──────────────────────────────────────────────────

def test_the_ladder_goes_by_tier_not_by_traffic(monkeypatch):
    """★★ 阶梯是 D → C → B → A，每一档放开的是**能自己做多少**。"""
    monkeypatch.delenv(release.ENV_KILL, raising=False)
    monkeypatch.setenv(release.ENV_MAX_TIER, "C")
    st = release.current_state()
    assert st.allows("D") and st.allows("C")
    assert not st.allows("B") and not st.allows("A"), "★ 上限是 C 却放行了更宽的档"


# ── ③ 闸门在收口里，写动作过不去 ────────────────────────────────────────

def test_a_halted_release_blocks_writes_before_they_execute(monkeypatch):
    """★★★ 判据不只是"返回了 refused"，是**那个写动作一次都没被执行**。

    返回 refused 但已经写出去了，这道闸等于没有。
    """
    monkeypatch.setenv(release.ENV_KILL, "1")
    out = _run(_gate(write_impl=_boom).invoke(
        tool="campaign.update_budget", arguments=WRITE_ARGS,
        ctx=DecisionContext(automation_tier="A"), param_source="user"))
    assert out.status == "refused" and out.error == "release_gate"


def test_reads_still_work_while_halted(monkeypatch):
    """★ 关掉的是**写**，读照常 —— 降级的意义是降级，不是失明。

    否则灰测一出事，连"出了什么事"都查不了。
    """
    monkeypatch.setenv(release.ENV_KILL, "1")
    out = _run(_gate().invoke(tool="campaign.get_metrics",
                              arguments={"campaign_id": "C1"},
                              ctx=DecisionContext(), param_source="system"))
    assert out.status == "ok"


def test_halting_is_traceable_in_the_event_stream(monkeypatch):
    """★★ 关闭必须**可追溯** —— 重新打开时要能按它找回受影响的 run。

    ⚠️ 只是"新的不许起"是不够的：已经停在 `waiting_for_user` 的 run
      没人去恢复的话会**永远卡在那里**。
    """
    monkeypatch.setenv(release.ENV_KILL, "1")
    events: list = []
    _run(_gate(events=events).invoke(
        tool="campaign.update_budget", arguments=WRITE_ARGS,
        ctx=DecisionContext(automation_tier="A"), param_source="user"))
    degraded = [e for e in events if e.get("kind") == "run.degraded"]
    assert degraded and degraded[-1]["payload"]["reason"] == "release_gate"
    assert degraded[-1]["payload"]["halt_reason"], "★ 关了却没记原因"


def test_the_gate_is_in_the_chokepoint_not_the_api(monkeypatch):
    """★ 闸门在 `ActionGate` 里 —— API 和 worker 两条路共用同一道闸。

    放在 API 层的话，worker 自己的编排**完全绕过**它。
    """
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    gate = (root / "syncopate" / "runtime" / "action_gate.py").read_text(encoding="utf-8")
    api = (root / "syncopate" / "runtime" / "api.py").read_text(encoding="utf-8")
    assert "current_state()" in gate
    assert "SYNCOPATE_RELEASE" not in api, "★ 闸门跑到 API 层去了，worker 那条路会绕过"
