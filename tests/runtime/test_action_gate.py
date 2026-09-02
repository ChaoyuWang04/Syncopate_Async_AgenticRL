"""`ActionGate`：生产级横切必须**绕不过去**，而不是"记得调用"。

★★★ 这一族测试钉的不是"功能对不对"，是"**能不能被绕过**"

`worker._execute` 原本是一段写死的两步计划，横切穿插在计划里 ——
"顺序对不对"由代码顺序保证。
⚠️ 一旦换成模型驱动的循环，这些横切就退化成"但愿那个循环记得调"，
而**「机制在，但没接上」是本项目记了十几次的第一失效形状**。

⇒ 所以边界是（`09 §4.5.2`）：

    模型能看到的      只有工具菜单 + observation
    模型能做的        提出一次工具调用
    模型**碰不到**的  权限 · 幂等 · 重试 · 成本闸 · 审批触发 · 审计 · 事件 · 步数上限

⚠️ 下面大多数用例**不需要数据库** —— 它们测的是"在打到外部世界之前就被拦住"，
   而被拦住的路径根本走不到 `ToolRuntime.call`。这是刻意的：
   **拦截逻辑不该依赖数据库能不能连上。**
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from syncopate.runtime.action_gate import (
    MAX_STEPS_PER_RUN, ActionGate, ToolBinding,
)
from syncopate.runtime.gateway import DecisionContext
from syncopate.runtime.tools import PermissionDenied, ToolRuntime


class _Recorder:
    """记下所有 emit / audit，供断言。"""

    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []
        self.audits: list[dict] = []

    async def emit(self, _db, *, org_id, run_id, kind, payload=None):  # noqa: ANN001
        self.events.append((kind, payload or {}))
        return len(self.events)

    async def audit(self, _db, *, org_id, run_id, action, object_key,  # noqa: ANN001
                    param_source, detail=None):
        self.audits.append({"action": action, "object_key": object_key,
                            "param_source": param_source, "detail": detail or {}})


def _gate(*, bindings=None, over_budget=False, rec=None, max_steps=MAX_STEPS_PER_RUN,
          tools=None, amount_threshold=None) -> tuple[ActionGate, _Recorder]:
    rec = rec or _Recorder()

    async def _ob() -> bool:
        return over_budget

    gate = ActionGate(
        db=None, tools=tools or ToolRuntime(db=None), bindings=bindings or {},
        org_id="org_1", run_id="run_1", over_budget=_ob,
        emit=rec.emit, audit=rec.audit, amount_threshold=amount_threshold,
        max_steps=max_steps)
    return gate, rec


def _run(coro):
    return asyncio.run(coro)


# ── 模型碰不到的东西 ──────────────────────────────────────────────────────

def test_param_source_must_be_explicit_and_valid():
    """★ `param_source` 没有默认值 —— 传错直接炸。

    这一列的意义是"这个参数是谁定的"。**"忘了传所以填了个默认值"会让整列失去意义** ——
    而那正是第一失效形状。⇒ 宁可炸，也不要悄悄记成 system。
    """
    gate, _ = _gate()
    with pytest.raises(ValueError, match="param_source"):
        _run(gate.invoke(tool="x", arguments={}, ctx=DecisionContext(),
                         param_source="whatever"))


def test_unknown_tool_reports_not_found_and_does_not_guess():
    """★ 模型会编工具名。**报"没有"，不许模糊匹配到一个名字相近的。**

    猜中一个相近的工具会**成功、会返回数据** —— 那是「量错对象」里最贵的形态。
    """
    gate, rec = _gate(bindings={"campaign.get_metrics": ToolBinding(_noop)})
    out = _run(gate.invoke(tool="campaign.get_metric",     # 少一个 s
                           arguments={}, ctx=DecisionContext(), param_source="model"))
    assert out.status == "failed"
    assert out.observation["error"].startswith("unknown_tool")


def test_step_cap_is_enforced_by_the_gate_not_the_loop():
    """★★ 步数上限记在**收口**里，不记在循环里。

    循环是会被换的（换模型、换 prompt、换编排），而"不许无限跑"是**生产约束**，
    不能跟着循环一起被换掉。
    """
    gate, rec = _gate(bindings={"t": ToolBinding(_noop)}, max_steps=2,
                      tools=_FakeTools(ok=True))
    ctx = DecisionContext()
    for _ in range(2):
        _run(gate.invoke(tool="t", arguments={}, ctx=ctx, param_source="model"))
    out = _run(gate.invoke(tool="t", arguments={}, ctx=ctx, param_source="model"))
    assert out.status == "refused" and out.error == "max_steps_exceeded"
    assert ("run.degraded", {"reason": "max_steps", "limit": 2}) in rec.events


def test_the_loop_cannot_supply_its_own_tool_implementation():
    """★★★ 堵洞：`ToolRuntime.call` 收 `invoke=` 由调用方传实现 ——
    循环可以绕过收口直接调 platform。

    判据写在**签名**上（「某集合应当完整」型）：`ActionGate.invoke` 不许有能传实现的形参。
    """
    import inspect
    params = set(inspect.signature(ActionGate.invoke).parameters)
    for forbidden in ("invoke", "platform", "impl", "callable"):
        assert forbidden not in params, f"收口不该让调用方传实现：{forbidden}"


# ── 成本闸：写之前必查，读不查 ────────────────────────────────────────────

def test_cost_gate_blocks_writes():
    gate, rec = _gate(bindings={"campaign.update_budget": ToolBinding(_boom)},
                      over_budget=True)
    out = _run(gate.invoke(tool="campaign.update_budget",
                           arguments={"campaign_id": "CMP_1", "new_budget": 1,
                                      "client_request_id": "r1"},
                           ctx=DecisionContext(), param_source="user"))
    assert out.status == "refused" and out.error == "daily_cost_cap_exceeded"
    assert rec.events[-1][0] == "run.degraded"
    assert rec.events[-1][1]["at"] == "before_write"


def test_cost_gate_does_not_block_reads():
    """★ 读不花平台的钱，超预算时仍然允许**查**。

    否则超预算的 run 连"为什么超了"都查不了 —— 而降级的意义是**降级**不是**失明**。
    ⚠️ 这条也钉住了闸门的位置：它在**写**之前，不在每一步之前。
    """
    gate, _ = _gate(bindings={"campaign.get_metrics": ToolBinding(_noop)},
                    over_budget=True, tools=_FakeTools(ok=True))
    out = _run(gate.invoke(tool="campaign.get_metrics",
                           arguments={"campaign_id": "CMP_1"},
                           ctx=DecisionContext(), param_source="system"))
    assert out.status == "ok"


# ── 审批：停在执行之前 ───────────────────────────────────────────────────

def test_triggers_halt_before_the_write_happens(monkeypatch):
    """★★ 审批必须停在**真正执行之前** —— 执行完再问人就只剩记账了。"""
    executed: list[str] = []

    async def _mark(**kw):                       # noqa: ANN003
        executed.append("did_write")
        return {}

    async def _fake_open_case(*a, **kw):         # noqa: ANN002, ANN003
        return "CASE_1"

    import syncopate.runtime.action_gate as ag
    monkeypatch.setattr(ag, "open_approval_case", _fake_open_case)

    gate, rec = _gate(bindings={"campaign.update_budget": ToolBinding(_mark)},
                      amount_threshold=100)
    ctx = DecisionContext(write_amount=999_999)   # 超阈值 ⇒ 必触发
    out = _run(gate.invoke(tool="campaign.update_budget",
                           arguments={"campaign_id": "CMP_1", "new_budget": 999_999,
                                      "client_request_id": "r1"},
                           ctx=ctx, param_source="user"))
    assert out.status == "halted" and out.case_ref == "CASE_1"
    assert executed == [], "★ 审批单开了，但写动作**已经执行了** —— 那审批就只是记账"
    # K4：run.waiting_for_user 由 open_approval_case 内的 transition_run 与状态同事务写；
    # 收口自己**不许再发**（发了就是重复事件）。真事件在 test_state_machine_k4 / test_worker 里验。
    assert all(kind != "run.waiting_for_user" for kind, _ in rec.events)


def test_decided_actions_skip_triggers():
    """已被人裁决过的动作不再过网关 —— 否则刚批准就又被同一个触发器拦下，
    run 会在 waiting_for_user 和 queued 之间来回弹，永远跑不完。"""
    gate, _ = _gate(bindings={"campaign.update_budget": ToolBinding(_noop)},
                    amount_threshold=100, tools=_FakeTools(ok=True))
    gate.skip_triggers = True
    out = _run(gate.invoke(tool="campaign.update_budget",
                           arguments={"campaign_id": "CMP_1", "new_budget": 999_999,
                                      "client_request_id": "r1"},
                           ctx=DecisionContext(write_amount=999_999),
                           param_source="human_review"))
    assert out.status == "ok"


# ── 给模型看的那一份 ─────────────────────────────────────────────────────

def test_observation_does_not_leak_internal_fields():
    """★★ 模型不该看到 `idempotency_key` / `attempts` / 触发器细节。

    它看到了就会开始**绕着判据走**，而判据一旦可被优化，就不再是判据。
    """
    gate, _ = _gate(bindings={"campaign.get_metrics": ToolBinding(_noop)},
                    tools=_FakeTools(ok=True))
    out = _run(gate.invoke(tool="campaign.get_metrics",
                           arguments={"campaign_id": "CMP_1"},
                           ctx=DecisionContext(), param_source="system"))
    leaked = {"idempotency_key", "attempts", "triggers", "case_ref", "replayed",
              # ★ 连 ok/tool 都不该在里面 —— 沙盒的 observation 只有工具数据本身
              "ok", "tool", "status"}
    assert not (leaked & set(out.observation)), f"泄漏了内部字段：{out.observation}"


def test_failures_are_visible_to_the_model():
    """失败要给模型看见 —— 沙盒里教的"失败之后怎么办"要在线上用得上。

    ⚠️ 反面是"重试到成功为止"，那会让那部分策略变成死代码（`tools.py` 同一条）。
    """
    gate, _ = _gate(bindings={"campaign.get_metrics": ToolBinding(_noop)},
                    tools=_FakeTools(ok=False, error="platform_down"))
    out = _run(gate.invoke(tool="campaign.get_metrics",
                           arguments={"campaign_id": "CMP_1"},
                           ctx=DecisionContext(), param_source="system"))
    assert out.status == "failed"
    # ★ 形状同沙盒：失败就是一个**只含 error** 的字典（不是 {"ok": False, ...}）
    assert out.observation == {"error": "platform_down"}


class _CrashingTools:
    """工具**实现**直接抛异常（不是返回 error）——模拟实现层 bug / 依赖崩溃。"""

    async def call(self, **_kw):                       # noqa: ANN003
        raise RuntimeError("operator is not unique: date + unknown")


def test_tool_implementation_crash_becomes_an_observation_not_run_death():
    """★ 工具实现崩了也要变成失败观测回给模型，不能带走整条 run。

    2026-08-20 压测实锤：calendar 的一个 SQL 类型错让 I11 8/8 全灭 ——
    异常一路穿到 run_once 兜底，「动作失败不终止循环」在崩溃这条路上是空话。
    ⚠️ 给模型的观测不含异常细节（SQL 栈不是模型该学的输入）；全文进审计。
    """
    gate, rec = _gate(bindings={"calendar.get_seasonal_context": ToolBinding(_noop)},
                      tools=_CrashingTools())
    out = _run(gate.invoke(tool="calendar.get_seasonal_context",
                           arguments={"region": "华东"},
                           ctx=DecisionContext(), param_source="model"))
    assert out.status == "failed"                      # 不是抛异常
    assert out.observation["error"].startswith("tool_crashed")
    assert "date + unknown" not in out.observation["error"]   # 细节不给模型
    assert rec.audits[-1]["action"] == "tool_crashed"
    assert "date + unknown" in rec.audits[-1]["detail"]["error"]  # 细节进审计


# ── 审计 ────────────────────────────────────────────────────────────────

def test_writes_are_audited_with_the_declared_param_source():
    """⚠️ 2026-08-20 起走**已裁决**路径才到得了执行：档位改由动作推导之后，
    不可逆写动作一律先要人点头（灰测默认档 C 的定义，release.py docstring 原话）。
    `skip_triggers=True` 就是 worker 在"人已裁决"时设的那个开关。"""
    gate, rec = _gate(bindings={"campaign.update_budget": ToolBinding(_noop)},
                      tools=_FakeTools(ok=True))
    gate.skip_triggers = True
    _run(gate.invoke(tool="campaign.update_budget",
                     arguments={"campaign_id": "CMP_7", "new_budget": 5,
                     "client_request_id": "r1"},
                     ctx=DecisionContext(), param_source="user"))
    assert rec.audits[-1]["param_source"] == "user"
    assert rec.audits[-1]["object_key"] == "CMP_7"


def test_object_key_reports_none_rather_than_guessing():
    """认不出作用对象就报 None —— 猜错的 `object_key` 比没有更糟：
    它让审计看起来完整，而实际指向另一个对象。"""
    from syncopate.runtime.action_gate import _object_key
    assert _object_key({"campaign_id": "CMP_1"}) == "CMP_1"
    assert _object_key({"note": "随便写点什么"}) is None
    assert _object_key({"campaign_id": 123}) is None      # 类型不对也不猜


def test_permission_denied_is_audited_not_swallowed():
    class _Denying(ToolRuntime):
        def __init__(self) -> None:
            pass

        async def call(self, **kw):               # noqa: ANN003
            raise PermissionDenied("缺少权限 budget:write")

    gate, rec = _gate(bindings={"campaign.update_budget": ToolBinding(_noop)},
                      tools=_Denying())
    gate.skip_triggers = True        # 同上：权限闸在审批之后，要先过得了审批这一关
    out = _run(gate.invoke(tool="campaign.update_budget",
                           arguments={"campaign_id": "C", "new_budget": 1,
                                      "client_request_id": "r1"},
                           ctx=DecisionContext(), param_source="model"))
    assert out.status == "failed"
    assert rec.audits[-1]["action"] == "permission_denied"


# ── 辅助 ────────────────────────────────────────────────────────────────

async def _noop(**kwargs: Any) -> dict:           # noqa: ANN401
    return {}


async def _boom(**kwargs: Any) -> dict:           # noqa: ANN401
    raise AssertionError("★ 这个工具不该被执行到 —— 有一道闸没拦住")


class _FakeTools(ToolRuntime):
    """替身：跳过数据库，只回报 ok/error。测的是**收口的顺序**，不是 ToolRuntime。"""

    def __init__(self, *, ok: bool, error: str | None = None) -> None:
        self._ok, self._error = ok, error

    async def call(self, **kw: Any) -> Any:       # noqa: ANN401
        from syncopate.runtime.tools import ToolOutcome
        return ToolOutcome(ok=self._ok, data={} if self._ok else None,
                           error=self._error, attempts=1, replayed=False,
                           idempotency_key=None)
