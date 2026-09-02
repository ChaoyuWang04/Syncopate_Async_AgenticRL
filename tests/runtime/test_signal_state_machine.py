"""v15 信令的状态机语义（`25 §R4` 门槛①：三信令各自触发正确状态与事件）。

★ 三条信令**不是同一种终止**，这是 N4「行为即动作」的全部意义所在：
    defer   → 本轮正常收工，复查靠 recheck_after_days 另行调度   (finished)
    clarify → 挂起等用户补充，和"开审批单等人"同族             (halted)
    reject  → 终止并审计；★ 归"取消"不归"失败"                  (exhausted/cancelled)
  归错了会让线上尺子把**模型做对的事**算成事故。
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import pytest


@dataclass
class _Proposal:
    kind: str
    final_answer: dict[str, Any] | None = None
    tool: str | None = None
    arguments: dict[str, Any] = field(default_factory=dict)
    rationale: str = ""
    thinking: str = ""
    param_source: str = "model"


class _OneShotDecider:
    def __init__(self, proposal):
        self._p = proposal

    async def decide(self, **_kw):
        return self._p


class _FakeGate:
    def __init__(self):
        self.step = 1
        self.events: list[tuple[str, dict]] = []

    async def stop_requested(self) -> bool:               # K5-5 安全点契约（假 gate 显式实现）
        return False

    async def budget_exceeded(self, *, model_calls, tokens):  # noqa: ANN001 —— K9-2 预算闸契约
        return None

    async def record_model_usage(self, *, call_no, tokens_in, tokens_out):  # noqa: ANN001 —— K9-3 记账契约
        return None

    def observation_for(self, tool, *, ok, data, error):    # noqa: ANN001
        return {"tool": tool, "ok": ok, "data": data, "error": error}

    async def emit_info(self, *, kind: str, payload: dict):
        self.events.append((kind, payload))


def _run(signal: str, args: dict):
    from syncopate.runtime import agent_loop as AL

    gate = _FakeGate()
    captured = {}

    async def _save(db, *, org_id, run_id, step, history):
        captured["history"] = history

    orig = AL.save_transcript
    AL.save_transcript = _save
    try:
        p = _Proposal(kind="final",
                      final_answer={"behavior": signal, "signal": signal,
                                    "arguments": args, "text": "好的。"})
        res = asyncio.run(AL.run_agent_loop(
            db=None, org_id="o", run_id="r", user_message="x",
            decider=_OneShotDecider(p), gate=gate))
    finally:
        AL.save_transcript = orig
    return res, gate


@pytest.mark.parametrize("signal,args,want_status", [
    ("defer", {"reason": "太新", "recheck_after_days": 3}, "finished"),
    ("clarify", {"question": "哪条?", "missing_fields": ["campaign_id"]}, "halted"),
    ("reject", {"reason_code": "unauthorized", "explanation": "越权"}, "exhausted"),
])
def test_each_signal_maps_to_its_own_terminal_state(signal, args, want_status):
    res, _ = _run(signal, args)
    assert res.status == want_status, f"{signal} 应当 {want_status}，得到 {res.status}"


@pytest.mark.parametrize("signal", ["defer", "clarify", "reject"])
def test_each_signal_emits_its_own_event(signal):
    """★ 事件按 kind 逐条校验 —— 没有事件 = 前端渲染不出卡片 = 机制没接上。"""
    _, gate = _run(signal, {"a": 1})
    kinds = [k for k, _ in gate.events]
    assert f"session.{signal}" in kinds, kinds


def test_signal_arguments_reach_the_event():
    """defer 的 recheck_after_days 必须到达事件层，否则挂起复查无从调度。"""
    _, gate = _run("defer", {"reason": "太新", "recheck_after_days": 7})
    payload = dict(gate.events)["session.defer"]
    assert payload["arguments"]["recheck_after_days"] == 7


def test_reject_carries_a_distinguishable_error():
    res, _ = _run("reject", {"reason_code": "policy", "explanation": "不行"})
    assert res.error == "session_reject"


def test_reject_is_cancelled_not_failed_in_worker():
    """★ 归类判据：session_reject 必须落在"取消"那一族。

    归"失败"会让人工修正率/失败率把**模型做对的拒绝**算成事故 —— 判据本身就错了。
    """
    import inspect

    from syncopate.runtime import worker

    src = inspect.getsource(worker)
    idx = src.index('"release_gate", "daily_cost_cap_exceeded"')
    window = src[idx: idx + 200]
    assert "session_reject" in window, "session_reject 没有和 release_gate 归在同一族"


def test_plain_text_final_still_finishes():
    """无信令的纯文本终答走原路径（v14 行为不变）。"""
    from syncopate.runtime import agent_loop as AL

    gate = _FakeGate()
    orig = AL.save_transcript

    async def _save(db, **_kw):
        return None

    AL.save_transcript = _save
    try:
        p = _Proposal(kind="final", final_answer={"behavior": None, "text": "好的。"})
        res = asyncio.run(AL.run_agent_loop(db=None, org_id="o", run_id="r",
                                            user_message="x",
                                            decider=_OneShotDecider(p), gate=gate))
    finally:
        AL.save_transcript = orig
    assert res.status == "finished"
    assert not [k for k, _ in gate.events if k.startswith("session.")]
