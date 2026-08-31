"""`session.report` 在 runtime 收口必须有 binding（v15，`25 §3.3`）。

⛔ 这条判据是**考场炸出来的**，不是想出来的：2026-08-30 v15r2 考场第一遍，
  50 道 L1 有 43 道 `cancelled` —— 事件流里 `session.report` 连续 6 次
  `unknown_tool`，模型放弃后改调 `session.reject`，run 被取消。
  训练侧同形的 allowlist 我早就修过；**运行时是另一份名单**，没跟着改。

判据形状取自「登记 ≠ 实现」：不查"名字在不在菜单里"（它在），查**真的调得动**。
"""
from __future__ import annotations

import asyncio

import pytest

from syncopate.core.contract import IS_V15, REPORT_TOOL

pytestmark = pytest.mark.skipif(not IS_V15, reason="v15 契约专有；v14 下信令族不存在")


def _bindings() -> dict:
    """拿 Worker 真正会用的那张表 —— 不重建一份"等价"的。

    绑定只是 `partial` 捕获，闭包里的 db/platform 在本测试里不会被**调用**，
    但建表时会取它们身上的方法引用，所以给一个什么属性都答应的替身。
    """
    from syncopate.runtime.worker import Worker, WorkerConfig
    w = Worker.__new__(Worker)
    class _Any:                      # 建表时会取几个方法引用，取到什么都行
        def __getattr__(self, _n):
            return _Any()

    w.db = w.platform = w.retrieval = _Any()
    w.config = WorkerConfig()
    return Worker._bindings(w, "org_demo", "run_x")


def test_report_has_a_runtime_binding() -> None:
    assert REPORT_TOOL in _bindings(), (
        f"{REPORT_TOOL} 在收口没有 binding ⇒ 线上判 unknown_tool（考场 43/50 取消）"
    )


def test_report_ack_shape_matches_sandbox() -> None:
    """观测形状必须和沙盒逐字节一样，否则模型学到的读法在生产上是错的。"""
    from syncopate.core.session_signals import ack_payload

    binding = _bindings()[REPORT_TOOL]
    got = asyncio.run(binding.invoke(conclusion="positive", lift=0.12))
    assert got == ack_payload(REPORT_TOOL, {"conclusion": "positive", "lift": 0.12})
    assert got["reported_fields"] == ["conclusion", "lift"]


def test_terminal_signals_stay_unbound() -> None:
    """⛔ 终止性信令**不许**有 binding：ack 成功会把"要拒绝"静默吞掉，
    比一个响的 unknown_tool 贵得多（它们在 decider 就被拦成 final）。"""
    from syncopate.core.contract import TERMINAL_SIGNALS

    bound = set(_bindings())
    assert not (bound & set(TERMINAL_SIGNALS)), "终止性信令不该走收口"
