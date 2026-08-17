"""下发侧记账（B4 / E08-c）的判据测试。

★ 为什么这几条要有测试守着：
这个仪器**装错位置整整一轮实验都不会报错** —— E08 量到「逐桶零差」（7200=7200），
读数干净得像个结论，实际上是因为记账点在「跑完之后」，
**被中途杀掉的长轨迹一条都没被记过**。⇒ 位置本身就是判据的一部分。
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from syncopate.train import verl_agent_loop as loop


def _drain(path, monkeypatch, coro_factory):
    monkeypatch.setattr(loop, "DISPATCH_LOG", str(path))
    monkeypatch.setattr(loop, "_dispatch_logged", False)
    # 每个测试用自己的锁：模块级的锁绑在别的事件循环上会 RuntimeError
    monkeypatch.setattr(loop, "_dispatch_lock", asyncio.Lock())
    asyncio.run(coro_factory())
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def test_three_event_types_are_written(tmp_path, monkeypatch):
    log = tmp_path / "dispatched.jsonl"
    bundle = SimpleNamespace(case_id="C1")
    output = SimpleNamespace(metrics={"num_steps": 7, "wall_seconds": 12.5, "truncated": 0})

    async def go():
        await loop.record_dispatch_start(bundle, "r1")
        await loop.record_dispatch(bundle, output, 0.75, rollout_id="r1",
                                   version_fields={"min_global_steps": 3, "max_global_steps": 5})
        await loop.record_dispatch_start(bundle, "r2")
        await loop.record_dispatch_abort(bundle, "r2", "cancelled")

    rows = _drain(log, monkeypatch, go)
    assert [r["event"] for r in rows] == ["dispatch", "complete", "dispatch", "abort"]
    # ★ 配对靠 rollout_id —— 没有它就算不出「发出去了但没回来」的那一批
    assert {r["rollout_id"] for r in rows} == {"r1", "r2"}
    done = rows[1]
    assert done["reward"] == 0.75 and done["num_steps"] == 7
    # staleness 的真值（这条轨迹横跨了几个策略版本）必须落在 complete 行上
    assert done["min_global_steps"] == 3 and done["max_global_steps"] == 5


def test_dispatch_row_carries_no_reward(tmp_path, monkeypatch):
    """⚠️ dispatch/abort **不许**带 reward 字段。

    带了就会被 `Pool.ingest` 当成真实得分吃进去 —— 而「还不知道」不是「0 分」。
    （TRACK-B §0.6 从 AReaL 源码里抄回来的硬约束，配套的守卫在 test_pool.py。）
    """
    log = tmp_path / "dispatched.jsonl"
    bundle = SimpleNamespace(case_id="C1")

    async def go():
        await loop.record_dispatch_start(bundle, "r1")
        await loop.record_dispatch_abort(bundle, "r1", "cancelled")

    for row in _drain(log, monkeypatch, go):
        assert "reward" not in row


def test_nothing_is_written_without_the_env_var(tmp_path, monkeypatch):
    """没设 SYNCOPATE_DISPATCH_LOG 时必须完全静默（默认不产生副作用）。"""
    log = tmp_path / "dispatched.jsonl"
    monkeypatch.setattr(loop, "DISPATCH_LOG", None)
    monkeypatch.setattr(loop, "_dispatch_lock", asyncio.Lock())
    asyncio.run(loop.record_dispatch_start(SimpleNamespace(case_id="C1"), "r1"))
    assert not log.exists()
