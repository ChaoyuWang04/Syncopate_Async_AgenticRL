"""B-5/E33 S0 · 一单旅程的分账插桩。

SYNCOPATE_STAGE_TIMING=1 时启用；默认关（所有热路径先查 ENABLED，零行为变化）。
观测原则：只计时不改逻辑；每单收尾由 worker 打一行机读 JSON（[stage-timing]），
分账表由 scripts/b5_ledger.py 事后拼装。

桶的定义（同一 asyncio task 内串行，靠 contextvars 归属到当前 run）：
    llm        decider.decide 的墙钟
    tool       工具 invoke 的墙钟（含工具内部的 DB 时间——见嵌套标记）
    db_wait    连接池 acquire 的等待（**不管谁借的都记**：这是 S2 扩池的判据）
    db_tx      工具外的事务体墙钟（emit/audit/记账等编排自身的 DB 时间）
    db_in_tool 工具内的事务体墙钟（信息桶，已含在 tool 里，防双记账用）
"""

from __future__ import annotations

import contextvars
import os
import time
from typing import Any

ENABLED = os.environ.get("SYNCOPATE_STAGE_TIMING", "0") == "1"

_acc: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "b5_stage_acc", default=None)


def begin_run(run_id: str) -> contextvars.Token | None:
    if not ENABLED:
        return None
    return _acc.set({"run_id": run_id, "t0": time.perf_counter(), "_tool_depth": 0})


def end_run(token: contextvars.Token | None) -> None:
    if token is None:
        return
    acc = _acc.get()
    _acc.reset(token)
    if not acc:
        return
    out = {k: (round(v, 4) if isinstance(v, float) else v)
           for k, v in acc.items() if not k.startswith("_")}
    out["exec_wall"] = round(time.perf_counter() - acc["t0"], 4)
    out.pop("t0", None)
    import json
    print("[stage-timing] " + json.dumps(out), flush=True)


def add(bucket: str, dt: float) -> None:
    acc = _acc.get()
    if acc is None:
        return
    acc[bucket] = acc.get(bucket, 0.0) + dt
    acc[bucket + "_n"] = acc.get(bucket + "_n", 0) + 1


def tool_enter() -> None:
    acc = _acc.get()
    if acc is not None:
        acc["_tool_depth"] += 1


def tool_exit() -> None:
    acc = _acc.get()
    if acc is not None:
        acc["_tool_depth"] -= 1


def in_tool() -> bool:
    acc = _acc.get()
    return bool(acc and acc["_tool_depth"] > 0)
