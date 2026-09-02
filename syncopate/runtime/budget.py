"""K9-2 · 预算闸（课件 CH9 §6 三层递进）：run 级四字段 + org 日预算两档。

  run 级   步数（ActionGate ①，已有）· max_model_calls · max_tokens · max_duration_s
           超限 ⇒ **转 waiting_for_user 不判死**（挡晋级不挡起跑：还有救的 run 留给人决定）
  org 级   日 token 预算：≥ warn_ratio 告警（判据行 [budget] warn）；≥ 100% **拒新建**（API 429），
           已在跑的不受影响（"超限拒新建"与"杀在飞的"是两件事）
  ⚠️ 没有对应 SLO 的层其正确性在线上不可验证 ⇒ 每条阈值都在 metrics 里有读数。
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

DEFAULT_ORG_DAILY_TOKENS = int(os.environ.get("SYNCOPATE_ORG_DAILY_TOKENS", "2000000"))
DEFAULT_ORG_DAILY_COST_MICROS = int(os.environ.get("SYNCOPATE_ORG_DAILY_COST_MICROS", "10000000"))
DEFAULT_WARN_RATIO = 0.8


@dataclass(frozen=True)
class RunBudget:
    max_model_calls: int
    max_tokens: int
    max_duration_s: int


def run_budget_exceeded(budget: RunBudget, *, model_calls: int, tokens: int,
                        elapsed_s: float) -> str | None:
    """返回超限原因（None = 没超）。三条各自独立，先撞哪条报哪条。"""
    if model_calls > budget.max_model_calls:
        return f"max_model_calls: {model_calls} > {budget.max_model_calls}"
    if tokens > budget.max_tokens:
        return f"max_tokens: {tokens} > {budget.max_tokens}"
    if elapsed_s > budget.max_duration_s:
        return f"max_duration_s: {elapsed_s:.0f}s > {budget.max_duration_s}s"
    return None


async def load_run_budget(db: Any, *, org_id: str, run_id: str) -> tuple[RunBudget, float]:
    async with db.tx() as conn:
        row = await conn.fetchrow(
            "SELECT max_model_calls, max_tokens, max_duration_s, "
            " COALESCE(extract(epoch FROM (now() - started_at)), 0) AS elapsed "
            "FROM agent_runs WHERE org_id=$1 AND run_id=$2", org_id, run_id)
    if row is None:
        raise LookupError(run_id)
    return (RunBudget(int(row["max_model_calls"]), int(row["max_tokens"]), int(row["max_duration_s"])),
            float(row["elapsed"]))


async def org_budget_state(db: Any, *, org_id: str) -> dict[str, Any]:
    """两档：ok / warn（≥ warn_ratio）/ over（≥ 100%）。读数来自 usage_records 当日聚合。"""
    async with db.tx() as conn:
        cfg = await conn.fetchrow("SELECT daily_tokens, daily_cost_micros, warn_ratio FROM org_budgets WHERE org_id=$1", org_id)
        used = await conn.fetchrow(
            "SELECT COALESCE(sum(tokens_in+tokens_out),0) AS tokens, COALESCE(sum(cost_micros),0) AS cost "
            "FROM usage_records WHERE org_id=$1 AND day=CURRENT_DATE", org_id)
    daily_tokens = int(cfg["daily_tokens"]) if cfg else DEFAULT_ORG_DAILY_TOKENS
    daily_cost = int(cfg["daily_cost_micros"]) if cfg else DEFAULT_ORG_DAILY_COST_MICROS
    warn_ratio = float(cfg["warn_ratio"]) if cfg else DEFAULT_WARN_RATIO
    tokens, cost = int(used["tokens"]), int(used["cost"])
    ratio = max(tokens / daily_tokens if daily_tokens else 0.0, cost / daily_cost if daily_cost else 0.0)
    state = "over" if ratio >= 1.0 else ("warn" if ratio >= warn_ratio else "ok")
    return {"state": state, "ratio": round(ratio, 3), "tokens": tokens, "daily_tokens": daily_tokens,
            "cost_micros": cost, "daily_cost_micros": daily_cost}
