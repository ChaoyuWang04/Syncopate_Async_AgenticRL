"""K7-2 · 事件分层（课件 CH7 §10）：**库里存全的，推出去的只留人话。**

三层：
  public    前端时间线要的：状态推进、工具结果的成败、思考文本（dev mode 刻意可见）
  internal  排障/对账要的：outbox/worker/lease/tool_call_id 这类基础设施事实——走 /trace（独立角色，K1-8）
  audit     不推、不进 trace 默认视图，只在审计导出里出现（本阶段登记，K9 消费）

payload 两视图：库里原样；public 推送前按 `INTERNAL_FIELDS` 剥字段（黑名单）+ 每种 kind 可选白名单。
⛔ **未登记的 kind 默认 internal（不外推）**——fail-closed。代价是"新事件忘了登记就看不见"，
   所以配一条结构测试：代码里每个 emit/append_event 的 kind 都必须在这里登记（K7 门槛③ 的另一半）。
"""
from __future__ import annotations

from typing import Any

PUBLIC, INTERNAL, AUDIT = "public", "internal", "audit"

KIND_LAYER: dict[str, str] = {
    # ---- run 状态推进（transition_run / create_run / dispatcher / cancel）----
    "run.created": PUBLIC,
    "run.enqueued": INTERNAL,
    "run.started": PUBLIC,
    "run.restarted": PUBLIC,
    "run.completed": PUBLIC,
    "run.failed": PUBLIC,
    "run.cancelled": PUBLIC,
    "run.waiting_for_user": PUBLIC,
    "run.resumed": PUBLIC,
    "run.cancel_requested": PUBLIC,
    "run.retry_scheduled": PUBLIC,
    "run.requeued_by_sweeper": INTERNAL,
    "run.degraded": PUBLIC,
    "run.awaiting_reconciliation": INTERNAL,
    # ---- 执行过程 ----
    "model.thinking": PUBLIC,            # dev mode：CoT 折叠展示是 Chaoyu 08-29 的要求
    "tool.result": PUBLIC,
    "retrieval.result": PUBLIC,
    "tool.manual_review": INTERNAL,
    "tool.repaired_from_intent_log": INTERNAL,
    "tool.response_lost": INTERNAL,
    "tool.repaired": INTERNAL,
    "run.waiting_too_long": PUBLIC,
    "run.stuck_queued": INTERNAL,
    "run.requeued_manually": INTERNAL,
    # ---- v15 信令族（decider/loop：session.<signal>）----
    "session.defer": PUBLIC,
    "session.clarify": PUBLIC,
    "session.reject": PUBLIC,
    "session.report": PUBLIC,
}

# public 视图里一律剥掉的字段（不管哪种 kind）
INTERNAL_FIELDS = frozenset({
    "worker_id", "lease_owner_was", "outbox_job_id", "tool_call_id", "prompt", "prompt_tokens",
    "tokens", "tokens_in", "tokens_out", "purpose", "request_json", "raw", "idempotency_key",
    "attempts", "actor", "halt_reason",
})

# 某些 kind 的 public 视图只留白名单字段（比黑名单更严）
PUBLIC_ALLOWLIST: dict[str, frozenset[str]] = {
    "tool.result": frozenset({"tool", "ok", "replayed", "error"}),
    "run.degraded": frozenset({"reason", "limit", "at", "tool", "tier"}),
}


def layer_of(kind: str) -> str:
    return KIND_LAYER.get(kind, INTERNAL)


def public_view(kind: str, payload: dict[str, Any] | None) -> dict[str, Any] | None:
    """返回可外推的 payload；None = 这条事件不推（internal/audit/未登记）。"""
    if layer_of(kind) != PUBLIC:
        return None
    data = dict(payload or {})
    allow = PUBLIC_ALLOWLIST.get(kind)
    if allow is not None:
        data = {k: v for k, v in data.items() if k in allow}
    return {k: v for k, v in data.items() if k not in INTERNAL_FIELDS}


def registered_kinds() -> frozenset[str]:
    return frozenset(KIND_LAYER)
