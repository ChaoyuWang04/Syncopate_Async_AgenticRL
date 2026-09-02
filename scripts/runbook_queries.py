#!/usr/bin/env python3
"""K11-2 · Runbook 六张卡的"05 第一步看什么"——每条查询**真跑一遍**（命令有效性验证，K11 门槛②）。

  python scripts/runbook_queries.py            # 逐卡打印读数；任一查询失败 ⇒ 退出码 1
读数来自表/端点，不许口算；卡片正文在 docs/syncopate/30-serving-release-checklist-and-runbook.md。
"""
from __future__ import annotations

import asyncio
import sys

from syncopate.runtime import metrics
from syncopate.runtime.db import Database

CARDS: dict[str, list[tuple[str, str]]] = {
    "01 queue lag 持续升高": [
        ("outbox pending / 最老 pending 秒", "SELECT count(*), COALESCE(max(extract(epoch FROM now()-created_at)),0) FROM outbox_jobs WHERE status='pending'"),
        ("queued run 数 / 最老 queued 秒", "SELECT count(*), COALESCE(max(extract(epoch FROM now()-created_at)),0) FROM agent_runs WHERE status='queued'"),
        ("dispatched 但未 started 的 run（消息丢了？）", "SELECT count(*) FROM agent_runs r WHERE r.status='queued' AND NOT EXISTS (SELECT 1 FROM outbox_jobs o WHERE o.payload->>'run_id'=r.run_id AND o.status='pending')"),
        ("running 数 / 活 lease 数", "SELECT count(*), count(*) FILTER (WHERE lease_expires_at >= now()) FROM agent_runs WHERE status='running'"),
    ],
    "02 卡死 run": [
        ("running ∧ lease 过期", "SELECT count(*) FROM agent_runs WHERE status='running' AND lease_expires_at < now()"),
        ("waiting_for_user 超 6h", "SELECT count(*) FROM agent_runs WHERE status='waiting_for_user' AND updated_at < now()-interval '6 hours'"),
        ("stuck_queued 告警未消（事件在、仍 queued）", "SELECT count(*) FROM run_events e JOIN agent_runs r ON r.org_id=e.org_id AND r.run_id=e.run_id WHERE e.kind='run.stuck_queued' AND r.status='queued'"),
        ("attempts 已到上限的 running", "SELECT count(*) FROM agent_runs WHERE status='running' AND attempts >= 3"),
    ],
    "03 写工具报错": [
        ("24h 写调用 失败/总", "SELECT count(*) FILTER (WHERE status IN ('failed','response_lost')), count(*) FROM tool_calls WHERE side_effect AND blocked_by IS NULL AND created_at > now()-interval '24 hours'"),
        ("response_lost 开放行（对账队列）", "SELECT count(*) FROM tool_calls WHERE side_effect AND status='response_lost'"),
        ("duplicate_prevented 总数（兜底生效次数）", "SELECT count(*) FROM tool_calls WHERE status='skipped_duplicate'"),
        ("按 error_json.code 分组（24h）", "SELECT error_json->>'code' AS code, count(*) FROM tool_calls WHERE created_at > now()-interval '24 hours' AND error_json IS NOT NULL GROUP BY 1 ORDER BY 2 DESC LIMIT 8"),
        ("死信（病历）", "SELECT count(*) FROM dead_letter_jobs WHERE reprocessed_at IS NULL"),
    ],
    "04 SSE 断线/时间线缺口": [
        ("seq 有洞的 run（分配器失效的信号）", "SELECT count(*) FROM (SELECT org_id, run_id, count(*) AS n, max(seq) AS m FROM run_events GROUP BY 1,2 HAVING count(*) <> max(seq)) x"),
        ("last_seq 与事件数不一致的 run", "SELECT count(*) FROM agent_runs r WHERE r.last_seq <> (SELECT count(*) FROM run_events e WHERE e.org_id=r.org_id AND e.run_id=r.run_id)"),
        ("终态但无终态事件的 run（挂死流的来源）", "SELECT count(*) FROM agent_runs r WHERE r.status IN ('succeeded','failed','cancelled') AND NOT EXISTS (SELECT 1 FROM run_events e WHERE e.org_id=r.org_id AND e.run_id=r.run_id AND e.kind IN ('run.completed','run.failed','run.cancelled'))"),
    ],
    "05 版本迁移失败": [
        ("alembic 版本", "SELECT version_num FROM alembic_version"),
        ("checkpoint 版本分布", "SELECT COALESCE(state->>'v','<none>') AS v, count(*) FROM checkpoints GROUP BY 1"),
        ("run 级版本分布（contract/prompt）", "SELECT contract_version, prompt_version, count(*) FROM agent_runs GROUP BY 1,2 ORDER BY 3 DESC LIMIT 5"),
        ("manual_review 事件（版本网关拒绝的）", "SELECT count(*) FROM run_events WHERE kind='run.manual_review' AND payload->>'reason'='unsupported_checkpoint_version'"),
    ],
    "06 token 消耗异常": [
        ("今日 org 用量 Top5", "SELECT org_id, sum(tokens_in+tokens_out) AS tokens, count(DISTINCT run_id) AS runs FROM usage_records WHERE day=CURRENT_DATE GROUP BY 1 ORDER BY 2 DESC LIMIT 5"),
        ("预算超限转 waiting 的 run", "SELECT count(*) FROM agent_runs WHERE budget_exceeded_at IS NOT NULL"),
        ("单 run token Top5（24h）", "SELECT run_id, sum(tokens_in+tokens_out) AS t FROM usage_records WHERE created_at > now()-interval '24 hours' GROUP BY 1 ORDER BY 2 DESC LIMIT 5"),
        ("org 预算配置", "SELECT org_id, daily_tokens, warn_ratio FROM org_budgets ORDER BY 1 LIMIT 10"),
    ],
}


async def main() -> int:
    db = Database()
    await db.connect(max_size=2)
    failed = 0
    try:
        for card, queries in CARDS.items():
            print(f"\n== RUNBOOK {card} ==")
            for label, sql in queries:
                try:
                    async with db.tx() as conn:
                        rows = await conn.fetch(sql)
                    vals = [tuple(r.values()) for r in rows][:5]
                    print(f"  ✅ {label}: {vals}")
                except Exception as exc:                 # noqa: BLE001
                    failed += 1
                    print(f"  🔴 {label}: {exc!r}")
        snap = await metrics.snapshot(db)
        print(f"\n== /metrics 快照关键项 == stuck_runs={snap['stuck_runs']} oldest_queued={snap['oldest_queued_run_age_s']:.0f}s "
              f"dead_letter={snap['dead_letter_open']} response_lost={snap['response_lost_open']} dup_prevented={snap['duplicate_prevented_total']}")
        for al in metrics.alerts(snap):
            print(f"[alert] {al['alert']}={al['value']} → {al['runbook']}")
    finally:
        await db.close()
    print(f"\n[runbook-queries] {'✅ 全部有效' if not failed else f'🔴 {failed} 条失败'}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
