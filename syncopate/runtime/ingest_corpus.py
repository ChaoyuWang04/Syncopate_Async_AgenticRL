"""把 RAG 语料入库到 runtime 的 PG（政策条款 + 复盘结论）。

    data/external/policy_corpus.seed.json
                    ↓
    PG: policy_clauses / insights        ← runtime 检索服务读这里

★ 和 `ingest_external.py` 的分工：那个管**结构化**的三样（安全线 / 素材标签 /
时令日历，设计 §13 的第 1、2、8、9 项），产物是 `ingested.json`；
这个管**半结构化 + 非结构化**（第 3–6 项），产物在数据库里。

★ 幂等：按 (scope, clause_id) upsert。语料是**增量追加**的，重跑不会重复。

    python -m syncopate.runtime.ingest_corpus                      # 入库种子语料（global 作用域）
    python -m syncopate.runtime.ingest_corpus --scope org_acme --from <file>   # 某 org 的私有 SOP
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from syncopate.runtime.db import Database
from syncopate.runtime.retrieval import upsert_insights, upsert_policy_clauses

DEFAULT = Path("data/external/policy_corpus.seed.json")


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="src", type=Path, default=DEFAULT)
    ap.add_argument("--scope", default="global",
                    help="'global' = 所有 org 都能查到（平台政策）；给 org_id = 该 org 私有（内部 SOP）")
    args = ap.parse_args()

    payload = json.loads(args.src.read_text(encoding="utf-8"))
    db = Database()
    await db.connect(max_size=2)
    try:
        n_p = await upsert_policy_clauses(db, payload.get("policy_clauses", []),
                                          scope=args.scope)
        n_i = await upsert_insights(db, payload.get("insights", []), scope=args.scope)
    finally:
        await db.close()
    print(f"[ingest-corpus] scope={args.scope} 政策条款 {n_p} 条 · 复盘结论 {n_i} 条 ← {args.src}")


if __name__ == "__main__":
    asyncio.run(main())
