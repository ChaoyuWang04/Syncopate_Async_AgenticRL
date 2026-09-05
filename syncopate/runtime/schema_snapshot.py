#!/usr/bin/env python3
"""schema 快照：把库里 public schema 的 表/列/约束/索引/触发器/函数 导成**规范文本**。

用途（K2，2026-09-02，28 P-04/P-05）：
  迁移链是唯一真相，但没有 ORM 就没有 autogenerate，漂移没人报。
  ⇒ 判据形状用守则①「两个东西应当相同」：
        干净库 `alembic upgrade head` 导出的快照  ==  仓库里提交的快照文件
        当前库导出的快照                          ==  仓库里提交的快照文件（有人手改库就红）

  python -m syncopate.runtime.schema_snapshot                 # 打印
  python -m syncopate.runtime.schema_snapshot --write         # 写 syncopate/runtime/schema.snapshot.txt
  python -m syncopate.runtime.schema_snapshot --check         # 与快照文件比对，不一致退出码 1（并打 diff）

连接：SYNCOPATE_PG_DSN（与 db.py 同一变量）；`--dsn` 可覆盖（测试用临时库）。
排除 alembic_version（它是迁移工具的账本，不是我们的 schema）。
"""
from __future__ import annotations

import argparse
import asyncio
import difflib
import os
import sys
from pathlib import Path

import asyncpg

SNAPSHOT = Path(__file__).resolve().parent / "schema.snapshot.txt"
DEFAULT_DSN = "postgresql://syncopate:syncopate@127.0.0.1:5432/syncopate"


async def dump(dsn: str) -> str:
    conn = await asyncpg.connect(dsn)
    try:
        out: list[str] = []
        tables = [r["table_name"] for r in await conn.fetch(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema='public' AND table_type='BASE TABLE' "
            "AND table_name <> 'alembic_version' ORDER BY table_name")]
        for t in tables:
            out.append(f"TABLE {t}")
            for c in await conn.fetch(
                    "SELECT column_name, data_type, is_nullable, column_default "
                    "FROM information_schema.columns WHERE table_schema='public' "
                    "AND table_name=$1 ORDER BY ordinal_position", t):
                out.append(f"  COL {c['column_name']} {c['data_type']} "
                           f"null={c['is_nullable']} default={c['column_default']}")
            for k in await conn.fetch(
                    "SELECT conname, pg_get_constraintdef(oid) AS def FROM pg_constraint "
                    "WHERE conrelid = $1::regclass ORDER BY conname", t):
                out.append(f"  CONSTRAINT {k['conname']} {k['def']}")
            for i in await conn.fetch(
                    "SELECT indexname, indexdef FROM pg_indexes WHERE schemaname='public' "
                    "AND tablename=$1 ORDER BY indexname", t):
                out.append(f"  INDEX {i['indexname']} {i['indexdef']}")
            for g in await conn.fetch(
                    "SELECT tgname, pg_get_triggerdef(oid) AS def FROM pg_trigger "
                    "WHERE tgrelid = $1::regclass AND NOT tgisinternal ORDER BY tgname", t):
                out.append(f"  TRIGGER {g['tgname']} {g['def']}")
        for f in await conn.fetch(
                "SELECT p.proname, pg_get_functiondef(p.oid) AS def FROM pg_proc p "
                "JOIN pg_namespace n ON n.oid = p.pronamespace WHERE n.nspname='public' "
                "ORDER BY p.proname"):
            out.append(f"FUNCTION {f['proname']}")
            out.extend("  " + line for line in f["def"].strip().splitlines())
        return "\n".join(out) + "\n"
    finally:
        await conn.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", default=os.environ.get("SYNCOPATE_PG_DSN", DEFAULT_DSN))
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    text = asyncio.run(dump(a.dsn))
    if a.write:
        SNAPSHOT.write_text(text, encoding="utf-8")
        print(f"[schema-snapshot] wrote {SNAPSHOT} ({len(text.splitlines())} lines)")
        return 0
    if a.check:
        want = SNAPSHOT.read_text(encoding="utf-8") if SNAPSHOT.exists() else ""
        if text == want:
            print("[schema-snapshot] ✅ live schema == snapshot")
            return 0
        sys.stdout.writelines(difflib.unified_diff(
            want.splitlines(True), text.splitlines(True), "snapshot", "live"))
        print("[schema-snapshot] 🔴 drift", file=sys.stderr)
        return 1
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
