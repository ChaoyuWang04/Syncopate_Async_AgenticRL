#!/usr/bin/env bash
# K11-4 · 灾备演练（附录 A §3.3："没演练过的备份等于没有备份"）：在**干净目录**里从仓库重建 serving 数据层
# 全链并计 RTO：PG（新 PGDATA、备用端口）→ Redis（新目录、备用端口）→ alembic upgrade head → 快照核对 → 冒烟。
# ⚠️ 模型权重/HF 仓库那半（bases/、candidate）本机没有，不在本演练内（登记在 30 号 §4）。
#
#   bash scripts/serving/dr_drill.sh              # 全程约 30–60s；结束时打印 [dr-drill] RTO=<秒>
#   bash scripts/serving/dr_drill.sh --keep       # 演练后不销毁临时实例
set -euo pipefail
ENV="${SYNCOPATE_INFRA_ENV:-$HOME/Downloads/ENTER/envs/syncopate-infra}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PYTHON:-$ROOT/.venv/bin/python}"
WORK="$(mktemp -d /tmp/syncopate-dr.XXXXXX)"
PGPORT_DR=${PGPORT_DR:-5499}; REDIS_PORT_DR=${REDIS_PORT_DR:-6399}
T0=$(date +%s.%N)
step() { echo "=== $* ==="; }
cleanup() {
  if [[ "${1:-}" != "--keep" ]]; then
    "$ENV/bin/pg_ctl" -D "$WORK/pgdata" stop -m immediate >/dev/null 2>&1 || true
    "$ENV/bin/redis-cli" -p "$REDIS_PORT_DR" -a dr-pass --no-auth-warning shutdown nosave >/dev/null 2>&1 || true
    rm -rf "$WORK"
  fi
}
trap 'cleanup ${1:-}' EXIT

step "① PG：initdb + 起（端口 $PGPORT_DR，PGDATA $WORK/pgdata）"
PGDATA="$WORK/pgdata" PGPORT="$PGPORT_DR" LOGFILE="$WORK/pg.log" \
  PG_HOME="$ENV" PG_SHARE="$ENV/share/postgresql" PG_LIB="$ENV/lib" \
  SYNCOPATE_PG_DSN="postgresql://syncopate:syncopate@127.0.0.1:$PGPORT_DR/syncopate" \
  bash "$ROOT/scripts/serving/pg_bootstrap.sh" | tail -3
step "② Redis：起（端口 $REDIS_PORT_DR，目录 $WORK/redis）"
REDIS_HOME="$ENV" REDIS_DIR="$WORK/redis" REDIS_PORT="$REDIS_PORT_DR" REDIS_PASS=dr-pass \
  bash "$ROOT/scripts/serving/redis_bootstrap.sh" | tail -2
step "③ 迁移链 + 快照核对（pg_bootstrap 已做，二次确认）"
SYNCOPATE_PG_DSN="postgresql://syncopate:syncopate@127.0.0.1:$PGPORT_DR/syncopate" "$PY" -m alembic current | tail -1
step "④ 冒烟：建 run → 领取 → 事件流（不起 worker，直接用库 API）"
SYNCOPATE_PG_DSN="postgresql://syncopate:syncopate@127.0.0.1:$PGPORT_DR/syncopate" "$PY" - <<'PYEOF'
import asyncio, uuid
from syncopate.runtime.db import Database, create_run, claim_run, finish_run
async def main():
    db = Database(); await db.connect(max_size=2)
    org, run = f"org_dr_{uuid.uuid4().hex[:6]}", f"run_{uuid.uuid4().hex[:8]}"
    await create_run(db, org_id=org, run_id=run, user_message="灾备冒烟")
    assert await claim_run(db, worker_id="dr", org_id=org, run_id=run)
    await finish_run(db, org_id=org, run_id=run, status="succeeded", result={"ok": True})
    async with db.tx() as c:
        kinds = [r["kind"] for r in await c.fetch("SELECT kind FROM run_events WHERE org_id=$1 ORDER BY seq", org)]
        n_tables = await c.fetchval("SELECT count(*) FROM information_schema.tables WHERE table_schema='public' AND table_name<>'alembic_version'")
    assert kinds == ["run.created", "run.started", "run.completed"], kinds
    print(f"[dr-smoke] ✅ 表 {n_tables} 张 · 事件 {kinds}")
    await db.close()
asyncio.run(main())
PYEOF
T1=$(date +%s.%N)
RTO=$(python3 -c "print(f'{$T1-$T0:.1f}')")
echo "[dr-drill] ✅ 干净目录重建数据层完成 RTO=${RTO}s（PG+Redis+迁移+冒烟；模型权重不在此演练内）"
mkdir -p "$ROOT/_audit/serving_k11"
echo "$(date -Iseconds) RTO=${RTO}s work=$WORK pgport=$PGPORT_DR redisport=$REDIS_PORT_DR" >> "$ROOT/_audit/serving_k11/dr_drill_log.txt"
