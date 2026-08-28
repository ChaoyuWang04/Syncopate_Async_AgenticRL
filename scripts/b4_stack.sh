#!/usr/bin/env bash
# E32 · 全栈起停（goodput@SLO 与 runtime_loadtest 的前置：PG + API:8000 + worker）。
#
#   bash scripts/b4_stack.sh start [worker并发]     # 默认 64（goodput 量 serving 层，
#                                                   #   编排层先不设瓶颈——槽位数记进产物）
#   bash scripts/b4_stack.sh stop
#
# ⚠️ worker 用 **org_acme**（loadtest 的 org；org_demo 归 chatbox 常驻、acme/globex
#   归测试——见 09 §0 抢单前科，这里是压测专用 worker）。vLLM(:8100) 由压测方自起。
set -u
cd "$(dirname "$0")/.."
set -a; . /workspace/.env; set +a
source .venv/bin/activate
D=logs/b4/stack; mkdir -p "$D"

case "${1:?start|stop}" in
stop)
  [ -f "$D/pids" ] && while read -r p; do kill "$p" 2>/dev/null; done < "$D/pids"
  rm -f "$D/pids"; echo "[stack] 已停"; exit 0 ;;
start) ;;
*) echo "用法: b4_stack.sh start|stop"; exit 1 ;;
esac

CONC=${2:-64}
pg_isready -h 127.0.0.1 -p 5432 >/dev/null 2>&1 || { echo "🔴 PG 没起（先 pg_bootstrap.sh）"; exit 1; }
: > "$D/pids"

uvicorn syncopate.runtime.api:app --host 127.0.0.1 --port 8000 > "$D/api.log" 2>&1 &
echo $! >> "$D/pids"
for _ in $(seq 1 30); do
  sleep 1; curl -sf http://127.0.0.1:8000/healthz >/dev/null 2>&1 && break
done
curl -sf http://127.0.0.1:8000/healthz >/dev/null || { echo "🔴 API 没起来："; tail -5 "$D/api.log"; exit 1; }
echo "[stack] API :8000 就绪"

SYNCOPATE_DECIDER_URL=http://127.0.0.1:8100 \
  python -m syncopate.runtime.worker --org-id org_acme --worker-id b4-loadtest \
  --concurrency "$CONC" > "$D/worker.log" 2>&1 &
echo $! >> "$D/pids"
sleep 3
kill -0 "$(tail -1 "$D/pids")" 2>/dev/null || { echo "🔴 worker 启动即死："; tail -5 "$D/worker.log"; exit 1; }
echo "[stack] worker org_acme 并发=$CONC 就绪（vLLM :8100 由压测方保证在跑）"
