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
  rm -f "$D/pids"
  # 等 :8000 真归还（uvicorn 父进程收尾要几秒；立刻重启会 bind 失败——08-28 r3 实录）
  for _ in $(seq 1 15); do ss -ltn 2>/dev/null | grep -q ':8000 ' || break; sleep 1; done
  echo "[stack] 已停"; exit 0 ;;
start) ;;
*) echo "用法: b4_stack.sh start|stop"; exit 1 ;;
esac

CONC=${2:-64}
# ⚠️ 不用 pg_isready——它不在 PATH（PG 装在 /workspace/tools/postgres 下），检查器指向
#   不存在的工具会把活着的 PG 判死（08-28 实录）。TCP 探活即可。
(exec 3<>/dev/tcp/127.0.0.1/5432) 2>/dev/null || { echo "🔴 PG 没起（先 pg_bootstrap.sh）"; exit 1; }
exec 3>&- 3<&- 2>/dev/null || true
: > "$D/pids"

# B-5 S1+S2（扩池×多进程必须连着落——S1 单独验证实测负收益：10 条连接原是天然限流阀，
# 单进程下拆阀=IO 回调全砸一个 GIL，llm/db_tx 反涨；E33 §6 记档）。
# 连接预算：4 API×12 + N worker×32 ≤ 300×0.7。
API_WORKERS="${B4_API_WORKERS:-4}"
N_WORKERS="${B4_WORKERS:-4}"
export SYNCOPATE_WORKER_DB_POOL="${SYNCOPATE_WORKER_DB_POOL:-40}"
export SYNCOPATE_API_DB_POOL="${SYNCOPATE_API_DB_POOL:-12}"
uvicorn syncopate.runtime.api:app --host 127.0.0.1 --port 8000 \
  --workers "$API_WORKERS" > "$D/api.log" 2>&1 &
echo $! >> "$D/pids"
for _ in $(seq 1 30); do
  sleep 1; curl -sf http://127.0.0.1:8000/healthz >/dev/null 2>&1 && break
done
curl -sf http://127.0.0.1:8000/healthz >/dev/null || { echo "🔴 API 没起来："; tail -5 "$D/api.log"; exit 1; }
echo "[stack] API :8000 就绪（$API_WORKERS 进程）"

# 压测 org 日预算抬 1000×（默认 10M micros 在 ~300 run 处刷爆 ⇒ 其后全部秒失败，
# goodput 阶梯必须先解除这个编排层瓶颈；生产 org 的默认值不动）
PER=$(( (CONC + N_WORKERS - 1) / N_WORKERS ))
for w in $(seq 1 "$N_WORKERS"); do
  # B4_DECIDER_MODE=direct ⇒ worker w 直连引擎 810w（"理想路由器"对照臂，量 router 自身开销）
  if [ "${B4_DECIDER_MODE:-router}" = "direct" ]; then
    DURL="http://127.0.0.1:$((8100 + w))"
  else
    DURL="${B4_DECIDER_URL:-http://127.0.0.1:8100}"
  fi
  SYNCOPATE_DECIDER_URL="$DURL" \
    python -m syncopate.runtime.worker --org-id org_acme --worker-id "b4-loadtest-$w" \
    --concurrency "$PER" --daily-cost-cap-micros "${B4_COST_CAP:-10000000000}" \
    > "$D/worker_$w.log" 2>&1 &
  echo $! >> "$D/pids"
done
sleep 3
alive=0
for p in $(tail -n "$N_WORKERS" "$D/pids"); do kill -0 "$p" 2>/dev/null && alive=$((alive+1)); done
[ "$alive" = "$N_WORKERS" ] || { echo "🔴 worker 启动即死（$alive/$N_WORKERS 活）："; tail -5 "$D/worker_1.log"; exit 1; }
# 分账采集向后兼容：verify 链读 stack/worker.log ⇒ 聚合软链
cat > "$D/collect_worker_logs.sh" <<'EOS'
cat "$(dirname "$0")"/worker_*.log
EOS
echo "[stack] $N_WORKERS 个 worker（各并发 $PER）就绪（vLLM :8100 由压测方保证在跑）"
