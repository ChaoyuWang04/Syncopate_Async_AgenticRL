#!/usr/bin/env bash
# E32 S2 · N×DP 副本 + 路由器编排（起/停）。
#
#   bash scripts/b4_serve_4x.sh start [N] [affinity|rr] [-- 额外 vllm 旗子]
#   bash scripts/b4_serve_4x.sh stop
#
# 引擎 i 在 GPU i / 端口 810(i+1)，taskset 绑本 GPU 的 NUMA CPU（无 numactl，
# 亲和近似；GPU0/1 共 NUMA3 ⇒ 拆 72-95 / 168-191）。路由器监听 :8100。
# PID 记 logs/b4/serve4x/pids，stop 全杀并等显存归还。
set -u
cd "$(dirname "$0")/.."
set -a; . /workspace/.env; set +a
source .venv/bin/activate

D=logs/b4/serve4x; mkdir -p "$D"
MODEL=models/Qwen3-4B-sft-v13r2-e1
ADAPTER=checkpoints/grpo/cand_v13r2_e1/adapter_global_step_25
CPUS=(72-95 168-191 48-71,144-167 0-23,96-119)   # nvidia-smi topo -m 实读（08-28 新机）
say() { echo "[B4x $(date +%H:%M:%S)] $*" | tee -a "$D/progress.log"; }

case "${1:?start|stop}" in
stop)
  [ -f "$D/pids" ] && while read -r p; do kill "$p" 2>/dev/null; done < "$D/pids"
  sleep 5; pkill -f 'VLLM::EngineCore' 2>/dev/null
  for i in $(seq 1 30); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
    [ "$used" -lt 2000 ] && break; sleep 5
  done
  say "已停（最大占用 ${used:-?} MiB）"; rm -f "$D/pids"; exit 0 ;;
start) shift ;;
*) echo "用法: b4_serve_4x.sh start|stop"; exit 1 ;;
esac

N=${1:-4}; [ "${1:-}" ] && shift
POLICY=${1:-affinity}; [ "${1:-}" ] && shift
EXTRA=(); [ "${1:-}" = "--" ] && { shift; EXTRA=("$@"); }

# ⚠️ 门禁等待期间绝不写 logs/（心跳会把静默期永久续住——08-27 rl_guard 同款坑）
until bash scripts/gpu_gate.sh >/dev/null 2>&1; do echo "[B4x] gpu_gate 未过，等 60s" >&2; sleep 60; done
: > "$D/pids"
BACKENDS=""
for i in $(seq 0 $((N - 1))); do
  port=$((8101 + i))
  say "起引擎 $i → GPU$i :$port cpus=${CPUS[$i]} extra=(${EXTRA[*]:-})"
  CUDA_VISIBLE_DEVICES=$i taskset -c "${CPUS[$i]}" vllm serve "$MODEL" \
    --served-model-name sft-base \
    --enable-lora --lora-modules candidate="$ADAPTER" \
    --max-lora-rank 32 --max-model-len 14336 --kv-cache-dtype fp8 \
    --host 127.0.0.1 --port "$port" "${EXTRA[@]}" > "$D/vllm_$i.log" 2>&1 &
  echo $! >> "$D/pids"
  BACKENDS="$BACKENDS${BACKENDS:+,}http://127.0.0.1:$port"
done

for i in $(seq 0 $((N - 1))); do
  port=$((8101 + i)); ok=0
  for _ in $(seq 1 90); do
    sleep 7; curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1 && { ok=1; break; }
  done
  [ "$ok" = 1 ] || { say "🔴 引擎 $i 没起来："; tail -5 "$D/vllm_$i.log"; bash "$0" stop; exit 1; }
  say "引擎 $i 就绪"
done

# 键=prompt[4409:4409+6144]：skip=全局公共前缀实测 4409 字符；窗口 6144=实测最小平衡窗
# （2048/4096 会塌到单副本——公共前缀后的模板段仍共享；6144 起 [140,146,98,128] 平衡且
#  零 case 分裂，E32 §7）
HASH_SKIP=${HASH_SKIP:-4409}; HASH_WINDOW=${HASH_WINDOW:-6144}
say "起路由器 :8100 policy=$POLICY backends=$BACKENDS hash=[$HASH_SKIP:+$HASH_WINDOW]"
.venv/bin/python scripts/b4_router.py --port 8100 --policy "$POLICY" \
  --backends "$BACKENDS" --hash-skip "$HASH_SKIP" --hash-window "$HASH_WINDOW" > "$D/router.log" 2>&1 &
echo $! >> "$D/pids"
for _ in $(seq 1 20); do
  sleep 2; curl -sf http://127.0.0.1:8100/health >/dev/null 2>&1 && { say "路由器就绪"; break; }
done
echo "$BACKENDS" > "$D/backends"
say "✅ $N 引擎 + router($POLICY) 全就绪；压测用：NO_SERVE=1 BASE_URL=http://127.0.0.1:8100 METRICS_URLS=$BACKENDS scripts/b4_bench.sh <arm>"
