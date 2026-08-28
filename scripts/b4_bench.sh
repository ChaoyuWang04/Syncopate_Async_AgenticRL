#!/usr/bin/env bash
# E32 S0.2 · 单臂编排：门禁→起服务→暖机→双轨压测→机读账本→拆服务等显存归还。
#
# 用法① 自起单引擎（S0/S1）：
#   scripts/b4_bench.sh <arm_name> [-- <额外 vllm 旗子...>]
# 用法② 压外部已起拓扑（S2 router/dp）：
#   NO_SERVE=1 BASE_URL=http://127.0.0.1:8100 METRICS_URLS=http://127.0.0.1:8101,... \
#     scripts/b4_bench.sh <arm_name>
#
# 双轨口径（E32 §5）：random 轨 = vllm bench serve in4200/out650（与 E19-c 逐字段可比，
# 打 sft-base）；trace 轨 = b4_replay 真实 512 条（打 candidate=生产 LoRA 口径，
# cache/路由结论只认这轨）。暖机固定 8 条低并发（各臂同罚，量的是臂内共享不是跨跑余温）。
# 可调：CONC(默认32) NUM_PROMPTS(默认48) TRACE_MODEL(默认candidate) RAND_MODEL(默认sft-base)
#
# 产物：logs/b4/<arm>/{vllm.log,bench_random.log,trace.json,arm.json}
set -u
cd "$(dirname "$0")/.."
set -a; . /workspace/.env; set +a
source .venv/bin/activate

ARM=${1:?用法: b4_bench.sh <arm_name> [-- vllm flags]}; shift
EXTRA=()
[ "${1:-}" = "--" ] && { shift; EXTRA=("$@"); }
D=logs/b4/$ARM; mkdir -p "$D"
CONC=${CONC:-32}; NUM_PROMPTS=${NUM_PROMPTS:-48}
TRACE_MODEL=${TRACE_MODEL:-candidate}; RAND_MODEL=${RAND_MODEL:-sft-base}
BASE_URL=${BASE_URL:-http://127.0.0.1:8100}
METRICS_URLS=${METRICS_URLS:-$BASE_URL}
MODEL=models/Qwen3-4B-sft-v13r2-e1
ADAPTER=checkpoints/grpo/cand_v13r2_e1/adapter_global_step_25
say() { echo "[B4 $(date +%H:%M:%S)] $*" | tee -a "$D/progress.log"; }

SRV=""
if [ -z "${NO_SERVE:-}" ]; then
  # ⚠️ 门禁等待期间绝不写 logs/（否则自己的心跳把静默期永久续住——08-27 rl_guard 同款坑）
  until GPUS="${SERVE_GPU:-0}" bash scripts/gpu_gate.sh >/dev/null 2>&1; do echo "[B4] gpu_gate 未过，等 60s" >&2; sleep 60; done
  say "起服务 arm=$ARM extra=(${EXTRA[*]:-})"
  CUDA_VISIBLE_DEVICES=${SERVE_GPU:-0} vllm serve "$MODEL" \
    --served-model-name sft-base \
    --enable-lora --lora-modules candidate="$ADAPTER" \
    --max-lora-rank 32 --max-model-len 14336 --kv-cache-dtype fp8 \
    --host 127.0.0.1 --port 8100 "${EXTRA[@]}" > "$D/vllm.log" 2>&1 &
  SRV=$!
  ok=0
  for i in $(seq 1 90); do
    sleep 7
    kill -0 "$SRV" 2>/dev/null || break
    curl -sf http://127.0.0.1:8100/health >/dev/null 2>&1 && { ok=1; break; }
  done
  if [ "$ok" != 1 ]; then
    alive=已死; kill -0 "$SRV" 2>/dev/null && alive=活着未就绪
    say "🔴 服务没起来（进程 $alive），vllm.log 尾部："
    tail -5 "$D/vllm.log" | tee -a "$D/progress.log"; kill "$SRV" 2>/dev/null; exit 1
  fi
  { grep -m2 -iE "model weights take|weights.*GiB" "$D/vllm.log"
    grep -m2 -iE "GPU KV cache size|kv cache.*tokens" "$D/vllm.log"; } | tee -a "$D/progress.log"
fi

say "暖机 8 条（并发 4）"
.venv/bin/python scripts/b4_replay.py --base-url "$BASE_URL" --model "$TRACE_MODEL" \
  --n 8 --concurrency 4 > "$D/warmup.log" 2>&1 || { say "🔴 暖机失败"; tail -5 "$D/warmup.log"; }

say "random 轨（bench serve in4200/out650 ×$NUM_PROMPTS，model=$RAND_MODEL）"
timeout 1800 vllm bench serve \
  --backend openai --base-url "$BASE_URL" --endpoint /v1/completions \
  --model "$RAND_MODEL" --tokenizer "$MODEL" \
  --dataset-name random --random-input-len 4200 --random-output-len 650 \
  --num-prompts "$NUM_PROMPTS" --ignore-eos > "$D/bench_random.log" 2>&1 \
  || say "⚠️ bench serve 非零退出"
grep -E "Median TTFT|P99 TTFT|Median TPOT|P99 TPOT|Output token throughput|Request throughput|Benchmark duration" \
  "$D/bench_random.log" | sed 's/^/  random /' | tee -a "$D/progress.log"

say "trace 轨（真实 512 条 ×并发 $CONC，model=$TRACE_MODEL）"
.venv/bin/python scripts/b4_replay.py --base-url "$BASE_URL" --model "$TRACE_MODEL" \
  --concurrency "$CONC" --metrics-urls "$METRICS_URLS" \
  --out "$D/trace.json" >> "$D/progress.log" 2>&1 || say "🔴 trace 轨有失败请求，看 trace.json"
.venv/bin/python - "$D" "$ARM" <<'EOF'
import json, re, sys
d, arm = sys.argv[1], sys.argv[2]
out = {"arm": arm}
try:
    out["trace"] = json.load(open(f"{d}/trace.json"))["summary"]
except Exception as e:
    out["trace_error"] = str(e)
rand = {}
try:
    txt = open(f"{d}/bench_random.log").read()
    for label, pat in [("ttft_median_ms", r"Median TTFT \(ms\):\s+([\d.]+)"),
                      ("ttft_p99_ms", r"P99 TTFT \(ms\):\s+([\d.]+)"),
                      ("tpot_median_ms", r"Median TPOT \(ms\):\s+([\d.]+)"),
                      ("tpot_p99_ms", r"P99 TPOT \(ms\):\s+([\d.]+)"),
                      ("output_tok_per_s", r"Output token throughput \(tok/s\):\s+([\d.]+)"),
                      ("req_per_s", r"Request throughput \(req/s\):\s+([\d.]+)"),
                      ("duration_s", r"Benchmark duration \(s\):\s+([\d.]+)")]:
        m = re.search(pat, txt)
        if m: rand[label] = float(m.group(1))
except Exception as e:
    rand["error"] = str(e)
out["random"] = rand
json.dump(out, open(f"{d}/arm.json", "w"), ensure_ascii=False, indent=2)
print(json.dumps(out, ensure_ascii=False))
EOF
say "账本 $D/arm.json"

if [ -n "$SRV" ]; then
  say "拆服务"
  kill "$SRV" 2>/dev/null; wait "$SRV" 2>/dev/null
  for i in $(seq 1 30); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${SERVE_GPU:-0}" | sort -n | tail -1)
    [ "$used" -lt 2000 ] && break
    [ "$i" = 12 ] && { say "⚠️ 60s 未归还，补杀 EngineCore"; pkill -f 'VLLM::EngineCore' 2>/dev/null; }
    sleep 5
  done
  say "显存归还（${used:-?} MiB）"
fi
