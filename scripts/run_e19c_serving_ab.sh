#!/usr/bin/env bash
# E19-c · serving 侧量化三臂 A/B（2026-08-21，Chaoyu 点头：先做最简单的）
# 臂（唯一变量=量化旗子，其余逐字段抄生产命令 logs/runtime/start_vllm.sh）：
#   bf16      生产现状
#   fp8kv     + --kv-cache-dtype fp8            （KV 池容量 ×2 的免费杠杆）
#   fp8w_kv   + --quantization fp8 + fp8 KV     （权重也压半，在线动态量化）
# 每臂量四类：①装载显存/KV 池容量（vLLM 日志 + nvidia-smi）②单流 TTFT/TPOT
# （measure_tpot.py，短/长 prompt）③并发压测（vllm bench serve，形状按真实负载
#  in≈4200/out≈650）④质量配对（评测臂在最后，4 卡 eval bf16 vs fp8，compare 定差）
cd "$(dirname "$0")/.."
set -a; . /workspace/.env; set +a
P=logs/e19c_progress.log
say() { echo "[E19C $(date +%H:%M:%S)] $*" | tee -a "$P"; }
MODEL=models/Qwen3-4B-sft-v13r2-e1
ADAPTER=checkpoints/grpo/cand_v13r2_e1/adapter_global_step_25

serve_arm() {
  local name=$1; shift
  say "ARM ${name} 起服务（$*）"
  until GPUS=0 bash scripts/gpu_gate.sh >/dev/null 2>&1; do sleep 60; done
  CUDA_VISIBLE_DEVICES=0 .venv/bin/vllm serve "$MODEL" \
    --served-model-name sft-base \
    --enable-lora --lora-modules candidate="$ADAPTER" \
    --max-lora-rank 32 --max-model-len 14336 \
    --host 127.0.0.1 --port 8100 "$@" > "logs/e19c_vllm_${name}.log" 2>&1 &
  local SRV=$!
  local ok=0
  for i in $(seq 1 60); do
    sleep 7
    if ! kill -0 $SRV 2>/dev/null; then break; fi
    curl -sf http://127.0.0.1:8100/health >/dev/null 2>&1 && { ok=1; break; }
  done
  if [ "$ok" != 1 ]; then
    say "🔴 ARM ${name} 服务没起来（进程$(kill -0 $SRV 2>/dev/null && echo 活着未就绪 || echo 已死)）"
    kill $SRV 2>/dev/null; sleep 10; return 1
  fi
  { echo "===== ${name} 装载账本 ====="
    grep -m2 -iE "model weights take|weights.*GiB" "logs/e19c_vllm_${name}.log"
    grep -m2 -iE "GPU KV cache size|kv cache.*tokens" "logs/e19c_vllm_${name}.log"
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | head -1 \
      | sed 's/^/  nvidia-smi GPU0: /'
  } | tee -a "$P"
  say "ARM ${name} 单流 TTFT/TPOT（measure_tpot）"
  timeout 600 .venv/bin/python scripts/measure_tpot.py \
    > "logs/e19c_tpot_${name}.log" 2>&1 || say "⚠️ ${name} measure_tpot 非零退出"
  say "ARM ${name} 并发压测（in4200/out650 ×48）"
  timeout 900 .venv/bin/vllm bench serve \
    --backend openai --base-url http://127.0.0.1:8100 --endpoint /v1/completions \
    --model sft-base --tokenizer "$MODEL" \
    --dataset-name random --random-input-len 4200 --random-output-len 650 \
    --num-prompts 48 --ignore-eos > "logs/e19c_bench_${name}.log" 2>&1 \
    || say "⚠️ ${name} bench serve 非零退出"
  grep -E "Median TTFT|P99 TTFT|Median TPOT|P99 TPOT|Median ITL|Output token throughput|Request throughput|Benchmark duration" \
    "logs/e19c_bench_${name}.log" | sed "s/^/  ${name} /" | tee -a "$P"
  kill $SRV 2>/dev/null; wait $SRV 2>/dev/null
  # 等显存真归还（wait_for_gpu 的坑：日志完了进程还在收尾）
  for i in $(seq 1 30); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "$used" -lt 2000 ] && break; sleep 5
  done
  say "ARM ${name} ✅ 服务臂完成"
}

eval_arm() {
  local name=$1 quant=$2 kv=$3
  say "EVAL ${name} 开始（quant=${quant:-无} kv=${kv:-auto}，4 卡）"
  until GPUS=0,1,2,3 bash scripts/gpu_gate.sh >/dev/null 2>&1; do sleep 60; done
  SYNCOPATE_EVAL_QUANT="$quant" SYNCOPATE_EVAL_KV_DTYPE="$kv" \
  MODEL="$MODEL" timeout 2400 bash scripts/eval_parallel.sh \
    "$ADAPTER" "_audit/e19c_eval_${name}.json" 4 \
    > "logs/e19c_eval_${name}.log" 2>&1
  if [ ! -f "_audit/e19c_eval_${name}.json.done" ]; then
    say "🔴 EVAL ${name} 无 done 标记"; return 1
  fi
  # 判据行核对：量化臂必须真的打了 [eval-quant]（机制在但没接上的守卫）
  if [ -n "$quant$kv" ]; then
    local n
    n=$(grep -rc "eval-quant" logs/eval_e19c_eval_${name}/ 2>/dev/null | awk -F: '{s+=$2} END{print s+0}')
    say "EVAL ${name} [eval-quant] 判据行 ×${n}（应=4 分片）"
  fi
  say "EVAL ${name} ✅"
}

say "E19-c 开始：serving 三臂 → 质量两臂"
serve_arm bf16
serve_arm fp8kv --kv-cache-dtype fp8
serve_arm fp8w_kv --quantization fp8 --kv-cache-dtype fp8
eval_arm bf16 "" ""
eval_arm fp8 fp8 fp8
say "质量配对（bf16 vs fp8，同尺子同 adapter）："
.venv/bin/python -m syncopate.train.compare _audit/e19c_eval_bf16.json _audit/e19c_eval_fp8.json 2>&1 \
  | grep -E "配对差值|结论|该 defer|均值" | tee -a "$P"
say "🏁 E19-c 全部结束"
touch logs/e19c_DONE
