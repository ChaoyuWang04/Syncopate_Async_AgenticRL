#!/usr/bin/env bash
# E19-c 第四臂 · NVFP4（W4A4）serving + 质量评测——接在三臂 A/B 之后自动跑
# 曲线定位：bf16 → fp8kv → fp8w_kv → nvfp4（显存/KV池/TTFT/TPOT/吞吐/任务分 全同尺子）
# ⚠️ 与前三臂的两个已知差异（记录在案，不静默）：
#   ① KV cache 无 FP4 选项（vLLM attention kernel 不吃）⇒ 本臂 = NVFP4 权重 + FP8 KV
#   ② LoRA 贴在量化基座上 vLLM 支持面不明 ⇒ serving 臂不挂 adapter（bench 打的是
#      sft-base 本体，与前三臂同对象）；eval 臂先试挂、失败则退到无 adapter 对
#      v13_sft_v13r2_e1_merged（同为无 adapter 的 bf16 读数）配对
cd "$(dirname "$0")/.."
set -a; . /workspace/.env; set +a
P=logs/e19c_progress.log
say() { echo "[NVFP4 $(date +%H:%M:%S)] $*" | tee -a "$P"; }
MODEL=models/Qwen3-4B-sft-v13r2-e1
QMODEL=models/Qwen3-4B-sft-v13r2-e1-NVFP4
ADAPTER=checkpoints/grpo/cand_v13r2_e1/adapter_global_step_25

say "等待三臂 A/B 收尾（logs/e19c_DONE）"
until [ -f logs/e19c_DONE ]; do sleep 60; done

# ⚠️ llmcompressor 只许住隔离 venv（2026-08-21 事故：装进生产 venv 会把 torch 2.9→2.13
#    整栈静默重解析，靠 uv.lock 才回滚回来）——量化产物是纯文件，serving 用主 venv 读
QPY=/workspace/venvs/quantize/bin/python
"$QPY" -c "import llmcompressor" 2>/dev/null || { say "🔴 隔离 venv 的 llmcompressor 没装好，退出"; exit 1; }

if [ ! -f "$QMODEL/config.json" ]; then
  say "离线量化开始（W4A4 · 校准=自家负载 128 条 · 隔离 venv）"
  until GPUS=0 bash scripts/gpu_gate.sh >/dev/null 2>&1; do sleep 60; done
  CUDA_VISIBLE_DEVICES=0 timeout 3600 "$QPY" scripts/quantize_nvfp4.py \
    > logs/e19c_quantize_nvfp4.log 2>&1 || { say "🔴 量化失败（logs/e19c_quantize_nvfp4.log）"; exit 1; }
  grep -E "\[verify\]" logs/e19c_quantize_nvfp4.log | tee -a "$P"
fi

say "ARM nvfp4 起服务（量化基座 + fp8 KV，无 adapter）"
until GPUS=0 bash scripts/gpu_gate.sh >/dev/null 2>&1; do sleep 60; done
CUDA_VISIBLE_DEVICES=0 .venv/bin/vllm serve "$QMODEL" \
  --served-model-name sft-base \
  --max-model-len 14336 --kv-cache-dtype fp8 \
  --host 127.0.0.1 --port 8100 > logs/e19c_vllm_nvfp4.log 2>&1 &
SRV=$!
ok=0
for i in $(seq 1 60); do
  sleep 7
  kill -0 $SRV 2>/dev/null || break
  curl -sf http://127.0.0.1:8100/health >/dev/null 2>&1 && { ok=1; break; }
done
if [ "$ok" != 1 ]; then
  say "🔴 ARM nvfp4 服务没起来（sm_120 的 NVFP4 kernel 支持面=探针结果之一，日志 logs/e19c_vllm_nvfp4.log）"
  kill $SRV 2>/dev/null
else
  { echo "===== nvfp4 装载账本 ====="
    grep -m2 -iE "model weights take|weights.*GiB" logs/e19c_vllm_nvfp4.log
    grep -m2 -iE "GPU KV cache size|kv cache.*tokens" logs/e19c_vllm_nvfp4.log
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | head -1 | sed 's/^/  nvidia-smi GPU0: /'
  } | tee -a "$P"
  say "ARM nvfp4 单流 TTFT/TPOT"
  timeout 600 .venv/bin/python scripts/measure_tpot.py > logs/e19c_tpot_nvfp4.log 2>&1 \
    || say "⚠️ nvfp4 measure_tpot 非零退出"
  say "ARM nvfp4 并发压测（in4200/out650 ×48）"
  timeout 900 .venv/bin/vllm bench serve \
    --backend openai --base-url http://127.0.0.1:8100 --endpoint /v1/completions \
    --model sft-base --tokenizer "$MODEL" \
    --dataset-name random --random-input-len 4200 --random-output-len 650 \
    --num-prompts 48 --ignore-eos > logs/e19c_bench_nvfp4.log 2>&1 \
    || say "⚠️ nvfp4 bench serve 非零退出"
  grep -E "Median TTFT|P99 TTFT|Median TPOT|P99 TPOT|Median ITL|Output token throughput|Request throughput|Benchmark duration" \
    logs/e19c_bench_nvfp4.log | sed 's/^/  nvfp4 /' | tee -a "$P"
  kill $SRV 2>/dev/null; wait $SRV 2>/dev/null
fi
for i in $(seq 1 30); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  [ "$used" -lt 2000 ] && break; sleep 5
done

say "EVAL nvfp4 开始（先试挂 adapter）"
until GPUS=0,1,2,3 bash scripts/gpu_gate.sh >/dev/null 2>&1; do sleep 60; done
MODEL="$QMODEL" timeout 2400 bash scripts/eval_parallel.sh \
  "$ADAPTER" "_audit/e19c_eval_nvfp4.json" 4 > logs/e19c_eval_nvfp4.log 2>&1
if [ ! -f _audit/e19c_eval_nvfp4.json.done ]; then
  say "⚠️ 挂 adapter 的 nvfp4 eval 未完成，退到无 adapter 臂"
  until GPUS=0,1,2,3 bash scripts/gpu_gate.sh >/dev/null 2>&1; do sleep 60; done
  MODEL="$QMODEL" timeout 2400 bash scripts/eval_parallel.sh \
    "" "_audit/e19c_eval_nvfp4_noadapter.json" 4 > logs/e19c_eval_nvfp4_na.log 2>&1
  if [ -f _audit/e19c_eval_nvfp4_noadapter.json.done ]; then
    say "质量配对（nvfp4 无 adapter vs bf16 SFT merged）："
    .venv/bin/python -m syncopate.train.compare _audit/v13_sft_v13r2_e1_merged.json \
      _audit/e19c_eval_nvfp4_noadapter.json 2>&1 \
      | grep -E "配对差值|结论|该 defer|均值" | tee -a "$P"
  else
    say "🔴 无 adapter 臂也未完成"
  fi
else
  say "质量配对（nvfp4+adapter vs bf16+adapter）："
  .venv/bin/python -m syncopate.train.compare _audit/e19c_eval_bf16.json \
    _audit/e19c_eval_nvfp4.json 2>&1 \
    | grep -E "配对差值|结论|该 defer|均值" | tee -a "$P"
fi
say "🏁 NVFP4 臂全部结束"
touch logs/e19c_nvfp4_DONE
