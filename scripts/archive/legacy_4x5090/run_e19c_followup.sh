#!/usr/bin/env bash
# E19-c 收尾链：① fp8 KV-only 质量臂（把 −0.010 归因到部件）
#              ② NVFP4A16 鉴别臂（W4A4 崩塌的凶手是权重量化还是激活量化）
cd "$(dirname "$0")/.."
set -a; . /workspace/.env; set +a
P=logs/e19c_progress.log
say() { echo "[FOLLOWUP $(date +%H:%M:%S)] $*" | tee -a "$P"; }
MODEL=models/Qwen3-4B-sft-v13r2-e1
QMODEL=models/Qwen3-4B-sft-v13r2-e1-NVFP4A16
ADAPTER=checkpoints/grpo/cand_v13r2_e1/adapter_global_step_25
QPY=/workspace/venvs/quantize/bin/python

# ---------- ① KV-only 质量臂 ----------
say "EVAL kvonly 开始（只 fp8 KV，权重 bf16）"
until GPUS=0,1,2,3 bash scripts/gpu_gate.sh >/dev/null 2>&1; do sleep 60; done
SYNCOPATE_EVAL_KV_DTYPE=fp8 MODEL="$MODEL" timeout 2400 bash scripts/eval_parallel.sh \
  "$ADAPTER" "_audit/e19c_eval_kvonly.json" 4 > logs/e19c_eval_kvonly.log 2>&1
if [ -f _audit/e19c_eval_kvonly.json.done ]; then
  n=$(grep -rc "eval-quant" logs/eval_e19c_eval_kvonly/ 2>/dev/null | awk -F: '{s+=$2} END{print s+0}')
  say "EVAL kvonly ✅（[eval-quant] 判据行 ×${n}，应=4）；配对 vs bf16："
  .venv/bin/python -m syncopate.train.compare _audit/e19c_eval_bf16.json _audit/e19c_eval_kvonly.json 2>&1 \
    | grep -E "配对差值|结论|该 defer|均值" | tee -a "$P"
else
  say "🔴 EVAL kvonly 未完成"
fi

# ---------- ② NVFP4A16 鉴别臂 ----------
if [ ! -f "$QMODEL/config.json" ]; then
  say "NVFP4A16 量化开始（只量权重，隔离 venv）"
  until GPUS=0 bash scripts/gpu_gate.sh >/dev/null 2>&1; do sleep 60; done
  CUDA_VISIBLE_DEVICES=0 timeout 3600 "$QPY" scripts/quantize_nvfp4.py --scheme NVFP4A16 \
    > logs/e19c_quantize_a16.log 2>&1 || { say "🔴 A16 量化失败"; exit 1; }
  grep -E "\[verify\]" logs/e19c_quantize_a16.log | tee -a "$P"
  # llmcompressor 0.13 的新字段与 vllm 0.12 的 compressed-tensors 不兼容，落盘后剥掉
  .venv/bin/python - <<EOF
import json
p = "$QMODEL/config.json"; cfg = json.load(open(p))
def strip(d):
    if isinstance(d, dict):
        for k in ("scale_dtype", "zp_dtype"): d.pop(k, None)
        for v in d.values(): strip(v)
    elif isinstance(d, list):
        for v in d: strip(v)
strip(cfg.get("quantization_config", {})); json.dump(cfg, open(p, "w"), indent=2)
EOF
  say "A16 config 兼容降级完成"
fi

say "ARM nvfp4a16 起服务（W4A16 + fp8 KV）"
until GPUS=0 bash scripts/gpu_gate.sh >/dev/null 2>&1; do sleep 60; done
CUDA_VISIBLE_DEVICES=0 .venv/bin/vllm serve "$QMODEL" \
  --served-model-name sft-base \
  --max-model-len 14336 --kv-cache-dtype fp8 \
  --host 127.0.0.1 --port 8100 > logs/e19c_vllm_nvfp4a16.log 2>&1 &
SRV=$!
ok=0
for i in $(seq 1 60); do
  sleep 7; kill -0 $SRV 2>/dev/null || break
  curl -sf http://127.0.0.1:8100/health >/dev/null 2>&1 && { ok=1; break; }
done
if [ "$ok" = 1 ]; then
  { echo "===== nvfp4a16 装载账本 ====="
    grep -m2 -iE "GPU KV cache size" logs/e19c_vllm_nvfp4a16.log
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | head -1 | sed 's/^/  nvidia-smi GPU0: /'
  } | tee -a "$P"
  timeout 600 .venv/bin/python scripts/measure_tpot.py > logs/e19c_tpot_nvfp4a16.log 2>&1 || true
  timeout 900 .venv/bin/vllm bench serve \
    --backend openai --base-url http://127.0.0.1:8100 --endpoint /v1/completions \
    --model sft-base --tokenizer "$MODEL" \
    --dataset-name random --random-input-len 4200 --random-output-len 650 \
    --num-prompts 48 --ignore-eos > logs/e19c_bench_nvfp4a16.log 2>&1 || true
  grep -E "Median TTFT|P99 TTFT|Mean TPOT|Output token throughput|Total Token|Benchmark duration" \
    logs/e19c_bench_nvfp4a16.log | sed 's/^/  nvfp4a16 /' | tee -a "$P"
  kill $SRV 2>/dev/null; wait $SRV 2>/dev/null
else
  say "🔴 ARM nvfp4a16 服务没起来"; kill $SRV 2>/dev/null
fi
for i in $(seq 1 30); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  [ "$used" -lt 2000 ] && break; sleep 5
done

say "EVAL nvfp4a16 开始（挂 adapter）"
until GPUS=0,1,2,3 bash scripts/gpu_gate.sh >/dev/null 2>&1; do sleep 60; done
MODEL="$QMODEL" timeout 2400 bash scripts/eval_parallel.sh \
  "$ADAPTER" "_audit/e19c_eval_nvfp4a16.json" 4 > logs/e19c_eval_nvfp4a16.log 2>&1
if [ -f _audit/e19c_eval_nvfp4a16.json.done ]; then
  say "质量配对（nvfp4a16 vs bf16）："
  .venv/bin/python -m syncopate.train.compare _audit/e19c_eval_bf16.json _audit/e19c_eval_nvfp4a16.json 2>&1 \
    | grep -E "配对差值|结论|该 defer|均值" | tee -a "$P"
else
  say "🔴 EVAL nvfp4a16 未完成"
fi
say "🏁 FOLLOWUP 全部结束"
touch logs/e19c_followup_DONE
