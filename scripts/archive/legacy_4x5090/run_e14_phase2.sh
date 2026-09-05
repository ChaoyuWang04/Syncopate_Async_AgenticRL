#!/usr/bin/env bash
# E14 phase2 · 乒乓修理 A/B 四臂 + 全开最终测速（2026-08-20 夜，接在 sweep 之后）
# A/B 判据：torch-prof 账本的 cudaStreamSynchronize 次数（预期 off≈912 → ③−96 → ②−584 → ②③≈230）
# 最终测速：修理全开 + CUDA graph；final2 另加 sweep 甜点候选 sync8/0.25（其质量已由 sweep 臂评过）
cd "$(dirname "$0")/.."
set -a; . /workspace/.env; set +a
P=logs/e14_sweep_progress.log
say() { echo "[PHASE2 $(date +%H:%M:%S)] $*" | tee -a "$P"; }

run_ab() {
  local name=$1 jag=$2 ri=$3
  say "AB ${name} 开始（FIX_JAGGED=$jag FIX_PG_RI=$ri，16步+torch-prof@3）"
  until GPUS=0,1,2,3 bash scripts/gpu_gate.sh >/dev/null 2>&1; do sleep 60; done
  SYNCOPATE_FIX_JAGGED=$jag SYNCOPATE_FIX_PG_RI=$ri SYNCOPATE_TORCH_PROF=3 \
  timeout 1500 .venv/bin/python -m syncopate.train.launch_rl \
    --model models/Qwen3-4B-sft-v13r2-e1 --lora-rank 32 \
    --mode fully_async --trainer-gpus 3 --rollout-gpus 1 \
    --steps 16 --save-freq 999 --experiment "e14f_${name}" \
    --save-path "checkpoints/grpo/e14f_${name}" \
    --wandb-mode offline --logger console > "logs/e14f_${name}.log" 2>&1
  local sync
  sync=$(grep -A3 "同步类算子账本" "logs/e14f_${name}.log" | grep -oE "cudaStreamSynchronize\s+×\s*[0-9]+" | grep -oE "[0-9]+$" | head -1)
  say "AB ${name} 完成：cudaStreamSynchronize ×${sync:-未捕获}"
  rm -rf "checkpoints/grpo/e14f_${name}/global_step_"* 2>/dev/null
}

run_final() {
  local name=$1; shift
  say "FINAL ${name} 开始（64步测速：$*）"
  until GPUS=0,1,2,3 bash scripts/gpu_gate.sh >/dev/null 2>&1; do sleep 60; done
  timeout 2400 .venv/bin/python -m syncopate.train.launch_rl \
    --model models/Qwen3-4B-sft-v13r2-e1 --lora-rank 32 \
    --mode fully_async --trainer-gpus 3 --rollout-gpus 1 \
    --steps 64 --save-freq 999 "$@" \
    --experiment "e14f_${name}" --save-path "checkpoints/grpo/e14f_${name}" \
    --wandb-mode offline --logger console > "logs/e14f_${name}.log" 2>&1
  say "FINAL ${name} timing："
  .venv/bin/python scripts/parse_fully_async_timing.py "logs/e14f_${name}.log" 2>/dev/null | head -7 | tee -a "$P"
  rm -rf "checkpoints/grpo/e14f_${name}/global_step_"* 2>/dev/null
}

say "phase2 开始：修理 A/B 四臂 → 全开最终测速 ×2"
run_ab off 0 0
run_ab jag 1 0
run_ab ri  0 1
run_ab both 1 1
# 最终①：修理全开（默认即开）+ CUDA graph，保守 staleness（sync4 精度已证安全）
run_final allon_s4 --enforce-eager False
# 最终②：再叠 sweep 甜点候选（质量由 sweep 的 s8_t025 臂评测背书/否决，本跑只测速）
run_final allon_s16t01 --enforce-eager False --sync-every 16 --staleness-threshold 0.1
say "🏁 phase2 全部结束——今夜任务闭环"
