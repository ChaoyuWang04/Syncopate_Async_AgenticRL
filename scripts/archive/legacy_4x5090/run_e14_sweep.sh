#!/usr/bin/env bash
# E14/R2 · staleness 甜点扫描 + 等时臂（2026-08-20 夜，Chaoyu 授权连跑）
# 每臂：门禁 → 训练 → 提 adapter → 4 卡评测 → compare（vs ctrl64 + vs SFT）
# 进度行全部写 logs/e14_sweep_progress.log（监视器盯它）；单臂失败记录后继续下一臂。
cd "$(dirname "$0")/.."
set -a; . /workspace/.env; set +a
P=logs/e14_sweep_progress.log
say() { echo "[SWEEP $(date +%H:%M:%S)] $*" | tee -a "$P"; }

run_arm() {
  local name=$1 sync=$2 stal=$3 steps=$4
  say "ARM ${name} 开始（sync=${sync} stal=${stal} steps=${steps}）"
  until GPUS=0,1,2,3 bash scripts/gpu_gate.sh >/dev/null 2>&1; do sleep 60; done
  timeout 2700 .venv/bin/python -m syncopate.train.launch_rl \
    --model models/Qwen3-4B-sft-v13r2-e1 --lora-rank 32 \
    --mode fully_async --trainer-gpus 3 --rollout-gpus 1 \
    --steps "$steps" --save-freq 999 --sync-every "$sync" --staleness-threshold "$stal" \
    --experiment "e14s_${name}" --save-path "checkpoints/grpo/e14s_${name}" \
    --wandb-mode offline --logger console > "logs/e14s_${name}.log" 2>&1
  local ck
  ck=$(ls -d checkpoints/grpo/e14s_${name}/global_step_* 2>/dev/null | sort -V | tail -1)
  if [ -z "$ck" ]; then say "🔴 ARM ${name} 训练无 ckpt，跳过"; return 1; fi
  say "ARM ${name} 训练完成（$(basename $ck)），提取 adapter"
  .venv/bin/python scripts/rl_ckpt_to_adapter.py "$ck/actor" --out "models/adapters/e14s_${name}" \
    >> "logs/e14s_${name}.log" 2>&1 || { say "🔴 ARM ${name} adapter 提取失败"; return 1; }
  say "ARM ${name} 评测开始"
  MODEL=models/Qwen3-4B-sft-v13r2-e1 timeout 1800 bash scripts/eval_parallel.sh \
    "models/adapters/e14s_${name}" "_audit/e14s_${name}.json" 4 \
    > "logs/e14s_${name}_eval.log" 2>&1
  if [ ! -f "_audit/e14s_${name}.json.done" ]; then say "🔴 ARM ${name} 评测无 done 标记"; return 1; fi
  say "ARM ${name} 评测完成；对比："
  { echo "===== ${name} vs ctrl64（臂对臂）====="
    .venv/bin/python -m syncopate.train.compare _audit/e14c_ctrl64.json "_audit/e14s_${name}.json" 2>&1 \
      | grep -E "配对差值|结论|该 defer|均值"
    echo "===== ${name} vs SFT ====="
    .venv/bin/python -m syncopate.train.compare _audit/v13_sft_v13r2_e1_merged.json "_audit/e14s_${name}.json" 2>&1 \
      | grep -E "配对差值|结论"
  } | tee -a "$P"
  say "ARM ${name} ✅ 全链完成"
}

say "扫点开始：4 臂（s8_t025 / s8_t01 / s16_t01 / isotime80）"
run_arm s8_t025 8 0.25 64
run_arm s8_t01 8 0.1 64
run_arm s16_t01 16 0.1 64
run_arm isotime80 16 0.5 80
say "🏁 全部扫点结束"
