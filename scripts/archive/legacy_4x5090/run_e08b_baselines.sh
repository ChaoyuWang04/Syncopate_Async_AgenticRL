#!/usr/bin/env bash
# E08-b · 同机分母：三种模式的配对对照
#
# 目的：所有加速比至今**没有同机分母**。这里把三种摆法放到同一把尺子上。
#
# ★ 实验设计纪律：**除了 `--mode`（和它必然带来的卡数差异），其余全部保持一致**。
#   特别是 `--bypass-mode`：M7 用的是 decoupled（False），
#   而旧的 one_step_off 32.6 s/步 基线是 bypass=True 跑的 —— **那两个数不可比**，
#   这正是本脚本要修掉的问题。
#
# 用法：bash scripts/run_e08b_baselines.sh
set -uo pipefail
cd "$(dirname "$0")/.."

COMMON=(
  --model models/Qwen3-4B-sft-v11-e1
  --train-file data/rl/v11/train.parquet --val-file data/rl/v11/val.parquet
  --lora-rank 32 --steps 12
  --train-batch-size 6 --rollout-n 8 --ppo-mini-batch-size 6 --micro-batch-size 1
  --max-num-seqs 64 --object-store-gb 2
  # ★ 长度预算与采样参数**不在这里传** —— 唯一来源是 syncopate/train/rollout_budget.py，
  #   launch_rl 的默认值从那里取。⚠️ 显式传会被 check_pipeline_invariants 的 contract 组判红
  #   （2026-08-18 就是各脚本各抄一份，抄着抄着漂成了 3584/1536 vs 5120/2048 两套）。
  --save-freq 999 --latency-scale 0.01 --wandb-mode offline
  --dynamic-bsz True --max-token-len-per-gpu 16384
)

run () {
  local name="$1"; shift
  echo "================ $name  $(date +%H:%M:%S) ================"
  .venv/bin/python -m syncopate.train.launch_rl "$@" "${COMMON[@]}" \
      --save-path "checkpoints/grpo/$name" --experiment "$name" \
      > "logs/$name.log" 2>&1
  echo "---- $name 退出码 $?  $(date +%H:%M:%S)"
  grep -oE "timing_s/step:[0-9.]+" "logs/$name.log" | tail -3
}

# ① colocate：单卡，rollout 和 train 同卡。**经典基线** —— "不上多卡时是什么样"
run e08b_colocate --mode colocate --trainer-gpus 1

# ② one_step_off：3 训练 + 1 rollout，落后一步
#    ⚠️ 与旧的 rl_best（32.6 s/步）不同：那次是 bypass=True，这次是 decoupled，
#       为的是和 fully_async 可比
run e08b_onestepoff --mode one_step_off --trainer-gpus 3 --rollout-gpus 1

echo "================ 全部完成 $(date +%H:%M:%S) ================"
echo "对照：fully_async 3+1 decoupled = 74.1 s/global step（M7，logs/m7_v11e1_fullyasync.log）"
