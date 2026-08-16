#!/usr/bin/env bash
# 2026-08-14 GPU 时段剩余任务，串行跑完，不用盯
#
# ① E12-d · layered_summon A/B（只变这一个变量）
#    基线已有：layered_summon=True（logs/e12c_step5b.log）
#      param_sync 稳态均值 63.48 s / trainer侧 60.13 s / rollout侧 63.42 s
#    假设：layered_summon 是为**分片 FSDP** 设计的（逐层 summon 省显存），
#         而本机 --fsdp-size 1 不分片 ⇒ 它把一个免费操作拆成 36 份收费操作
#    ★ 预测（跑之前写死）：trainer 侧 60.13 s → **< 10 s**；
#      若没降，说明成本不在 layered_summon，回去查 send_weights
#    ⚠️ 盯显存：False 会走整体 summon_full_params，DDP 下理论上不额外占，但要看 actor 峰值
#
# ② E08-b · 三方同机分母（colocate / one_step_off，fully_async 已有）
#    除 mode 外全部一致，都用 decoupled，layered_summon 保持 True 以与 M7 可比
set -uo pipefail
cd "$(dirname "$0")/.."

COMMON=(
  --model models/Qwen3-4B-sft-v11-e1
  --train-file data/rl/v11/train.parquet --val-file data/rl/v11/val.parquet
  --lora-rank 32 --steps 12
  --train-batch-size 6 --rollout-n 8 --ppo-mini-batch-size 6 --micro-batch-size 1
  --max-num-seqs 64 --object-store-gb 2
  --max-prompt-length 3584 --max-response-length 1536
  --save-freq 999 --latency-scale 0.01 --wandb-mode offline
  --dynamic-bsz True --max-token-len-per-gpu 16384
)

run () {
  local name="$1"; shift
  echo "================ $name  开始 $(date +%H:%M:%S)"
  SYNCOPATE_SYNC_TIMING=1 .venv/bin/python -m syncopate.train.launch_rl "$@" "${COMMON[@]}" \
      --save-path "checkpoints/grpo/$name" --experiment "$name" \
      > "logs/$name.log" 2>&1
  echo "---- $name 退出码 $?  $(date +%H:%M:%S)"
  echo "   param_sync: $(grep -o 'param_sync: [0-9.]* seconds' "logs/$name.log" | tr '\n' ' ')"
  echo "   trainer侧:  $(grep -oE 'trainer侧[^ ]* [0-9.]+ s' "logs/$name.log" | grep -oE '[0-9.]+ s' | tr '\n' ' ')"
  echo "   step:       $(grep -oE 'timing_s/step:[0-9.]+' "logs/$name.log" | tr '\n' ' ')"
}

# ① layered_summon 对照
run e12d_nolayered --mode fully_async --trainer-gpus 3 --rollout-gpus 1 --layered-summon False

# ② 同机分母
run e08b_colocate    --mode colocate       --trainer-gpus 1
run e08b_onestepoff  --mode one_step_off   --trainer-gpus 3 --rollout-gpus 1

echo "================ 全部完成 $(date +%H:%M:%S)"
