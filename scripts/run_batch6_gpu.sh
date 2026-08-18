#!/usr/bin/env bash
# 第 1.9 批（batch5 之后）—— 都要新写码，写完了才排进来
#
#   A17   FSDP 分片补到 16 字节 → **端到端** ZeRO-3 对照   ★ 上游 issue 欠的最后一段
#   E19b  FP8 用在 ref 上的数值对拍                        ★ FP8 接线的前置判据
set -uo pipefail
cd "$(dirname "$0")/.."
STAMP="$(date +%m%d_%H%M)"; QUEUE_LOG="logs/batch6_queue_${STAMP}.log"
mkdir -p logs _audit/infra
log () { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$QUEUE_LOG"; }
COMMON=(
  --model models/Qwen3-4B-sft-v13-e1
  --train-file data/rl/v13/train.parquet --val-file data/rl/v13/val.parquet
  --lora-rank 32 --train-batch-size 6 --rollout-n 8 --ppo-mini-batch-size 6 --micro-batch-size 1
  # ★ 2026-08-18 由 3584/1536 提到 5120/2048：训练与评测此前不同源（评测宽 43%/33%），
  #   见 docs/syncopate/20 §P0-1。⚠️ max_model_len 随之 5120→7168，首次跑请盯 vLLM KV cache。
  --max-num-seqs 64 --object-store-gb 2 --max-prompt-length 5120 --max-response-length 2048
  --save-freq 999 --wandb-mode offline --logger console --dynamic-bsz False --max-token-len-per-gpu 16384
)
RUN_TIMEOUT="${RUN_TIMEOUT:-5400}"
want () { [ $# -le 1 ] && return 0; local t="$1"; shift
          for a in "$@"; do [ "$a" = "$t" ] && return 0; done; return 1; }
TARGETS=("$@")

# ═══════════════════════════════════════════════════════════════════════
# ① A17 · 「改这一行就好了」的端到端证据（3 卡 ZeRO-3，只改对齐补丁）
#
# ★ 预测：对齐之后 update_actor 从 ~96 s（本机 Simple 档，带 NCCL_DEBUG 的那次口径）
#   降到 LL128 档附近（~30 s）。**判据有两条，缺一不可**：
#     ① 快了；② **grad_norm 与未打补丁的那跑同量级** —— 补丁改的是 flat param 尺寸，
#        切错位会静默地训练错的东西，只看变快是不够的。
# ⚠️ 两跑都强制 NCCL_PROTO=Simple（不然自动 LL128 会把悬崖绕过去，对照就没意义了）
if want a17 "${TARGETS[@]}"; then
  for arm in off on; do
    name="a17_align_${arm}"
    [ "$arm" = "on" ] && export SYNCOPATE_FSDP_ALIGN=1 || unset SYNCOPATE_FSDP_ALIGN
    log "════════ $name 开始（对齐补丁=$arm）"
    ( set -x; timeout "$RUN_TIMEOUT" .venv/bin/python -m syncopate.train.launch_rl \
        --mode colocate --trainer-gpus 3 --fsdp-size 3 --steps 2 --rollout-gpu-util 0.35 \
        "${COMMON[@]}" --weight-sync-bucket-mb 512 \
        --save-path "checkpoints/grpo/$name" --experiment "$name" \
        "++ray_kwargs.ray_init.runtime_env.env_vars.NCCL_PROTO=Simple" \
        '++actor_rollout_ref.actor.fsdp_config.reshard_after_forward=True' \
    ) > "logs/${name}.log" 2>&1
    log "──────── $name 退出码 $?"
    log "   判据①(快没快) update_actor: $(grep -oE 'timing_s/update_actor:[0-9.]+' logs/${name}.log | tail -2 | tr '\n' ' ')"
    log "   判据②(对不对) grad_norm:    $(grep -oE 'actor/grad_norm:[0-9.e-]+' logs/${name}.log | tail -2 | tr '\n' ' ')"
    log "   补丁判据行:                 $(grep -c '16 字节对齐已启用' logs/${name}.log) 条"
    rm -rf "checkpoints/grpo/$name"/global_step_* 2>/dev/null
  done
  unset SYNCOPATE_FSDP_ALIGN
fi

# ═══════════════════════════════════════════════════════════════════════
# ② E19b · FP8 在 ref 那一遍上的数值代价（真实 lm_head 权重）
if want e19b "${TARGETS[@]}"; then
  log "════════ e19b · FP8 logprob/KL 数值对拍"
  ( set -x; CUDA_VISIBLE_DEVICES=0 timeout 900 .venv/bin/python \
      scripts/probe_fp8_logprob_error.py --json logs/e19b_fp8_logprob.json ) 2>&1 | tee -a "$QUEUE_LOG"
fi

log "════════ batch6 结束"
echo "batch6 done $(date '+%F %T')" >> logs/BATCH_DONE
