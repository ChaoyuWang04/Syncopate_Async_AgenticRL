#!/usr/bin/env bash
# 第 1.7 批（2026-08-18 02:2x 排定）—— batch3 之后的续跑
#
# ⚠️⚠️ **起这一批的教训**：batch3 在 21:56 就跑完了，而我没给它挂完成守卫
#    ⇒ **四张卡空转了 4 个多小时**。本脚本结束时会往 `logs/BATCH_DONE` 写一行，
#    守卫脚本盯那个文件即可（见 §末尾）。
#
# 顺序（按「能不能把某个 before 变成 after」排）：
#   B16  B12 的长跑两臂 + B5 任务级尺子   ★ 项目自定纪律「只报吞吐不报精度不算完成」至今一次没过
#   B15  分步计时（**修好接线后**重跑）    那 6 倍到底掉在哪一格
#   A9   4bit MoE 加载路径（探针已接住 OOM） A2 的前置
set -uo pipefail
cd "$(dirname "$0")/.."

STAMP="$(date +%m%d_%H%M)"
QUEUE_LOG="logs/batch4_queue_${STAMP}.log"
mkdir -p logs _audit/infra
log () { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$QUEUE_LOG"; }

COMMON=(
  --model models/Qwen3-4B-sft-v13-e1
  --train-file data/rl/v13/train.parquet --val-file data/rl/v13/val.parquet
  --lora-rank 32
  --train-batch-size 6 --rollout-n 8 --ppo-mini-batch-size 6 --micro-batch-size 1
  --max-num-seqs 64 --object-store-gb 2
  # ★ 2026-08-18 由 3584/1536 提到 5120/2048：训练与评测此前不同源（评测宽 43%/33%），
  #   见 docs/syncopate/20 §P0-1。⚠️ max_model_len 随之 5120→7168，首次跑请盯 vLLM KV cache。
  --max-prompt-length 5120 --max-response-length 2048
  --wandb-mode offline --logger console
  --dynamic-bsz False --max-token-len-per-gpu 16384
)
BUCKET_DEFAULT=512
RUN_TIMEOUT="${RUN_TIMEOUT:-7200}"

run () {   # COMMON 在前、单跑参数在后、Hydra 位置参数最后
  local name="$1"; shift
  local logf="logs/${name}.log"
  log "════════ $name 开始"
  ( set -x; timeout "$RUN_TIMEOUT" .venv/bin/python -m syncopate.train.launch_rl \
      "${COMMON[@]}" --weight-sync-bucket-mb "$BUCKET_DEFAULT" \
      --save-path "checkpoints/grpo/$name" --experiment "$name" \
      "$@" ) > "$logf" 2>&1
  local rc=$?
  log "──────── $name 退出码 $rc"
  [ "$rc" = "124" ] && log "🔴 $name 撞到 ${RUN_TIMEOUT}s 超时"
  .venv/bin/python scripts/parse_fully_async_timing.py "$logf" \
      --json "_audit/infra/${name}_timing.json" 2>&1 | tee -a "$QUEUE_LOG"
}

want () { [ $# -le 1 ] && return 0; local t="$1"; shift
          for a in "$@"; do [ "$a" = "$t" ] && return 0; done; return 1; }
TARGETS=("$@")

# ═══════════════════════════════════════════════════════════════════════
# ① B16 · E17 的长跑两臂（★ 保留 ckpt，B5 要用）
#
# 短跑（12 步）已证吞吐省 12.7%，但**过不了任务级尺子** —— 12 步动不了模型多少
# （位移 = lr × 步数，见记忆 rl-step-size-is-lr-times-steps）。
# ⇒ 这里跑 **60 步**（≈15 个 param version），两臂只差 `use_kl_loss`。
# ★ 预测：① 吞吐差仍在 12–15%；② EVAL 配对**无显著差异**（MDE≈0.05）；
#         ③ ref 关掉那臂的 `actor/entropy` 掉得更快。
# ⚠️ **不删 ckpt**（其余批次都删）—— 收尾那次 force 保存就是 B5 要评的东西。
if want b16 "${TARGETS[@]}"; then
  run b16_ref_on_60  --mode fully_async --trainer-gpus 3 --rollout-gpus 1 --steps 60 \
      --sync-every 4 --save-freq 999 --lr 3e-5
  run b16_ref_off_60 --mode fully_async --trainer-gpus 3 --rollout-gpus 1 --steps 60 \
      --sync-every 4 --save-freq 999 --lr 3e-5 \
      '++actor_rollout_ref.actor.use_kl_loss=False'
  log "   两臂 ckpt（B5 要用，别删）："
  ls -d checkpoints/grpo/b16_ref_*/global_step_* 2>/dev/null | tee -a "$QUEUE_LOG"
  log "   熵曲线对照（P3）："
  for a in on off; do
    log "     ref_$a: $(grep -oE 'actor/entropy:[0-9.]+' logs/b16_ref_${a}_60.log | tail -4 | tr '\n' ' ')"
  done
fi

# ═══════════════════════════════════════════════════════════════════════
# ② B15 · 分步计时（**接线已修**：钩子原来嵌在 grad_probe 里，只有开那个才装）
#
# ★ 判据行：`[verl-patch] 权重同步分步计时已启用` —— **没有这行就是又没接上，直接停**。
# ★ 预测：仍是第 5 步占 ~99%，但绝对值从 E12 记录的 59.76 s 降到 ~9 s。
if want b15 "${TARGETS[@]}"; then
  ( export SYNCOPATE_SYNC_TIMING=1
    run b15b_synctiming --mode fully_async --trainer-gpus 3 --rollout-gpus 1 --steps 12 \
        --sync-every 4 --save-freq 999 )
  n=$(grep -c "权重同步分步计时已启用" logs/b15b_synctiming.log || true)
  log "   判据行 $n 条（要 ≥1，0 就是**又没接上**）"
  grep -E "\[sync-timing\]" logs/b15b_synctiming.log | tail -24 | tee -a "$QUEUE_LOG"
  rm -rf checkpoints/grpo/b15b_synctiming/global_step_* 2>/dev/null
fi

# ═══════════════════════════════════════════════════════════════════════
# ③ A9 · 4bit MoE 加载路径（探针已改成接住 OOM 并继续）
if want a9 "${TARGETS[@]}"; then
  log "════════ a9 · 三阶段（盘剩余 $(df -h /workspace | awk 'NR==2{print $4}')）"
  ( set -x; CUDA_VISIBLE_DEVICES=0 timeout 1800 .venv/bin/python scripts/probe_moe_4bit_load.py \
      --stage online --json logs/a9_online.json ) 2>&1 | tail -20 | tee -a "$QUEUE_LOG"
  ( set -x; CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      timeout 3600 .venv/bin/python scripts/probe_moe_4bit_load.py \
      --stage save --out models/Qwen3-30B-A3B-nf4 ) 2>&1 | tail -20 | tee -a "$QUEUE_LOG"
  ( set -x; CUDA_VISIBLE_DEVICES=0 timeout 1800 .venv/bin/python scripts/probe_moe_4bit_load.py \
      --stage preload --pre models/Qwen3-30B-A3B-nf4 --json logs/a9_preload.json ) 2>&1 \
      | tail -20 | tee -a "$QUEUE_LOG"
fi

log "════════ batch4 结束"
echo "batch4 done $(date '+%F %T')" >> logs/BATCH_DONE     # ★ 给守卫看的
