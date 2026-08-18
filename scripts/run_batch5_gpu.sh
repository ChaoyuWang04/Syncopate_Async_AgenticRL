#!/usr/bin/env bash
# 第 1.8 批（2026-08-18 03:5x）—— 全是**不需要新写码**的实验，先把卡填满
#
# 排序依据：B3 的意外发现「**真正的杠杆是同步频率，不是藏没藏住生成**」
#   （one_step_off 把生成藏得更彻底却仍慢 10%，差别在每步 8.42 s×2 的同步）
#   ⇒ 而 `sync_every` 这个旋钮**从来没扫过**，它现在是队列里最便宜的一块。
#
#   B19  sync_every 扫描 2/4/8/16    ~45 min   ★ 新：杠杆本身没量过
#   B20  dynamic_bsz @1536 复测      ~20 min   README §7 附自己写着「正式配置下值得复测」
#   A9b  4bit MoE 预量化存盘 + 加载   ~40 min   stage① 已复现 OOM，②③ 上次没跑到
set -uo pipefail
cd "$(dirname "$0")/.."

STAMP="$(date +%m%d_%H%M)"; QUEUE_LOG="logs/batch5_queue_${STAMP}.log"
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
  --save-freq 999 --wandb-mode offline --logger console
)
BUCKET_DEFAULT=512; RUN_TIMEOUT="${RUN_TIMEOUT:-5400}"

run () {
  local name="$1"; shift
  local logf="logs/${name}.log"
  log "════════ $name 开始"
  ( set -x; timeout "$RUN_TIMEOUT" .venv/bin/python -m syncopate.train.launch_rl \
      "${COMMON[@]}" --weight-sync-bucket-mb "$BUCKET_DEFAULT" \
      --save-path "checkpoints/grpo/$name" --experiment "$name" \
      "$@" ) > "$logf" 2>&1
  local rc=$?
  log "──────── $name 退出码 $rc"
  .venv/bin/python scripts/parse_fully_async_timing.py "$logf" \
      --json "_audit/infra/${name}_timing.json" 2>&1 | tee -a "$QUEUE_LOG"
  rm -rf "checkpoints/grpo/$name"/global_step_* 2>/dev/null
}
want () { [ $# -le 1 ] && return 0; local t="$1"; shift
          for a in "$@"; do [ "$a" = "$t" ] && return 0; done; return 1; }
TARGETS=("$@")

# ═══════════════════════════════════════════════════════════════════════
# ① B19 · `sync_every` 扫描（同步频率 = B3 指认出来的那个真杠杆）
#
# ⚠️⚠️ **口径陷阱，必须处理**：fully_async 的 `global_steps` 每个 fit step 跳 `sync_every`，
#    而 `--steps` 卡的是 global_steps ⇒ 直接固定 `--steps 12` 去扫 sync_every，
#    **四档做的训练更新次数完全不同**（16 档只更新 1 次），比出来的东西没有意义。
#    ⇒ 这里按 `steps = 4 × sync_every` 给，**保证四档都恰好做 4 次训练更新**。
#
# ★ 预测（跑之前写死）：
#   P1 每步墙钟随 sync_every 增大而下降，且**主要来自同步摊薄**
#      （param_sync ≈ 9 s / sync_every）⇒ 2→4 省得多，8→16 边际收益很小
#   P2 代价是陈旧度上升 ⇒ `stale_trajectory_processed` 单调上升、IS 方差变大
#   P3 若 16 档反而变慢，说明陈旧度已经开始伤到别的东西（那才是有意思的发现）
if want b19 "${TARGETS[@]}"; then
  for se in 2 4 8 16; do
    run "b19_sync${se}" --mode fully_async --trainer-gpus 3 --rollout-gpus 1 \
        --steps $(( 4 * se )) --sync-every "$se"
    log "   陈旧轨迹：$(grep -oE 'stale_trajectory_processed:[0-9.]+' logs/b19_sync${se}.log | tail -1) · IS 序列 std：$(grep -oE 'rollout_corr/rollout_is_seq_std:[0-9.]+' logs/b19_sync${se}.log | tail -1)"
  done
fi

# ═══════════════════════════════════════════════════════════════════════
# ② B20 · `dynamic_bsz` 在**正式序列长度**下复测
#
# README §7 附自己写着：上次是在 max_response_length=512 下测的，只有 +4~5%，
# 而**正式跑是 1536**，「打包收益来自序列长度的方差 ⇒ 长序列下可能更高，值得复测」。
# ★ 预测：1536 下收益 > 5%（方差更大 ⇒ 打包更值），但**不会回到旧记录的 27%**
#   （那是另一台机器 + 另一个 flash-attn 构建）。
if want b20 "${TARGETS[@]}"; then
  run b20_dynbsz_off --mode fully_async --trainer-gpus 3 --rollout-gpus 1 --steps 12 \
      --sync-every 4 --dynamic-bsz False --max-token-len-per-gpu 16384
  run b20_dynbsz_on  --mode fully_async --trainer-gpus 3 --rollout-gpus 1 --steps 12 \
      --sync-every 4 --dynamic-bsz True  --max-token-len-per-gpu 16384
fi

# ═══════════════════════════════════════════════════════════════════════
# ③ A9b · 4bit MoE：预量化存盘 → 从盘加载（stage① 已确认 OOM，13.32/17.43 GB）
# ★ 判据：preload 阶段碎片 < 1 GB 且**不靠 expandable_segments** ⇒ A2 的加载路径就用它
if want a9b "${TARGETS[@]}"; then
  log "════════ a9b · 预量化存盘（盘剩余 $(df -h /workspace | awk 'NR==2{print $4}')）"
  ( set -x; CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      timeout 3600 .venv/bin/python scripts/probe_moe_4bit_load.py \
      --stage save --out models/Qwen3-30B-A3B-nf4 ) 2>&1 | tail -12 | tee -a "$QUEUE_LOG"
  ( set -x; CUDA_VISIBLE_DEVICES=0 timeout 1800 .venv/bin/python scripts/probe_moe_4bit_load.py \
      --stage preload --pre models/Qwen3-30B-A3B-nf4 --json logs/a9_preload.json ) 2>&1 \
      | tail -12 | tee -a "$QUEUE_LOG"
fi

log "════════ batch5 结束"
echo "batch5 done $(date '+%F %T')" >> logs/BATCH_DONE
