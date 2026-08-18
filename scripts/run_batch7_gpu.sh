#!/usr/bin/env bash
# 第 2.0 批 —— E17 §5.1.1 提出的那个**被改写过的问题**
#
# 昨晚 E17 的结论从「ref 能不能不要」改成了「**kl_loss_coef=0.001 是不是小到不起作用**」：
#   位移实测 —— KL 开每次更新 0.00783%、KL 关 0.00790%，**差 0.9%，在噪声里**
#   ⇒ 现在的局面是「付 14% 的算力，维持一个不起作用的约束」
#   ⇒ 那就先把系数扫开，看它**什么时候才开始起作用**。
#
#   E17-D  kl_loss_coef ∈ {0.001, 0.01, 0.05}，各 60 步，量位移 + 熵
#
# ★ 预测（跑之前写死）：
#   P1  0.001 → 位移与「关掉 KL」几乎相同（已知，作为基线复现）
#   P2  0.01  → 位移开始被拉住（每次更新的位移下降 >5%）
#   P3  0.05  → 位移明显被拉住，但**熵可能被压住**（KL 拉回基座 = 拉回更确定的分布）
#   ⛔ 若 0.05 都拉不住 ⇒ 说明**在这个位移尺度上 KL 根本不是主导项**，
#      那么「ref 那 14% 买到了什么」这个问题的答案就是「几乎没有」——
#      **那才是 E17 真正的结论**，而且它比「省 12.7%」有分量得多。
set -uo pipefail
cd "$(dirname "$0")/.."
STAMP="$(date +%m%d_%H%M)"; QUEUE_LOG="logs/batch7_queue_${STAMP}.log"
mkdir -p logs _audit/infra
log () { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$QUEUE_LOG"; }
COMMON=(
  --model models/Qwen3-4B-sft-v13-e1
  --train-file data/rl/v13/train.parquet --val-file data/rl/v13/val.parquet
  --lora-rank 32 --train-batch-size 6 --rollout-n 8 --ppo-mini-batch-size 6 --micro-batch-size 1
  --max-num-seqs 64 --object-store-gb 2 --max-prompt-length 3584 --max-response-length 1536
  --save-freq 999 --wandb-mode offline --logger console --dynamic-bsz False --max-token-len-per-gpu 16384
)
RUN_TIMEOUT="${RUN_TIMEOUT:-7200}"

for coef in 0.001 0.01 0.05; do
  name="e17d_kl${coef}"
  log "════════ $name 开始（kl_loss_coef=$coef）"
  ( set -x; timeout "$RUN_TIMEOUT" .venv/bin/python -m syncopate.train.launch_rl \
      "${COMMON[@]}" --weight-sync-bucket-mb 512 \
      --mode fully_async --trainer-gpus 3 --rollout-gpus 1 --steps 60 --sync-every 4 --lr 3e-5 \
      --save-path "checkpoints/grpo/$name" --experiment "$name" \
      "++actor_rollout_ref.actor.kl_loss_coef=$coef" ) > "logs/${name}.log" 2>&1
  log "──────── $name 退出码 $?"
  .venv/bin/python scripts/parse_fully_async_timing.py "logs/${name}.log" \
      --json "_audit/infra/${name}_timing.json" 2>&1 | grep -E "^\s+step |三次前向" | tee -a "$QUEUE_LOG"
  log "   kl_loss: $(grep -oE 'actor/kl_loss:[0-9.e-]+' logs/${name}.log | tail -3 | tr '\n' ' ')"
  log "   熵:      $(grep -oE 'actor/entropy:[0-9.]+' logs/${name}.log | tail -3 | tr '\n' ' ')"
  # ★ 位移必须在 ckpt 还在的时候算，算完再删
  ck=$(ls -d checkpoints/grpo/$name/global_step_*/actor 2>/dev/null | tail -1)
  if [ -n "$ck" ]; then
    .venv/bin/python scripts/rl_ckpt_drift.py "$ck" 2>&1 | tee -a "$QUEUE_LOG"
  else
    log "   ⚠️ 没找到 ckpt —— 位移算不了（收尾保存没落？）"
  fi
  rm -rf "checkpoints/grpo/$name"/global_step_* 2>/dev/null
done

log "════════ batch7 结束"
echo "batch7 done $(date '+%F %T')" >> logs/BATCH_DONE
