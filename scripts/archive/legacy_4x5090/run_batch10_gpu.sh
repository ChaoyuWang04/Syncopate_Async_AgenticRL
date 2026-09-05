#!/usr/bin/env bash
# 第 2.2 批 —— ★ 「权衡是不是被消掉了」：token 级 IS × 大 sync_every
#
# 背景（E20 §3.8）：异步的**全部价值就在"少同步"上** ——
#   sync_every=1 时每步 54.48 s，而 colocate 是 55.28 s，**差 1.4%，异步等于白做**。
# 而 B19 测出 sync_every 4→8→16 能到 29.37 / 27.88 s（相对 colocate 1.88× / 1.98×），
# **代价是陈旧度**。此前这是一个"吞吐 vs 训练崩不崩"的权衡。
#
# ⇒ 但 batch8 证明 **token 级 IS 在 sync_every=4 下让 ESS 稳在 1.000**。
#   **那么这个权衡还在吗？** 如果 token IS 在 8/16 下 ESS 依然不塌，
#   我们就**同时拿到 2 倍吞吐和健康的训练** —— 权衡消失，不是找平衡点。
#
# ★ 判据：ESS/N ≥0.95 视为"权衡消失"；0.8–0.95 是"还剩一点"；<0.8 是"权衡仍在"。
set -uo pipefail
cd "$(dirname "$0")/.."
STAMP="$(date +%m%d_%H%M)"; QUEUE_LOG="logs/batch10_queue_${STAMP}.log"
mkdir -p logs _audit/infra
log () { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$QUEUE_LOG"; }
COMMON=(
  --model models/Qwen3-4B-sft-v13-e1
  --train-file data/rl/v13/train.parquet --val-file data/rl/v13/val.parquet
  --lora-rank 32 --train-batch-size 6 --rollout-n 8 --ppo-mini-batch-size 6 --micro-batch-size 1
  # ★ 长度预算与采样参数**不在这里传** —— 唯一来源是 syncopate/train/rollout_budget.py，
  #   launch_rl 的默认值从那里取。⚠️ 显式传会被 check_pipeline_invariants 的 contract 组判红
  #   （2026-08-18 就是各脚本各抄一份，抄着抄着漂成了 3584/1536 vs 5120/2048 两套）。
  --max-num-seqs 64 --object-store-gb 2
  --save-freq 999 --wandb-mode offline --logger console --dynamic-bsz False --max-token-len-per-gpu 16384
  --weight-sync-bucket-mb 512 --rollout-is token --lr 3e-5
)
RUN_TIMEOUT="${RUN_TIMEOUT:-9000}"
for se in 8 16; do
  name="e20h_tokenis_sync${se}"
  log "════════ $name 开始"
  ( set -x; timeout "$RUN_TIMEOUT" .venv/bin/python -m syncopate.train.launch_rl \
      "${COMMON[@]}" --mode fully_async --trainer-gpus 3 --rollout-gpus 1 \
      --steps $(( 4 * se )) --sync-every "$se" \
      --save-path "checkpoints/grpo/$name" --experiment "$name" ) > "logs/${name}.log" 2>&1
  log "──────── $name 退出码 $?"
  .venv/bin/python - "$name" <<'PYEOF' 2>&1 | tee -a "$QUEUE_LOG"
import re, sys
n=sys.argv[1]; t=open(f"logs/{n}.log",errors="replace").read()
S=lambda k:[float(x) for x in re.findall(rf"{re.escape(k)}:([0-9.e+-]+)",t)]
e,g,d=S("rollout_corr/rollout_is_eff_sample_size"),S("actor/grad_norm"),S("rollout_corr/log_ppl_diff")
st=[float(x) for x in re.findall(r"timing_s/step:([0-9.]+)",t)]
print(f"  【{n}】更新 {len(g)} 次")
if e: print(f"    ESS/N        {e[0]:.3f} → {e[-1]:.3f}   ← ≥0.95 = 权衡消失 / <0.8 = 权衡仍在")
if g: print(f"    grad_norm    {g[0]:.5f} → {g[-1]:.5f}")
if d: print(f"    log_ppl_diff {d[0]:.5f} → {d[-1]:.5f}   ← 陈旧度本身（会更大，这是预期的）")
PYEOF
  .venv/bin/python scripts/parse_fully_async_timing.py "logs/${name}.log" 2>/dev/null | grep -E "^\s+step " | tee -a "$QUEUE_LOG"
  # ⛔ 2026-08-18：**不再删 ckpt** —— 主线 17 §4.3 指出唯一能判定这条线的
  #    是「同位移下的任务级配对 + 三个计数」，而我把 e20_tokenis 那臂的 ckpt 删了
  #    ⇒ 唯一的判据用不了。**凡是要过任务尺子的跑，ckpt 必须留。**
  du -sh "checkpoints/grpo/$name" 2>/dev/null | tee -a "$QUEUE_LOG"
done
log "════════ batch10 结束"
echo "batch10 done $(date '+%F %T')" >> logs/BATCH_DONE
