#!/usr/bin/env bash
# 第 2.1 批 —— E20 的第二半：**估计量修好了，现在补更新量**
#
# batch8 已证：token 级 IS 让 ESS 从 0.449 回到 **1.000**、grad_norm 从缩小 4× 变成增大 2.3×。
# 但**位移几乎没动**（0.107% → 0.110%）—— 因为 13–15 次更新本来就太少（E20 §3.6）。
# ⇒ 现在估计量能扛了，把更新量加上去：
#
#   E20-e  token IS + ppo_mini_batch_size 6→2   一个 fit step 从 1 次更新变 3 次
#   E20-f  token IS + lr 3e-5→1e-4              单次步长 ×3.3
#   E20-g  两个一起                              ★ 看能不能把位移推进 0.5–5% 那个正常区间
#
# ★ 判据（四件一起看，任何一件坏了就说明推过头了）：
#     位移 ↑（目标 >0.3%）· ESS 不塌（<0.8 就是警告）· grad_norm 不炸也不缩 · 响应长度不崩
# ⚠️ **token 级 IS 是这三跑的共同底座**，别再和序列级混着比。
set -uo pipefail
cd "$(dirname "$0")/.."
STAMP="$(date +%m%d_%H%M)"; QUEUE_LOG="logs/batch9_queue_${STAMP}.log"
mkdir -p logs _audit/infra
log () { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$QUEUE_LOG"; }
COMMON=(
  --model models/Qwen3-4B-sft-v13-e1
  --train-file data/rl/v13/train.parquet --val-file data/rl/v13/val.parquet
  --lora-rank 32 --train-batch-size 6 --rollout-n 8 --micro-batch-size 1
  --max-num-seqs 64 --object-store-gb 2 --max-prompt-length 3584 --max-response-length 1536
  --save-freq 999 --wandb-mode offline --logger console --dynamic-bsz False --max-token-len-per-gpu 16384
  --weight-sync-bucket-mb 512 --rollout-is token
)
RUN_TIMEOUT="${RUN_TIMEOUT:-9000}"
e20run () {
  local name="$1"; shift
  log "════════ $name 开始"
  ( set -x; timeout "$RUN_TIMEOUT" .venv/bin/python -m syncopate.train.launch_rl \
      "${COMMON[@]}" --mode fully_async --trainer-gpus 3 --rollout-gpus 1 --steps 60 --sync-every 4 \
      --save-path "checkpoints/grpo/$name" --experiment "$name" "$@" ) > "logs/${name}.log" 2>&1
  log "──────── $name 退出码 $?"
  .venv/bin/python - "$name" <<'PYEOF' 2>&1 | tee -a "$QUEUE_LOG"
import re, sys
n = sys.argv[1]; t = open(f"logs/{n}.log", errors="replace").read()
S = lambda k: [float(x) for x in re.findall(rf"{re.escape(k)}:([0-9.e+-]+)", t)]
e, g, L = S("rollout_corr/rollout_is_eff_sample_size"), S("actor/grad_norm"), S("response_length/mean")
print(f"  【{n}】更新 {len(g)} 次")
if e: print(f"    ESS/N        {e[0]:.3f} → {e[-1]:.3f}      ← <0.8 是警告")
if g: print(f"    grad_norm    {g[0]:.5f} → {g[-1]:.5f}")
if L: print(f"    响应长度      {L[0]:.0f} → {L[-1]:.0f}")
PYEOF
  local ck=$(ls -d checkpoints/grpo/$name/global_step_*/actor 2>/dev/null | tail -1)
  [ -n "$ck" ] && .venv/bin/python scripts/rl_ckpt_drift.py "$ck" 2>&1 | grep "★" | tee -a "$QUEUE_LOG"
  # ⛔ 2026-08-18：**不再删 ckpt** —— 主线 17 §4.3 指出唯一能判定这条线的
  #    是「同位移下的任务级配对 + 三个计数」，而我把 e20_tokenis 那臂的 ckpt 删了
  #    ⇒ 唯一的判据用不了。**凡是要过任务尺子的跑，ckpt 必须留。**
  du -sh "checkpoints/grpo/$name" 2>/dev/null | tee -a "$QUEUE_LOG"
}

e20run e20e_mini2      --ppo-mini-batch-size 2 --lr 3e-5
e20run e20f_lr1e4      --ppo-mini-batch-size 6 --lr 1e-4
e20run e20g_mini2_lr1e4 --ppo-mini-batch-size 2 --lr 1e-4

log "════════ batch9 结束"
echo "batch9 done $(date '+%F %T')" >> logs/BATCH_DONE
