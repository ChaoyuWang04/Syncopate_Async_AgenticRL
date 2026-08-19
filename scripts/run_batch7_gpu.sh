#!/usr/bin/env bash
# 第 2.0 批 —— ★★ E20：RL 学不动（**当前优先级最高，因为它影响正确性不是速度**）
#
# ⚠️ 本批原定是 E17-D（KL 系数扫描），2026-08-18 按 Chaoyu 的新排序原则**降级**：
#    「影响正确性的 > 影响速度的」。KL 系数是速度侧的定价问题；
#    而**模型学不动是正确性问题** —— 100+ 步后能力几乎没变化。
#
# 诊断见 docs/infra_exp/E20-rl-not-learning.md：
#   序列级 IS 权重 w = exp(L · Δlogp̄)，L 已经 694 token ⇒ 指数脆弱
#   实测 lr=3e-5 时 IS 均值 0.99→0.49、min→0、梯度反而变小，而权重只动了 0.10%
#   ⇒ 不是「lr 太大」，是「序列级 IS 在这个长度上不能用」
#
#   E20-a  rollout_is: sequence → **token**    ★ 最高优先级，且不牺牲吞吐
#   E20-b  同 lr 下 sync_every ∈ {1, 4}        看陈旧度对 ESS 崩塌速度的贡献
#
# ★ 四件判据一起看（缺一件都可能得出错误结论）：
#     位移   scripts/rl_ckpt_drift.py    目标：**起得来**（现在 0.10%，正常 LoRA 0.5–5%）
#     ESS    从 rollout_is_seq_{mean,std} 算 1/(1+CV²)   目标：**不随步数塌**
#     梯度   actor/grad_norm 趋势          目标：**不随步数缩小**
#     长度   response_length/mean          它是崩塌的自变量，必须一起记
set -uo pipefail
cd "$(dirname "$0")/.."
STAMP="$(date +%m%d_%H%M)"; QUEUE_LOG="logs/batch7_queue_${STAMP}.log"
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
  --weight-sync-bucket-mb 512
)
RUN_TIMEOUT="${RUN_TIMEOUT:-9000}"

# 跑一个 E20 实验，跑完立刻把四件判据打出来
e20run () {
  local name="$1"; shift
  log "════════ $name 开始"
  ( set -x; timeout "$RUN_TIMEOUT" .venv/bin/python -m syncopate.train.launch_rl \
      "${COMMON[@]}" --mode fully_async --trainer-gpus 3 --rollout-gpus 1 \
      --steps 60 --lr 3e-5 \
      --save-path "checkpoints/grpo/$name" --experiment "$name" \
      "$@" ) > "logs/${name}.log" 2>&1
  log "──────── $name 退出码 $?"
  .venv/bin/python - "$name" <<'PYEOF' 2>&1 | tee -a "$QUEUE_LOG"
import re, sys
n = sys.argv[1]
t = open(f"logs/{n}.log", errors="replace").read()
def series(k):
    return [float(x) for x in re.findall(rf"{re.escape(k)}:([0-9.e+-]+)", t)]
m, s = series("rollout_corr/rollout_is_seq_mean"), series("rollout_corr/rollout_is_seq_std")
g, L = series("actor/grad_norm"), series("response_length/mean")
d = series("rollout_corr/log_ppl_diff")
def ess(mm, ss):
    cv = ss / mm if mm else float("inf"); return 1 / (1 + cv * cv)
print(f"  【{n}】四件判据")
if m and s:
    print(f"    ESS/N        首 {ess(m[0],s[0]):.3f} → 末 {ess(m[-1],s[-1]):.3f}   "
          f"(IS 均值 {m[0]:.3f}→{m[-1]:.3f})   ← 目标：**不塌**")
if g: print(f"    grad_norm    首 {g[0]:.5f} → 末 {g[-1]:.5f}                    ← 目标：**不缩小**")
if d: print(f"    log_ppl_diff 首 {d[0]:.5f} → 末 {d[-1]:.5f}                    ← 第一个数=数值失配地板")
if L: print(f"    响应长度      首 {L[0]:.0f} → 末 {L[-1]:.0f}                     ← 崩塌的自变量")
PYEOF
  local ck=$(ls -d checkpoints/grpo/$name/global_step_*/actor 2>/dev/null | tail -1)
  [ -n "$ck" ] && .venv/bin/python scripts/rl_ckpt_drift.py "$ck" 2>&1 | grep -E "★|被适配" | tee -a "$QUEUE_LOG"
  rm -rf "checkpoints/grpo/$name"/global_step_* 2>/dev/null
}

# ① 基线复现（序列级 IS，lr 3e-5）—— B16 已有，但那次没记 response_length 的完整序列，重跑一次做同尺子分母
# ⚠️ 用 launch_rl 自带的 `--rollout-is`（它本来就有这个参数），别用 Hydra override ——
#    launch_rl 自己会写一条 `algorithm.rollout_correction.rollout_is=…`，两条并存要靠顺序，容易错。
e20run e20_seqis_sync4 --sync-every 4 --rollout-is sequence
# ② ★ token 级 IS —— 本批的正题
e20run e20_tokenis_sync4 --sync-every 4 --rollout-is token
# ③ 陈旧度对照：同 lr、同 IS 口径，只把同步频率拉到每步一次
e20run e20_seqis_sync1 --sync-every 1 --rollout-is sequence

log "════════ batch7（E20）结束"
echo "batch7 done $(date '+%F %T')" >> logs/BATCH_DONE
