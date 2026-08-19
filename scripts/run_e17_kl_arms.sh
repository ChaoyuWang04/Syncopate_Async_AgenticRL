#!/usr/bin/env bash
# E17 · KL 两臂（A 现状 / B 全砍）—— 串行跑，判据是**任务尺子 + defer 双向率**，不是吞吐。
#
# ★ 为什么现在跑（2026-08-19）
#   吞吐侧 12.7% 已实测（b12，2026-08-17），但那是在 E21/E22 修复**之前**；
#   精度侧的 b16 两跑同样在修复之前，且权重已删 ⇒ **任务尺子那一关从来没过过。**
#
# ★ 白捡的第二个问题：这是**第一次在 max_prompt_length=5120 下跑的任务分实验**。
#   夜跑 19 项全在 3584 下，100% 被左截断（中位砍 573 token，砍掉的正是
#   「每步只输出一个 tool call」「调查先于任何结论」等规则，存活 0/659）。
#   ⇒ A 臂与 r1_tokenis **只差 prompt 预算这一个变量**
#   ⇒ 若 defer 从 83% 回到 ~97%，那 defer 崩塌就是**截断造成的**，不是 reward 设计。
#
# ⚠️ 契约参数一个都不传（纪律⑨）：长度预算与采样参数唯一来源是 rollout_budget.py。
# ⚠️ 两臂都必须 keep ckpt —— 不过任务尺子的跑等于没跑。
set -uo pipefail
cd "$(dirname "$0")/.."
set -a; . /workspace/.env 2>/dev/null || true; set +a

Q=logs/e17kl
mkdir -p "$Q" _audit/infra
BOARD="$Q/BOARD.md"
say () { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$Q/queue.log"; }

COMMON=(
  --model models/Qwen3-4B-sft-v13-e1
  --train-file data/rl/v13/train.parquet --val-file data/rl/v13/val.parquet
  --lora-rank 32 --train-batch-size 6 --rollout-n 8 --micro-batch-size 1
  --max-num-seqs 64 --object-store-gb 2
  --save-freq 999 --wandb-mode offline --logger console --dynamic-bsz False
  --max-token-len-per-gpu 16384 --mode fully_async --trainer-gpus 3 --rollout-gpus 1
  --rollout-is token --lr 3e-5 --ppo-mini-batch-size 6 --steps 60 --sync-every 4
)
export SYNCOPATE_SYNC_PAYLOAD=1
export SYNCOPATE_SYNC_REF=75.377708
export SYNCOPATE_SYNC_WATCH="model.layers.0.self_attn.q_proj.base_layer.weight"

preflight () {
  local free_g; free_g=$(df -BG --output=avail /workspace | tail -1 | tr -dc '0-9')
  if [ "${free_g:-0}" -lt 40 ]; then say "🔴 $1：磁盘只剩 ${free_g}G（<40G）⇒ 跳过"; return 1; fi
  local waited=0 busiest
  while :; do
    busiest=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
    [ "${busiest:-99999}" -lt 2000 ] && break
    [ "$waited" -ge 900 ] && { say "⚠️ $1：等了 900s，最忙的卡仍 ${busiest}MB，继续"; break; }
    sleep 15; waited=$((waited+15))
  done
  say "前置检查通过（磁盘 ${free_g}G · 最忙的卡 ${busiest}MB · 等了 ${waited}s）"
}

verdict () {   # $1=名字 $2=日志
  local name="$1" log="$2" out="$Q/${name}.done"
  {
    echo "# $name"; echo "结束时间: $(date '+%F %T')"
    [ -f "$log" ] || { echo "🔴 没有日志"; exit; }
    echo "判据① list_loras : $(grep -o 'list_loras()=\[[0-9]*\]' "$log" | sort | uniq -c | tr '\n' ' ')"
    echo "判据② 载荷       : $(grep -o '本次同步推出去：[0-9]* 个张量 / [0-9,.]* MiB / 其中 lora_ [0-9]* 个' "$log" | sort -u | tr '\n' ' ')"
    echo "判据③ kl 轨迹    : $(grep -o 'rollout_corr/kl:[0-9.e+-]*' "$log" | sed 's/.*://' | awk '{printf "%.5f ",$1}')"
    # ★ 本实验专属判据：prompt 截断必须归零（这是它同时回答第二个问题的地方）
    echo "★ prompt_length/clip_ratio : $(grep -o 'prompt_length/clip_ratio:[0-9.]*' "$log" | sed 's/.*://' | sort -u | tr '\n' ' ')  ← **必须是 0**"
    echo "★ prompt_length/mean       : $(grep -o 'prompt_length/mean:[0-9.]*' "$log" | sed 's/.*://' | sort -u | head -3 | tr '\n' ' ')"
    # ★ B 臂的机制判据：timing_s/ref 必须一次都不出现
    echo "★ timing_s/ref 出现次数    : $(grep -c 'timing_s/ref:' "$log")  ← B 臂必须是 0"
    for k in timing_s/step timing_s/update_actor timing_s/old_log_prob timing_s/ref \
             actor/kl_loss rollout_corr/rollout_is_eff_sample_size critic/score/mean; do
      v=$(grep -o "${k}:[0-9.e+-]*" "$log" | sed 's/.*://' | awk 'NR==1{f=$1} {l=$1; s+=$1} END{if(NR)printf "%s → %s（中位约 %.3f，%d 次）",f,l,s/NR,NR}')
      [ -n "$v" ] && echo "  ${k}: $v"
    done
    echo "错误: $(grep -c -i 'RayTaskError' "$log") 处 RayTaskError"
    echo "UserWarning: $(grep -c 'UserWarning' "$log") 条（★ 新出现的必须有人看过）"
  } > "$out" 2>&1
  { echo; echo "## $name"; sed 's/^/    /' "$out"; } >> "$BOARD"
  say "── $name 判据已落盘 → $out"
}

b5 () {        # $1=实验名 —— 过任务尺子，配对基线用 merged（E24）
  local name="$1"
  local ck; ck=$(ls -d checkpoints/grpo/${name}/global_step_*/actor 2>/dev/null | tail -1)
  if [ -z "$ck" ]; then echo "🔴 找不到 $name 的 ckpt" > "$Q/${name}_b5.done"; return; fi
  .venv/bin/python scripts/rl_ckpt_to_adapter.py "$ck" --out "models/adapters/${name}" \
    > "$Q/${name}_adapter.log" 2>&1
  if [ ! -d "models/adapters/${name}" ]; then
    { echo "🔴 adapter 转换失败"; tail -10 "$Q/${name}_adapter.log"; } > "$Q/${name}_b5.done"; return
  fi
  MODEL=models/Qwen3-4B-sft-v13-e1 timeout 3600 \
    bash scripts/eval_parallel.sh "models/adapters/${name}" "_audit/${name}.json" \
    > "$Q/${name}_eval.log" 2>&1
  {
    echo "# B5 · $name"; echo "结束时间: $(date '+%F %T')"
    if [ -f "_audit/${name}.json.done" ]; then
      echo "✅ 与合法基线（_audit/v13_sft_e1_merged.json）配对"
      .venv/bin/python -m syncopate.train.compare _audit/v13_sft_e1_merged.json "_audit/${name}.json" 2>&1 | tail -30
    else
      echo "🔴 没有 .done"; tail -20 "$Q/${name}_eval.log"
    fi
  } > "$Q/${name}_b5.done" 2>&1
  { echo; echo "## B5 · $name"; sed 's/^/    /' "$Q/${name}_b5.done"; } >> "$BOARD"
  say "── $name 任务尺子已落盘"
}

arm () {       # $1=名字 $2..=额外参数
  local name="$1"; shift
  preflight "$name" || return
  say "════════ $name 开始"
  ( set -x; timeout 7200 .venv/bin/python -m syncopate.train.launch_rl \
      "${COMMON[@]}" --save-path "checkpoints/grpo/$name" --experiment "$name" "$@" ) \
      > "logs/${name}.log" 2>&1
  say "──────── $name 退出码 $?"
  verdict "$name" "logs/${name}.log"
  b5 "$name"
}

echo "# E17 · KL 两臂 · 开始于 $(date '+%F %T')" > "$BOARD"
say "════════════ E17 KL 两臂启动（A 现状 → B 全砍）════════════"

arm e17a_kl_on  --use-kl-loss True
arm e17b_kl_off --use-kl-loss False

# ★ 最后一步：两臂**直接配对**（不经基线），这才是本实验真正的判据
say "════════ 直接配对 A vs B"
{
  echo "# E17 · A（KL 开）vs B（KL 关）直接配对"
  echo "★ 这是本实验的主判据：看 defer 双向率与三计数，不看均值"
  if [ -f _audit/e17a_kl_on.json.done ] && [ -f _audit/e17b_kl_off.json.done ]; then
    .venv/bin/python -m syncopate.train.compare _audit/e17a_kl_on.json _audit/e17b_kl_off.json 2>&1 | tail -30
  else
    echo "🔴 两臂的审计没齐，无法配对"
  fi
} > "$Q/AB_paired.done" 2>&1
{ echo; echo "## A vs B 直接配对"; sed 's/^/    /' "$Q/AB_paired.done"; } >> "$BOARD"
touch "$Q/ALL_DONE"
say "════════════ 全部完成 ════════════"
