#!/usr/bin/env bash
# 第四段队列 —— 等补跑 v3 结束后跑，把 9 小时填满。
#
# 三项，每一项都是**今晚的实验直接逼出来的新问题**（不是补跑）：
#
#   A · 优化器步数到底是多少（E20 §7.8）
#       今晚发现：`training/global_step` / dump 文件数 / param_version **三个都不是更新次数**，
#       而 E20 原因②「一个 epoch 只更新 110 次」整条结论建立在这个数上。
#       ⇒ 装上真正的计数器（SYNCOPATE_OPT_STEP_PROBE=1），同一配置跑 mini_batch 6 与 3 各一次，
#         **直接数 optimizer_step 被调了几次**。
#
#   B · 那 11.4% 省在哪（E08 §5.3）
#       param_sync 只占一步 0.2–0.7%，而 sync_every 4→16 省了 11.4% ⇒ **差 23 倍**。
#       ⇒ 开分步计时（SYNCOPATE_SYNC_TIMING=1），在**新载荷（252 MB）**下重拆那 8 步。
#       ⚠️ E12 拆过，但那是在"每次搬 8.4 GB"的错前提下 —— 构成已经完全变了。
#
#   C · 序列级 IS 的 ESS 会不会真的塌（E20 §7.7.3 的最大 caveat）
#       60 步里它是 0.909 → 0.852，**在衰减**。判决说两臂任务分打平，
#       但那条曲线只画到 60 步。⇒ 拉到 120 步看它到底会不会塌下去。
set -uo pipefail
cd "$(dirname "$0")/.."
set -a; . /workspace/.env 2>/dev/null || true; set +a
Q=logs/queue9h; mkdir -p "$Q"
say () { echo "[$(date '+%H:%M:%S')] [q4] $*" | tee -a "$Q/queue.log"; }
export SYNCOPATE_SYNC_PAYLOAD=1 SYNCOPATE_SYNC_REF=75.377708
export SYNCOPATE_SYNC_WATCH="model.layers.0.self_attn.q_proj.base_layer.weight"

COMMON=(
  --model models/Qwen3-4B-sft-v13-e1
  --train-file data/rl/v13/train.parquet --val-file data/rl/v13/val.parquet
  --lora-rank 32 --train-batch-size 6 --rollout-n 8 --micro-batch-size 1
  --max-num-seqs 64 --object-store-gb 2
  --save-freq 999 --wandb-mode offline --logger console --dynamic-bsz False
  --max-token-len-per-gpu 16384 --mode fully_async --trainer-gpus 3 --rollout-gpus 1
  --weight-sync-bucket-mb 512 --rollout-is token
)
wait_gpu () { while :; do b=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
  [ "${b:-99999}" -lt 2000 ] && break; sleep 15; done; }
disk_ok () { local g; g=$(df -BG --output=avail /workspace | tail -1 | tr -dc '0-9')
  [ "${g:-0}" -ge 40 ] || { say "🔴 磁盘只剩 ${g}G，跳过"; return 1; }; return 0; }

runq () {  # $1=标签 $2=名字 $3..=参数
  local tag="$1" name="$2"; shift 2
  disk_ok || return 0
  wait_gpu; say "════════ $tag · $name 开始"
  rm -rf "checkpoints/grpo/$name"
  timeout 9000 .venv/bin/python -m syncopate.train.launch_rl "${COMMON[@]}" \
    --save-path "checkpoints/grpo/$name" --experiment "$name" "$@" > "logs/${name}.log" 2>&1
  say "──────── $tag 退出码 $?"
  local L="logs/${name}.log"
  { echo "# $tag · $name"; echo "结束时间: $(date '+%F %T')"
    echo "★ **真实**优化器更新次数: $(grep -o '\\[opt-step\\] 本 rank 累计 optimizer_step 调用 = [0-9]*' "$L" | tail -1 | grep -o '[0-9]*$')"
    echo "  （对照：global_step=$(grep -o 'training/global_step:[0-9]*' "$L" | tail -1 | sed 's/.*://')"
    echo "   · dump 文件数=$(ls checkpoints/grpo/${name}/rollout_dumps/*.jsonl 2>/dev/null | wc -l)"
    echo "   · 每 dump 行数=$(wc -l < "$(ls checkpoints/grpo/${name}/rollout_dumps/*.jsonl 2>/dev/null | head -1)" 2>/dev/null)）"
    echo "判据① list_loras : $(grep -o 'list_loras()=\[[0-9]*\]' "$L" | sort | uniq -c | tr '\n' ' ')"
    echo "判据③ kl        : $(grep -o 'rollout_corr/kl:[0-9.e+-]*' "$L" | sed 's/.*://' | awk '{printf "%.5f ",$1}')"
    for k in rollout_corr/rollout_is_eff_sample_size rollout_corr/chi2_seq actor/grad_norm critic/score/mean; do
      v=$(grep -o "${k}:[0-9.e+-]*" "$L" | sed 's/.*://' | awk 'NR==1{f=$1}{l=$1}END{if(NR)printf "%s → %s（%d 次）",f,l,NR}')
      [ -n "$v" ] && echo "  ${k}: $v"; done
    echo "分步计时（若开了 SYNC_TIMING）:"; grep -o '\[sync-timing\].*' "$L" | tail -12
    echo "错误: $(grep -c -i 'RayTaskError' "$L") 处"
  } > "$Q/${tag}.done" 2>&1
  say "── $tag 判据已落盘"
  find "checkpoints/grpo/${name}" -name "*.pt" -delete 2>/dev/null
}

say "等补跑 v3 结束…"
until [ -f "$Q/RETRY3_ALL_DONE" ]; do
  pgrep -f "run_queue_retry3[.]sh" >/dev/null 2>&1 || { say "补跑 v3 已消失"; break; }
  sleep 60
done

# A · 真实优化器步数：同配置只改 mini_batch
export SYNCOPATE_OPT_STEP_PROBE=1
runq A1 opt_mb6 --lr 3e-5 --ppo-mini-batch-size 6 --steps 24 --sync-every 4
runq A2 opt_mb3 --lr 3e-5 --ppo-mini-batch-size 3 --steps 24 --sync-every 4
unset SYNCOPATE_OPT_STEP_PROBE

# B · 那 11.4% 省在哪：新载荷下重拆同步的 8 步
export SYNCOPATE_SYNC_TIMING=1
runq B1 synctiming_new --lr 3e-5 --ppo-mini-batch-size 6 --steps 24 --sync-every 4
unset SYNCOPATE_SYNC_TIMING

# C · 序列级 IS 拉长到 120 步，看 ESS 到底塌不塌
runq C1 seqis_long120 --rollout-is sequence --lr 3e-5 --ppo-mini-batch-size 6 --steps 120 --sync-every 4

touch "$Q/Q4_ALL_DONE"; say "════════ 第四段结束"
