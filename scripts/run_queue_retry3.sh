#!/usr/bin/env bash
# 补跑队列 v3 —— 等补跑 v2 结束后跑。
#
# 为什么加这一段（2026-08-18 队列 T7 之后的判断）：
#   T7（lr 1e-4）训练分 +0.123 略高于 T2（lr 3e-5）的 +0.109，
#   **但有两个方向相反的信号**：`grad_norm` 在降（0.1415→0.0919）、`response_length` 在缩（591→561）。
#   而 T4 已经显示 token 臂有「该拒绝时少拒绝 14 个点」的代价。
#   ⇒ **lr 更大是不是把这条代价放大了？只有任务尺子能答。**
#   ⚠️ 而 T7 的 ckpt 按队列策略被清了（排队列时只给 R1 两臂留） ⇒ 必须带 ckpt 重跑一次。
#
# ★ 顺带补上 §5.2 的第 6 条里最便宜的一个探针（Q4：失败注入在组内是否确定性）——
#   它不吃 GPU，读已有的 rollout_dumps 就行。
set -uo pipefail
cd "$(dirname "$0")/.."
set -a; . /workspace/.env 2>/dev/null || true; set +a
Q=logs/queue9h; mkdir -p "$Q"
say () { echo "[$(date '+%H:%M:%S')] [retry3] $*" | tee -a "$Q/queue.log"; }
export SYNCOPATE_SYNC_PAYLOAD=1 SYNCOPATE_SYNC_REF=75.377708
export SYNCOPATE_SYNC_WATCH="model.layers.0.self_attn.q_proj.base_layer.weight"

say "等补跑 v2 结束…"
until [ -f "$Q/RETRY_ALL_DONE" ]; do
  pgrep -f "run_queue_retry2[.]sh" >/dev/null 2>&1 || { say "补跑 v2 进程已消失"; break; }
  sleep 60
done
while :; do b=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
  [ "${b:-99999}" -lt 2000 ] && break; sleep 15; done

# ── T7b · lr 1e-4，**这次留 ckpt**，要过任务尺子
say "════════ T7b · e20f_lr1e4_keep（带 ckpt）"
rm -rf checkpoints/grpo/e20f_lr1e4_keep
timeout 7200 .venv/bin/python -m syncopate.train.launch_rl \
  --model models/Qwen3-4B-sft-v13-e1 \
  --train-file data/rl/v13/train.parquet --val-file data/rl/v13/val.parquet \
  --lora-rank 32 --train-batch-size 6 --rollout-n 8 --micro-batch-size 1 \
  --max-num-seqs 64 --object-store-gb 2 --max-prompt-length 3584 --max-response-length 1536 \
  --save-freq 999 --wandb-mode offline --logger console --dynamic-bsz False \
  --max-token-len-per-gpu 16384 --mode fully_async --trainer-gpus 3 --rollout-gpus 1 \
  --weight-sync-bucket-mb 512 --rollout-is token --lr 1e-4 --ppo-mini-batch-size 6 \
  --steps 60 --sync-every 4 \
  --save-path checkpoints/grpo/e20f_lr1e4_keep --experiment e20f_lr1e4_keep \
  > logs/e20f_lr1e4_keep.log 2>&1
say "──────── T7b 退出码 $?"

ck=$(ls -d checkpoints/grpo/e20f_lr1e4_keep/global_step_*/actor 2>/dev/null | tail -1)
if [ -n "$ck" ]; then
  .venv/bin/python scripts/rl_ckpt_to_adapter.py "$ck" --out models/adapters/e20f_lr1e4 > "$Q/T7b_adapter.log" 2>&1
  if [ -d models/adapters/e20f_lr1e4 ]; then
    while :; do b=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
      [ "${b:-99999}" -lt 2000 ] && break; sleep 15; done
    MODEL=models/Qwen3-4B-sft-v13-e1 timeout 3600 \
      bash scripts/eval_parallel.sh models/adapters/e20f_lr1e4 _audit/e20f_lr1e4.json > "$Q/T7b_eval.log" 2>&1
  fi
  { echo "# T7b · lr 1e-4 过任务尺子"; echo "结束时间: $(date '+%F %T')"
    if [ -f _audit/e20f_lr1e4.json.done ]; then
      echo "★ 与合法基线配对（看总分与三计数）："
      .venv/bin/python -m syncopate.train.compare _audit/v13_sft_e1_merged.json _audit/e20f_lr1e4.json 2>&1 | head -18
      echo; echo "★★ 与 lr 3e-5 那一臂直接配对（**lr 是不是把「少拒绝」这条代价放大了**）："
      .venv/bin/python -m syncopate.train.compare _audit/r1_tokenis.json _audit/e20f_lr1e4.json 2>&1 | head -18
    else echo "🔴 没有 .done"; tail -20 "$Q/T7b_eval.log"; fi
  } > "$Q/T7b.done" 2>&1
  say "── T7b 判据已落盘"
  find checkpoints/grpo/e20f_lr1e4_keep -name "*.pt" -delete 2>/dev/null
fi

# ── 管线不变量复查（不吃 GPU）⚠️ 原计划的 Q4 做不了：dump 里没有注入字段
say "════════ 管线不变量在干净基线上复查（不占卡）"
.venv/bin/python scripts/probe_pipeline_invariants_on_clean_runs.py > "$Q/INV.done" 2>&1 || true
say "── 管线不变量判据已落盘"

touch "$Q/RETRY3_ALL_DONE"; say "════════ 补跑 v3 结束"
