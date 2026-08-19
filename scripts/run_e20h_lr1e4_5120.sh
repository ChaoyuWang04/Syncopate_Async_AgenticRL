#!/usr/bin/env bash
# 队首任务 ① · 5120 预算下重测 lr 1e-4（00-INFRA-HANDOFF §5.1-1）
#
# 为什么重测：夜跑 e20f（lr 1e-4）「defer→0%」量在 prompt 100% 截断之下（E20 §7.12 翻案）。
# 现在 rollout_budget.py 共用预算已是 5120/2048 ⇒ 同配方重跑天然就是干净输入版。
#
# 配方 = run_queue_retry3.sh T7b 原样，仅改名（不覆盖 3584 的旧审计）。
# 判据（跑完读 logs/queue_e20h/T.done）：
#   ① 与合法基线 v13_sft_e1_merged 配对：总分/三计数/行为读数
#   ② 与 lr 3e-5 @5120 臂（e17a_kl_on）直接配对 ⇒ 单变量 = lr
#   ③ 核心问题：干净输入下 lr 1e-4 还会不会把「该 defer」打掉（旧数据 97%→0% 是截断造的假）
#   ④ prompt_length/clip_ratio 必须 0.0000（第四常驻判据）
set -u
cd /workspace/Syncopate_Async_AgenticRL
set -a; . /workspace/.env; set +a
Q=logs/queue_e20h; mkdir -p "$Q"
say() { echo "[$(date '+%T')] $*"; }

say "════════ e20h · lr 1e-4 @5120（重测，带 ckpt 过任务尺子）"
rm -rf checkpoints/grpo/e20h_lr1e4_5120
SYNCOPATE_SYNC_PAYLOAD=1 SYNCOPATE_SYNC_REF=75.377708 \
SYNCOPATE_SYNC_WATCH="model.layers.0.self_attn.q_proj.base_layer.weight" \
timeout 10800 .venv/bin/python -m syncopate.train.launch_rl \
  --model models/Qwen3-4B-sft-v13-e1 \
  --train-file data/rl/v13/train.parquet --val-file data/rl/v13/val.parquet \
  --lora-rank 32 --train-batch-size 6 --rollout-n 8 --micro-batch-size 1 \
  --max-num-seqs 64 --object-store-gb 2 \
  --save-freq 999 --wandb-mode offline --logger console --dynamic-bsz False \
  --max-token-len-per-gpu 16384 --mode fully_async --trainer-gpus 3 --rollout-gpus 1 \
  --weight-sync-bucket-mb 512 --rollout-is token --lr 1e-4 --ppo-mini-batch-size 6 \
  --steps 60 --sync-every 4 \
  --save-path checkpoints/grpo/e20h_lr1e4_5120 --experiment e20h_lr1e4_5120 \
  > logs/e20h_lr1e4_5120.log 2>&1
say "──────── 训练退出码 $?"

ck=$(ls -d checkpoints/grpo/e20h_lr1e4_5120/global_step_*/actor 2>/dev/null | tail -1)
if [ -n "$ck" ]; then
  .venv/bin/python scripts/rl_ckpt_to_adapter.py "$ck" --out models/adapters/e20h_lr1e4_5120 > "$Q/adapter.log" 2>&1
  if [ -d models/adapters/e20h_lr1e4_5120 ]; then
    while :; do b=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
      [ "${b:-99999}" -lt 2000 ] && break; sleep 15; done
    MODEL=models/Qwen3-4B-sft-v13-e1 timeout 3600 \
      bash scripts/eval_parallel.sh models/adapters/e20h_lr1e4_5120 _audit/e20h_lr1e4_5120.json > "$Q/eval.log" 2>&1
  fi
  { echo "# e20h · lr 1e-4 @5120 重测"; echo "结束时间: $(date '+%F %T')"
    echo "clip_ratio 检查（必须全 0）:"
    grep -oE "prompt_length/clip_ratio:[0-9.]+" logs/e20h_lr1e4_5120.log | sort | uniq -c
    if [ -f _audit/e20h_lr1e4_5120.json.done ]; then
      echo; echo "★ 与合法基线（v13_sft_e1_merged）配对："
      .venv/bin/python -m syncopate.train.compare _audit/v13_sft_e1_merged.json _audit/e20h_lr1e4_5120.json 2>&1 | head -24
      echo; echo "★★ 与 lr 3e-5 @5120 臂（e17a_kl_on）直接配对 —— 单变量 = lr："
      .venv/bin/python -m syncopate.train.compare _audit/e17a_kl_on.json _audit/e20h_lr1e4_5120.json 2>&1 | head -24
      echo; echo "★★★ 与 3584 的旧 lr 1e-4 臂（e20f）配对 —— 只差长度预算："
      .venv/bin/python -m syncopate.train.compare _audit/e20f_lr1e4.json _audit/e20h_lr1e4_5120.json 2>&1 | head -24
    else echo "🔴 没有 .done"; tail -20 "$Q/eval.log"; fi
  } > "$Q/T.done" 2>&1
  say "── 判据已落盘 $Q/T.done"
  find checkpoints/grpo/e20h_lr1e4_5120 -name "*.pt" -delete 2>/dev/null
else
  echo "🔴 没有 ckpt，训练没跑完" > "$Q/T.done"
fi
say "════════ ALL DONE"
