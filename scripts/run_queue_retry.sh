#!/usr/bin/env bash
# 补跑队列 —— 等主队列结束后，重跑因**解析器崩溃**而失败的那一项。
#
# 起因：queue9h 的 T3（r1_seqis）3 分钟就死在 `parsing.py:108 payload.get("name")`
#   —— 模型吐出一个 payload 是**字符串**的 tool_call，合法 JSON 但不是对象 ⇒ AttributeError
#   ⇒ 整个 rollout 被打崩、拖垮一整跑。已修（畸形丢弃而非崩溃）+ 回归测试。
#
# ★ 为什么 T2 不用重跑：那条 bug 只以**崩溃**的形式表现，而 T2 没崩
#   ⇒ T2 实际走过的代码路径在修复前后**逐字节相同**，修复对它是 no-op。
set -uo pipefail
cd "$(dirname "$0")/.."
set -a; . /workspace/.env 2>/dev/null || true; set +a
Q=logs/queue9h; mkdir -p "$Q"
say () { echo "[$(date '+%H:%M:%S')] [retry] $*" | tee -a "$Q/queue.log"; }

say "等主队列结束…"
until [ -f "$Q/QUEUE_ALL_DONE" ]; do
  pgrep -f "run_queue_9h.sh" >/dev/null 2>&1 || { say "主队列进程已消失"; break; }
  sleep 60
done
while :; do b=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
  [ "${b:-99999}" -lt 2000 ] && break; sleep 15; done
say "════════ R3' · r1_seqis 补跑（解析器已修）"

export SYNCOPATE_SYNC_PAYLOAD=1 SYNCOPATE_SYNC_REF=75.377708
export SYNCOPATE_SYNC_WATCH="model.layers.0.self_attn.q_proj.base_layer.weight"
rm -rf checkpoints/grpo/r1_seqis
timeout 7200 .venv/bin/python -m syncopate.train.launch_rl \
  --model models/Qwen3-4B-sft-v13-e1 \
  --train-file data/rl/v13/train.parquet --val-file data/rl/v13/val.parquet \
  --lora-rank 32 --train-batch-size 6 --rollout-n 8 --micro-batch-size 1 \
  --max-num-seqs 64 --object-store-gb 2 --max-prompt-length 3584 --max-response-length 1536 \
  --save-freq 999 --wandb-mode offline --logger console --dynamic-bsz False \
  --max-token-len-per-gpu 16384 --mode fully_async --trainer-gpus 3 --rollout-gpus 1 \
  --rollout-is sequence --lr 3e-5 --ppo-mini-batch-size 6 --steps 60 --sync-every 4 \
  --weight-sync-bucket-mb 512 \
  --save-path checkpoints/grpo/r1_seqis --experiment r1_seqis > logs/r1_seqis.log 2>&1
say "──────── r1_seqis 补跑退出码 $?"

{ echo "# R3' · r1_seqis（补跑）"; echo "结束时间: $(date '+%F %T')"
  L=logs/r1_seqis.log
  echo "判据① list_loras : $(grep -o 'list_loras()=\[[0-9]*\]' $L | sort | uniq -c | tr '\n' ' ')"
  echo "判据③ kl 轨迹    : $(grep -o 'rollout_corr/kl:[0-9.e+-]*' $L | sed 's/.*://' | awk '{printf "%.5f ",$1}')"
  for k in rollout_corr/rollout_is_eff_sample_size rollout_corr/chi2_token rollout_corr/chi2_seq \
           rollout_corr/log_ppl_diff actor/grad_norm critic/score/mean; do
    v=$(grep -o "${k}:[0-9.e+-]*" $L | sed 's/.*://' | awk 'NR==1{f=$1}{l=$1}END{if(NR)printf "%s → %s（%d 次）",f,l,NR}')
    [ -n "$v" ] && echo "  ${k}: $v"; done
  echo "更新次数: $(ls checkpoints/grpo/r1_seqis/rollout_dumps/*.jsonl 2>/dev/null | wc -l)"
  echo "错误: $(grep -c -i 'RayTaskError' $L) 处"
} > "$Q/T3b.done" 2>&1
say "── T3b 判据已落盘"

# 接着过任务尺子，与 T1 的合法基线配对
ck=$(ls -d checkpoints/grpo/r1_seqis/global_step_*/actor 2>/dev/null | tail -1)
if [ -n "$ck" ]; then
  .venv/bin/python scripts/rl_ckpt_to_adapter.py "$ck" --out models/adapters/r1_seqis > "$Q/T5b_adapter.log" 2>&1
  [ -d models/adapters/r1_seqis ] && MODEL=models/Qwen3-4B-sft-v13-e1 timeout 3600 \
    bash scripts/eval_parallel.sh models/adapters/r1_seqis _audit/r1_seqis.json > "$Q/T5b_eval.log" 2>&1
  { echo "# T5b · B5 · r1_seqis（补跑）"; echo "结束时间: $(date '+%F %T')"
    if [ -f _audit/r1_seqis.json.done ]; then
      .venv/bin/python -m syncopate.train.compare _audit/v13_sft_e1_merged.json _audit/r1_seqis.json 2>&1 | tail -25
    else echo "🔴 没有 .done"; tail -20 "$Q/T5b_eval.log"; fi
  } > "$Q/T5b.done" 2>&1
  say "── T5b 判据已落盘"
fi
touch "$Q/RETRY_ALL_DONE"; say "════════ 补跑结束"
