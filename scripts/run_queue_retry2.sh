#!/usr/bin/env bash
# 补跑队列 v2 —— 等主队列结束后，重跑三个失败项。
#
# 失败原因各不相同，都已修：
#   T3 r1_seqis         解析器被模型**畸形输出**打崩（parsing.py:108，已修 + 回归测试）
#   T6 e20e_mini2       `mini_batch 2 × rollout_n 8 = 16` **不能被 3 个 rank 整除**
#   T8 e20g_mini2_lr1e4 同上
#   ⇒ 后两个改用**合法值 mini_batch=3**（3×8=24，24%3=0）
#     更新次数从 1 次/fit-step 变成 **2 次**（不是原计划的 3 次 —— mini_batch=2 在本机非法）
#   ⇒ 并已给 launch_rl 加启动守卫：非法组合秒炸并列出可用值 [3,6,9,12,15]
set -uo pipefail
cd "$(dirname "$0")/.."
set -a; . /workspace/.env 2>/dev/null || true; set +a
Q=logs/queue9h; mkdir -p "$Q"
say () { echo "[$(date '+%H:%M:%S')] [retry2] $*" | tee -a "$Q/queue.log"; }
export SYNCOPATE_SYNC_PAYLOAD=1 SYNCOPATE_SYNC_REF=75.377708
export SYNCOPATE_SYNC_WATCH="model.layers.0.self_attn.q_proj.base_layer.weight"

COMMON=(
  --model models/Qwen3-4B-sft-v13-e1
  --train-file data/rl/v13/train.parquet --val-file data/rl/v13/val.parquet
  --lora-rank 32 --train-batch-size 6 --rollout-n 8 --micro-batch-size 1
  --max-num-seqs 64 --object-store-gb 2 --max-prompt-length 3584 --max-response-length 1536
  --save-freq 999 --wandb-mode offline --logger console --dynamic-bsz False
  --max-token-len-per-gpu 16384 --mode fully_async --trainer-gpus 3 --rollout-gpus 1
  --weight-sync-bucket-mb 512
)
wait_gpu () { while :; do b=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
  [ "${b:-99999}" -lt 2000 ] && break; sleep 15; done; }

verdict () {  # $1=标签 $2=实验名
  local tag="$1" name="$2" L="logs/${2}.log"
  { echo "# $tag · $name（补跑）"; echo "结束时间: $(date '+%F %T')"
    echo "判据① list_loras : $(grep -o 'list_loras()=\[[0-9]*\]' "$L" 2>/dev/null | sort | uniq -c | tr '\n' ' ')"
    echo "判据③ kl 轨迹    : $(grep -o 'rollout_corr/kl:[0-9.e+-]*' "$L" 2>/dev/null | sed 's/.*://' | awk '{printf "%.5f ",$1}')"
    for k in rollout_corr/rollout_is_eff_sample_size rollout_corr/chi2_token rollout_corr/chi2_seq \
             rollout_corr/log_ppl_diff actor/grad_norm critic/score/mean response_length/mean; do
      v=$(grep -o "${k}:[0-9.e+-]*" "$L" 2>/dev/null | sed 's/.*://' | awk 'NR==1{f=$1}{l=$1}END{if(NR)printf "%s → %s（%d 次）",f,l,NR}')
      [ -n "$v" ] && echo "  ${k}: $v"; done
    echo "更新次数: $(ls checkpoints/grpo/${name}/rollout_dumps/*.jsonl 2>/dev/null | wc -l)"
    echo "错误: $(grep -c -i 'RayTaskError' "$L" 2>/dev/null) 处"
  } > "$Q/${tag}.done" 2>&1
  say "── $tag 判据已落盘"
}

run1 () {  # $1=标签 $2=实验名 $3=keep $4..=参数
  local tag="$1" name="$2" keep="$3"; shift 3
  wait_gpu; say "════════ $tag · $name 开始"
  rm -rf "checkpoints/grpo/$name"
  timeout 7200 .venv/bin/python -m syncopate.train.launch_rl "${COMMON[@]}" \
    --save-path "checkpoints/grpo/$name" --experiment "$name" "$@" > "logs/${name}.log" 2>&1
  say "──────── $tag 退出码 $?"
  verdict "$tag" "$name"
  local ck; ck=$(ls -d "checkpoints/grpo/${name}"/global_step_*/actor 2>/dev/null | tail -1)
  [ -n "$ck" ] && .venv/bin/python scripts/extract_ckpt_fingerprint.py "$ck" >/dev/null 2>&1
  [ "$keep" = "1" ] || { find "checkpoints/grpo/${name}" -name "*.pt" -delete 2>/dev/null; say "   ckpt 已清"; }
}

say "等主队列结束…"
until [ -f "$Q/QUEUE_ALL_DONE" ]; do
  pgrep -f "run_queue_9h[.]sh" >/dev/null 2>&1 || { say "主队列进程已消失"; break; }
  sleep 60
done

# ── R1 的另一半：序列级 IS（解析器已修）★ ckpt 要留，要过 B5
run1 T3b r1_seqis 1 --rollout-is sequence --lr 3e-5 --ppo-mini-batch-size 6 --steps 60 --sync-every 4

# ── B5 · 序列级那一臂 + ★★ 与 token 级直接配对（这才是 R1 真正要答的问题）
ck=$(ls -d checkpoints/grpo/r1_seqis/global_step_*/actor 2>/dev/null | tail -1)
if [ -n "$ck" ]; then
  .venv/bin/python scripts/rl_ckpt_to_adapter.py "$ck" --out models/adapters/r1_seqis > "$Q/T5b_adapter.log" 2>&1
  if [ -d models/adapters/r1_seqis ]; then
    wait_gpu
    MODEL=models/Qwen3-4B-sft-v13-e1 timeout 3600 \
      bash scripts/eval_parallel.sh models/adapters/r1_seqis _audit/r1_seqis.json > "$Q/T5b_eval.log" 2>&1
  fi
  { echo "# T5b · B5 · r1_seqis（补跑）"; echo "结束时间: $(date '+%F %T')"
    if [ -f _audit/r1_seqis.json.done ]; then
      echo "★ 与合法基线配对："
      .venv/bin/python -m syncopate.train.compare _audit/v13_sft_e1_merged.json _audit/r1_seqis.json 2>&1 | head -18
      echo; echo "★★ 与 token 级那一臂直接配对（R1 真正要答的问题）："
      .venv/bin/python -m syncopate.train.compare _audit/r1_seqis.json _audit/r1_tokenis.json 2>&1 | head -18
    else echo "🔴 没有 .done"; tail -20 "$Q/T5b_eval.log"; fi
  } > "$Q/T5b.done" 2>&1
  say "── T5b 判据已落盘"
fi

# ── E20 原因② · 更新次数（**合法值 mini_batch=3**）
run1 T6b e20e_mini3       0 --rollout-is token --lr 3e-5 --ppo-mini-batch-size 3 --steps 60 --sync-every 4
run1 T8b e20g_mini3_lr1e4 0 --rollout-is token --lr 1e-4 --ppo-mini-batch-size 3 --steps 60 --sync-every 4

touch "$Q/RETRY_ALL_DONE"; say "════════ 补跑 v2 结束"
