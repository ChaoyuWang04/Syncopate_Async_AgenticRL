#!/usr/bin/env bash
# E26 同尺子吞吐 A/B（00-INFRA-HANDOFF §5.1 队首，Chaoyu 2026-08-19 定）
#
# 三臂，一次只变一个变量（同 seed / 同数据 / 同 20 步 / 同 sync-every 4）：
#   on_mb8   SYNCOPATE_PREFIX_GROUPER=1 · micro-batch 8    ← 被测者
#   off_mb8  补丁关                     · micro-batch 8    ← 拆变量：mb=8 本身值多少
#   off_mb1  补丁关                     · micro-batch 1    ← 生产现状基线（E25 定的最优）
#
# 判据（跑完读 logs/queue_e26ab/AB.done）：
#   ① 每臂 parse_fully_async_timing.py：step / update_actor / old_log_prob / ref
#      —— 掐掉前 2 步暖机，报中位数；加速比 = off_mb1 → on_mb8
#   ② on 臂必须出现 [prefix-grouper] 打包前向已生效（对照计数：组构成）
#   ③ 三臂 prompt_length/clip_ratio 全 0.0000
#   ④ 吞吐探针短跑：ckpt 跑完即删（dispatched.jsonl / rollout_dumps 留）
set -u
cd /workspace/Syncopate_Async_AgenticRL
set -a; . /workspace/.env; set +a
Q=logs/queue_e26ab; mkdir -p "$Q"
say() { echo "[$(date '+%T')] $*"; }

COMMON="--model models/Qwen3-4B-sft-v13-e1 \
  --train-file data/rl/v13/train.parquet --val-file data/rl/v13/val.parquet \
  --lora-rank 32 --train-batch-size 6 --rollout-n 8 --ppo-mini-batch-size 6 \
  --max-num-seqs 64 --object-store-gb 2 --seed 42 \
  --save-freq 999 --wandb-mode offline --logger console --dynamic-bsz False \
  --max-token-len-per-gpu 16384 --mode fully_async --trainer-gpus 3 --rollout-gpus 1 \
  --weight-sync-bucket-mb 512 --rollout-is token --steps 20 --sync-every 4"

wait_gpu() { while :; do b=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
  [ "${b:-99999}" -lt 2000 ] && break; sleep 15; done; }

run_arm() {  # $1=臂名  $2=PG开关  $3=micro-batch
  local name=$1 pg=$2 mb=$3
  say "════════ 臂 $name（PG=$pg · micro-batch=$mb）"
  wait_gpu
  rm -rf "checkpoints/grpo/e26ab_$name"
  SYNCOPATE_PREFIX_GROUPER=$pg timeout 5400 .venv/bin/python -m syncopate.train.launch_rl \
    $COMMON --micro-batch-size "$mb" \
    --save-path "checkpoints/grpo/e26ab_$name" --experiment "e26ab_$name" \
    > "logs/e26ab_$name.log" 2>&1
  say "──────── 臂 $name 退出码 $?"
  find "checkpoints/grpo/e26ab_$name" -name "*.pt" -delete 2>/dev/null
}

run_arm on_mb8  1 8
run_arm off_mb8 0 8
run_arm off_mb1 0 1

{ echo "# E26 同尺子吞吐 A/B（20 步 × 3 臂，seed 42）"; echo "结束时间: $(date '+%F %T')"
  for a in on_mb8 off_mb8 off_mb1; do
    echo; echo "══ 臂 $a"
    echo "clip_ratio（必须全 0）:"; grep -oE "prompt_length/clip_ratio:[0-9.]+" "logs/e26ab_$a.log" | sort | uniq -c
    echo "判据A 次数: $(grep -c '打包前向已生效' "logs/e26ab_$a.log")（on 臂必须 >0，off 臂必须 =0）"
    echo "报错计数: $(grep -cE 'RuntimeError|Traceback \(most' "logs/e26ab_$a.log")"
    .venv/bin/python scripts/parse_fully_async_timing.py "logs/e26ab_$a.log" 2>&1 | tail -14
  done
} > "$Q/AB.done" 2>&1
say "── 判据已落盘 $Q/AB.done"
say "════════ ALL DONE"
