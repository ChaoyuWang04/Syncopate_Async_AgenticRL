#!/bin/bash
cd /workspace/Syncopate_Async_AgenticRL
set -a; . /workspace/.env; set +a
source .venv/bin/activate
RLPID=$(cat /tmp/p3_rl.pid)
BASE=models/Qwen3-4B-sft-v14.5-epoch3
CK=checkpoints/grpo/smoke
while kill -0 $RLPID 2>/dev/null; do sleep 300; done
echo "[watch2] RL 退出，逐点评（先评 s25-s100 四点=RL-100..400 口径）"
for s in 25 50 75 100; do
  d="$CK/adapter_global_step_$s"; [ -d "$d" ] || d="$CK/global_step_$s"
  [ -d "$d" ] || { echo "  缺 step_$s"; continue; }
  rm -f "_audit/v145_rl_s$s.json.done"
  MODEL=$BASE bash scripts/eval_parallel.sh "$d" "_audit/v145_rl_s$s.json" 4 || { echo EVAL-s$s-FAIL; continue; }
  until [ -f "_audit/v145_rl_s$s.json.done" ]; do sleep 15; done
  .venv/bin/python -m syncopate.train.compare _audit/v145_e3_merged.json "_audit/v145_rl_s$s.json" > "logs/u_route/p3_cmp_s$s.txt" 2>&1
  echo "  s$s: $(grep -m1 '配对差值' logs/u_route/p3_cmp_s$s.txt | tr -s ' ')"
done
echo P3-WATCH2-DONE
