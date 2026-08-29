#!/bin/bash
# P3 跑后接管 watcher（独立于发射链——编辑运行中 bash 的偏移错乱风险规避）
cd /workspace/Syncopate_Async_AgenticRL
set -a; . /workspace/.env; set +a
source .venv/bin/activate
RLPID=$(cat /tmp/p3_rl.pid)
BASE=models/Qwen3-4B-sft-v14.5-epoch3
CKDIR=checkpoints/grpo/smoke
say(){ echo "[W3 $(date +%H:%M:%S)] $*"; }
say "接管：等 RL($RLPID) 跑完"
while kill -0 $RLPID 2>/dev/null; do sleep 120; done
say "RL 退出，提取+逐点评测"
EVALED=""
for d in "$CKDIR"/adapter_global_step_* "$CKDIR"/global_step_*; do
  [ -d "$d" ] || continue
  s=$(basename "$d" | grep -oE '[0-9]+$')
  case " $EVALED " in *" $s "*) continue;; esac
  EVALED="$EVALED $s"
  if [ -d "$d/actor" ]; then
    ad="$CKDIR/adapter_global_step_$s"
    [ -d "$ad" ] || .venv/bin/python scripts/rl_ckpt_to_adapter.py "$d/actor" --out "$ad" \
      || { say "  提取失败 step_$s"; continue; }
    d="$ad"
  fi
  say "  评 step_$s（≈RL-$((s*16)) rollout 步）"
  rm -f "_audit/v145_rl_s$s.json.done"
  MODEL=$BASE bash scripts/eval_parallel.sh "$d" "_audit/v145_rl_s$s.json" 4 || { say "EVAL-s$s-FAIL"; continue; }
  until [ -f "_audit/v145_rl_s$s.json.done" ]; do sleep 15; done
  .venv/bin/python -m syncopate.train.compare _audit/v145_e3_merged.json "_audit/v145_rl_s$s.json" \
    > "logs/u_route/p3_cmp_s$s.txt" 2>&1
  say "  s$s: $(grep -m1 '配对差值' logs/u_route/p3_cmp_s$s.txt | tr -s ' ')"
done
echo P3-WATCH3-DONE
