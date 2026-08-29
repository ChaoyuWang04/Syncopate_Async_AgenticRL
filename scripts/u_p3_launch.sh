#!/usr/bin/env bash
# v14.5 · P3 RL 发射链（06 §1 清单执行件）：静态检查 → merged 配对基线 → fully_async 3+1
# 起跑（守卫+瘦身挂载）→ 存点即评 watch 循环（每 global 25 步=RL-100 一个点）。
set -u
cd /workspace/Syncopate_Async_AgenticRL
set -a; . /workspace/.env; set +a
source .venv/bin/activate
say(){ echo "[P3 $(date +%H:%M:%S)] $*"; }
BASE=models/Qwen3-4B-sft-v14.5-epoch3
EXP=cand_v145_e3

say "① 静态检查器 + 磁盘"
python scripts/check_pipeline_invariants.py > logs/u_route/p3_invariants.log 2>&1 \
  || { grep -E "🔴|违反" logs/u_route/p3_invariants.log | head -8; echo INVARIANTS-FAIL; exit 1; }
python scripts/disk_report.py | tail -2

say "② merged 配对基线（RL 尺子=起点模型自己的冻结 EVAL·merged 形态）"
if [ ! -f _audit/v145_e3_merged.json.done ]; then
  MODEL=$BASE bash scripts/eval_parallel.sh "" _audit/v145_e3_merged.json 4 || { echo BASELINE-FAIL; exit 1; }
  until [ -f _audit/v145_e3_merged.json.done ]; do sleep 15; done
fi
.venv/bin/python -m syncopate.train.compare _audit/v13_sft_v13r2_e1_merged.json _audit/v145_e3_merged.json \
  > logs/u_route/p3_base_cmp.txt 2>&1
grep -E "配对差值|结论" logs/u_route/p3_base_cmp.txt | head -2

say "③ 起 RL（fully_async 3+1 · candidate · 守卫+瘦身挂载）"
SYNCOPATE_SYNC_PAYLOAD=1 SYNCOPATE_SYNC_REF=75.378174 \
SYNCOPATE_SYNC_WATCH="model.layers.0.self_attn.q_proj.base_layer.weight" \
nohup .venv/bin/python -m syncopate.train.launch_rl --model "$BASE" \
  --lora-rank 32 --mode fully_async --trainer-gpus 3 --rollout-gpus 1 \
  --train-file data/rl/v13/train.parquet --val-file data/rl/v13/val.parquet \
  --steps 400 --purpose candidate --experiment "$EXP" \
  --save-freq 25 --test-freq 25 > logs/u_route/p3_rl.log 2>&1 &
RLPID=$!
echo $RLPID > /tmp/p3_rl.pid
sleep 30
kill -0 $RLPID 2>/dev/null || { tail -20 logs/u_route/p3_rl.log; echo RL-DEAD-EARLY; exit 1; }
CKDIR=checkpoints/grpo/$EXP
nohup bash scripts/rl_guard.sh logs/u_route/p3_rl.log "$CKDIR" --kill > logs/u_route/p3_guard.log 2>&1 &
nohup bash scripts/rl_ckpt_rolling_prune.sh "$CKDIR" > logs/u_route/p3_prune.log 2>&1 &

say "④ 起跑后 10 分钟判据行自查"
sleep 600
for pat in "\[pool\]" "\[agent-loop\]" "\[lora-probe\]" "\[sync-payload\]"; do
  if grep -q "$pat" logs/u_route/p3_rl.log; then echo "  ✅ $pat"; else echo "  🔴 缺判据行 $pat"; fi
done
grep -m2 "clip_ratio" logs/u_route/p3_rl.log || echo "  （clip_ratio 行未出现，稍后看 wandb）"
echo "  UserWarning 计数: $(grep -c UserWarning logs/u_route/p3_rl.log)"

say "⑤ 等 RL 跑完（fully_async 3+1 吃满四卡，训练期无卡可评——存点回头逐点评）"
while kill -0 $RLPID 2>/dev/null; do
  sleep 300
  tail -1 logs/u_route/p3_rl.log | cut -c1-120
done
say "RL 进程退出，逐存点评测（每 global25=RL-100 一个点）"
for d in "$CKDIR"/adapter_global_step_*; do
  [ -d "$d" ] || continue
  s=$(basename "$d" | grep -oE '[0-9]+$')
  say "  评 step_$s（=RL-$((s*4)) 步）"
  rm -f "_audit/v145_rl_s$s.json.done"
  MODEL=$BASE bash scripts/eval_parallel.sh "$d" "_audit/v145_rl_s$s.json" 4 \
    || { echo "EVAL-s$s-FAIL"; continue; }
  until [ -f "_audit/v145_rl_s$s.json.done" ]; do sleep 15; done
  .venv/bin/python -m syncopate.train.compare _audit/v145_e3_merged.json "_audit/v145_rl_s$s.json" \
    > "logs/u_route/p3_cmp_s$s.txt" 2>&1
  say "  s$s: $(grep -m1 '配对差值' logs/u_route/p3_cmp_s$s.txt | tr -s ' ')"
done
echo P3-RL-DONE
