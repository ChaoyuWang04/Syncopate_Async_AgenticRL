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

say "⓪ 清废跑残留（含孤儿 Ray 集群——守卫只杀 launcher 会留集群吃卡）"
ray stop --force >/dev/null 2>&1; sleep 5
nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | while read -r p; do kill -9 "$p" 2>/dev/null; done
sleep 3
rm -rf checkpoints/grpo/smoke
mkdir -p checkpoints/grpo/smoke

say "① 静态检查器 + 磁盘"
python scripts/check_pipeline_invariants.py > logs/u_route/p3_invariants.log 2>&1 || true
# 遗留红白名单（06 §1「只剩登记在案的遗留红」；均为 08-27 历史审计，与 v14.5 无关）：
#   ① v13_sft_e1 vs v13_rl_s110 基座不可配对（v13 时代审计口径注记）
#   ② e31s12 1/400 重复题（infra E31 已收官实验工件，登记于 E31 §）
new_reds=$(grep "🔴" logs/u_route/p3_invariants.log | grep -v "v13_sft_e1\|e31s12\|条判据被违反" || true)
if [ -n "$new_reds" ]; then
  echo "$new_reds" | head -8
  echo INVARIANTS-FAIL; exit 1
fi
echo "  检查器：仅剩 2 条已登记遗留红（v13 配对注记 · e31s12），无新红 ⇒ 放行"
python scripts/disk_report.py | tail -2

say "② merged 配对基线（RL 尺子=起点模型自己的冻结 EVAL·merged 形态）"
if [ ! -f _audit/v145_e3_merged.json.done ]; then
  MODEL=$BASE bash scripts/eval_parallel.sh "" _audit/v145_e3_merged.json 4 || { echo BASELINE-FAIL; exit 1; }
  until [ -f _audit/v145_e3_merged.json.done ]; do sleep 15; done
fi
.venv/bin/python -m syncopate.train.compare _audit/v13_sft_v13r2_e1_merged.json _audit/v145_e3_merged.json \
  > logs/u_route/p3_base_cmp.txt 2>&1
grep -E "配对差值|结论" logs/u_route/p3_base_cmp.txt | head -2

# ⚠️ fully_async 语义（08-29 实测）：save_freq 挂在 param_version（每 16 global step
#    同步一次）而非训练步——save_freq=25 会恰好只存终点；6 ⇒ 每 ~96 步一存 = 4-5 个点
# ⚠️ 不传 --test-freq：verl 内置 validate 在 fully_async 下 reward 聚合翻倍断言炸
#    （len(lst)=330 vs 165，10:56 实证）；评测走外部冻结 EVAL，内置验证纯冗余
say "③ 起 RL（fully_async 3+1 · candidate）——先清废跑残留（dispatched/pool_state 污染守卫判读）"
rm -rf checkpoints/grpo/smoke
mkdir -p checkpoints/grpo/smoke
SYNCOPATE_SYNC_PAYLOAD=1 SYNCOPATE_SYNC_REF=75.378174 \
SYNCOPATE_SYNC_WATCH="model.layers.0.self_attn.q_proj.base_layer.weight" \
nohup .venv/bin/python -m syncopate.train.launch_rl --model "$BASE" \
  --lora-rank 32 --mode fully_async --trainer-gpus 3 --rollout-gpus 1 \
  --train-file data/rl/v13/train.parquet --val-file data/rl/v13/val.parquet \
  --steps 400 --purpose candidate --experiment "$EXP" \
  --save-freq 6 > logs/u_route/p3_rl.log 2>&1 &
RLPID=$!
echo $RLPID > /tmp/p3_rl.pid
sleep 30
kill -0 $RLPID 2>/dev/null || { tail -20 logs/u_route/p3_rl.log; echo RL-DEAD-EARLY; exit 1; }
CKDIR=checkpoints/grpo/smoke
nohup bash scripts/rl_ckpt_rolling_prune.sh "$CKDIR" > logs/u_route/p3_prune.log 2>&1 &

say "④ 起跑后 10 分钟判据行自查（守卫在自查后挂载——冷启动窗口会误触 defer 连零）"
sleep 600
# ⚠️ D 族连零门槛 25 是旧采样制度反填值——fully_async 动态分池会把已学好的题排出
#    有效池（659→~400），defer 题学好即被排除 ⇒ 自然连零远超旧上界（11:09 误杀实测
#    streak=67 而模型行为健康：defer 终答在、该 defer 100%）。defer_watch 注释自己
#    预警过要按新制度反填。本跑 D 族观察模式（999=只记录），跑完用 dump 实测反填新门槛。
MAX_DEFER_ZERO_STREAK=999 RL_PIDFILE=/tmp/p3_rl.pid nohup bash scripts/rl_guard.sh logs/u_route/p3_rl.log "$CKDIR" --kill > logs/u_route/p3_guard.log 2>&1 &
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
