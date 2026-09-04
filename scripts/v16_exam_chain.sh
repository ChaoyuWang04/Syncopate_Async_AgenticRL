#!/usr/bin/env bash
# v16 考场链：起端点 → API/worker → 考场 v4 N 遍 → 判卷 → 三查（由 scripts/v16_pipeline.sh 的 exam stage 调；smoke=1 遍 40 题 · candidate=4 遍）
#   bash scripts/v16_exam_chain.sh <合并模型> <臂名> [考卷]
# ⚠️ 四遍是**方差要求**不是保险：考场单遍方差实测 29pp ≫ 8pp 闸（24 §4-P2）。
# 09-05 三处修：日志目录先建再写 · 判卷接多遍文件 · 三查只读**本次**判卷产物（此前默认读 v15-R3 旧文件）；三查 rc 2 = 有缺口，报不拦（挡晋级不挡起跑）
set -u
export SYNCOPATE_CONTRACT="${SYNCOPATE_CONTRACT:-v15}" SYNCOPATE_THINK="${SYNCOPATE_THINK:-1}"
# 09-04 固定管线：不再写死 /workspace；仓库根 = 本文件上两级；.env 有则读；解释器优先容器 venv（/env）再本机 .venv
cd "$(dirname "${BASH_SOURCE[0]}")/.."
[ -f /workspace/.env ] && { set -a; . /workspace/.env; set +a; }
PY="${PY:-$( [ -x /env/.venv/bin/python ] && echo /env/.venv/bin/python || echo .venv/bin/python )}"
export PATH="$(dirname "$PY"):$PATH"
MERGED="${1:?合并模型路径}"; ARM="${2:-v16sft}"; EXAM="${3:-context_v4}"   # 09-03：考卷 v4 为默认
mkdir -p logs/v15_r5 logs/u_route
PASSES="${EXAM_PASSES:-4}"; LIMIT_ARG="${EXAM_LIMIT:+--limit $EXAM_LIMIT}"
say(){ echo "[V16-EXAM $(date +%H:%M:%S)] $*"; }

# ⛔ 08-30：上一轮的 worker 没停干净 ⇒ 它认旧模型名，和新 worker 抢同一个队列，
#   109/277 条 run 拿 404 死掉，而链路看着"跑完了"。⇒ 起链之前先拒绝重叠。
if pgrep -f "syncopate.runtime.worker" >/dev/null; then
  echo "🔴 已有 worker 在跑（下面这些），先停掉再起考场 —— 两个 worker 抢同一个队列："
  ps -eo pid,etime,cmd | grep "[s]yncopate.runtime.worker"
  exit 1
fi

say "① 起端点（GPU0，:8100）"
CUDA_VISIBLE_DEVICES=0 nohup $PY -m vllm.entrypoints.openai.api_server --model "$MERGED" \
  --max-model-len 24576 --host 127.0.0.1 --port 8100 --gpu-memory-utilization 0.85 \
  > logs/v15_r5/exam_vllm.log 2>&1 &
V=$!
until curl -sf http://127.0.0.1:8100/health >/dev/null 2>&1; do
  sleep 5; kill -0 $V 2>/dev/null || { echo VLLM-DEAD; tail -5 logs/v15_r5/exam_vllm.log; exit 1; }
done
say "② 起 API + worker（契约 v15 显式可见）"
SYNCOPATE_CONTRACT=v15 SYNCOPATE_API_DB_POOL=12 nohup $PY -m uvicorn syncopate.runtime.api:app \
  --host 127.0.0.1 --port 8000 --workers 2 > logs/v15_r5/exam_api.log 2>&1 &
A=$!
sleep 8
SYNCOPATE_CONTRACT=v15 SYNCOPATE_DECIDER_URL=http://127.0.0.1:8100 \
  SYNCOPATE_DECIDER_TOKENIZER="$MERGED" SYNCOPATE_DECIDER_MODEL="$MERGED" \
  nohup $PY -m syncopate.runtime.worker --org-id org_demo --worker-id v15-exam \
  --daily-cost-cap-micros 10000000000 > logs/v15_r5/exam_worker.log 2>&1 &
W=$!
sleep 5
say "③ 考场 $EXAM 跑 $PASSES 遍（方差前置条件：四遍；冒烟可 EXAM_PASSES=1）"
for i in $(seq 1 "$PASSES"); do
  $PY scripts/v16_exam_run.py --exam "$EXAM" --arm "${ARM}_r$i" --concurrency 4 $LIMIT_ARG \
    > "logs/v15_r5/exam_v3_r$i.log" 2>&1 || echo "🔴 第 $i 遍失败"
  say "  第 $i 遍完成"
done
say "④ 收尾：停端点与服务"
kill $W $A $V 2>/dev/null; sleep 8
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sort -u); do kill -9 $p 2>/dev/null; done
say "⑤ 判卷 v4 + 三查"
JL=""; for i in $(seq 1 "$PASSES"); do JL="$JL logs/u_route/run_${ARM}_r${i}_${EXAM}.jsonl"; done
$PY scripts/v16_exam_judge.py --context $JL > "logs/v15_r5/judge_${ARM}.log" 2>&1 || { echo "🔴 判卷失败"; tail -20 "logs/v15_r5/judge_${ARM}.log"; exit 1; }
tail -25 "logs/v15_r5/judge_${ARM}.log"
$PY scripts/v16_gate_triage.py --judged "logs/u_route/judged_${ARM}_r*_${EXAM}.jsonl" --out "_audit/v16/gate_triage_${ARM}.json" \
  > "logs/v15_r5/triage_${ARM}.log" 2>&1; TRC=$?
case "$TRC" in 0) ;; 2) echo "🟡 三查有缺口（rc 2：报不拦，门槛表待 v16 首考读数重登记）";; *) echo "🔴 三查失败 rc=$TRC"; tail -20 "logs/v15_r5/triage_${ARM}.log"; exit 1;; esac
tail -8 "logs/v15_r5/triage_${ARM}.log"
say "✅ $PASSES 遍完成 → logs/u_route/run_${ARM}_r{1..$PASSES}_${EXAM}.jsonl · 判卷 logs/v15_r5/judge_${ARM}.log · 三查 logs/v15_r5/triage_${ARM}.log"
