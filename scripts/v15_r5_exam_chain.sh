#!/usr/bin/env bash
# v15 · R5 U4：考场 v3 四遍聚合（`25 §R5` 门槛②③⑤⑥ + R3④ 的方差闸）
#   bash scripts/v15_r5_exam_chain.sh <合并模型> <臂名>
# ⚠️ 四遍是**方差要求**不是保险：考场单遍方差实测 29pp ≫ 8pp 闸（24 §4-P2）。
set -u
cd /workspace/Syncopate_Async_AgenticRL
set -a; . /workspace/.env; set +a
source .venv/bin/activate
MERGED="${1:?合并模型路径}"; ARM="${2:-v15sft}"
say(){ echo "[V15-EXAM $(date +%H:%M:%S)] $*"; }

say "① 起端点（GPU0，:8100）"
CUDA_VISIBLE_DEVICES=0 nohup vllm serve "$MERGED" \
  --max-model-len 14336 --host 127.0.0.1 --port 8100 --gpu-memory-utilization 0.85 \
  > logs/v15_r5/exam_vllm.log 2>&1 &
V=$!
until curl -sf http://127.0.0.1:8100/health >/dev/null 2>&1; do
  sleep 5; kill -0 $V 2>/dev/null || { echo VLLM-DEAD; tail -5 logs/v15_r5/exam_vllm.log; exit 1; }
done
say "② 起 API + worker（契约 v15 显式可见）"
SYNCOPATE_CONTRACT=v15 SYNCOPATE_API_DB_POOL=12 nohup uvicorn syncopate.runtime.api:app \
  --host 127.0.0.1 --port 8000 --workers 2 > logs/v15_r5/exam_api.log 2>&1 &
A=$!
sleep 8
SYNCOPATE_CONTRACT=v15 SYNCOPATE_DECIDER_URL=http://127.0.0.1:8100 \
  SYNCOPATE_DECIDER_TOKENIZER="$MERGED" SYNCOPATE_DECIDER_MODEL="$MERGED" \
  nohup python -m syncopate.runtime.worker --org-id org_demo --worker-id v15-exam \
  --daily-cost-cap-micros 10000000000 > logs/v15_r5/exam_worker.log 2>&1 &
W=$!
sleep 5
say "③ 考场 v3 跑四遍（方差闸要求）"
for i in 1 2 3 4; do
  python scripts/u_exam_run.py --exam context_v3 --arm "${ARM}_r$i" --concurrency 4 \
    > "logs/v15_r5/exam_v3_r$i.log" 2>&1 || echo "🔴 第 $i 遍失败"
  say "  第 $i 遍完成"
done
say "④ 收尾：停端点与服务"
kill $W $A $V 2>/dev/null; sleep 8
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sort -u); do kill -9 $p 2>/dev/null; done
say "✅ 四遍完成 → logs/u_route/run_${ARM}_r{1..4}_context_v3.jsonl"
