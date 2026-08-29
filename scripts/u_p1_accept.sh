#!/usr/bin/env bash
# U 路 P1 验收链：任务分配对(4卡) → 考场三件(单卡栈) → 汇总判定
set -u
cd /workspace/Syncopate_Async_AgenticRL
set -a; . /workspace/.env; set +a
source .venv/bin/activate
AD=checkpoints/opd/p1_r1/final
say(){ echo "[P1A $(date +%H:%M:%S)] $*"; }

say "① 冻结考场评测（4 卡并行）"
rm -f _audit/u_opd_p1.json.done
MODEL=models/Qwen3-4B-sft-v13r2-e1 bash scripts/eval_parallel.sh "$AD" _audit/u_opd_p1.json 4 || { echo EVAL-FAIL; exit 1; }
until [ -f _audit/u_opd_p1.json.done ]; do sleep 20; done
say "② 配对比较 vs candidate"
.venv/bin/python -m syncopate.train.compare _audit/cand_v13r2_rl_s100.json _audit/u_opd_p1.json \
  | tee logs/u_route/p1_compare.txt | tail -20

say "③ 起单卡栈（adapter=P1）跑考场"
CUDA_VISIBLE_DEVICES=0 nohup vllm serve models/Qwen3-4B-sft-v13r2-e1 \
  --served-model-name sft-base --enable-lora --lora-modules candidate="$AD" \
  --max-lora-rank 32 --max-model-len 14336 --kv-cache-dtype fp8 \
  --max-num-batched-tokens 16384 --scheduling-policy priority \
  --host 127.0.0.1 --port 8100 > logs/u_route/p1_vllm.log 2>&1 &
SRV=$!
until curl -sf http://127.0.0.1:8100/health >/dev/null 2>&1; do sleep 5; kill -0 $SRV 2>/dev/null || { echo VLLM-DEAD; exit 1; }; done
SYNCOPATE_API_DB_POOL=12 nohup uvicorn syncopate.runtime.api:app --host 127.0.0.1 --port 8000 --workers 2 > logs/u_route/p1_api.log 2>&1 &
API=$!
until curl -sf http://127.0.0.1:8000/healthz >/dev/null 2>&1; do sleep 2; done
SYNCOPATE_DECIDER_URL=http://127.0.0.1:8100 SYNCOPATE_WORKER_DB_POOL=32 \
nohup python -m syncopate.runtime.worker --org-id org_demo --worker-id p1-accept \
  --daily-cost-cap-micros 10000000000 > logs/u_route/p1_worker.log 2>&1 &
WK=$!
sleep 8
.venv/bin/python scripts/u_exam_run.py --exam context --arm p1 --concurrency 4 > logs/u_route/p1_context.log 2>&1 || echo CTX-RUN-FAIL
.venv/bin/python scripts/u_exam_run.py --exam talk --arm p1 --concurrency 4 > logs/u_route/p1_talk.log 2>&1 || echo TALK-RUN-FAIL
kill $WK $API $SRV 2>/dev/null; sleep 5
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sort -u); do kill -9 $p 2>/dev/null; done

say "④ 机判 + 首步调工具率"
.venv/bin/python scripts/u_exam_judge.py --context logs/u_route/run_p1_context.jsonl | head -6
.venv/bin/python - <<'PY'
import json
rows=[json.loads(x) for x in open('logs/u_route/run_p1_context.jsonl')]
task_rows=[r for r in rows if r['level'] in ('L2','L3')]
first_tool=sum(1 for r in task_rows if r['turns'][0]['tools'])
print(f"首步调工具率: {first_tool}/{len(task_rows)}（门槛=全过）")
PY
echo P1-ACCEPT-DONE
