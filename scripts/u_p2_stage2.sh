#!/usr/bin/env bash
# U 路 P2 验收后半链：合并胜者 epoch2 → 单卡栈服务化 → 两考场 → 机判 + 盲评包
set -u
cd /workspace/Syncopate_Async_AgenticRL
set -a; . /workspace/.env; set +a
source .venv/bin/activate
MERGED=models/Qwen3-4B-sft-v14r1-e2
say(){ echo "[P2B $(date +%H:%M:%S)] $*"; }

say "① 合并 epoch2（配对 +0.058 t=+4.7 胜出；e1=-0.105 淘汰）"
if [ ! -d "$MERGED" ]; then
  python -m syncopate.train.merge_adapter --base models/Qwen3-4B \
    --adapter checkpoints/sft/v14_r1/epoch2 --out "$MERGED" || { echo MERGE-FAIL; exit 1; }
fi

say "② 起单卡栈（merged 直接顶 candidate 名）"
CUDA_VISIBLE_DEVICES=0 nohup vllm serve "$MERGED" \
  --served-model-name candidate --max-model-len 14336 --kv-cache-dtype fp8 \
  --max-num-batched-tokens 16384 --scheduling-policy priority \
  --host 127.0.0.1 --port 8100 > logs/u_route/p2_vllm.log 2>&1 &
SRV=$!
until curl -sf http://127.0.0.1:8100/health >/dev/null 2>&1; do sleep 5; kill -0 $SRV 2>/dev/null || { echo VLLM-DEAD; exit 1; }; done
SYNCOPATE_API_DB_POOL=12 nohup uvicorn syncopate.runtime.api:app --host 127.0.0.1 --port 8000 --workers 2 > logs/u_route/p2_api.log 2>&1 &
API=$!
until curl -sf http://127.0.0.1:8000/healthz >/dev/null 2>&1; do sleep 2; done
SYNCOPATE_DECIDER_URL=http://127.0.0.1:8100 SYNCOPATE_DECIDER_TOKENIZER="$MERGED" \
SYNCOPATE_WORKER_DB_POOL=32 \
nohup python -m syncopate.runtime.worker --org-id org_demo --worker-id p2-accept \
  --daily-cost-cap-micros 10000000000 > logs/u_route/p2_worker.log 2>&1 &
WK=$!
sleep 8
say "③ 考场两件"
.venv/bin/python scripts/u_exam_run.py --exam context --arm v14sft --concurrency 4 > logs/u_route/p2_context.log 2>&1 || echo CTX-RUN-FAIL
.venv/bin/python scripts/u_exam_run.py --exam talk --arm v14sft --concurrency 4 > logs/u_route/p2_talk.log 2>&1 || echo TALK-RUN-FAIL
kill $WK $API $SRV 2>/dev/null; sleep 5
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sort -u); do kill -9 $p 2>/dev/null; done

say "④ 机判 + 首步调工具率 + 盲评包（v14sft vs p1，比「说人话不低于 P1」）"
.venv/bin/python scripts/u_exam_judge.py --context logs/u_route/run_v14sft_context.jsonl | head -8
.venv/bin/python - <<'PY'
import json
rows=[json.loads(x) for x in open('logs/u_route/run_v14sft_context.jsonl')]
task_rows=[r for r in rows if r['level'] in ('L2','L3')]
first_tool=sum(1 for r in task_rows if r['turns'][0]['tools'])
print(f"首步调工具率: {first_tool}/{len(task_rows)}（门槛=全过）")
PY
.venv/bin/python scripts/u_exam_judge.py --blind logs/u_route/run_v14sft_talk.jsonl logs/u_route/run_p1_talk.jsonl
echo P2-STAGE2-DONE
