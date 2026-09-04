#!/usr/bin/env bash
# U 路 P2 · v14.1 全链：SFT 重训 → e1/e2 评测配对 → 选优合并 → 考场 → 机判
set -u
cd /workspace/Syncopate_Async_AgenticRL
set -a; . /workspace/.env; set +a
source .venv/bin/activate
say(){ echo "[P2C $(date +%H:%M:%S)] $*"; }

say "① SFT v14_r2（852 行=v14+100 L1 概念行）"
.venv/bin/python -m syncopate.train.sft --model models/Qwen3-4B \
  --train-file data/sft/v14/train.parquet --val-file data/sft/v14/val.parquet \
  --out checkpoints/sft/v14_r2 --epochs 3 --wandb-run sft_v14_r2 \
  > logs/u_route/sft_v14_r2.log 2>&1 || { tail -5 logs/u_route/sft_v14_r2.log; echo SFT-FAIL; exit 1; }
tail -3 logs/u_route/sft_v14_r2.log

best=""; bestd=-999
for ep in epoch1 epoch2; do
  say "② 评 $ep（4 卡）"
  AD=checkpoints/sft/v14_r2/$ep
  rm -f "_audit/v141_sft_$ep.json.done"
  MODEL=models/Qwen3-4B bash scripts/eval_parallel.sh "$AD" "_audit/v141_sft_$ep.json" 4 || { echo EVAL-FAIL; exit 1; }
  until [ -f "_audit/v141_sft_$ep.json.done" ]; do sleep 20; done
  .venv/bin/python -m syncopate.train.compare _audit/v13_sft_v13r2_e1_merged.json "_audit/v141_sft_$ep.json" \
    | tee "logs/u_route/p2_r2_cmp_$ep.txt" | grep -E "配对差值|结论|有梯度"
  d=$(grep -m1 "配对差值" "logs/u_route/p2_r2_cmp_$ep.txt" | grep -oE '[+-][0-9.]+' | head -1)
  awk "BEGIN{exit !($d > $bestd)}" && { bestd=$d; best=$ep; }
done
say "胜者=$best（Δ=$bestd）"
[ -n "$best" ] || { echo NO-WINNER; exit 1; }

MERGED=models/Qwen3-4B-sft-v14r2-$best
say "③ 合并 $best -> $MERGED"
python -m syncopate.train.merge_adapter --base models/Qwen3-4B \
  --adapter "checkpoints/sft/v14_r2/$best" --out "$MERGED" || { echo MERGE-FAIL; exit 1; }

say "④ 单卡栈 + 考场两件"
CUDA_VISIBLE_DEVICES=0 nohup vllm serve "$MERGED" \
  --served-model-name candidate --max-model-len 14336 --kv-cache-dtype fp8 \
  --max-num-batched-tokens 16384 --scheduling-policy priority \
  --host 127.0.0.1 --port 8100 > logs/u_route/p2r2_vllm.log 2>&1 &
SRV=$!
until curl -sf http://127.0.0.1:8100/health >/dev/null 2>&1; do sleep 5; kill -0 $SRV 2>/dev/null || { echo VLLM-DEAD; exit 1; }; done
SYNCOPATE_API_DB_POOL=12 nohup uvicorn syncopate.runtime.api:app --host 127.0.0.1 --port 8000 --workers 2 > logs/u_route/p2r2_api.log 2>&1 &
API=$!
until curl -sf http://127.0.0.1:8000/healthz >/dev/null 2>&1; do sleep 2; done
SYNCOPATE_DECIDER_URL=http://127.0.0.1:8100 SYNCOPATE_DECIDER_TOKENIZER="$MERGED" \
SYNCOPATE_WORKER_DB_POOL=32 \
nohup python -m syncopate.runtime.worker --org-id org_demo --worker-id p2r2-accept \
  --daily-cost-cap-micros 10000000000 > logs/u_route/p2r2_worker.log 2>&1 &
WK=$!
sleep 8
.venv/bin/python scripts/v16_exam_run.py --exam context --arm v141 --concurrency 4 > logs/u_route/p2r2_context.log 2>&1 || echo CTX-RUN-FAIL
.venv/bin/python scripts/v16_exam_run.py --exam talk --arm v141 --concurrency 4 > logs/u_route/p2r2_talk.log 2>&1 || echo TALK-RUN-FAIL
kill $WK $API $SRV 2>/dev/null; sleep 5
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sort -u); do kill -9 $p 2>/dev/null; done

say "⑤ 机判 + 首步 + 盲评包"
.venv/bin/python scripts/u_exam_judge.py --context logs/u_route/run_v141_context.jsonl | head -8
.venv/bin/python - <<'PY'
import json
rows=[json.loads(x) for x in open('logs/u_route/run_v141_context.jsonl')]
task_rows=[r for r in rows if r['level'] in ('L2','L3')]
first_tool=sum(1 for r in task_rows if r['turns'][0]['tools'])
print(f"首步调工具率: {first_tool}/{len(task_rows)}（门槛=全过）")
PY
.venv/bin/python scripts/u_exam_judge.py --blind logs/u_route/run_v141_talk.jsonl logs/u_route/run_p1_talk.jsonl
echo P2-R2-CHAIN-DONE
