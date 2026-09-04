#!/usr/bin/env bash
# v14.5 · P2 全链（24 §4-P2 最终设计执行件）：教师起栈 → S4 建库（门禁内置）→ 四卡 SFT
# → 五点谱评测 → 双判据选点 → 合并 → 双卷考场 → 机判。每段落盘判据行，失败即停。
set -u
cd /workspace/Syncopate_Async_AgenticRL
set -a; . /workspace/.env; set +a
source .venv/bin/activate
say(){ echo "[V145 $(date +%H:%M:%S)] $*"; }

say "① 教师起栈（4B@8210 GPU0 · 8B@8211 GPU1）"
CUDA_VISIBLE_DEVICES=0 nohup vllm serve models/Qwen3-4B --served-model-name t \
  --max-model-len 8192 --host 127.0.0.1 --port 8210 --gpu-memory-utilization 0.85 \
  > logs/u_route/v145_t4b.log 2>&1 &
T4=$!
CUDA_VISIBLE_DEVICES=1 nohup vllm serve models/Qwen3-8B --served-model-name t \
  --max-model-len 10240 --host 127.0.0.1 --port 8211 --gpu-memory-utilization 0.85 \
  > logs/u_route/v145_t8b.log 2>&1 &
T8=$!
for p in 8210 8211; do
  until curl -sf http://127.0.0.1:$p/health >/dev/null 2>&1; do
    sleep 5; kill -0 $T4 2>/dev/null || { echo T4B-DEAD; exit 1; }
    kill -0 $T8 2>/dev/null || { echo T8B-DEAD; exit 1; }
  done
done
say "教师就绪"

say "② S4 建库（门禁内置：份额/密度/OOV/泄漏/冻结）"
python scripts/v16_build_sft.py > logs/u_route/build_v145.log 2>&1 \
  || { tail -15 logs/u_route/build_v145.log; kill $T4 $T8 2>/dev/null; echo BUILD-FAIL; exit 1; }
grep -E "份额|密度|✅" logs/u_route/build_v145.log | tail -8
kill $T4 $T8 2>/dev/null; sleep 8
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sort -u); do kill -9 $p 2>/dev/null; done
sleep 3

say "③ SFT v14_5_r1（CLI 直跑=自动四卡；五点谱 e1/1.5/2/2.5/3）"
python -m syncopate.train.sft --model models/Qwen3-4B \
  --train-file data/sft/v14_5/train.parquet --val-file data/sft/v14_5/val.parquet \
  --out checkpoints/sft/v14_5_r1 --epochs 3 --wandb-run sft_v14_5_r1 \
  > logs/u_route/sft_v14_5_r1.log 2>&1 || { tail -8 logs/u_route/sft_v14_5_r1.log; echo SFT-FAIL; exit 1; }
grep -E "^\[epoch|ΔW" logs/u_route/sft_v14_5_r1.log | tail -6

say "④ 五点谱评测 + 双判据选点"
for pt in epoch1 sel_f1.5 epoch2 sel_f2.5 epoch3; do
  AD=checkpoints/sft/v14_5_r1/$pt
  [ -d "$AD" ] || { say "⚠️ 缺 $pt"; continue; }
  tag=${pt//./_}
  rm -f "_audit/v145_sft_$tag.json.done"
  MODEL=models/Qwen3-4B bash scripts/eval_parallel.sh "$AD" "_audit/v145_sft_$tag.json" 4 || { echo EVAL-FAIL; exit 1; }
  until [ -f "_audit/v145_sft_$tag.json.done" ]; do sleep 15; done
  .venv/bin/python -m syncopate.train.compare _audit/v13_sft_v13r2_e1_merged.json "_audit/v145_sft_$tag.json" \
    > "logs/u_route/v145_cmp_$tag.txt" 2>&1
  say "  $pt: $(grep -m1 '配对差值' logs/u_route/v145_cmp_$tag.txt | tr -s ' ')"
done
.venv/bin/python - <<'PY' > logs/u_route/v145_winner.txt 2>&1 || { cat logs/u_route/v145_winner.txt; echo WINNER-FAIL; exit 1; }
import re
MDE = 0.025
pts = {}
for pt in ("epoch1", "sel_f1_5", "epoch2", "sel_f2_5", "epoch3"):
    try:
        txt = open(f"logs/u_route/v145_cmp_{pt}.txt").read()
    except FileNotFoundError:
        continue
    d = float(re.search(r"配对差值\s+([+-][0-9.]+)", txt).group(1))
    g = int(re.findall(r"有梯度\s+(\d+)", txt)[-1])
    pts[pt] = (d, g)
assert pts, "没有任何点的 compare"
best_d = max(d for d, _ in pts.values())
elig = {k: v for k, v in pts.items() if v[0] >= best_d - MDE}
winner = max(elig, key=lambda k: elig[k][1])
print(f"谱: { {k: v for k, v in pts.items()} }")
print(f"双判据（任务分距最优<{MDE} 内选有梯度格最多）胜者: {winner} Δ={pts[winner][0]} 有梯度={pts[winner][1]}")
open("logs/u_route/v145_winner.id", "w").write(winner.replace("_f1_5", "_f1.5").replace("_f2_5", "_f2.5"))
PY
cat logs/u_route/v145_winner.txt
WIN=$(cat logs/u_route/v145_winner.id)
say "胜者=$WIN"

MERGED=models/Qwen3-4B-sft-v14.5-${WIN//./_}
say "⑤ 合并 $WIN -> $MERGED + 单卡栈 + 三场考试"
python -m syncopate.train.merge_adapter --base models/Qwen3-4B \
  --adapter "checkpoints/sft/v14_5_r1/$WIN" --out "$MERGED" || { echo MERGE-FAIL; exit 1; }
CUDA_VISIBLE_DEVICES=0 nohup vllm serve "$MERGED" \
  --served-model-name candidate --max-model-len 14336 --kv-cache-dtype fp8 \
  --max-num-batched-tokens 16384 --scheduling-policy priority \
  --host 127.0.0.1 --port 8100 > logs/u_route/v145_vllm.log 2>&1 &
SRV=$!
until curl -sf http://127.0.0.1:8100/health >/dev/null 2>&1; do sleep 5; kill -0 $SRV 2>/dev/null || { echo VLLM-DEAD; exit 1; }; done
SYNCOPATE_API_DB_POOL=12 nohup uvicorn syncopate.runtime.api:app --host 127.0.0.1 --port 8000 --workers 2 > logs/u_route/v145_api.log 2>&1 &
API=$!
until curl -sf http://127.0.0.1:8000/healthz >/dev/null 2>&1; do sleep 2; kill -0 $API 2>/dev/null || { echo API-DEAD; exit 1; }; done
SYNCOPATE_DECIDER_URL=http://127.0.0.1:8100 SYNCOPATE_DECIDER_TOKENIZER="$MERGED" \
SYNCOPATE_WORKER_DB_POOL=32 \
nohup python -m syncopate.runtime.worker --org-id org_demo --worker-id v145-accept \
  --daily-cost-cap-micros 10000000000 > logs/u_route/v145_worker.log 2>&1 &
WK=$!
sleep 8
.venv/bin/python scripts/v16_exam_run.py --exam context_v2 --arm v145 --concurrency 4 > logs/u_route/v145_ctxv2.log 2>&1 || echo CTXV2-RUN-FAIL
.venv/bin/python scripts/v16_exam_run.py --exam context --arm v145 --concurrency 4 > logs/u_route/v145_ctx.log 2>&1 || echo CTX-RUN-FAIL
.venv/bin/python scripts/v16_exam_run.py --exam talk --arm v145 --concurrency 4 > logs/u_route/v145_talk.log 2>&1 || echo TALK-RUN-FAIL
kill $WK $API $SRV 2>/dev/null; sleep 5
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sort -u); do kill -9 $p 2>/dev/null; done

say "⑥ 机判（v2 双词表 + v1 对照）+ 首步 + 盲评包"
.venv/bin/python scripts/v16_exam_judge_core.py --context logs/u_route/run_v145_context_v2.jsonl | head -14
.venv/bin/python scripts/u_exam_judge.py --context logs/u_route/run_v145_context.jsonl | head -6
.venv/bin/python - <<'PY'
import json
rows = [json.loads(x) for x in open('logs/u_route/run_v145_context_v2.jsonl')]
task_rows = [r for r in rows if r['level'] in ('L2', 'L3')]
first_tool = sum(1 for r in task_rows if r['turns'][0]['tools'])
print(f"首步调工具率: {first_tool}/{len(task_rows)}（门槛=全过）")
PY
.venv/bin/python scripts/u_exam_judge.py --blind logs/u_route/run_v145_talk.jsonl logs/u_route/run_p1_talk.jsonl
echo P2-V145-CHAIN-DONE
