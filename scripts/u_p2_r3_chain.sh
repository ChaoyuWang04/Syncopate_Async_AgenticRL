#!/usr/bin/env bash
# U 路 P2 · v14.2 全链（r3）：重建 → 冻结校验 → 四卡 SFT → 五点谱 → 选优合并 → 考场 → 机判
# 迭代依据（08-29 05:05 判定，详见 24 §4-P2）：r2 考场 L1=60 不过 85——词表内 5/5 过、
# 词表外 20 挂 10 = 背词条没学会规则 ⇒ v14.2 词表 20→61 · L1 行 100→250 · 句式 5→8。
set -u
cd /workspace/Syncopate_Async_AgenticRL
set -a; . /workspace/.env; set +a
source .venv/bin/activate
say(){ echo "[P2R3 $(date +%H:%M:%S)] $*"; }

say "① 重建 v14.2"
.venv/bin/python scripts/u_build_v14.py > logs/u_route/build_v142.log 2>&1 || { tail -5 logs/u_route/build_v142.log; echo BUILD-FAIL; exit 1; }
# 增量重建冻结校验（[incremental-rebuild-freeze]）：非 L1 桶行数必须与 v14.1 逐个相等
.venv/bin/python - <<'PY' || { echo FREEZE-FAIL; exit 1; }
import json
m = json.load(open('data/sft/v14/manifest.json'))
exp = {"v13_train": 419, "l2_multiturn": 200, "chat_distill": 74, "cot_distill": 59}
for k, v in exp.items():
    assert m["sources"][k] == v, f"冻结破坏：{k}={m['sources'][k]}≠{v}"
assert m["sources"]["l1_concept"] >= 200, f"L1 行只有 {m['sources']['l1_concept']}"
print(f"冻结校验过：{m['sources']}  total={m['total']}")
PY

say "② SFT v14_r3（CLI 直跑=自动四卡；五点谱 e1/1.5/2/2.5/3）"
python -m syncopate.train.sft --model models/Qwen3-4B \
  --train-file data/sft/v14/train.parquet --val-file data/sft/v14/val.parquet \
  --out checkpoints/sft/v14_r3 --epochs 3 --wandb-run sft_v14_r3 \
  > logs/u_route/sft_v14_r3.log 2>&1 || { tail -8 logs/u_route/sft_v14_r3.log; echo SFT-FAIL; exit 1; }
grep -E "^\[epoch|ΔW" logs/u_route/sft_v14_r3.log | tail -6

best=""; bestd=-999
for pt in epoch1 sel_f1.5 epoch2 sel_f2.5 epoch3; do
  AD=checkpoints/sft/v14_r3/$pt
  [ -d "$AD" ] || { say "⚠️ 缺 $pt，跳过"; continue; }
  tag=${pt//./_}
  say "③ 评 $pt（4 卡）"
  rm -f "_audit/v142_sft_$tag.json.done"
  MODEL=models/Qwen3-4B bash scripts/eval_parallel.sh "$AD" "_audit/v142_sft_$tag.json" 4 || { echo EVAL-FAIL; exit 1; }
  until [ -f "_audit/v142_sft_$tag.json.done" ]; do sleep 15; done
  .venv/bin/python -m syncopate.train.compare _audit/v13_sft_v13r2_e1_merged.json "_audit/v142_sft_$tag.json" \
    > "logs/u_route/p2_r3_cmp_$tag.txt" 2>&1
  d=$(grep -m1 "配对差值" "logs/u_route/p2_r3_cmp_$tag.txt" | grep -oE '[+-][0-9.]+' | head -1)
  say "  $pt Δ=$d"
  awk "BEGIN{exit !($d > $bestd)}" && { bestd=$d; best=$pt; }
done
[ -n "$best" ] || { echo NO-WINNER; exit 1; }
say "任务分胜者=$best（Δ=$bestd）——cap 干净度复核在链外做（并列时可能改判）"

MERGED=models/Qwen3-4B-sft-v14r3-${best//./_}
say "④ 合并 $best -> $MERGED"
python -m syncopate.train.merge_adapter --base models/Qwen3-4B \
  --adapter "checkpoints/sft/v14_r3/$best" --out "$MERGED" || { echo MERGE-FAIL; exit 1; }

say "⑤ 单卡栈 + 考场两件"
CUDA_VISIBLE_DEVICES=0 nohup vllm serve "$MERGED" \
  --served-model-name candidate --max-model-len 14336 --kv-cache-dtype fp8 \
  --max-num-batched-tokens 16384 --scheduling-policy priority \
  --host 127.0.0.1 --port 8100 > logs/u_route/p2r3_vllm.log 2>&1 &
SRV=$!
until curl -sf http://127.0.0.1:8100/health >/dev/null 2>&1; do sleep 5; kill -0 $SRV 2>/dev/null || { echo VLLM-DEAD; exit 1; }; done
SYNCOPATE_API_DB_POOL=12 nohup uvicorn syncopate.runtime.api:app --host 127.0.0.1 --port 8000 --workers 2 > logs/u_route/p2r3_api.log 2>&1 &
API=$!
until curl -sf http://127.0.0.1:8000/healthz >/dev/null 2>&1; do sleep 2; kill -0 $API 2>/dev/null || { echo API-DEAD; exit 1; }; done
SYNCOPATE_DECIDER_URL=http://127.0.0.1:8100 SYNCOPATE_DECIDER_TOKENIZER="$MERGED" \
SYNCOPATE_WORKER_DB_POOL=32 \
nohup python -m syncopate.runtime.worker --org-id org_demo --worker-id p2r3-accept \
  --daily-cost-cap-micros 10000000000 > logs/u_route/p2r3_worker.log 2>&1 &
WK=$!
sleep 8
.venv/bin/python scripts/u_exam_run.py --exam context --arm v142 --concurrency 4 > logs/u_route/p2r3_context.log 2>&1 || echo CTX-RUN-FAIL
.venv/bin/python scripts/u_exam_run.py --exam talk --arm v142 --concurrency 4 > logs/u_route/p2r3_talk.log 2>&1 || echo TALK-RUN-FAIL
kill $WK $API $SRV 2>/dev/null; sleep 5
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sort -u); do kill -9 $p 2>/dev/null; done

say "⑥ 机判 + 首步 + 盲评包"
.venv/bin/python scripts/u_exam_judge.py --context logs/u_route/run_v142_context.jsonl | head -8
.venv/bin/python - <<'PY'
import json
rows = [json.loads(x) for x in open('logs/u_route/run_v142_context.jsonl')]
task_rows = [r for r in rows if r['level'] in ('L2', 'L3')]
first_tool = sum(1 for r in task_rows if r['turns'][0]['tools'])
print(f"首步调工具率: {first_tool}/{len(task_rows)}（门槛=全过）")
PY
.venv/bin/python scripts/u_exam_judge.py --blind logs/u_route/run_v142_talk.jsonl logs/u_route/run_p1_talk.jsonl
echo P2-R3-CHAIN-DONE
