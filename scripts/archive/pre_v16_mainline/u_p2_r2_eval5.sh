#!/usr/bin/env bash
# U 路 P2 r2 · 多点选点评测（Chaoyu 08-29：把整个 epoch 谱扫一遍，看清最优点在哪）
# 本轮可评点：e1 / e1.875(sel_f0.625 旧单位) / e2 / e2.25(sel_f0.75) / e2.5(sel_f2.5) / e3
# （e1.5 在断点前，本轮存不到；下轮起标准谱=e1/1.5/2/2.5/3）
set -u
cd /workspace/Syncopate_Async_AgenticRL
set -a; . /workspace/.env; set +a
source .venv/bin/activate
say(){ echo "[P2R2E $(date +%H:%M:%S)] $*"; }
CK=checkpoints/sft/v14_r2
BASE=_audit/v13_sft_v13r2_e1_merged.json

for pt in epoch1 sel_f0.625 epoch2 sel_f0.75 sel_f2.5 epoch3; do
  [ -d "$CK/$pt" ] || { say "⚠️ 缺 $pt，跳过"; continue; }
  tag=${pt//./_}
  say "评 $pt（4 卡）"
  rm -f "_audit/v141_sft_$tag.json.done"
  MODEL=models/Qwen3-4B bash scripts/eval_parallel.sh "$CK/$pt" "_audit/v141_sft_$tag.json" 4 \
    || { echo "EVAL-$pt-FAIL"; exit 1; }
  until [ -f "_audit/v141_sft_$tag.json.done" ]; do sleep 15; done
  .venv/bin/python -m syncopate.train.compare "$BASE" "_audit/v141_sft_$tag.json" \
    > "logs/u_route/p2_r2_cmp_$tag.txt" 2>&1
  grep -E "配对差值|结论" "logs/u_route/p2_r2_cmp_$tag.txt" | head -2
done
say "全点对照汇总："
for pt in epoch1 sel_f0.625 epoch2 sel_f0.75 sel_f2.5 epoch3; do
  tag=${pt//./_}
  [ -f "logs/u_route/p2_r2_cmp_$tag.txt" ] && \
    echo "  $pt: $(grep -m1 '配对差值' logs/u_route/p2_r2_cmp_$tag.txt | tr -s ' ')"
done
echo P2-R2-EVAL5-DONE
