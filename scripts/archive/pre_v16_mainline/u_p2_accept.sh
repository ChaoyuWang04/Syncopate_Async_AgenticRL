#!/usr/bin/env bash
# U 路 P2 验收链：e1/e2 两点选点(A-2 日常口径) → merge → 配对 → 考场 → 七门槛材料
set -u
cd /workspace/Syncopate_Async_AgenticRL
set -a; . /workspace/.env; set +a
source .venv/bin/activate
say(){ echo "[P2A $(date +%H:%M:%S)] $*"; }
CK=checkpoints/sft/v14_r1

for ep in epoch1 epoch2; do
  AD="$CK/$ep"
  [ -d "$AD" ] || AD="$CK/sel_$ep"
  [ -d "$AD" ] || { say "找不到 $ep ckpt，目录内容："; ls "$CK"; continue; }
  say "评 $ep（4 卡）"
  rm -f "_audit/v14_sft_$ep.json.done"
  MODEL=models/Qwen3-4B bash scripts/eval_parallel.sh "$AD" "_audit/v14_sft_$ep.json" 4 || { echo EVAL-$ep-FAIL; exit 1; }
  until [ -f "_audit/v14_sft_$ep.json.done" ]; do sleep 15; done
done
say "两点对照（vs v13 SFT merged 基线）"
for ep in epoch1 epoch2; do
  [ -f "_audit/v14_sft_$ep.json" ] && \
  .venv/bin/python -m syncopate.train.compare _audit/v13_sft_v13r2_e1_merged.json "_audit/v14_sft_$ep.json" \
    > "logs/u_route/p2_compare_$ep.txt" 2>&1 && grep -E "配对差值|结论|零梯度|有梯度" "logs/u_route/p2_compare_$ep.txt" | head -6
done
echo P2-EVAL-DONE
