#!/usr/bin/env bash
# v15 · R5 U2：五点谱评测（四卡分片）。用法：bash scripts/v15_r5_eval5.sh <ckpt根> <前缀>
#   ⚠️ 考场是单端点（要驱动真 runtime 栈），**评测不是** —— 343 条冻结 EVAL 走四卡分片。
set -u
cd /workspace/Syncopate_Async_AgenticRL
set -a; . /workspace/.env; set +a
ROOT="${1:?ckpt 根目录}"; PFX="${2:-r3}"
say(){ echo "════ $* $(date +%H:%M:%S)"; }
for pt in epoch1 sel_f1.5 epoch2 sel_f2.5 epoch3; do
  AD="$ROOT/$pt"
  [ -d "$AD" ] || { say "⚠️ 缺 $pt"; continue; }
  tag=${pt//./}; tag=${tag/sel_f/sel_f}
  OUT="_audit/v15_r5/${PFX}_${tag}.json"
  say "$pt"
  rm -f "$OUT.done"
  SYNCOPATE_CONTRACT=v15 MODEL=models/Qwen3-4B bash scripts/eval_parallel.sh "$AD" "$OUT" 4 \
    || { echo "🔴 EVAL-FAIL $pt"; exit 1; }
  until [ -f "$OUT.done" ]; do sleep 15; done
  say "  ✅ $pt"
done
echo "✅ 五点谱评测完成"
