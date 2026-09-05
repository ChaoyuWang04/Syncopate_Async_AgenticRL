#!/usr/bin/env bash
# E32 S1 · 单卡批调度扫参（OFAT：从默认中心一次动一个旋钮；全交叉 18 臂不值那 4 个小时——
# 判据是"收益 > 2× 噪声地板"，OFAT 足以回答 P1"默认已优"与否；有胜手再组合复测）。
#
#   bash scripts/serving/b4_sweep.sh          # 6 臂，每臂 ~15min，产物 logs/b4/s1_*/arm.json
#
# 中心 = vllm 0.12 默认（chunked prefill 开 · mnbt 8192 · gpu_util 0.9）+ 生产旗子。
set -u
cd "$(dirname "$0")/../.."

run() { bash scripts/serving/b4_bench.sh "$1" -- "${@:2}"; }

# 中心臂 = base_fp8kv_s0/r2/r3（S0 噪声地板三跑与 S1 中心同配置，一鱼两吃，不重跑）
run s1_mnbt2048   --max-num-batched-tokens 2048
run s1_mnbt16384  --max-num-batched-tokens 16384
run s1_mns64      --max-num-seqs 64
run s1_mns256     --max-num-seqs 256
run s1_util085    --gpu-memory-utilization 0.85

echo "==== S1 汇总 ===="
for d in logs/b4/base_fp8kv_s0/arm.json logs/b4/s1_*/arm.json; do
  .venv/bin/python -c "
import json,sys; a=json.load(open('$d'))
t=a.get('trace',{}); r=a.get('random',{})
print(f\"{a['arm']:14s} trace_tok/s={t.get('output_tok_per_s','-'):>7} ttft_p90={t.get('ttft_s',{}).get('p90','-'):>6} \"
      f\"tpot_p90={t.get('tpot_ms',{}).get('p90','-'):>6} rand_tok/s={r.get('output_tok_per_s','-'):>7}\")"
done
