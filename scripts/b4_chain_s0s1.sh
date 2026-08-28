#!/usr/bin/env bash
set -u
cd /workspace/Syncopate_Async_AgenticRL
for arm in base_fp8kv_s0 base_fp8kv_r2 base_fp8kv_r3; do
  echo "== $arm =="
  bash scripts/b4_bench.sh "$arm" || { echo "$arm-FAILED"; exit 1; }
  [ -f "logs/b4/$arm/arm.json" ] || { echo "$arm-NO-ARMJSON"; exit 1; }
done
echo "== S1 sweep =="; bash scripts/b4_sweep.sh || echo "SWEEP-FAILED"
echo NIGHT1-ALL-DONE
