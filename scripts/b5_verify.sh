#!/usr/bin/env bash
# B-5 · 单遍验证：确保 fleet 在 → 重起栈（插桩开）→ 阶梯 8/64/96 → 拼账本
# 用法: bash /tmp/b5_verify.sh <tag> [levels]
set -u
cd /workspace/Syncopate_Async_AgenticRL
set -a; . /workspace/.env; set +a
source .venv/bin/activate
TAG=${1:?tag}; LEVELS=${2:-8,64,96}
D=logs/b5; say() { echo "[B5V $(date +%H:%M:%S)] $*"; }

if ! curl -sf http://127.0.0.1:8100/health >/dev/null 2>&1; then
  say "fleet 不在，拉起（冻结口径）"
  bash scripts/b4_serve_4x.sh start 4 affinity -- --max-num-batched-tokens 16384 \
    --speculative-config '{"method":"ngram","num_speculative_tokens":4,"prompt_lookup_max":4,"prompt_lookup_min":2}' \
    || { say "🔴 fleet 失败"; exit 1; }
fi
bash scripts/b4_stack.sh stop >/dev/null 2>&1
SYNCOPATE_STAGE_TIMING=1 ${STACK_ENV:-} bash scripts/b4_stack.sh start 512 || { say "🔴 stack 失败"; exit 1; }
timeout 3600 .venv/bin/python scripts/b4_goodput.py --tag "$TAG" --levels "$LEVELS" \
  --out "$D/gp_$TAG.json" > "$D/gp_$TAG.log" 2>&1 || say "⚠️ 阶梯非零退出（破线属预期时忽略）"
cat logs/b4/stack/worker_*.log > "$D/worker_$TAG.log" 2>/dev/null || cp logs/b4/stack/worker.log "$D/worker_$TAG.log"
bash scripts/b4_stack.sh stop
.venv/bin/python scripts/b5_ledger.py --goodput "$D/gp_$TAG.json" \
  --worker-log "$D/worker_$TAG.log" --out "$D/ledger_$TAG.json" > "$D/ledger_$TAG.print" 2>&1 \
  || { say "🔴 ledger 失败"; tail -3 "$D/ledger_$TAG.print"; exit 1; }
echo "B5V-$TAG-DONE"
