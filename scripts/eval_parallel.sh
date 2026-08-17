#!/usr/bin/env bash
# 四卡并行跑冻结 EVAL，跑完自动合并成一份审计。
#
#   bash scripts/eval_parallel.sh <adapter> <out.json> [卡数]
#
# ★ 为什么是按 case 分片而不是 tensor parallel：
#   4B 模型单卡装得下，而这台机器 **PCIe P2P 全关**，TP 的通信会成为瓶颈 ⇒ 大概率净亏。
#   而评测天生可分（每条 case 的 Sandbox 独立），分片是纯赚：278 条 / 4 卡 ≈ 时间除以四。
#
# ⚠️ 分片用**交错取**不是切块（见 eval_local 的 --shard 注释）：模板按 case_id 聚在一起，
#    切块会让某片全是 14 步的 GEO、另一片全是 1 步的 HIGH，负载差好几倍。
set -euo pipefail
ADAPTER="${1:?用法: eval_parallel.sh <adapter> <out.json> [卡数]}"
OUT="${2:?}"
N="${3:-4}"
BATCH="${BATCH:-data/batches/v13}"
SPLIT="${SPLIT:-data/splits/v13}"
MODEL="${MODEL:-models/Qwen3-4B}"
PY="${PY:-.venv/bin/python}"

TMPDIR_SHARDS="$(mktemp -d)"
pids=()
for i in $(seq 0 $((N-1))); do
  CUDA_VISIBLE_DEVICES="$i" "$PY" -m syncopate.train.eval_local \
    --model "$MODEL" --adapter "$ADAPTER" --batch "$BATCH" --split-dir "$SPLIT" \
    --shard "$i/$N" --out "$TMPDIR_SHARDS/shard_$i.json" \
    > "logs/eval_shard_$i.log" 2>&1 &
  pids+=($!)
done
echo "[eval] $N 片已起：${pids[*]}  日志 logs/eval_shard_*.log"
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
[ "$fail" -eq 0 ] || { echo "🔴 有分片失败，看 logs/eval_shard_*.log"; exit 1; }

"$PY" -m syncopate.train.merge_eval_shards --shards "$TMPDIR_SHARDS" --out "$OUT"
rm -rf "$TMPDIR_SHARDS"
