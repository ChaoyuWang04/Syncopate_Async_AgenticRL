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
# ⚠️⚠️ 两个踩过的坑，都写在这：
#
# ① **日志文件名不能固定**。第一版四片都写 `logs/eval_shard_$i.log`，
#    起第二批时直接**覆盖掉第一批的日志** —— 出了事根本查不出是谁的。
#    ⇒ 每次运行一个带 tag 的目录。
#
# ② **别用「输出文件存在」当完成判据**。调用方写
#    `until [ -f _audit/xxx.json ]; do sleep; done` 等它 ——
#    而那个文件**上一版评测就存在**，循环立刻通过、第二批当场起来抢同样四张卡，
#    结果第二批全部 OOM 挂掉。⇒ 本脚本跑完才写 `<out>.done` 标记，等就等那个。
#    ★ 这是「判据看起来完整、其实量的不是那件事」的又一例。
set -euo pipefail
# ⚠️ 2026-08-18 修（infra）：原来是 `${1:?}`，而 `:?` 把**空字符串**也当成"没给"
#    ⇒ 文档里那条评基座的命令 `eval_parallel.sh "" out.json` **直接报错跑不通**。
#    改成 `${1?}`：**参数必须存在**（保住"别忘了想清楚 adapter"这个用意），
#    但**允许显式给空** = 不贴 adapter，只评基座本身。
ADAPTER="${1?用法: eval_parallel.sh <adapter> <out.json> [卡数]；不贴 adapter 就显式传空串 \"\"}"
OUT="${2:?}"
N="${3:-4}"
BATCH="${BATCH:-data/batches/v13}"
SPLIT="${SPLIT:-data/splits/v13}"
# ⚠️⚠️ **没有默认值** —— 2026-08-17 踩过：RL 的 LoRA 是在**SFT 合并模型**之上训的，
# 而这里默认成裸基座 `models/Qwen3-4B` ⇒ 等于把 SFT 学到的东西全丢掉再评。
# 症状极具迷惑性：不是报错，是**分数全线暴跌**（CLAR 0.844→0.000、REJ 0.812→0.094），
# 看着像"RL 把模型训坏了"。⇒ 基座必须显式给，给错了宁可报错。
MODEL="${MODEL:?必须显式指定基座：RL adapter 要贴在 SFT 合并模型上，不是裸基座}"
PY="${PY:-.venv/bin/python}"

# 先确认没有别的评测在占卡 —— 抢卡会让新起的这批直接 OOM
if pgrep -f 'train[.]eval_local' >/dev/null; then
  echo "🔴 已经有评测在跑，先等它结束（抢卡会 OOM）"; exit 1
fi
TAG="$(basename "${OUT%.json}")"
LOGDIR="logs/eval_${TAG}"
mkdir -p "$LOGDIR"
rm -f "${OUT}.done"
TMPDIR_SHARDS="$(mktemp -d)"
pids=()
for i in $(seq 0 $((N-1))); do
  CUDA_VISIBLE_DEVICES="$i" "$PY" -m syncopate.train.eval_local \
    --model "$MODEL" --adapter "$ADAPTER" --batch "$BATCH" --split-dir "$SPLIT" \
    --shard "$i/$N" --out "$TMPDIR_SHARDS/shard_$i.json" \
    > "$LOGDIR/shard_$i.log" 2>&1 &
  pids+=($!)
done
echo "[eval] $N 片已起：${pids[*]}  日志 $LOGDIR/"
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
[ "$fail" -eq 0 ] || { echo "🔴 有分片失败，看 $LOGDIR/"; exit 1; }

EXPECT="$("$PY" -c "import json;print(len(json.load(open('$SPLIT/eval_cases.json'))['case_ids']))")"
"$PY" -m syncopate.train.merge_eval_shards --shards "$TMPDIR_SHARDS" --out "$OUT" --expect "$EXPECT"
rm -rf "$TMPDIR_SHARDS"
touch "${OUT}.done"        # ★ 完成判据：等这个，不要等 $OUT（它可能是上一版留下的）
echo "[eval] 完成标记 ${OUT}.done"
