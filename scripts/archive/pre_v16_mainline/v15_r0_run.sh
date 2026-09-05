#!/usr/bin/env bash
# v15 · R0 强通道假说验证 —— 双臂微调（25 §4-R0）
#
#   bash scripts/v15_r0_run.sh            # 两臂顺序跑，单卡 GPU3
#
# ★ 两臂唯一的差别就是训练文件；模型/超参/seed/epoch 数**逐字相同**（公平性口径 §R0③）。
#
# ⚠️ 起点选择的**已知偏向**（结果怎么读，取决于这段）：
#   起点 = v14.5-epoch3，它**已经把壳契约练熟了**，而工具臂要从零学一套没见过的契约。
#   ⇒ 这个设计**偏向壳臂 = 保守（不利于假说）**。
#     · 工具臂仍赢 ≥15pp  ⇒ 结论很强（逆风还赢）
#     · 工具臂输          ⇒ **区分不了「假说错」和「工具臂没练够」**
#   ⇒ 分辨这两者的尺子就是门槛②a（信令语法合法率 ≥99%）：合法率够高 = 契约学会了，
#     那时候输才算假说输。
#
# ★ epoch 数的定法（不拍脑袋）：整 epoch 全部存档，e1..e6 一次跑出来；
#   评测时按**预注册规则**选点 —— 取「两臂都过②a 的最小 epoch」，避免事后挑对自己有利的点。
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; . /workspace/.env; set +a

MODEL=models/Qwen3-4B-sft-v14.5-epoch3
EPOCHS=6
SEED=1500
mkdir -p logs/v15_r0

for ARM in shell tool; do
  OUT=checkpoints/v15_r0/arm_${ARM}
  LOG=logs/v15_r0/sft_${ARM}.log
  echo "════════ 臂 ${ARM} ════════ $(date +%H:%M:%S)"
  CUDA_VISIBLE_DEVICES=3 SYNCOPATE_SFT_SINGLE=1 \
  .venv/bin/python -m syncopate.train.sft \
      --model "$MODEL" \
      --train-file "data/v15_r0/arm_${ARM}/train.parquet" \
      --val-file   "data/v15_r0/arm_${ARM}/val.parquet" \
      --out "$OUT" --epochs "$EPOCHS" --seed "$SEED" \
      --save-epochs "" --no-wandb \
      > "$LOG" 2>&1 || { echo "🔴 臂 ${ARM} 训练失败"; tail -20 "$LOG"; exit 1; }
  echo "  存档: $(ls -d ${OUT}/epoch* 2>/dev/null | tr '\n' ' ')"
  grep -E "^\[epoch" "$LOG" | tail -3
done
echo "✅ 两臂训练完成 $(date +%H:%M:%S)"
