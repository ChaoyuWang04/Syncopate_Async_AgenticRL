#!/usr/bin/env bash
# RL 跑中滚动瘦身：只留**最新一个**全量 ckpt（崩溃续跑用），更早的提成 adapter 后删。
#
#   bash scripts/tools/rl_ckpt_rolling_prune.sh checkpoints/grpo/<run> &
#
# 为什么需要：verl 的 FSDP checkpoint 存全量 state_dict，LoRA 下一个存点 ≈27GB
# 而训练产物只有 253MB。save-freq 50 × 400 步 = 8 个存点 = 216GB ⇒ 盘装不下。
# 滚动后：全程磁盘占用 ≈ 27GB（最新全量）+ N×253MB（历史 adapter，选点/staleness 全够用）。
#
# 安全性：
#   ① 只动「不是最新」的 global_step_*（最新那个可能正在写/要用于续跑）
#   ② 提取走 rl_ckpt_to_adapter（内含跨 rank 一致性断言），验证产物 >1MB 才删
#   ③ 训练结束（pidfile 消失）后再处理一轮，然后自动退出
set -uo pipefail
cd "$(dirname "$0")/../.."
RUN="${1:?用法: rl_ckpt_rolling_prune.sh <run_dir>}"
PIDFILE="${RL_PIDFILE:-logs/rl_train.pid}"

prune_older() {
  # ⚠️ 按**步号**排序（2026-08-19 事故：sort -t_ -k3 切的是整条路径，路径里
  #   cand_v13r2_e1/global_step 全是下划线 ⇒ 键错位 ⇒ 把最新的 100 当旧的删了，
  #   丢掉 step 400 的优化器状态。basename 提数字再排，别信字段位置。）
  mapfile -t steps < <(ls -d "$RUN"/global_step_* 2>/dev/null | awk -F'global_step_' '{print $NF" "$0}' | sort -n | cut -d' ' -f2-)
  local n=${#steps[@]}
  (( n <= 1 )) && return 0
  for gs in "${steps[@]:0:n-1}"; do
    [ -d "$gs/actor" ] || continue
    out="$RUN/adapter_$(basename "$gs")"
    if [ ! -f "$out/adapter_config.json" ]; then
      echo "[rolling] 提取 $(basename "$gs") -> $out"
      .venv/bin/python -m syncopate.train.ckpt_to_adapter "$gs/actor" --out "$out" || { echo "[rolling] 🔴 提取失败，跳过删除"; continue; }
    fi
    size=$(du -sb "$out" 2>/dev/null | cut -f1 || echo 0)
    if [ -f "$out/adapter_config.json" ] && [ "$size" -gt 1000000 ]; then
      echo "[rolling] 提取物验证过 ⇒ 删全量 $(basename "$gs")（$(du -sh "$gs" | cut -f1)）"
      rm -rf "$gs"
    fi
  done
}

echo "[rolling] 盯 $RUN（保留最新全量，其余提 adapter 后删）"
while true; do
  prune_older
  if [ ! -f "$PIDFILE" ] || ! kill -0 "$(cat "$PIDFILE" 2>/dev/null)" 2>/dev/null; then
    sleep 60; prune_older            # 收尾保存可能晚到，等一拍再扫最后一轮
    echo "[rolling] 训练进程已退，收尾扫描完成，退出"; exit 0
  fi
  sleep 120
done
