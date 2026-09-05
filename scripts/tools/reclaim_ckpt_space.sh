#!/usr/bin/env bash
# 按 disk_report 的处置回收旧 ckpt：先提 adapter（信息保留）再删 global_step_*（体积回收）。
#
#   bash scripts/tools/reclaim_ckpt_space.sh e17a_kl_on e17b_kl_off ...
#
# ⚠️ 三道闸，缺一不删（「先验证提取物，再删原件」——删了就回不来）：
#   ① rl_ckpt_to_adapter 退出码 0（内含 assert_ranks_identical，E21 前的分叉 ckpt 会拦）
#   ② 提取物 adapter_config.json + safetensors 存在且 > 1 MB
#   ③ 只删 global_step_*；dispatched.jsonl / rollout_dumps / pool_state 一律不碰
set -euo pipefail
cd "$(dirname "$0")/../.."

for run in "$@"; do
  root="checkpoints/grpo/$run"
  [ -d "$root" ] || { echo "⏭ $run 不存在"; continue; }
  for gs in "$root"/global_step_*; do
    [ -d "$gs/actor" ] || continue
    out="$root/adapter_$(basename "$gs")"
    if [ ! -f "$out/adapter_config.json" ]; then
      echo "== $run/$(basename "$gs") 提取 -> $out"
      .venv/bin/python -m syncopate.train.ckpt_to_adapter "$gs/actor" --out "$out"
    fi
    size=$(du -sb "$out" 2>/dev/null | cut -f1 || echo 0)
    if [ -f "$out/adapter_config.json" ] && [ "$size" -gt 1000000 ]; then
      echo "   提取物 $(du -sh "$out" | cut -f1) 已验证 ⇒ 删 $(du -sh "$gs" | cut -f1) 的 $(basename "$gs")"
      rm -rf "$gs"
    else
      echo "   🔴 提取物缺失或过小（$size B）—— 不删 $gs"
    fi
  done
done
df -h /workspace | tail -1
