#!/usr/bin/env bash
# 磁盘守卫：夜间无人值守时，盘要满了就**先保住队列**，牺牲当前这一跑。
#
# ★ 起因（2026-08-17 20:19）：A5 的 nsys 采样 180 秒，在 /workspace/tmp 下产生
#   **72 GB** 中间文件（injection storage），可用空间从 139 G 掉到 67 G。
#   —— profiler 的中间产物比它最后写出的 .nsys-rep（386 MB）大两个数量级。
set -uo pipefail
MIN_GB="${MIN_GB:-15}"
while true; do
    free_gb=$(df -BG --output=avail /workspace | tail -1 | tr -dc '0-9')
    if [ "${free_gb:-999}" -lt "$MIN_GB" ]; then
        echo "[disk-guard] 🔴 只剩 ${free_gb}G < ${MIN_GB}G —— 杀掉 nsys，保住后面的队列"
        pkill -f "nsys profile" && echo "[disk-guard] 已杀 nsys"
        exit 1
    fi
    sleep 60
done
