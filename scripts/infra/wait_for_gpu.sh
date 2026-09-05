#!/usr/bin/env bash
# 等显存空出来再往下走。
#
# ★ 为什么需要它：2026-08-13 踩过 —— SFT 的日志打出 `[OK] adapter -> ...` 之后
# 我立刻起了评测，结果四步全 OOM。那行日志只表示 adapter 存完了，**训练进程还在
# 收尾（wandb 上传），模型仍占着 27GB 显存**。
#
# ⇒ 该等的是**进程退出 / 显存释放**，不是日志行。
# 日志行是"某件事做完了"的证据，不是"资源还回来了"的证据。
set -u
NEED_MB=${1:-24000}      # 需要多少 MB 空闲
TIMEOUT=${2:-900}        # 最多等多久（秒）
start=$(date +%s)
while true; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    free=$(( total - used ))
    [ "$free" -ge "$NEED_MB" ] && { echo "[GPU] 空闲 ${free}MB ≥ ${NEED_MB}MB，继续"; exit 0; }
    now=$(date +%s)
    [ $(( now - start )) -ge "$TIMEOUT" ] && { echo "[GPU] 等了 ${TIMEOUT}s 仍只有 ${free}MB 空闲，放弃" >&2; exit 1; }
    echo "[GPU] 空闲 ${free}MB < ${NEED_MB}MB，等待中…"
    sleep 10
done
