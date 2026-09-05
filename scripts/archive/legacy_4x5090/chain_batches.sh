#!/usr/bin/env bash
# 批次链：上一批的**GPU 真的空了**就起下一批 —— 不让卡空转。
#
# ⛔⛔ 2026-08-18 07:02 修：第一版用 `pgrep -f "run_batch[0-9]*_gpu.sh"` 判断"上一批还在吗"，
#    而**链自己的命令行里就带着 `run_batch6_gpu.sh run_batch7_gpu.sh`**
#    ⇒ 它匹配到自己，永远在等自己退出 ⇒ batch5 在 05:38 结束，卡空转到 07:02（1h24m）。
#    ★ 这是「**判据把自己算进去了**」—— 和「判据行挂在别的开关下」「解析器认错格式」
#      「deviceId 是进程局部的」同一族：**判据看起来在工作，量的却不是那件事。**
#    ⇒ 改成**直接问 GPU**：`nvidia-smi --query-compute-apps` 空了才算上一批真的结束。
#      这个判据**不可能匹配到自己**，而且量的正是我们关心的东西（卡有没有被占）。
#
# 用法：nohup setsid bash scripts/chain_batches.sh run_batch7_gpu.sh &
set -uo pipefail
cd "$(dirname "$0")/.."

gpu_busy () { [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]; }

for nxt in "$@"; do
    while gpu_busy; do sleep 60; done
    sleep 60                       # 让显存真正还回来（日志说完了 ≠ 资源还回来了）
    while gpu_busy; do sleep 60; done   # 60 秒里又有人占上了就接着等
    echo "[chain] $(date '+%F %T') 起 $nxt"
    set -a; . /workspace/.env 2>/dev/null; set +a
    bash "scripts/$nxt" > "logs/$(basename "$nxt" .sh)_nohup.log" 2>&1
    echo "[chain] $(date '+%F %T') $nxt 结束（退出码 $?）"
done
echo "[chain] 全部批次结束 $(date '+%F %T')"
