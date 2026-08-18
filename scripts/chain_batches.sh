#!/usr/bin/env bash
# 批次链：前一批的进程一退出就起下一批 —— 不让卡空转。
#
# ★ 起因（2026-08-18 02:19）：batch3 在 21:56 跑完，没人接班，**四张卡空转了 4 个多小时**。
#   ⇒ 从此每批都用这个脚本串起来，而不是靠人（或靠我）记得回来看。
#   用法：nohup bash scripts/chain_batches.sh run_batch6_gpu.sh run_batch7_gpu.sh &
set -uo pipefail
cd "$(dirname "$0")/.."
for nxt in "$@"; do
    # 等当前所有 batch 脚本退出
    while pgrep -f "run_batch[0-9]*_gpu.sh" >/dev/null; do sleep 60; done
    # ⚠️ 再等 60 秒让 Ray/vLLM 把显存真正还回来（日志说完了 ≠ 资源还回来了）
    sleep 60
    echo "[chain] $(date '+%F %T') 起 $nxt"
    set -a; . /workspace/.env 2>/dev/null; set +a
    bash "scripts/$nxt" > "logs/$(basename "$nxt" .sh)_nohup.log" 2>&1
    echo "[chain] $(date '+%F %T') $nxt 结束（退出码 $?）"
done
echo "[chain] 全部批次结束 $(date '+%F %T')"
