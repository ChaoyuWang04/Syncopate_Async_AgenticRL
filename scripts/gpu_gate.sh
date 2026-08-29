#!/usr/bin/env bash
# ⛔⛔ 抢卡门禁：infra 线的实验**只有在主线训练+评测全部结束后**才允许起。
#
# 用户 2026-08-17 明令：**训练是最高优先级，严禁抢卡。**
# 而「主线跑完了」这件事有三个不同的判据，只查一个都会误判 —— 三个都查：
#
#   ① 进程    主线的 RL / 评测 / SFT 进程是否都退出了
#   ② 显存    四张卡是否都把显存还回来了
#             ★ `scripts/wait_for_gpu.sh` 开头记着这个坑：日志打完 `[OK] adapter -> …`
#               之后训练进程还在收尾（wandb 上传）、模型仍占 27 GB ⇒ 立刻起评测四步全 OOM。
#               **该等的是资源归还，不是日志行。**
#   ③ 静默期  产物目录在最近 N 分钟内没有新的写入
#             ★ 这条防的是**两个阶段之间的空档**：RL 刚退出、评测还没起来，
#               ①② 会同时通过，而主线其实还没完。
#               `eval_parallel.sh` 的注释里记着同型的坑（「用输出文件存在当判据」）。
#
# 用法：
#   scripts/gpu_gate.sh              # 查一次，全过 exit 0
#   scripts/gpu_gate.sh --watch      # 一直查到全过为止（给后台监控用）
#   QUIET_MIN=15 scripts/gpu_gate.sh # 把静默期从默认 10 分钟改成 15
set -uo pipefail
cd "$(dirname "$0")/.."

QUIET_MIN="${QUIET_MIN:-10}"        # 静默期（分钟）
FREE_MB="${FREE_MB:-28000}"         # 每张卡至少要空出多少 MB（32 GB 卡，留 4 GB 余量）
# ★ 圈卡模式（2026-08-20，对应 /MAINLINE-INFRA「GPU0 vLLM 常驻、1–3 可用」的共存约定）：
#   GPUS=1,2,3 gpu_gate.sh ⇒ ②显存/计算进程只查圈内卡；③静默期放过 logs/runtime/（端点自己的日志）。
#   ①主线训练/评测进程照查不放松——共存约定让的是卡，不是让训练撞车。不设 GPUS = 原样全查。
GPUS="${GPUS:-}"
WATCH_EVERY="${WATCH_EVERY:-120}"   # --watch 的轮询间隔（秒）

# 主线会起的进程（我们自己的实验不在此列 —— 门禁只在起跑之前用）
MAINLINE_PATTERNS='main_ppo_pool|train[.]eval_local|train[.]sft|launch_rl|fully_async_main|ray::'
# 主线产物落在这些地方；静默期看它们最近有没有被写
WATCH_DIRS=(logs _audit checkpoints models data/splits outputs)

check_once () {
    local ok=1

    # ① 进程
    local procs
    procs="$(pgrep -af "$MAINLINE_PATTERNS" 2>/dev/null | grep -v gpu_gate | head -5)"
    if [ -n "$procs" ]; then
        echo "🔴 ① 还有主线进程在跑："
        echo "$procs" | sed 's/^/      /' | cut -c1-140
        ok=0
    else
        echo "🟢 ① 没有主线进程（RL / eval / SFT 全退了）"
    fi

    # ② 显存
    # ⚠️ nvidia-smi 的 csv 输出**带逗号和空格**（`0, 12148, 32607`）⇒ 必须设 IFS 拆，
    #    否则 `$(( total - used ))` 会拿到 `16428,` 直接语法错（2026-08-17 踩过）。
    local bad=0
    while IFS=', ' read -r idx used total; do
        [ -z "${total:-}" ] && continue
        if [ -n "$GPUS" ] && ! echo ",$GPUS," | grep -q ",$idx,"; then
            continue                             # 圈卡模式：圈外卡（主线端点）不查
        fi
        local free=$(( total - used ))
        if [ "$free" -lt "$FREE_MB" ]; then
            echo "🔴 ② GPU$idx 只空 ${free} MB（要 ≥ ${FREE_MB}）"
            bad=1
        fi
    done < <(nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits)
    if [ "$bad" -eq 0 ]; then
        echo "🟢 ② ${GPUS:-全部} 卡显存都空着（每张空 ≥ ${FREE_MB} MB）"
    else
        ok=0
    fi
    # 补一刀：有没有任何进程还挂在卡上（显存可能已释放但进程还在）。圈卡模式按 uuid 归卡后只看圈内。
    local apps
    if [ -n "$GPUS" ]; then
        apps="$(nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader 2>/dev/null \
                | while IFS=', ' read -r pid uuid mem; do
                    gidx="$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader 2>/dev/null \
                            | grep "$uuid" | cut -d, -f1)"
                    echo ",$GPUS," | grep -q ",$gidx," && echo "GPU$gidx:pid$pid:${mem}"
                  done)"
    else
        apps="$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null)"
    fi
    if [ -n "$apps" ]; then
        echo "🔴 ② 圈内卡上还有计算进程：$(echo "$apps" | tr '\n' ' ')"
        ok=0
    fi

    # ③ 静默期
    # ⚠️ 排除 **我们自己**（infra 线）写的东西 —— 否则本窗口一边更新文档/解析日志，
    #    一边永远等不到静默期。判据要量的是「主线还在动」，不是「有人在动」。
    local recent
    local extra_excl='^$'
    [ -n "$GPUS" ] && extra_excl='logs/runtime/'   # 圈卡模式：端点常驻日志不算「主线在动」
    recent="$(find "${WATCH_DIRS[@]}" -type f -mmin "-${QUIET_MIN}" \
                -not -path '*/.git/*' -not -name '*.log.lock' 2>/dev/null \
              | grep -Ev '(e12d_|batch2_|_timing\.json|infra_|gpu_gate|e14|e19|e29|nsys|anatomy|torchprof|b4_|/b4/)' \
              | grep -Ev "$extra_excl" | head -5)"
    if [ -n "$recent" ]; then
        # ⚠️ Chaoyu 2026-08-29 裁定：③ 降为**只报告不拦截**——单实验体制下不存在
        #    两跑抢卡；原始理由（阶段间隙里 ①② 双绿但主线未完 ⇒ 第二批抢卡 OOM，
        #    eval_parallel 头注②的事故）保留在案，若恢复多实验并行要把 ok=0 加回来。
        echo "🟡 ③ 最近 ${QUIET_MIN} 分钟内产物目录有写入（仅提示，不拦截）："
        echo "$recent" | sed 's/^/      /'
    else
        echo "🟢 ③ 产物目录静默 ≥ ${QUIET_MIN} 分钟"
    fi

    return $(( 1 - ok ))
}

if [ "${1:-}" = "--watch" ]; then
    while true; do
        echo "──────── $(date '+%F %T') 门禁检查"
        if check_once; then
            echo "✅ 三条判据全过 —— 可以起 infra 实验了"
            exit 0
        fi
        sleep "$WATCH_EVERY"
    done
else
    echo "──────── $(date '+%F %T') 门禁检查"
    if check_once; then
        echo "✅ 三条判据全过 —— 可以起 infra 实验了"
        exit 0
    fi
    echo "⛔ 未通过 —— **不许起实验**"
    exit 1
fi
