#!/usr/bin/env bash
# 第 1.6 批 · 后半（batch2 跑完之后再起）
#
# ⚠️⚠️ **为什么另开一个文件而不是往 batch2 里加**：
#    bash 是**按字节偏移增量读脚本**的 —— 改一个正在跑的脚本，它会从新内容的旧偏移处接着读，
#    行为不可预测（可能执行到半行）。⇒ **正在跑的脚本一个字都不能动。**
#
# 顺序（依据 handoff §5 第 1.6 批，A16 是 2026-08-17 新增的）：
#   A16  对齐悬崖在**真实分片尺寸**上复现     ~5 min   ★ A15 提 issue 前欠的那一步
#   B11  拓扑感知的放置（2+2 才有的问题）      ~1 h
#   B10  陈旧度节流的代价曲线                  ~2 h
set -uo pipefail
cd "$(dirname "$0")/.."

STAMP="$(date +%m%d_%H%M)"
QUEUE_LOG="logs/batch3_queue_${STAMP}.log"
mkdir -p logs _audit/infra
log () { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$QUEUE_LOG"; }

if [ "${SKIP_GATE:-0}" != "1" ]; then
    bash scripts/gpu_gate.sh >> "$QUEUE_LOG" 2>&1 || { log "⛔ 门禁未过，退出"; exit 1; }
fi

COMMON=(
  --model models/Qwen3-4B-sft-v13-e1
  --train-file data/rl/v13/train.parquet --val-file data/rl/v13/val.parquet
  --lora-rank 32
  --train-batch-size 6 --rollout-n 8 --ppo-mini-batch-size 6 --micro-batch-size 1
  --max-num-seqs 64 --object-store-gb 2
  --max-prompt-length 3584 --max-response-length 1536
  --save-freq 999 --wandb-mode offline --logger console
  --dynamic-bsz False --max-token-len-per-gpu 16384
)
BUCKET_DEFAULT=512
RUN_TIMEOUT="${RUN_TIMEOUT:-5400}"

run () {   # ⚠️ COMMON 在前、单跑参数在后、Hydra 位置参数最后（同 batch2 的两条教训）
  local name="$1"; shift
  local logf="logs/${name}.log"
  log "════════ $name 开始"
  ( set -x; timeout "$RUN_TIMEOUT" .venv/bin/python -m syncopate.train.launch_rl \
      "${COMMON[@]}" --weight-sync-bucket-mb "$BUCKET_DEFAULT" \
      --save-path "checkpoints/grpo/$name" --experiment "$name" \
      "$@" ) > "$logf" 2>&1
  local rc=$?
  log "──────── $name 退出码 $rc"
  [ "$rc" = "124" ] && log "🔴 $name 撞到 ${RUN_TIMEOUT}s 超时"
  .venv/bin/python scripts/parse_fully_async_timing.py "$logf" \
      --json "_audit/infra/${name}_timing.json" 2>&1 | tee -a "$QUEUE_LOG"
  rm -rf "checkpoints/grpo/$name"/global_step_* 2>/dev/null
}

want () { [ $# -le 1 ] && return 0; local t="$1"; shift
          for a in "$@"; do [ "$a" = "$t" ] && return 0; done; return 1; }
TARGETS=("$@")

# ═══════════════════════════════════════════════════════════════════════
# ① A16 · 对齐悬崖在**真实分片尺寸**上复现（A15 提 issue 前欠的那一步）
#
# A14 实测 verl ZeRO-3 每 rank 分块 = 67,287,212 B（%16=12）。
# 现在有的两条证据是：
#   ① 微基准上（整齐的 24 MB + 偏移）补齐就恢复 —— E18 §10.3
#   ② 真实路径上确实错位 —— E18 §12
# **缺的是「在真实那个尺寸上补 4 字节就恢复」**，而 issue 要的正是这一条。
#
# ★ 预测：67,287,212（%16=12）≈ 1 GB/s 档；67,287,216（%16=0）≈ 13 GB/s 档，差 ~10×。
#   若两者接近，说明悬崖只在特定尺寸区间出现 ⇒ E18 §10.4 的「正解是补齐」要收窄成立范围。
if want a16 "${TARGETS[@]}"; then
  log "════════ a16 对齐悬崖 @ 真实分片尺寸 67,287,212"
  ( set -x; CLIFF_BASE=67287212 CLIFF_OFFS=0,4,8,12,16 \
      timeout 900 .venv/bin/python scripts/probe_alignment_cliff.py ) 2>&1 | tee -a "$QUEUE_LOG"
fi

# ═══════════════════════════════════════════════════════════════════════
# ② B11 · 拓扑感知的放置（2+2 才有的问题，E08-e）
#
# 机器是 2 socket、GPU0/1@node0 · GPU2/3@node1；组内 28.8 / 跨 socket 22.2 GB/s。
# 当前 trainer=0,1,2 **跨 socket** ⇒ DDP 的 all_reduce 每步走 UPI。
# ★ 预测：差别很小（<3%）。依据是 focus-migration §1 的换算：
#   DDP 梯度 260 MB，跨 socket 净代价 1.2 ms/步 = 一步的 0.004%。
#   **如果实测差别明显大于 3%，说明代价不在 all_reduce 上**，那才是新发现。
# ⚠️ launch_rl 目前按序号取卡，两种摆法靠 CUDA_VISIBLE_DEVICES 换序实现。
if want b11 "${TARGETS[@]}"; then
  ( export CUDA_VISIBLE_DEVICES=0,1,2,3
    run b11_trainer012_rollout3 --mode fully_async --trainer-gpus 3 --rollout-gpus 1 --steps 12 --sync-every 4 )
  ( export CUDA_VISIBLE_DEVICES=2,3,0,1
    run b11_trainer230_rollout1 --mode fully_async --trainer-gpus 3 --rollout-gpus 1 --steps 12 --sync-every 4 )
fi

# ═══════════════════════════════════════════════════════════════════════
# ③ B10 · 陈旧度节流的代价曲线（占空比成因②，至今没量过）
#
# ★ 预测：放宽 staleness_threshold 0.1 → 0.5 能显著提吞吐（rollout 卡 82.5% 在闲），
#   代价是陈旧度上升；ESS 会跟着掉。**ESS/N < 0.3 是停止条件**（自定纪律）。
# ⚠️ 这条动的是**算法有效性**，不是纯吞吐 ⇒ 结论必须带 ESS，且要过 B5 的任务级尺子才算数。
if want b10 "${TARGETS[@]}"; then
  for th in 0.1 0.3 0.5; do
    run "b10_stale${th}" --mode fully_async --trainer-gpus 3 --rollout-gpus 1 --steps 12 \
        --sync-every 4 --staleness-threshold "$th"
    log "   ESS（rollout_corr/*）：$(grep -oE 'rollout_corr/[a-z_]*ess[a-z_]*:[0-9.]+' "logs/b10_stale${th}.log" | tail -3 | tr '\n' ' ')"
  done
fi

# ═══════════════════════════════════════════════════════════════════════
# ④ B2-redo · 只有在 batch2 的 2048 档 OOM 时才跑（`bash scripts/run_batch3_gpu.sh b2redo`）
#
# 历史坑：bucket 2048 会在**第一次权重同步**时 OOM —— CheckpointEngine 在**目标卡**上
# 分配一个 bucket 大小的暂存区，而 rollout 卡上还住着 vLLM（默认 gpu_util 0.85）。
# ⇒ 重跑时把两档**同时**降到 0.65，保持「只改 bucket 这一个变量」。
# ⚠️ 降 gpu_util 会缩小 KV 池 ⇒ 生成会变慢 ⇒ **这一对的绝对值不能和别的跑比**，只看两档之差。
if want b2redo "${TARGETS[@]}"; then
  for mb in 512 2048; do
    run "b2redo_bucket${mb}" --mode fully_async --trainer-gpus 3 --rollout-gpus 1 --steps 12 \
        --sync-every 4 --rollout-gpu-util 0.65 --weight-sync-bucket-mb "$mb"
    log "   param_sync：$(grep -oE 'param_sync: [0-9.]+ seconds' "logs/b2redo_bucket${mb}.log" | tr '\n' ' ')"
  done
fi

log "════════ batch3 结束"
