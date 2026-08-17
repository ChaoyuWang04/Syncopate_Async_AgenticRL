#!/usr/bin/env bash
# 第 1.6 批 · 夜间 GPU 批（2026-08-17 排定，见 00-INFRA-HANDOFF §5「第 1.6 批」）
#
# ⛔⛔ **训练是最高优先级**：本脚本开头会跑 `scripts/gpu_gate.sh`，三条判据不全过就直接退出。
#      不许绕过它（用户 2026-08-17 明令）。
#
# 纪律（infra_exp/README §4，每一项都照做）：
#   ① 先写预测再跑 —— 预测写在每个 run 函数上面，跑完对照
#   ② 每个加速比都要有同分母，一次只变一个变量
#   ③ 跑完就删 27 GB 的 ckpt（`--save-freq 999` 挡不住收尾那次保存）
#
# 用法：
#   bash scripts/run_batch2_gpu.sh              # 全跑
#   bash scripts/run_batch2_gpu.sh a14 b2       # 只跑指定项
#   SKIP_GATE=1 bash scripts/run_batch2_gpu.sh  # ⛔ 只有在人工确认卡空之后才准用
set -uo pipefail
cd "$(dirname "$0")/.."

STAMP="$(date +%m%d_%H%M)"
QUEUE_LOG="logs/batch2_queue_${STAMP}.log"
mkdir -p logs _audit/infra

log () { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$QUEUE_LOG"; }

if [ "${SKIP_GATE:-0}" != "1" ]; then
    if ! bash scripts/gpu_gate.sh 2>&1 | tee -a "$QUEUE_LOG" | tail -3; then
        log "⛔ 门禁未过，退出。主线还在跑就等着 —— 严禁抢卡。"
        exit 1
    fi
fi

# ───────────────────────── 公共参数（同尺子的基础：只有被点名的变量才许变）
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
# ⚠️ bucket 单独拎出来：B2 要把它当唯一变量扫，塞进 COMMON 里就没法只改它
BUCKET_DEFAULT=512

# 跑一个 RL 实验：run <名字> <额外参数...>
run () {
  local name="$1"; shift
  local logf="logs/${name}.log"
  log "════════ $name 开始"
  ( set -x; .venv/bin/python -m syncopate.train.launch_rl "$@" "${COMMON[@]}" \
      --weight-sync-bucket-mb "$BUCKET_DEFAULT" \
      --save-path "checkpoints/grpo/$name" --experiment "$name" ) > "$logf" 2>&1
  local rc=$?
  log "──────── $name 退出码 $rc"
  .venv/bin/python scripts/parse_fully_async_timing.py "$logf" \
      --json "_audit/infra/${name}_timing.json" 2>&1 | tee -a "$QUEUE_LOG"
  # ③ 27 GB/个，跑完就删（dispatched.jsonl 和 rollout_dumps 要留）
  rm -rf "checkpoints/grpo/$name"/global_step_* 2>/dev/null
  log "──────── $name ckpt 已清理"
}

want () {  # 没给参数 = 全跑；给了就只跑点名的
  [ $# -eq 0 ] && return 0
  local t="$1"; shift
  for a in "$@"; do [ "$a" = "$t" ] && return 0; done
  return 1
}
TARGETS=("$@")

# ═══════════════════════════════════════════════════════════════════════
# ① A14 · 真实 ZeRO-3 的分片是不是 16 字节错位（E18 唯一没闭的环）
#
# ★ 预测（跑之前写死）：
#   P1 —— `AllGather: N Bytes` 里 **%16 != 0 的按字节加权占比 > 80%**。
#         若 < 20%，则 6.02× 另有原因，E18 §10 的因果链就还差一环（这是可证伪的）。
#   P2 —— 强制 `NCCL_PROTO=Simple` 时 update_actor 复现 ~47.9 s；自动 LL128 时 ~14.4 s。
#
# ⚠️ `--fsdp-size 3` 会让 launch_rl 自动加 `+…NCCL_PROTO="LL128"` ⇒ 要压回默认必须用 `++`
#    （Hydra：键已存在时 `+` 会报 "Could not append to config"，见 E18 §7-3）。
# ⚠️ 统计**必须按字节加权**，不能按调用次数 —— 小张量再多也解释不了 6.02×。
if want a14 "${TARGETS[@]}"; then
  for proto in Simple LL128; do
    name="a14_zero3_${proto,,}"
    logf="logs/${name}.log"
    log "════════ $name 开始（NCCL_PROTO=$proto，抓 AllGather 字节数）"
    ( set -x; NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=TUNING \
      .venv/bin/python -m syncopate.train.launch_rl \
        --mode colocate --trainer-gpus 3 \
        --fsdp-size 3 --steps 2 --rollout-gpu-util 0.35 \
        "${COMMON[@]}" --weight-sync-bucket-mb "$BUCKET_DEFAULT" \
        --save-path "checkpoints/grpo/$name" --experiment "$name" \
        "++ray_kwargs.ray_init.runtime_env.env_vars.NCCL_PROTO=$proto" \
        '++actor_rollout_ref.actor.fsdp_config.reshard_after_forward=True' \
    ) > "$logf" 2>&1
    log "──────── $name 退出码 $?"
    .venv/bin/python scripts/analyze_allgather_alignment.py "$logf" \
        --json "_audit/infra/${name}_align.json" 2>&1 | tee -a "$QUEUE_LOG"
    rm -rf "checkpoints/grpo/$name"/global_step_* 2>/dev/null
  done
fi

# ═══════════════════════════════════════════════════════════════════════
# ② B2 · bucket 512 vs 2048 的干净分母（E12-e）
#
# ★ 预测：param_sync 稳态 512 档 ≈ 8.4 s（主线野外值）、2048 档 ≈ 55 s（E12 记录），
#   即**耗时随 bucket 走而不是随传输量走**（传的恒是 132 MB）。
#   若两档接近，说明主线那个 6.6× 来自别的变量（数据版本/dynamic_bsz/…），
#   §7.4 的推论就要撤回。
if want b2 "${TARGETS[@]}"; then
  for mb in 512 2048; do
    name="b2_bucket${mb}"
    logf="logs/${name}.log"
    log "════════ $name 开始"
    ( set -x; .venv/bin/python -m syncopate.train.launch_rl \
        --mode fully_async --trainer-gpus 3 --rollout-gpus 1 --steps 12 \
        --sync-every 4 --staleness-threshold 0.1 \
        "${COMMON[@]}" --weight-sync-bucket-mb "$mb" \
        --save-path "checkpoints/grpo/$name" --experiment "$name" \
    ) > "$logf" 2>&1
    log "──────── $name 退出码 $?"
    .venv/bin/python scripts/parse_fully_async_timing.py "$logf" \
        --json "_audit/infra/${name}_timing.json" 2>&1 | tee -a "$QUEUE_LOG"
    rm -rf "checkpoints/grpo/$name"/global_step_* 2>/dev/null
  done
fi

# ═══════════════════════════════════════════════════════════════════════
# ③ B3 · 三模式同尺子（E08-b）—— 兑现物里那句「三模式同尺子比较」目前是假的
#
# ★ 预测：fully_async < one_step_off < colocate(3卡)，且 fully_async 相对 colocate
#   的收益落在长尾比 1.37–2.75× 之间（P-B1）。超出上限 ⇒ 口径错了（多半是分母不同机）。
if want b3 "${TARGETS[@]}"; then
  run b3_colocate3     --mode colocate     --trainer-gpus 3 --steps 8 --rollout-gpu-util 0.40
  run b3_onestepoff    --mode one_step_off --trainer-gpus 3 --rollout-gpus 1 --steps 8
  run b3_fullyasync    --mode fully_async  --trainer-gpus 3 --rollout-gpus 1 --steps 12 --sync-every 4
fi

log "════════ 队列结束（本脚本只含 ①②③；④⑤⑥⑦⑧ 由窗口按结果决定后续，见 handoff §5 第 1.6 批）"
