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
  # ★ 长度预算与采样参数**不在这里传** —— 唯一来源是 syncopate/train/rollout_budget.py，
  #   launch_rl 的默认值从那里取。⚠️ 显式传会被 check_pipeline_invariants 的 contract 组判红
  #   （2026-08-18 就是各脚本各抄一份，抄着抄着漂成了 3584/1536 vs 5120/2048 两套）。
  --save-freq 999 --wandb-mode offline --logger console
  --dynamic-bsz False --max-token-len-per-gpu 16384
)
# ⚠️ bucket 单独拎出来：B2 要把它当唯一变量扫，塞进 COMMON 里就没法只改它
BUCKET_DEFAULT=512

# 跑一个 RL 实验：run <名字> <该跑特有的参数...> [Hydra override…]
#
# ⚠️ 参数顺序有讲究，两条都踩过或差点踩：
#   ① **COMMON 在前、单跑参数在后** —— argparse 里后出现的同名参数覆盖前面的，
#      B2 要把 bucket 当唯一变量扫，就靠这个顺序。
#   ② **Hydra 的位置参数（`++x=y`）必须排在最后** —— argparse 的 `nargs="*"`
#      位置参数和选项交错时会解析失败，报错还很难看懂。
# ⚠️ 每跑套 timeout：夜里没人盯，一次挂死不能把整批吃掉。
RUN_TIMEOUT="${RUN_TIMEOUT:-5400}"     # 单跑上限（秒）
run () {
  local name="$1"; shift
  local logf="logs/${name}.log"
  log "════════ $name 开始"
  ( set -x; timeout "$RUN_TIMEOUT" .venv/bin/python -m syncopate.train.launch_rl \
      "${COMMON[@]}" --weight-sync-bucket-mb "$BUCKET_DEFAULT" \
      --save-path "checkpoints/grpo/$name" --experiment "$name" \
      "$@" ) > "$logf" 2>&1
  local rc=$?
  [ "$rc" = "124" ] && log "🔴 $name 撞到 ${RUN_TIMEOUT}s 超时被杀 —— 后面的照跑，别让它吃掉整批"
  log "──────── $name 退出码 $rc"
  .venv/bin/python scripts/parse_fully_async_timing.py "$logf" \
      --json "_audit/infra/${name}_timing.json" 2>&1 | tee -a "$QUEUE_LOG"
  # ③ 27 GB/个，跑完就删（dispatched.jsonl 和 rollout_dumps 要留）
  rm -rf "checkpoints/grpo/$name"/global_step_* 2>/dev/null
  log "──────── $name ckpt 已清理"
}

want () {  # 没给筛选参数 = 全跑；给了就只跑点名的
  # ⚠️ 调用形式是 `want a14 "${TARGETS[@]}"` ⇒ 没有筛选时 $# 是 **1**（只有名字）不是 0。
  #    第一版写 `[ $# -eq 0 ]`，结果整批一项都没跑、还打了「队列结束」——
  #    ★ 又一个「判据看起来完整、其实永远为假」：脚本正常退出、日志正常收尾。
  [ $# -le 1 ] && return 0
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
      timeout "$RUN_TIMEOUT" .venv/bin/python -m syncopate.train.launch_rl \
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
    ( set -x; timeout "$RUN_TIMEOUT" .venv/bin/python -m syncopate.train.launch_rl \
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

# ═══════════════════════════════════════════════════════════════════════
# ④ A5 · E01 的阶段归属（nsys + NVTX）—— B12/E17 的门槛
#
# ⚠️ 为什么必须是**我们自己**的一跑：`nsys` **只能包住启动，不能事后 attach**；
#    而且要打 NVTX（`--nvtx`），改的是启动路径。
#    ⇒ 「挂在别人的跑上零成本」只对不需要改被测对象的观测成立（E01 §6.1 的教训）。
#
# ★ 预测：三次前向（update_actor/old_log_prob/ref）在 **kernel 层**的占比与 timing 行
#   一致（±10 个百分点内）。若差很多，说明有一大块时间**不在 kernel 上**
#   （等 / H2D / Python），那本身就是更值钱的发现。
if want a5 "${TARGETS[@]}"; then
  NSYS=/opt/nvidia/nsight-compute/2025.1.1/host/target-linux-x64/nsys
  name=a5_e01_nvtx
  log "════════ $name 开始（nsys 包住启动 + NVTX）"
  ( set -x; timeout "$RUN_TIMEOUT" "$NSYS" profile -o logs/nsys/${name} --force-overwrite true \
      --trace=cuda,nvtx,osrt --delay 420 --duration 180 \
      .venv/bin/python -m syncopate.train.launch_rl \
        --mode fully_async --trainer-gpus 3 --rollout-gpus 1 --steps 16 \
        --sync-every 4 --nvtx \
        "${COMMON[@]}" --weight-sync-bucket-mb "$BUCKET_DEFAULT" \
        --save-path "checkpoints/grpo/$name" --experiment "$name" \
  ) > "logs/${name}.log" 2>&1
  log "──────── $name 退出码 $?"
  # 判据①：两侧各一行 NVTX 判据（只有一行 = 作用域漏了一半）
  grep -c "NVTX 阶段标注 ✓" "logs/${name}.log" | xargs -I{} log "   NVTX 判据行 {} 条（要 ≥2）"
  "$NSYS" stats --report cuda_gpu_kern_sum --format csv \
      --output "_audit/infra/nsys/${name}" "logs/nsys/${name}.nsys-rep" >> "$QUEUE_LOG" 2>&1
  .venv/bin/python scripts/analyze_nsys_step.py "logs/nsys/${name}.sqlite" \
      --json "_audit/infra/${name}_nsys.json" 2>&1 | tee -a "$QUEUE_LOG"
  rm -rf "checkpoints/grpo/$name"/global_step_* 2>/dev/null
  # ⚠️ sqlite 中间产物 9 GB 级，留 .nsys-rep 就够（要用再 export）
  rm -f "logs/nsys/${name}.sqlite"
  log "──────── $name 清理完成"
fi

# ═══════════════════════════════════════════════════════════════════════
# ⑤ B12 / E17 · 三次前向里 `ref` 那一遍值不值（占空比最大的一块）
#
# ★ 预测写在 E17 §1，一句话：关掉 KL ⇒ 每步墙钟降 12–15%，而 KL 项当前只贡献
#   损失的 0.011%（E17 §4.1 的零 GPU 账）。
# ★ 机制是读码确证的：use_kl_in_reward=False + use_kl_loss=False
#   ⇒ need_reference_policy=False ⇒ **整个 ref 段被跳过**（verl/trainer/ppo/utils.py:79）。
# **判据**：B 臂的 timing 行里**不该再有 `timing_s/ref`**。有就是没关掉。
# ⚠️ 吞吐赢了**不算完成** —— 必须再过一次 B5 的任务级尺子（E17 §8-2）。
if want b12 "${TARGETS[@]}"; then
  run b12_ref_on   --mode fully_async --trainer-gpus 3 --rollout-gpus 1 --steps 12 --sync-every 4
  run b12_ref_off  --mode fully_async --trainer-gpus 3 --rollout-gpus 1 --steps 12 --sync-every 4 \
                   '++actor_rollout_ref.actor.use_kl_loss=False'
  log "   ★ 判据：ref_off 那跑里 timing_s/ref 应当消失 —— 实际出现次数："
  grep -c "timing_s/ref" logs/b12_ref_off.log | xargs -I{} log "     {} 次（要 0）"
fi

log "════════ 队列结束（本脚本含 ①A14 ②B2 ③B3 ④A5 ⑤B12；⑥B11 ⑦B10 ⑧A9 由窗口按结果决定，见 handoff §5 第 1.6 批）"
