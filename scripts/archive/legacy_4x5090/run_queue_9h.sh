#!/usr/bin/env bash
# 9 小时任务队列 —— 按 `docs/infra_exp/00-INFRA-HANDOFF §5.1 + §5.2` 的优先级串行跑完。
#
# ★ 这条队列的前提（2026-08-18 全部就位，缺一条都不该跑）：
#     E21 修复（梯度真的同步）· 0-A（归约口径 1.000000）· E22 修法①（adapter 真的到达 rollout）
#   ⇒ 在此之前的所有"学习类"数字都作废，本队列就是去把它们重测出来的。
#
# 每一项都做四件事，缺一不可：
#   ① 前置检查   磁盘 ≥ 40G、显存已释放（**日志说完了 ≠ 资源还回来了**，wait_for_gpu 的教训）
#   ② 跑
#   ③ 判据解析   三条常驻判据（list_loras / 载荷含 lora_ / kl 回地板）+ 本项自己的指标
#   ④ ckpt 策略  **要过任务尺子的留，其余提指纹后删** —— 每跑收尾落 27 GB，11 跑会撑爆盘
#
# ⚠️ 单项失败不许拖垮队列：每项 `set +e`，失败记一行继续往下。
# ⚠️ 进度写在 logs/queue9h/  —— 每项完成落一个 T<N>.done，里面是判据摘要（供外部分析）。
set -uo pipefail
cd "$(dirname "$0")/.."
set -a; . /workspace/.env 2>/dev/null || true; set +a

Q=logs/queue9h
mkdir -p "$Q" _audit/infra
BOARD="$Q/BOARD.md"
say () { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$Q/queue.log"; }

COMMON=(
  --model models/Qwen3-4B-sft-v13-e1
  --train-file data/rl/v13/train.parquet --val-file data/rl/v13/val.parquet
  --lora-rank 32 --train-batch-size 6 --rollout-n 8 --micro-batch-size 1
  --max-num-seqs 64 --object-store-gb 2
  --save-freq 999 --wandb-mode offline --logger console --dynamic-bsz False
  --max-token-len-per-gpu 16384 --mode fully_async --trainer-gpus 3 --rollout-gpus 1
)
# 三条常驻判据靠这两个探针产出（判据行见 handoff §5.1.1）
export SYNCOPATE_SYNC_PAYLOAD=1
export SYNCOPATE_SYNC_REF=75.377708
export SYNCOPATE_SYNC_WATCH="model.layers.0.self_attn.q_proj.base_layer.weight"

preflight () {   # $1 = 任务名
  local free_g; free_g=$(df -BG --output=avail /workspace | tail -1 | tr -dc '0-9')
  if [ "${free_g:-0}" -lt 40 ]; then
    say "🔴 $1 前置检查失败：磁盘只剩 ${free_g}G（<40G）⇒ 跳过，避免写盘被截断"
    return 1
  fi
  # ⚠️ 不能用 scripts/wait_for_gpu.sh —— 它只看 **GPU0**（`head -1`）。
  #    无人值守 9 小时，残留在 GPU2 上的进程会让下一跑直接 OOM。
  #    ⇒ 这里看**全部四张卡里最忙的那张**。
  local waited=0 busiest
  while :; do
    busiest=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
    [ "${busiest:-99999}" -lt 2000 ] && break
    [ "$waited" -ge 900 ] && { say "⚠️ $1：等了 900s，最忙的卡仍占 ${busiest}MB，仍继续"; break; }
    sleep 15; waited=$((waited+15))
  done
  say "前置检查通过（磁盘 ${free_g}G · 最忙的卡 ${busiest}MB · 等了 ${waited}s）"
  return 0
}

# 三条常驻判据 + 本项指标，写进 T<N>.done
verdict () {   # $1=编号 $2=名字 $3=日志
  local n="$1" name="$2" log="$3" out="$Q/T${n}.done"
  {
    echo "# T$n · $name"
    echo "结束时间: $(date '+%F %T')"
    [ -f "$log" ] || { echo "🔴 没有日志"; return; }
    local lora payload kl
    lora=$(grep -o "list_loras()=\[[0-9]*\]" "$log" | sort | uniq -c | tr '\n' ' ')
    payload=$(grep -o "本次同步推出去：[0-9]* 个张量 / [0-9,.]* MiB / 其中 lora_ [0-9]* 个" "$log" | sort -u | tr '\n' ' ')
    kl=$(grep -o "rollout_corr/kl:[0-9.e+-]*" "$log" | sed 's/.*://' | awk '{printf "%.5f ",$1}')
    echo "判据① list_loras : ${lora:-（无）}"
    echo "判据② 载荷       : ${payload:-（无）}"
    echo "判据③ kl 轨迹    : ${kl:-（无）}"
    for k in rollout_corr/rollout_is_eff_sample_size rollout_corr/chi2_token rollout_corr/chi2_seq \
             rollout_corr/log_ppl_diff actor/grad_norm response_length/mean critic/score/mean; do
      v=$(grep -o "${k}:[0-9.e+-]*" "$log" | sed 's/.*://' | awk 'NR==1{f=$1} {l=$1} END{if(NR)printf "%s → %s（%d 次）",f,l,NR}')
      [ -n "$v" ] && echo "  ${k}: $v"
    done
    echo "更新次数(dump 文件数): $(ls checkpoints/grpo/${name}/rollout_dumps/*.jsonl 2>/dev/null | wc -l)"
    echo "错误: $(grep -c -i 'RayTaskError' "$log") 处 RayTaskError"
    echo "新出现的 UserWarning: $(grep -c 'UserWarning' "$log") 条（★ 纪律：新出现的必须有人看过）"
  } > "$out" 2>&1
  say "── T$n 判据已落盘 → $out"
  { echo; echo "## T$n · $name"; cat "$out" | sed 's/^/    /'; } >> "$BOARD"
}

# ckpt 策略：keep=1 留着过任务尺子；否则提指纹后删（每跑 27 GB）
sweep () {   # $1=实验名 $2=keep
  local name="$1" keep="${2:-0}"
  local ck; ck=$(ls -d checkpoints/grpo/${name}/global_step_*/actor 2>/dev/null | tail -1)
  [ -n "$ck" ] || return 0
  .venv/bin/python scripts/extract_ckpt_fingerprint.py "$ck" >/dev/null 2>&1 || true
  if [ "$keep" = "1" ]; then
    say "   ckpt 保留（要过任务尺子）：$(du -sh checkpoints/grpo/${name} | cut -f1)"
  else
    find "checkpoints/grpo/${name}" -name "*.pt" -delete 2>/dev/null
    say "   ckpt 已清（指纹已留）；磁盘剩 $(df -BG --output=avail /workspace | tail -1 | tr -d ' ')"
  fi
}

rl_run () {   # $1=编号 $2=实验名 $3=keep $4..=额外参数
  local n="$1" name="$2" keep="$3"; shift 3
  preflight "T$n·$name" || { echo "跳过（前置检查失败）" > "$Q/T${n}.done"; return; }
  say "════════ T$n · $name 开始"
  ( set -x; timeout 7200 .venv/bin/python -m syncopate.train.launch_rl \
      "${COMMON[@]}" --save-path "checkpoints/grpo/$name" --experiment "$name" "$@" ) \
      > "logs/${name}.log" 2>&1
  say "──────── T$n · $name 退出码 $?"
  verdict "$n" "$name" "logs/${name}.log"
  sweep "$name" "$keep"
}

echo "# 9 小时队列 · 开始于 $(date '+%F %T')" > "$BOARD"
say "════════════════ 队列启动 ════════════════"

# ───────────────────────── §5.1 ─────────────────────────
# T1 · ⑥ 重基线评测（4 卡 ~15 min）—— 最短且是读 R1 任务分的前提，排第一
preflight "T1·rebaseline" && {
  say "════════ T1 · ⑥ 重基线评测（RL 真起点的审计）"
  MODEL=models/Qwen3-4B-sft-v13-e1 timeout 3600 \
    bash scripts/eval_parallel.sh "" _audit/v13_sft_e1_merged.json > "$Q/T1_eval.log" 2>&1
  say "──────── T1 退出码 $?"
  {
    echo "# T1 · ⑥ 重基线评测"; echo "结束时间: $(date '+%F %T')"
    if [ -f _audit/v13_sft_e1_merged.json.done ]; then
      echo "✅ .done 标记已出现（★ 判据是它，不是"输出文件存在"）"
      .venv/bin/python -m syncopate.train.compare _audit/v13_sft_e1.json _audit/v13_sft_e1_merged.json 2>&1 | tail -25
    else
      echo "🔴 没有 .done 标记 ⇒ 评测没跑完"; tail -20 "$Q/T1_eval.log"
    fi
  } > "$Q/T1.done" 2>&1
  { echo; echo "## T1 · ⑥ 重基线评测"; sed 's/^/    /' "$Q/T1.done"; } >> "$BOARD"
  say "── T1 判据已落盘"
}

# T2/T3 · R1 · E20 全套重测：token 级 vs 序列级 IS（★ ckpt 必须留，要过 B5）
rl_run 2 r1_tokenis  1 --rollout-is token    --lr 3e-5 --ppo-mini-batch-size 6 --steps 60 --sync-every 4 --weight-sync-bucket-mb 512
rl_run 3 r1_seqis    1 --rollout-is sequence --lr 3e-5 --ppo-mini-batch-size 6 --steps 60 --sync-every 4 --weight-sync-bucket-mb 512

# T4/T5 · B5 · 把两臂的 ckpt 过任务级尺子（三计数，不看均值）
for pair in "4 r1_tokenis" "5 r1_seqis"; do
  set -- $pair; n="$1"; name="$2"
  preflight "T$n·B5($name)" && {
    say "════════ T$n · B5 任务尺子 · $name"
    ck=$(ls -d checkpoints/grpo/${name}/global_step_*/actor 2>/dev/null | tail -1)
    if [ -z "$ck" ]; then
      echo "🔴 找不到 $name 的 ckpt ⇒ 跳过" > "$Q/T${n}.done"
    else
      .venv/bin/python scripts/rl_ckpt_to_adapter.py "$ck" --out "models/adapters/${name}" \
        > "$Q/T${n}_adapter.log" 2>&1
      if [ -d "models/adapters/${name}" ]; then
        MODEL=models/Qwen3-4B-sft-v13-e1 timeout 3600 \
          bash scripts/eval_parallel.sh "models/adapters/${name}" "_audit/${name}.json" \
          > "$Q/T${n}_eval.log" 2>&1
        {
          echo "# T$n · B5 · $name"; echo "结束时间: $(date '+%F %T')"
          if [ -f "_audit/${name}.json.done" ]; then
            echo "✅ 评测完成 —— 与**合法基线**（T1 的 merged 审计）配对比三计数"
            .venv/bin/python -m syncopate.train.compare _audit/v13_sft_e1_merged.json "_audit/${name}.json" 2>&1 | tail -25
          else
            echo "🔴 没有 .done"; tail -20 "$Q/T${n}_eval.log"
          fi
        } > "$Q/T${n}.done" 2>&1
      else
        echo "🔴 adapter 转换失败" > "$Q/T${n}.done"; tail -10 "$Q/T${n}_adapter.log" >> "$Q/T${n}.done"
      fi
    fi
    { echo; echo "## T$n · B5 · $name"; sed 's/^/    /' "$Q/T${n}.done"; } >> "$BOARD"
    say "── T$n 判据已落盘"
  }
done

# ───────────────────────── §5.2 ─────────────────────────
# T6-T8 · E20 原因② · 更新次数（token 级 IS 是共同底座）
#   ⚠️ 必须在 R1 之后：批内走得更远会让序列级 IS 崩得更快
rl_run 6 e20e_mini2       0 --rollout-is token --lr 3e-5 --ppo-mini-batch-size 2 --steps 60 --sync-every 4 --weight-sync-bucket-mb 512
rl_run 7 e20f_lr1e4       0 --rollout-is token --lr 1e-4 --ppo-mini-batch-size 6 --steps 60 --sync-every 4 --weight-sync-bucket-mb 512
rl_run 8 e20g_mini2_lr1e4 0 --rollout-is token --lr 1e-4 --ppo-mini-batch-size 2 --steps 60 --sync-every 4 --weight-sync-bucket-mb 512

# T9-T10 · R2 · B19 陈旧度代价重测（★ 现在"陈旧度"这个旋钮才第一次真的接上）
rl_run 9  b19r_sync8  0 --rollout-is token --lr 3e-5 --ppo-mini-batch-size 6 --steps 64 --sync-every 8  --weight-sync-bucket-mb 512
rl_run 10 b19r_sync16 0 --rollout-is token --lr 3e-5 --ppo-mini-batch-size 6 --steps 64 --sync-every 16 --weight-sync-bucket-mb 512

# T11-T12 · R2 · B10 陈旧度阈值重测
rl_run 11 b10r_stale03 0 --rollout-is token --lr 3e-5 --ppo-mini-batch-size 6 --steps 60 --sync-every 4 --weight-sync-bucket-mb 512 --staleness-threshold 0.3
rl_run 12 b10r_stale05 0 --rollout-is token --lr 3e-5 --ppo-mini-batch-size 6 --steps 60 --sync-every 4 --weight-sync-bucket-mb 512 --staleness-threshold 0.5

say "════════════════ 队列全部结束 ════════════════"
echo "queue9h done $(date '+%F %T')" >> logs/BATCH_DONE
touch "$Q/QUEUE_ALL_DONE"
