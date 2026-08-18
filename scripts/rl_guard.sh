#!/usr/bin/env bash
# RL 长跑的运行时守卫 —— 盯停止条件，触发就停。
#
#   bash scripts/rl_guard.sh <日志文件> <ckpt目录> [--kill]
#
# ★★★ 为什么这个脚本必须在仓库里（2026-08-18）
#
# 此前它是**跑的时候临时写的一段 shell**，产出 `logs/rl_guard.log` 之后就没了，
# 而且只盯 ESS / 熵 / 磁盘 —— 全是「模型学得怎么样」那一族。
# ⇒ 两个基石级 bug（**梯度没跨 rank 同步** · **权重从没推给 rollout engine**）
#   **静默跑完了整整两轮训练，一条线都没响**。
#
# ⇒ 判据分两族，见 `docs/syncopate/06-rl-run-protocol.md §2`：
#     A 族「系统还活着吗」 ← 今天之前一条都没有，两个 bug 就是从这个缺口漏过去的
#     C 族「模型学得怎么样」← 原有
#
# ⚠️ 两个记录在案的坑，这里都绕开了：
#   ① `pgrep -f <模式>` 会自匹配（执行的 shell 自己也含该模式）—— 犯过 6 次 ⇒ 用 PID 文件
#   ② 别用「输出文件存在」当判据 —— 那文件上一版就在 ⇒ 用显式的完成标记
set -uo pipefail
LOG="${1:?用法: rl_guard.sh <日志文件> <ckpt目录> [--kill]}"
CKPT="${2:?}"
KILL="${3:-}"
OUT="logs/rl_guard_$(date +%m%d_%H%M).log"
PIDFILE="${RL_PIDFILE:-logs/rl_train.pid}"

# ── 阈值 ⚠️ 24 与 0.40 是**按当前 batch（6×8=48）反填的工程值，不是推导出来的**。
#    第一次重跑之后用实测反填（06 §2.E）。
MIN_EFFECTIVE_SEQS=24        # A4：绝对有效条数，不是比例
MAX_CLIPPED_FRACTION=0.40    # 2.B：被截到上下界的比例
MIN_ENTROPY=0.05             # 2.C
MIN_DISK_GB=40               # 磁盘（nsys 那次的教训）
BATCH_SEQS="${BATCH_SEQS:-48}"   # 一次更新的序列数 = mini_batch × rollout_n

say () { echo "$(date '+%H:%M:%S') $*" | tee -a "$OUT"; }
stop () {
  say "🔴 触发停机：$*"
  if [ "$KILL" = "--kill" ] && [ -f "$PIDFILE" ]; then
    kill "$(cat "$PIDFILE")" 2>/dev/null && say "   已发 SIGTERM 给 $(cat "$PIDFILE")"
  else
    say "   （未加 --kill，只报警不停机）"
  fi
  exit 1
}
num () { echo "${1:-0}" | awk '{print ($1=="" ? 0 : $1)}'; }

say "守卫启动 · 日志=$LOG · ckpt=$CKPT · 阈值: 有效条数≥$MIN_EFFECTIVE_SEQS ·"\
    "截断比例≤$MAX_CLIPPED_FRACTION · 熵≥$MIN_ENTROPY · 磁盘≥${MIN_DISK_GB}G"

FLOOR=""          # A1：首个 kl，就是「数值失配地板」
# ★ 先查后睡：启动就做一次完整检查 —— 配置错了要立刻知道，不要等一分钟
#   （「短冒烟证明不了长跑」的反面：长跑的守卫也得先证明自己能跑通一次）
INTERVAL="${GUARD_INTERVAL:-60}"
while true; do
  if [ ! -f "$LOG" ]; then say "等日志出现…"; sleep "$INTERVAL"; continue; fi

  last () { grep -o "$1:[0-9.eE+-]*" "$LOG" | tail -1 | cut -d: -f2; }
  ESS=$(last "rollout_corr/rollout_is_eff_sample_size")
  KL=$(last "rollout_corr/kl")
  FLO=$(last "rollout_corr/rollout_is_ratio_fraction_low")
  FHI=$(last "rollout_corr/rollout_is_ratio_fraction_high")
  ENT=$(last "actor/entropy")
  GN=$(last "actor/grad_norm")
  DISK=$(df -BG --output=avail . 2>/dev/null | tail -1 | tr -dc '0-9')

  # ── A1 · 权重同步真的发生了吗 ────────────────────────────────────────
  # 判据：每次同步后 kl 必须回落到**首步那个数量级**。不回落 = 权重没推过去。
  # ⚠️ 这条判据是 2026-08-18 那个 bug 反推出来的：它跑了两轮训练，零报警。
  if [ -n "$KL" ]; then
    # ⚠️ 地板必须取**这一跑的第一个** kl，不是"守卫第一次看到的那个"。
    #    守卫起晚了或重启过，后者会锁到一个已经涨上去的值 ⇒ 判据永远不会响。
    #    这正是「判据看起来正常、实际指向了另一件事」那个形状。
    if [ -z "$FLOOR" ]; then
      FLOOR=$(grep -o "rollout_corr/kl:[0-9.eE+-]*" "$LOG" | head -1 | cut -d: -f2)
      say "A1 地板值锁定 kl=$FLOOR（**日志里的第一个** param_version）"
    fi
    RATIO=$(awk -v a="$KL" -v b="$FLOOR" 'BEGIN{print (b>0)? a/b : 0}')
    OVER=$(awk -v r="$RATIO" 'BEGIN{print (r>10)?1:0}')
    [ "$OVER" = "1" ] && say "⚠️ A1 kl=$KL 是地板 $FLOOR 的 ${RATIO}× 且从未回落 —— 疑似权重同步没生效"
  fi

  # ── A4 · 绝对有效条数（不是比例）────────────────────────────────────
  if [ -n "$ESS" ]; then
    EFF=$(awk -v e="$ESS" -v n="$BATCH_SEQS" 'BEGIN{printf "%.1f", e*n}')
    LOWEFF=$(awk -v e="$EFF" -v m="$MIN_EFFECTIVE_SEQS" 'BEGIN{print (e<m)?1:0}')
    [ "$LOWEFF" = "1" ] && stop "有效样本数 $EFF < $MIN_EFFECTIVE_SEQS 条（ESS/N=$ESS × N=$BATCH_SEQS）"
  fi

  # ── 2.B · 估计量退化（被截到界上的比例）──────────────────────────────
  if [ -n "$FLO" ] && [ -n "$FHI" ]; then
    CLIP=$(awk -v a="$FLO" -v b="$FHI" 'BEGIN{print a+b}')
    BAD=$(awk -v c="$CLIP" -v m="$MAX_CLIPPED_FRACTION" 'BEGIN{print (c>m)?1:0}')
    [ "$BAD" = "1" ] && stop "被截断比例 $CLIP > $MAX_CLIPPED_FRACTION —— IS 修正退化成常数缩放（H3）"
  fi
  # ⚠️ ESS<0.3 **不停机**：上游原话是"换聚合口径"，不是"停"（06 §2.B）
  if [ -n "$ESS" ]; then
    SW=$(awk -v e="$ESS" 'BEGIN{print (e<0.3)?1:0}')
    [ "$SW" = "1" ] && say "🟡 ESS/N=$ESS < 0.3 ⇒ 建议换 --rollout-is token（**不停机**）"
  fi

  # ── 2.C · 模型学得怎么样 ─────────────────────────────────────────────
  if [ -n "$ENT" ]; then
    LOWE=$(awk -v e="$ENT" -v m="$MIN_ENTROPY" 'BEGIN{print (e<m)?1:0}')
    [ "$LOWE" = "1" ] && stop "决策位熵 $ENT < $MIN_ENTROPY —— 探索死了"
  fi
  case "$GN" in *nan*|*inf*) stop "grad_norm=$GN —— 非有限（flash-attn 反向坏掉就是这个症状）";; esac

  # ── 磁盘（M7 丢掉最终 ckpt 的那个形状）──────────────────────────────
  if [ -n "$DISK" ] && [ "$DISK" -lt "$MIN_DISK_GB" ]; then
    stop "剩余空间 ${DISK}G < ${MIN_DISK_GB}G —— 收尾那次 27 GB 写盘会被截断"
  fi

  NCKPT=$(ls -d "$CKPT"/global_step_* 2>/dev/null | wc -l)
  say "kl=${KL:-–} ESS=${ESS:-–} 有效=${EFF:-–} 截断=${CLIP:-–} 熵=${ENT:-–} ckpt=$NCKPT 磁盘=${DISK}G"
  sleep "$INTERVAL"
done
