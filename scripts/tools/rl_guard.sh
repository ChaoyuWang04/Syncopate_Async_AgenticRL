#!/usr/bin/env bash
# RL 长跑的运行时守卫 —— 盯停止条件，触发就停。
#
#   bash scripts/tools/rl_guard.sh <日志文件> <ckpt目录> [--kill]
#
# ★★★ 为什么这个脚本必须在仓库里（2026-08-18）
#
# 此前它是**跑的时候临时写的一段 shell**，产出 `logs/rl_guard.log` 之后就没了，
# 而且只盯 ESS / 熵 / 磁盘 —— 全是「模型学得怎么样」那一族。
# ⇒ 两个基石级 bug（**梯度没跨 rank 同步** · **权重从没推给 rollout engine**）
#   **静默跑完了整整两轮训练，一条线都没响**。
#
# ⇒ 判据分两族，见 `docs/syncopate/04-TRAINING.md`：
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
# 2.C 熵：⛔ **只报不停**（2026-08-19 同一天两次误杀后除名，证据链完整）：
#   ① 绝对门槛 0.05 误杀第一跑（键 actor/entropy 是 token 平均熵，v13r2 起点天然 0.031–0.05）
#   ② 锚定 30% 误杀第二跑 —— 健康跑 e17a/e17b 实测下探到起点的 **31–42%** 再回升
#   ③ ★ 真塌陷跑 e20f（defer→0%）的熵**全程 ≥0.0265，从没报过警**
#   ⇒ token 平均熵对我们的失效模式**既会误报也不会真报** —— 不是阈值问题，是警报器
#     对不上火灾类型。塌陷的硬停由 D 族（defer 连零，正是抓到 e20f 的那条）负责。
ENTROPY_COLLAPSE_RATIO=0.30   # 仅用于 ⚠️ 提示行，不停机
ENTROPY_HARD_FLOOR=0.005
MIN_DISK_GB=40               # 磁盘（nsys 那次的教训）
# D 族：连续多少步没有 defer 就算塌陷。⚠️ 同样是反填的工程值，见 defer_watch.py 的表
#   [实测] 3584 时代：健康跑最长连零 9 / 16 / 18 步；崩塌那跑 56 步。取 25。
#   [复核 2026-08-19] 用 5120 干净跑重反填（infra 回信第 5 条要求的）：
#     e17a 11 · e17b 4 · e17d×2 0（各 60 步）⇒ 健康侧上界 11，25 仍有 ≥2.3× 余量
#     ⇒ **维持 25**。⚠️ 上界样本都是 60 步跑；137 步整 epoch 的自然连零可能更长，别急着收紧。
MAX_DEFER_ZERO_STREAK="${MAX_DEFER_ZERO_STREAK:-25}"
BATCH_SEQS="${BATCH_SEQS:-48}"   # 一次更新的序列数 = mini_batch × rollout_n

say () { echo "$(date '+%H:%M:%S') $*" | tee -a "$OUT"; }
stop () {
  say "🔴 触发停机：$*"
  if [ "$KILL" = "--kill" ] && [ -f "$PIDFILE" ]; then
    kill "$(cat "$PIDFILE")" 2>/dev/null && say "   已发 SIGTERM 给 $(cat "$PIDFILE")"
    # ⚠️ 只杀 launcher 会留下孤儿 Ray 集群继续吃满四卡（08-29 实测：守卫杀后引擎
    #    仍 2284 tok/s 空转，下一跑撞残留集群当场死）——停机必须连集群一起收
    sleep 10; ray stop --force >/dev/null 2>&1
    sleep 5; nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
      | while read -r p; do kill -9 "$p" 2>/dev/null; done
    say "   Ray 集群与 GPU 进程已随停机清收"
  else
    say "   （未加 --kill，只报警不停机）"
  fi
  exit 1
}
num () { echo "${1:-0}" | awk '{print ($1=="" ? 0 : $1)}'; }

say "守卫启动 · 日志=$LOG · ckpt=$CKPT · 阈值: 有效条数≥$MIN_EFFECTIVE_SEQS ·"\
    "截断比例≤$MAX_CLIPPED_FRACTION · 熵只报不停(2.C) · 磁盘≥${MIN_DISK_GB}G ·"\
    "prompt截断=0（P族） · defer连零<$MAX_DEFER_ZERO_STREAK（D族）"

FLOOR=""          # A1：首个 kl，就是「数值失配地板」
ENT0=""           # 2.C：首个 actor/entropy，就是本跑的探索起点
# ★ 先查后睡：启动就做一次完整检查 —— 配置错了要立刻知道，不要等一分钟
#   （「短冒烟证明不了长跑」的反面：长跑的守卫也得先证明自己能跑通一次）
INTERVAL="${GUARD_INTERVAL:-60}"
while true; do
  if [ ! -f "$LOG" ]; then say "等日志出现…"; sleep "$INTERVAL"; continue; fi

  last () { grep -o "$1:[0-9.eE+-]*" "$LOG" | tail -1 | cut -d: -f2; }
  ESS=$(last "rollout_corr/rollout_is_eff_sample_size")
  KL=$(last "rollout_corr/kl")
  CLIPR=$(last "prompt_length/clip_ratio")
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

  # ── P 族 · 输入完整性：prompt 一条都不许被截（2026-08-19 第四条常驻判据）──
  # 判据：`prompt_length/clip_ratio` 必须恒为 0。
  # ⚠️ 为什么是停机不是报警：3584 那次 100% 截断把「调查先于结论（含拒绝）」
  #   整段砍掉，模型训练时看不见"可以拒绝"这个选项 ⇒ defer 97%→0%，
  #   还制造了一整条「reward 在教不拒绝」的错误归因链（E20 §7.12 翻案）。
  #   截断污染的是**输入分布**，跑得越久废得越多 —— 早停就是省钱。
  # ⚠️ 此前这条只活在 run_e17_kl_arms.sh 的收尾 echo 里（跑完才看，只覆盖那一个实验）
  #   —— 正是「机制在但没接上」。现在挪进守卫，跑中每分钟看一次。
  if [ -n "$CLIPR" ]; then
    CLIPBAD=$(awk -v c="$CLIPR" 'BEGIN{print (c>0)?1:0}')
    [ "$CLIPBAD" = "1" ] && stop "prompt_length/clip_ratio=$CLIPR > 0 —— prompt 被截断，输入分布已污染（预算见 rollout_budget.py）"
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
    # ⚠️ 2026-08-19：默认已改回 sequence ⇒ 这句从"默认动作"变成**逃生口**。
    #   序列级在干净基线 120 步上无衰减（ESS 0.78–0.94），真跌破 0.3 才换 token。
    [ "$SW" = "1" ] && say "🟡 ESS/N=$ESS < 0.3 ⇒ 逃生口：换 --rollout-is token（**不停机**）"
  fi

  # ── D 族 · 行为塌陷（2026-08-19 新增）────────────────────────────────
  # 判据：**连续 N 步一条 defer 都没有** ⇒ 拒绝能力可能已经塌了。
  #
  # ⚠️ 为什么不看 defer **率**：`r1_seqis` 完全健康（评测该 defer 97%），
  #   总体率却从 5.1% 降到 1.6% —— 降的是**误 defer**，那是好事。
  #   **水平分不开好坏，"持续为零"才分得开。**
  # ⚠️ 为什么这条要停机而不是只报警：它**不可逆** —— 塌了之后那些题
  #   组内 std=0 ⇒ advantage 恒为 0 ⇒ 再也不产生梯度，RL 自己爬不出来。
  #   （其余 C 族的指标坏了，继续跑还有救；这条没有。）
  if [ -d "$CKPT/rollout_dumps" ]; then
    STREAK=$(.venv/bin/python scripts/tools/defer_watch.py "$CKPT" --streak 2>/dev/null)
    if [ -n "$STREAK" ] && [ "$STREAK" -ge "$MAX_DEFER_ZERO_STREAK" ]; then
      stop "连续 $STREAK 步没有任何 defer（门槛 $MAX_DEFER_ZERO_STREAK）—— 拒绝能力塌陷，**不可逆**"
    fi
  fi

  # ── 2.C · 熵：**只报不停**（除名理由见文件头；读数仍进心跳行与面板）──────
  if [ -n "$ENT" ]; then
    if [ -z "$ENT0" ]; then
      ENT0=$(grep -o "actor/entropy:[0-9.eE+-]*" "$LOG" | head -1 | cut -d: -f2)
      say "2.C 熵起点 ENT0=$ENT0（token 平均熵，只观察不停机）"
    fi
    LOWENT=$(awk -v e="$ENT" -v a="$ENT0" -v r="$ENTROPY_COLLAPSE_RATIO" -v h="$ENTROPY_HARD_FLOOR" 'BEGIN{print (e < a*r || e < h)?1:0}')
    [ "$LOWENT" = "1" ] && say "⚠️ 熵 $ENT 低于起点 $ENT0 的 ${ENTROPY_COLLAPSE_RATIO} 倍 —— 观察项（健康跑也会下探到 31–42% 再回升；真塌陷看 D 族）"
  fi
  case "$GN" in *nan*|*inf*) stop "grad_norm=$GN —— 非有限（flash-attn 反向坏掉就是这个症状）";; esac

  # ── 磁盘（M7 丢掉最终 ckpt 的那个形状）──────────────────────────────
  if [ -n "$DISK" ] && [ "$DISK" -lt "$MIN_DISK_GB" ]; then
    stop "剩余空间 ${DISK}G < ${MIN_DISK_GB}G —— 收尾那次 27 GB 写盘会被截断"
  fi

  NCKPT=$(ls -d "$CKPT"/global_step_* 2>/dev/null | wc -l)
  say "kl=${KL:-–} ESS=${ESS:-–} 有效=${EFF:-–} 截断=${CLIP:-–} prompt截断=${CLIPR:-–} 熵=${ENT:-–} ckpt=$NCKPT 磁盘=${DISK}G"
  sleep "$INTERVAL"
done
