#!/usr/bin/env bash
# E27 · thinking 三臂评测探针（eval-only，无训练；判据与预测见 E27-thinking-probe.md）
#
#   bash scripts/run_e27_think_probe.sh <新SFT的adapter路径>
#
# 三臂（同一份 eval 脚本、同冻结 EVAL、同采样契约；一臂只变一个东西）：
#   A  think-off · 裸基座      ⇒ _audit/e27_base_off.json   ★ 修复后管线的**永久基线**
#   B  think-on  · 裸基座      ⇒ _audit/e27_base_on.json    探索臂（SYNCOPATE_THINK=1）
#   C  think-off · 新 SFT      ⇒ _audit/e27_sft_new.json    最新系统+版本的 SFT
#
# ⚠️ B 臂开跑前必须看到一行 [think-mode]（对照计数）；跑完必查 truncation=tokens 比例
#   —— 思考被截断连答案都没有（E20 §7.12 同族），比例高就加预算重跑，别硬比。
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
set -a; . /workspace/.env; set +a

SFT_ADAPTER="${1:?用法: run_e27_think_probe.sh <新SFT的adapter路径>}"
[ -d "$SFT_ADAPTER" ] || { echo "🔴 adapter 目录不存在: $SFT_ADAPTER"; exit 1; }
BASE=models/Qwen3-4B
Q=logs/queue_e27; mkdir -p "$Q"
say() { echo "[$(date '+%T')] $*"; }

bash scripts/gpu_gate.sh || { echo "⛔ 门禁没过，不起"; exit 1; }

wait_gpu() { while :; do b=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
  [ "${b:-99999}" -lt 2000 ] && break; sleep 15; done; }

run_arm() {  # $1=臂名 $2=adapter $3=THINK(0/1) $4=EVAL_EXTRA
  local name=$1 adapter=$2 think=$3 extra=${4:-}
  say "════════ 臂 $name（adapter='${adapter}' think=$think extra='${extra}'）"
  wait_gpu
  SYNCOPATE_THINK=$think EVAL_EXTRA="$extra" MODEL=$BASE timeout 7200 \
    bash scripts/eval_parallel.sh "$adapter" "_audit/e27_${name}.json" > "$Q/${name}.log" 2>&1 \
    || say "──────── 臂 $name 退出码 $?（继续下一臂，判据文件里会显形）"
  say "──────── 臂 $name 完成"
}

# ★ 裸基座两臂单轮上限都给 2048（A 臂不给的话 256 会砍长输出，截断与真实弱分不开
#   —— v13_base 就是这么作废的）；SFT 臂用生产默认 256（实测 0 token 截断）。
#   ⇒ A vs B 单变量 = thinking；A vs C 的差异里含单轮上限，但 C 实测碰不到上限。
run_arm base_off ""             0 "--max-new-tokens 2048"
run_arm base_on  ""             1
run_arm sft_new  "$SFT_ADAPTER" 0

{ echo "# E27 · thinking 三臂（$(date '+%F %T')）"
  echo; echo "══ 判据 0：B 臂 [think-mode] 行数（必须 >0，off 两臂必须 =0）："
  for a in base_off base_on sft_new; do
    echo "  $a: $(grep -c '\[think-mode\]' "$Q/$a.log" || true)"
  done
  echo; echo "══ 判据 1：三臂截断（truncation=tokens；B 臂过高 ⇒ 预算不够，结论无效）："
  for a in base_off base_on sft_new; do
    echo "  $a: $(grep -oE 'truncation[^,}]*tokens' "$Q/$a.log" | wc -l) 处日志提及"
  done
  for a in base_off base_on sft_new; do
    [ -f "_audit/e27_${a}.json.done" ] || echo "🔴 $a 没有 .done —— 该臂作废"
  done
  echo; echo "══ A vs B（thinking 的净效果，裸基座同底）："
  .venv/bin/python -m syncopate.train.compare _audit/e27_base_off.json _audit/e27_base_on.json 2>&1 | head -24
  echo; echo "══ A vs C（新 SFT 的净效果，think-off 同底）："
  .venv/bin/python -m syncopate.train.compare _audit/e27_base_off.json _audit/e27_sft_new.json 2>&1 | head -24
} > "$Q/E27.done" 2>&1
say "── 判据已落盘 $Q/E27.done"
say "════════ ALL DONE"
