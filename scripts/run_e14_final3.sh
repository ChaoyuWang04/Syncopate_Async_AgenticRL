#!/usr/bin/env bash
# E14/R2 · 收官三件之①②（2026-08-21）：s16/0.1 多种子复核 ×2 + CUDA graph 正式精度闸
# 每臂：门禁 → 训练 → 提 adapter → 4 卡评测 → compare；进度写 logs/e14_sweep_progress.log
# 臂设计（唯一变量原则）：
#   seed2/seed3   与 08-20 夜 sweep 的 s16_t01 臂逐字段同（eager·修理默认开·64步），只换 --seed
#   graphgate     与 ctrl64 逐字段同（sync4/0.1·seed1234·修理②③显式关——ctrl64 是修理落地前跑的），
#                 唯一变量 --enforce-eager False ⇒ 配对差 = graph 本身
cd "$(dirname "$0")/.."
set -a; . /workspace/.env; set +a
P=logs/e14_sweep_progress.log
say() { echo "[FINAL3 $(date +%H:%M:%S)] $*" | tee -a "$P"; }

run_arm() {
  local name=$1 seed=$2 sync=$3 stal=$4 eager=$5 fixes=$6
  say "ARM ${name} 开始（seed=${seed} sync=${sync}/${stal} eager=${eager} fixes=${fixes}）"
  until GPUS=0,1,2,3 bash scripts/gpu_gate.sh >/dev/null 2>&1; do sleep 60; done
  local envs=()
  [ "$fixes" = off ] && envs=(SYNCOPATE_FIX_JAGGED=0 SYNCOPATE_FIX_PG_RI=0)
  env "${envs[@]}" timeout 2700 .venv/bin/python -m syncopate.train.launch_rl \
    --model models/Qwen3-4B-sft-v13r2-e1 --lora-rank 32 \
    --mode fully_async --trainer-gpus 3 --rollout-gpus 1 \
    --steps 64 --save-freq 999 --seed "$seed" \
    --sync-every "$sync" --staleness-threshold "$stal" --enforce-eager "$eager" \
    --experiment "e14x_${name}" --save-path "checkpoints/grpo/e14x_${name}" \
    --wandb-mode offline --logger console > "logs/e14x_${name}.log" 2>&1
  local ck
  ck=$(ls -d checkpoints/grpo/e14x_${name}/global_step_* 2>/dev/null | sort -V | tail -1)
  if [ -z "$ck" ]; then say "🔴 ARM ${name} 训练无 ckpt，跳过"; return 1; fi
  # 跑中最小判据四条（00-START §6）——在这里就地留痕，晋级前人工复核
  { echo "===== ${name} 跑中判据 ====="
    grep -c "lora-probe" "logs/e14x_${name}.log" | sed 's/^/  lora-probe 行数: /'
    grep -oE "clip_ratio[^,]*" "logs/e14x_${name}.log" | sort | uniq -c | head -3
  } | tee -a "$P"
  say "ARM ${name} 训练完成（$(basename $ck)），timing："
  .venv/bin/python scripts/parse_fully_async_timing.py "logs/e14x_${name}.log" 2>/dev/null | head -5 | tee -a "$P"
  .venv/bin/python scripts/rl_ckpt_to_adapter.py "$ck/actor" --out "models/adapters/e14x_${name}" \
    >> "logs/e14x_${name}.log" 2>&1 || { say "🔴 ARM ${name} adapter 提取失败"; return 1; }
  say "ARM ${name} 评测开始"
  MODEL=models/Qwen3-4B-sft-v13r2-e1 timeout 1800 bash scripts/eval_parallel.sh \
    "models/adapters/e14x_${name}" "_audit/e14x_${name}.json" 4 \
    > "logs/e14x_${name}_eval.log" 2>&1
  if [ ! -f "_audit/e14x_${name}.json.done" ]; then say "🔴 ARM ${name} 评测无 done 标记"; return 1; fi
  say "ARM ${name} 评测完成；对比："
  { echo "===== ${name} vs ctrl64（臂对臂）====="
    .venv/bin/python -m syncopate.train.compare _audit/e14c_ctrl64.json "_audit/e14x_${name}.json" 2>&1 \
      | grep -E "配对差值|结论|该 defer|均值"
    echo "===== ${name} vs SFT ====="
    .venv/bin/python -m syncopate.train.compare _audit/v13_sft_v13r2_e1_merged.json "_audit/e14x_${name}.json" 2>&1 \
      | grep -E "配对差值|结论"
  } | tee -a "$P"
  say "ARM ${name} ✅ 全链完成"
}

say "收官三件之①②开始：s16_t01 seed2/seed3 + graphgate（sync4 精度闸）"
run_arm s16t01_seed2 2    16 0.1 True  on
run_arm s16t01_seed3 3    16 0.1 True  on
run_arm graphgate    1234 4  0.1 False off
say "🏁 FINAL3 全部结束"
touch logs/e14_final3_DONE   # 小写 e14 前缀：gpu_gate ③ 的排除表按它认自己人
