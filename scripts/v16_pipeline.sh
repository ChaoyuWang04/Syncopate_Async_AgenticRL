#!/usr/bin/env bash
# ★ 固定管线 runbook（2026-09-04 Chaoyu：每一段都必须是固定脚本、默认值直接跑就健康；不许依赖谁临场敲参数）
#
#   bash scripts/v16_pipeline.sh [--dry-run] [--profile smoke|candidate] <stage|all>
#
# 每个 stage 只做一件事，输入/输出路径全部从仓库常量派生（DATA_VERSION · model_paths · rollout_budget），
# 这里**不写任何数字**；要改数字去改那份常量并重新注册（守则⑨⑬）。stage 之间用文件交接，任何一段都可单独重跑（幂等）。
# 云上（modal_app/stack_probe.py）与本机都只许调这一份；探针里不再各写各的命令。
#
# 数据 → SFT+eval → RL+eval → OPD+eval：
#   cases       题库生成（configs/buckets/<DV>.yaml → data/batches/<DV>，gold 实跑验证）
#   menus       case.tool_menu（verifier/routing 用；v15 契约 prompt 一律全量菜单；09-05 起不再并入任何旧模型的评测审计）
#   split       三桶（EVAL 冻结 / SFT / RL），互斥实测
#   gates       D1–D11 多样性门禁 + 三桶隔离复核（题库层）
#   rl-data     data/rl/<DV>/{train,val}.parquet（出口隔离闸）
#   teacher     起 27B 教师端点（:8210，两角色同端点）—— sft-data 的前置，长驻进程
#   sft-data    data/sft/<DV>/{train,val}.parquet（v16_build_sft：教师物料 → 六桶 → 全部闸 → 出厂体检 → 隔离复核 → 画廊）
#   sft-train   checkpoints/sft/<DV>/{epoch*, sel_f*}（LoRA attn_shared）
#   sft-eval    每个候选：entropy + eval_local（冻结 EVAL，8 样本）→ _audit/<DV>_{entropy,eval}_<候选>.json
#   sft-select  按决策位熵+有梯度格子选点，--prune 删临时点 → checkpoints/sft/<DV>/SELECTED
#   merge       models/<学生>-sft-<DV>（RL 起点；launch_rl_v1 断言不含 lora_adapter/）
#   exam        考场 v4 N 遍 + 判卷 + 三查（scripts/v16_exam_chain.sh，PG/Redis 需已起）
#   rl-train    launch_rl_v1 --profile <profile>（candidate：合并模型起点，≥400 步）
#   rl-adapter  最新 global_step 的 actor → PEFT adapter（verl 0.9 model_merger；FSDP2 分片）
#   rl-eval     eval_local --model 合并模型 --adapter RL adapter
#   opd-train   torchrun opd（学生=RL/SFT adapter；教师 27B；vocab 断言）
#   opd-eval    eval_local --adapter OPD adapter
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export SYNCOPATE_CONTRACT="${SYNCOPATE_CONTRACT:-v15}" SYNCOPATE_THINK="${SYNCOPATE_THINK:-1}"
PY="${PY:-$( [ -x /env/.venv/bin/python ] && echo /env/.venv/bin/python || echo .venv/bin/python )}"
DRY=0; PROFILE="${PROFILE:-candidate}"
while [ $# -gt 0 ]; do case "$1" in
  --dry-run) DRY=1; shift;; --profile) PROFILE="$2"; shift 2;; *) break;; esac; done
STAGE="${1:?用法: v16_pipeline.sh [--dry-run] [--profile smoke|candidate] <stage|all>}"

# ── 常量只从代码取（不在这里写第二份）──
eval "$($PY - 2>/dev/null <<'PYEOF' | grep -E '^(DV|STUDENT)='
import sys; sys.path.insert(0, ".")
from syncopate.pipeline.split import DATA_VERSION, DEFAULT_BATCH_DIR, DEFAULT_SPLIT_DIR, DEFAULT_SFT_DIR, DEFAULT_RL_DIR
from syncopate.core.model_paths import STUDENT_MODEL, TEACHER_MODEL
from syncopate.train.rollout_budget import MAX_PROMPT_LENGTH, MAX_RESPONSE_LENGTH
from pathlib import Path
print(f'DV="{DATA_VERSION}"; BATCH="{DEFAULT_BATCH_DIR}"; SPLIT="{DEFAULT_SPLIT_DIR}"; SFT_DIR="{DEFAULT_SFT_DIR}"; RL_DIR="{DEFAULT_RL_DIR}"')
print(f'STUDENT="{STUDENT_MODEL}"; TEACHER="{TEACHER_MODEL}"; STUDENT_NAME="{Path(STUDENT_MODEL).name}"; MAX_MODEL_LEN={MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH}')
PYEOF
)"
SFT_OUT="checkpoints/sft/$DV"; MERGED="models/${STUDENT_NAME}-sft-$DV"
RL_OUT="checkpoints/grpo/${DV}_${PROFILE/candidate/cand}"; RL_ADAPTER="models/adapters/rl_${DV}_${PROFILE}"
OPD_OUT="checkpoints/opd/${DV}_${PROFILE}"; AUD="_audit"
TEACHER_URL="${SYNCOPATE_TEACHER_LANG_URL:-http://127.0.0.1:8210/v1}"
say(){ echo "[v16-pipeline $(date +%H:%M:%S)] $*"; }
run(){ echo "  \$ $*"; [ "$DRY" = 1 ] || eval "$@"; }
need(){ for f in "$@"; do [ -e "$f" ] || { echo "🔴 缺前置：$f"; [ "$DRY" = 1 ] || exit 1; }; done; }

stage_cases(){ say "[stage cases] 题库 $DV"; run "$PY -m syncopate cases generate --spec configs/buckets/$DV.yaml --out $BATCH"; }
stage_menus(){ say "[stage menus]"; need "$BATCH/manifest.json"; run "$PY scripts/set_tool_menus.py --batch $BATCH"; }
stage_split(){ say "[stage split]"; need "$BATCH/manifest.json"; run "$PY -m syncopate data split --batch $BATCH --out $SPLIT"; }
stage_gates(){ say "[stage gates] D1–D11 + 三桶互斥"; need "$SPLIT/sft_cases.json"; run "$PY scripts/check_data_gates.py --batch $BATCH --split-dir $SPLIT"; }
stage_supply(){ say "[stage supply] 本机供给核对：SFT 桶每类底题供给 vs 建库数量闸（不调教师；run26 那种 144<280 在这里就红）"; need "$SPLIT/sft_cases.json"; run "$PY scripts/check_supply_vs_floors.py"; }
stage_rl_data(){ say "[stage rl-data]"; need "$SPLIT/rl_cases.json"; run "$PY -m syncopate data build --pool rl --batch $BATCH --split-dir $SPLIT --out $RL_DIR --val-every 5"; run "$PY scripts/check_split_isolation.py $RL_DIR/train.parquet $RL_DIR/val.parquet --pool rl"; }
stage_teacher(){ say "[stage teacher] 27B @8210（长驻；已起则跳过）"; if curl -sf "${TEACHER_URL%/v1}/health" >/dev/null 2>&1; then say "  教师已在线"; else
  run "nohup $PY -m vllm.entrypoints.openai.api_server --model $TEACHER --served-model-name t --max-model-len $MAX_MODEL_LEN --gpu-memory-utilization 0.90 --port 8210 --limit-mm-per-prompt '{\"image\": 0, \"video\": 0}' --max-num-seqs 64 > logs/teacher_$DV.log 2>&1 &"
  [ "$DRY" = 1 ] || until curl -sf http://127.0.0.1:8210/health >/dev/null; do sleep 5; done; fi; }
stage_sft_data(){ say "[stage sft-data] 建库（教师物料→六桶→全部闸→出厂体检→隔离→画廊）"; need "$SPLIT/sft_cases.json"
  run "SYNCOPATE_TEACHER_LANG_URL=$TEACHER_URL SYNCOPATE_TEACHER_THINK_URL=$TEACHER_URL $PY scripts/v16_build_sft.py 2>&1 | tee $AUD/$DV/build.log"
  run "$PY scripts/v16_prompt_budget_gate.py --prompt-budget $SFT_DIR/train.parquet"
  run "$PY scripts/check_split_isolation.py $SFT_DIR/train.parquet $SFT_DIR/val.parquet --pool sft"
  run "$PY scripts/v16_data_gallery.py --parquet $SFT_DIR/train.parquet > $AUD/$DV/gallery.md"; }
# 本机离线全量建库：教师材料全部来自云盘缓存（modal volume get），缺一条就红——改闸之后先在本机把整条建库跑到底，不上云试错
stage_sft_data_offline(){ say "[stage sft-data-offline] 拉云盘缓存 → 离线建库（全部闸 + 出厂体检）"; need "$SPLIT/sft_cases.json"
  run "mkdir -p data/u_route && for f in v16_ballast_replies v16_defs v16_chat_mat v16_l2l1_rows v16_fam_rows v16_cot_rows; do modal volume get syncopate-home _audit/$DV/cache/\$f.json data/u_route/\$f.json --force >/dev/null || echo \"⚠️ 缓存缺 \$f\"; done"
  run "SYNCOPATE_TEACHER_OFFLINE=1 $PY scripts/v16_build_sft.py 2>&1 | tee $AUD/$DV/build_offline.log"
  run "$PY scripts/v16_prompt_budget_gate.py --prompt-budget $SFT_DIR/train.parquet"
  run "$PY scripts/check_split_isolation.py $SFT_DIR/train.parquet $SFT_DIR/val.parquet --pool sft"; }
stage_sft_train(){ say "[stage sft-train] $PROFILE"; need "$SFT_DIR/train.parquet"
  if [ "$PROFILE" = smoke ]; then run "$PY -m syncopate.train.sft --out ${SFT_OUT}_smoke --epochs 1 --batch-size 1 --grad-accum 8 --max-steps 30 --no-wandb --wandb-run sft_${DV}_smoke"
  else run "$PY -m syncopate.train.sft --out $SFT_OUT --wandb-run sft_$DV"; fi; }
stage_sft_eval(){ say "[stage sft-eval] 每个候选：entropy + eval_local（冻结 EVAL，8 样本）"; need "$SFT_OUT"
  for c in "$SFT_OUT"/epoch* "$SFT_OUT"/sel_f*; do [ -d "$c" ] || continue; n=$(basename "$c")
    run "$PY -m syncopate.train.entropy --adapter $c --limit 24 --out $AUD/${DV}_entropy_${n}.json"
    run "$PY -m syncopate.train.eval_local --adapter $c --samples-per-case 8 --out $AUD/${DV}_eval_${n}.json"; done; }
stage_sft_select(){ say "[stage sft-select] 决策位熵+有梯度格子 → SELECTED，删临时点"; need "$SFT_OUT"
  run "$PY scripts/select_sft_ckpt.py $SFT_OUT --auto --prune"; }
stage_merge(){ say "[stage merge] → $MERGED"; need "$SFT_OUT/SELECTED"; run "$PY -m syncopate.train.merge_adapter --adapter $SFT_OUT/SELECTED --out $MERGED"; }
stage_exam(){ say "[stage exam] 考场 v4 → 判卷 → 三查（PG/Redis 需已起：scripts/pg_bootstrap.sh · redis_bootstrap.sh；smoke=1 遍 40 题）"; need "$MERGED"
  if [ "$PROFILE" = smoke ]; then run "EXAM_PASSES=1 EXAM_LIMIT=40 bash scripts/v16_exam_chain.sh $MERGED ${DV}_smoke context_v4"
  else run "EXAM_PASSES=4 bash scripts/v16_exam_chain.sh $MERGED ${DV}_sft context_v4"; fi; }
stage_rl_train(){ say "[stage rl-train] $PROFILE"; [ "$PROFILE" = smoke ] || need "$MERGED" "$RL_DIR/train.parquet"; run "$PY -m syncopate.train.launch_rl_v1 --profile $PROFILE --save-path $RL_OUT"; }
stage_rl_adapter(){ say "[stage rl-adapter] 最新 global_step → PEFT adapter（verl 0.9 model_merger，FSDP2 分片）"; need "$RL_OUT"
  last=$(ls -d "$RL_OUT"/global_step_* 2>/dev/null | sort -t_ -k3 -n | tail -1 || true); [ -n "$last" ] || { echo "🔴 $RL_OUT 下没有 global_step_*"; [ "$DRY" = 1 ] || exit 1; last="$RL_OUT/global_step_N"; }
  run "$PY -m verl.model_merger merge --backend fsdp --local_dir $last/actor --target_dir $RL_ADAPTER"
  run "$PY scripts/check_lora_adapter.py $RL_ADAPTER/lora_adapter"; }
stage_rl_eval(){ say "[stage rl-eval]"; need "$MERGED"; run "$PY -m syncopate.train.eval_local --model $MERGED --adapter $RL_ADAPTER/lora_adapter --samples-per-case 8 --out $AUD/${DV}_eval_rl_${PROFILE}.json"; }
# OPD 的学生起点：优先 RL adapter（训在合并 SFT 模型之上 ⇒ 底座=MERGED），否则 SFT 选中点（底座=学生底座）；评测同底
opd_base(){ if [ -d "$RL_ADAPTER/lora_adapter" ]; then echo "$MERGED"; else echo "$STUDENT"; fi; }
opd_adapter(){ if [ -d "$RL_ADAPTER/lora_adapter" ]; then echo "$RL_ADAPTER/lora_adapter"; else echo "$SFT_OUT/SELECTED"; fi; }
stage_opd_train(){ say "[stage opd-train] 学生@GPU0 · 教师+锚@GPU1"; ad="$(opd_adapter)"; base="$(opd_base)"
  if [ "$PROFILE" = smoke ]; then extra="--max-steps 6 --batch 4 --max-new 160 --save-every 5 --probe-every 3"; adopt=""; base="$STUDENT"; else extra=""; adopt="--adapter $ad"; need "$ad"; fi
  run "CUDA_VISIBLE_DEVICES=0,1 OPD_AUX_GPUS=1 $PY -m torch.distributed.run --nproc_per_node=1 --master_port 29517 -m syncopate.train.opd --base $base --out $OPD_OUT $adopt $extra"; }
stage_opd_eval(){ say "[stage opd-eval]"; need "$OPD_OUT/final"; base="$(opd_base)"; [ "$PROFILE" = smoke ] && base="$STUDENT"
  run "$PY -m syncopate.train.eval_local --model $base --adapter $OPD_OUT/final --samples-per-case 8 --out $AUD/${DV}_eval_opd_${PROFILE}.json"; }

ALL=(cases menus split gates supply rl-data teacher sft-data sft-train sft-eval sft-select merge exam rl-train rl-adapter rl-eval opd-train opd-eval)
run_stage(){ local s="$1"; local fn="stage_${s//-/_}"; declare -F "$fn" >/dev/null || { echo "🔴 未知 stage：$s（可选：${ALL[*]} all）"; exit 2; }; "$fn"; }
if [ "$STAGE" = all ]; then for s in "${ALL[@]}"; do run_stage "$s"; done; else run_stage "$STAGE"; fi
say "done ($STAGE, profile=$PROFILE, dry=$DRY)"
