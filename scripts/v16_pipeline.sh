#!/usr/bin/env bash
# ★ 固定管线 runbook（2026-09-04 Chaoyu：每一段都必须是固定脚本、默认值直接跑就健康；不许依赖谁临场敲参数）
#
#   bash scripts/v16_pipeline.sh [--dry-run] [--resume] [--profile smoke|candidate]
#        [--gate-mode observe|strict] [--run-id ID] <stage|train-all|all>
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
#   supply      SFT 桶各类底题供给 ≥ 建库数量下限（不调教师）
#   rl-data     data/rl/<DV>/{train,val}.parquet（出口隔离闸）
#   teacher     起 27B 教师端点（:8210，两角色同端点）—— sft-data 的前置，长驻进程
#   sft-data    data/sft/<DV>/{train,val}.parquet（syncopate.pipeline.build_sft：教师物料 → 六桶 → 全部闸 → 出厂体检 → 隔离复核 → 画廊）
#   sft-train   checkpoints/sft/<DV>/{epoch*, sel_f*}（LoRA attn_shared）
#   sft-eval    每个候选：entropy + eval_local（冻结 EVAL，8 样本）→ _audit/<DV>_{entropy,eval}_<候选>.json
#   sft-select  按决策位熵+有梯度格子选点，--prune 删临时点 → checkpoints/sft/<DV>/SELECTED
#   merge       models/<学生>-sft-<DV>（RL 起点；launch_rl_v1 断言不含 lora_adapter/）
#   exam        考场 v4 N 遍 + 判卷 + 三查（scripts/v16/exam_chain.sh，PG/Redis 需已起）
#   rl-train    launch_rl_v1 --profile <profile>（candidate：合并模型起点，≥400 步）
#   rl-adapter  最新 global_step 的 LoRA-only FSDP2 actor → 经严格结构闸的 PEFT adapter
#   rl-eval     eval_local --model 合并模型 --adapter RL adapter
#   opd-train   torchrun opd（学生=RL/SFT adapter；教师 27B；vocab 断言）
#   opd-eval    eval_local --adapter OPD adapter
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export SYNCOPATE_CONTRACT="${SYNCOPATE_CONTRACT:-v15}" SYNCOPATE_THINK="${SYNCOPATE_THINK:-1}"
PY="${PY:-$( [ -x /env/.venv/bin/python ] && echo /env/.venv/bin/python || echo .venv/bin/python )}"
DRY=0; RESUME=0; PROFILE="${PROFILE:-smoke}"; GATE_MODE="${GATE_MODE:-}"; RUN_ID="${RUN_ID:-}"; RUN_SCOPED=0
[ -n "$RUN_ID" ] && RUN_SCOPED=1
while [ $# -gt 0 ]; do case "$1" in
  --dry-run) DRY=1; shift;;
  --resume) RESUME=1; shift;;
  --profile) PROFILE="${2:?--profile 缺值}"; shift 2;;
  --gate-mode) GATE_MODE="${2:?--gate-mode 缺值}"; shift 2;;
  --run-id) RUN_ID="${2:?--run-id 缺值}"; RUN_SCOPED=1; shift 2;;
  *) break;;
esac; done
STAGE="${1:?用法: v16_pipeline.sh [--dry-run] [--resume] [--profile smoke|candidate] [--gate-mode observe|strict] [--run-id ID] <stage|train-all|all>}"
[ "$PROFILE" = smoke ] || [ "$PROFILE" = candidate ] || { echo "🔴 未知 profile：$PROFILE"; exit 2; }
if [ -z "$GATE_MODE" ]; then [ "$PROFILE" = smoke ] && GATE_MODE=observe || GATE_MODE=strict; fi
[ "$GATE_MODE" = observe ] || [ "$GATE_MODE" = strict ] || { echo "🔴 未知 gate mode：$GATE_MODE"; exit 2; }
if [ -z "$RUN_ID" ]; then
  if [ "$STAGE" = all ] || [ "$STAGE" = train-all ]; then RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"; RUN_SCOPED=1
  else RUN_ID="${PROFILE}_manual"; fi
fi
[[ "$RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "🔴 run-id 只能含字母、数字、点、下划线和短横线"; exit 2; }

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
if [ "$PROFILE" = smoke ]; then
  SFT_OUT="checkpoints/sft/${DV}_smoke"; MERGED="models/${STUDENT_NAME}-sft-${DV}_smoke"
  RL_OUT="checkpoints/grpo/${DV}_smoke"; RL_ADAPTER="models/adapters/rl_${DV}_smoke"; OPD_OUT="checkpoints/opd/${DV}_smoke"
else
  SFT_OUT="checkpoints/sft/$DV"; MERGED="models/${STUDENT_NAME}-sft-$DV"
  RL_OUT="checkpoints/grpo/${DV}_cand"; RL_ADAPTER="models/adapters/rl_${DV}_candidate"; OPD_OUT="checkpoints/opd/${DV}_candidate"
fi
if [ "$RUN_SCOPED" = 1 ]; then
  SFT_OUT="${SFT_OUT}_${RUN_ID}"; MERGED="${MERGED}_${RUN_ID}"; RL_OUT="${RL_OUT}_${RUN_ID}"
  RL_ADAPTER="${RL_ADAPTER}_${RUN_ID}"; OPD_OUT="${OPD_OUT}_${RUN_ID}"
fi
AUD="_audit"; RUN_AUD="$AUD/$DV/runs/$RUN_ID"; RUN_MANIFEST="$RUN_AUD/manifest.json"
EXAM_ARM="${DV}_${PROFILE}_${RUN_ID}"
TEACHER_URL="${SYNCOPATE_TEACHER_LANG_URL:-http://127.0.0.1:8210/v1}"
TEACHER_STARTED_PID=""
say(){ echo "[v16-pipeline $(date +%H:%M:%S)] $*"; }
run(){ echo "  \$ $*"; [ "$DRY" = 1 ] || eval "$@"; }
need(){
  for f in "$@"; do
    [ -e "$f" ] && continue
    if [ "$DRY" = 1 ]; then
      echo "  ○ dry-run：前置将由真实前段生成：$f"
    else
      echo "🔴 缺前置：$f"
      return 1
    fi
  done
}
record_stage(){ [ "$DRY" = 1 ] || "$PY" -m syncopate.pipeline.run_state \
  --manifest "$RUN_MANIFEST" --run-id "$RUN_ID" --profile "$PROFILE" --gate-mode "$GATE_MODE" \
  --stage "$1" --status "$2" --returncode "$3"; }
quality_run(){
  echo "  \$ $*"
  [ "$DRY" = 1 ] && return 0
  local rc=0
  eval "$@" || rc=$?
  [ "$rc" = 0 ] && return 0
  if [ "$rc" = 2 ]; then
    if [ "$GATE_MODE" = observe ]; then
      echo "🟡 质量/读数门槛未过：observe 模式记录后继续；产物不得晋级 candidate"
      STAGE_WARN=1
      return 0
    fi
    echo "🔴 质量/读数门槛未过：strict 模式阻止下一段"
    return 20
  fi
  return "$rc"
}

stage_cases(){ say "[stage cases] 题库 $DV"; run "$PY -m syncopate cases generate --spec configs/buckets/$DV.yaml --out $BATCH" || return; }
stage_menus(){ say "[stage menus]"; need "$BATCH/manifest.json" || return; run "$PY -m syncopate.pipeline.tool_menus --batch $BATCH" || return; }
stage_split(){ say "[stage split]"; need "$BATCH/manifest.json" || return; run "$PY -m syncopate data split --batch $BATCH --out $SPLIT" || return; }
stage_gates(){ say "[stage gates] D1–D11 + 三桶互斥"; need "$SPLIT/sft_cases.json" || return; run "$PY -m syncopate.pipeline.data_gates --batch $BATCH --split-dir $SPLIT" || return; }
stage_supply(){ say "[stage supply] 本机供给核对：SFT 桶每类底题供给 vs 建库数量闸"; need "$SPLIT/sft_cases.json" || return; run "$PY -m syncopate.pipeline.supply_gate" || return; }
stage_rl_data(){ say "[stage rl-data]"; need "$SPLIT/rl_cases.json" || return
  run "$PY -m syncopate data build --pool rl --batch $BATCH --split-dir $SPLIT --out $RL_DIR --val-every 5" || return
  run "$PY -m syncopate.pipeline.split_isolation $RL_DIR/train.parquet $RL_DIR/val.parquet --pool rl" || return; }

cleanup_teacher(){
  [ -n "$TEACHER_STARTED_PID" ] || return 0
  if kill -0 "$TEACHER_STARTED_PID" 2>/dev/null; then
    say "[teacher-stop] 只停止本轮启动的教师 PID=$TEACHER_STARTED_PID"
    kill "$TEACHER_STARTED_PID" 2>/dev/null || true
    for _ in $(seq 1 30); do kill -0 "$TEACHER_STARTED_PID" 2>/dev/null || break; sleep 1; done
    if kill -0 "$TEACHER_STARTED_PID" 2>/dev/null; then kill -9 "$TEACHER_STARTED_PID" 2>/dev/null || true; fi
  fi
  TEACHER_STARTED_PID=""
}
stage_teacher(){ say "[stage teacher] 27B @8210（已起则复用；all 只清理自己启动的 PID）"
  if curl -sf "${TEACHER_URL%/v1}/health" >/dev/null 2>&1; then say "  教师已在线（不是本轮启动，不会停止）"; return 0; fi
  if [ "$DRY" = 1 ]; then
    run "nohup $PY -m vllm.entrypoints.openai.api_server --model $TEACHER --served-model-name t --max-model-len $MAX_MODEL_LEN --gpu-memory-utilization 0.90 --port 8210 --limit-mm-per-prompt '{\"image\": 0, \"video\": 0}' --max-num-seqs 64 > $RUN_AUD/teacher.log 2>&1 &" || return
    return 0
  fi
  mkdir -p "$RUN_AUD"
  nohup "$PY" -m vllm.entrypoints.openai.api_server --model "$TEACHER" --served-model-name t \
    --max-model-len "$MAX_MODEL_LEN" --gpu-memory-utilization 0.90 --port 8210 \
    --limit-mm-per-prompt '{"image": 0, "video": 0}' --max-num-seqs 64 > "$RUN_AUD/teacher.log" 2>&1 &
  TEACHER_STARTED_PID=$!
  local waited=0
  until curl -sf "${TEACHER_URL%/v1}/health" >/dev/null 2>&1; do
    kill -0 "$TEACHER_STARTED_PID" 2>/dev/null || { echo "🔴 教师进程提前退出"; tail -20 "$RUN_AUD/teacher.log"; return 1; }
    [ "$waited" -lt 1500 ] || { echo "🔴 教师 1500 秒仍未就绪"; return 1; }
    sleep 5; waited=$((waited + 5))
  done
}
stage_teacher_stop(){ say "[stage teacher-stop]"; cleanup_teacher; }
stage_sft_data(){ say "[stage sft-data] 建库（教师物料→六桶→全部闸→出厂体检→隔离→画廊）"; need "$SPLIT/sft_cases.json" || return
  [ "$DRY" = 1 ] || mkdir -p "$RUN_AUD"
  if [ "${SKIP_BEHAVIOR_PROBE:-0}" != 1 ]; then
    quality_run "$PY -m syncopate.pipeline.behavior_think_probe --n 20 --teacher $TEACHER_URL --out $RUN_AUD/behavior_think_probe.json" || return
  fi
  run "SYNCOPATE_TEACHER_LANG_URL=$TEACHER_URL SYNCOPATE_TEACHER_THINK_URL=$TEACHER_URL $PY -m syncopate.pipeline.build_sft 2>&1 | tee $RUN_AUD/build.log" || return
  run "$PY -m syncopate.pipeline.prompt_budget_gate --prompt-budget $SFT_DIR/train.parquet" || return
  run "$PY -m syncopate.pipeline.split_isolation $SFT_DIR/train.parquet $SFT_DIR/val.parquet --pool sft" || return
  run "$PY -m syncopate.pipeline.data_gallery --parquet $SFT_DIR/train.parquet > $RUN_AUD/gallery.md" || return; }
# 本机离线全量建库：教师材料全部来自云盘缓存（modal volume get），缺一条就红——改闸之后先在本机把整条建库跑到底，不上云试错
stage_sft_data_offline(){ say "[stage sft-data-offline] 拉云盘缓存 → 离线建库（全部闸 + 出厂体检）"; need "$SPLIT/sft_cases.json" || return
  [ "$DRY" = 1 ] || mkdir -p "$RUN_AUD"
  # cache_split 必须先于三份行缓存到位；否则构造器不知道缓存属于哪版切分，会正确地把它们作废。
  run "mkdir -p data/u_route; for f in v16_cache_split v16_ballast_replies v16_defs v16_chat_mat v16_l2l1_rows v16_fam_rows v16_cot_rows; do if [ -s data/u_route/\$f.json ] && [ \"\${REFRESH_TEACHER_CACHE:-0}\" != 1 ]; then echo \"  ○ 复用本机缓存：\$f.json\"; else modal volume get syncopate-home _audit/$DV/cache/\$f.json data/u_route/\$f.json --force >/dev/null || return 1; fi; done" || return
  run "SYNCOPATE_TEACHER_OFFLINE=1 $PY -m syncopate.pipeline.build_sft 2>&1 | tee $RUN_AUD/build_offline.log" || return
  run "$PY -m syncopate.pipeline.prompt_budget_gate --prompt-budget $SFT_DIR/train.parquet" || return
  run "$PY -m syncopate.pipeline.split_isolation $SFT_DIR/train.parquet $SFT_DIR/val.parquet --pool sft" || return; }
stage_sft_train(){ say "[stage sft-train] $PROFILE（默认单卡；B04 验明双卡更快后才改默认）"; need "$SFT_DIR/train.parquet" || return
  [ "$DRY" = 1 ] || mkdir -p "$RUN_AUD"
  if [ "$PROFILE" = smoke ]; then
    run "$PY -m syncopate.train.sft --out $SFT_OUT --epochs 1 --batch-size 1 --gpus 1 --effective-batch 8 --max-steps 30 --no-wandb --wandb-run sft_${DV}_smoke 2>&1 | tee $RUN_AUD/sft_train.log" || return
    quality_run "$PY -m syncopate.train.sft_run_gate --log $RUN_AUD/sft_train.log --adapter $SFT_OUT --expected-steps 30 --out $RUN_AUD/sft_run_gate.json" || return
  else
    run "$PY -m syncopate.train.sft --out $SFT_OUT --gpus 1 --effective-batch 16 --wandb-run sft_$DV 2>&1 | tee $RUN_AUD/sft_train.log" || return
  fi
  run "$PY -m syncopate.train.lora_adapter_check $SFT_OUT" || return; }
stage_sft_eval(){ say "[stage sft-eval] 每个候选：entropy + eval_local（冻结 EVAL，8 样本）"; need "$SFT_OUT" || return
  local found=0 candidates=()
  for c in "$SFT_OUT"/epoch* "$SFT_OUT"/sel_f*; do [ -d "$c" ] && candidates+=("$c"); done
  [ "$DRY" = 1 ] && [ "${#candidates[@]}" = 0 ] && candidates+=("$SFT_OUT/epochN")
  for c in "${candidates[@]}"; do n=$(basename "$c")
    found=1
    if [ "$PROFILE" = smoke ]; then
      run "$PY -m syncopate.train.entropy --adapter $c --limit 8 --out $RUN_AUD/sft_entropy_${n}.json" || return
      run "$PY -m syncopate.train.eval_local --adapter $c --limit 8 --samples-per-case 2 --out $RUN_AUD/sft_eval_${n}.json" || return
    else
      run "$PY -m syncopate.train.entropy --adapter $c --limit 24 --out $RUN_AUD/sft_entropy_${n}.json" || return
      run "$PY -m syncopate.train.eval_local --adapter $c --samples-per-case 8 --out $RUN_AUD/sft_eval_${n}.json" || return
    fi
  done
  [ "$DRY" = 1 ] || [ "$found" = 1 ] || { echo "🔴 $SFT_OUT 下没有 epoch*/sel_f* 候选"; return 1; }; }
stage_sft_select(){ say "[stage sft-select] 决策位熵+有梯度格子 → SELECTED，删临时点"; need "$SFT_OUT" || return
  run "$PY -m syncopate.train.select_sft_ckpt $SFT_OUT --audit-dir $RUN_AUD --auto --prune" || return; }
stage_merge(){ say "[stage merge] → $MERGED"; need "$SFT_OUT/SELECTED" || return
  run "$PY -m syncopate.train.merge_adapter --adapter $SFT_OUT/SELECTED --out $MERGED" || return; }
stage_exam(){ say "[stage exam] 考场 v4 → 判卷 → 本轮门禁（PG/Redis 需已起：scripts/serving/{pg,redis}_bootstrap.sh；smoke=1 遍 40 题）"; need "$MERGED" || return
  if [ "$PROFILE" = smoke ]; then
    run "EXAM_PROFILE=$PROFILE EXAM_GATE_MODE=$GATE_MODE EXAM_AUDIT_DIR=$RUN_AUD/exam EXAM_PASSES=1 EXAM_LIMIT=40 bash scripts/v16/exam_chain.sh $MERGED $EXAM_ARM context_v4" || return $?
  else
    run "EXAM_PROFILE=$PROFILE EXAM_GATE_MODE=$GATE_MODE EXAM_AUDIT_DIR=$RUN_AUD/exam EXAM_PASSES=4 bash scripts/v16/exam_chain.sh $MERGED $EXAM_ARM context_v4" || return $?
  fi; }
stage_rl_train(){ say "[stage rl-train] $PROFILE（官方均匀采样基线；动态分池只在 B05 显式 A/B）"; need "$MERGED" "$RL_DIR/train.parquet" || return
  [ "$DRY" = 1 ] || mkdir -p "$RUN_AUD"
  if [ "$PROFILE" = smoke ]; then logger=console; else logger=console,wandb; fi
  run "$PY -m syncopate.train.launch_rl_v1 --profile $PROFILE --model $MERGED --save-path $RL_OUT --logger $logger 2>&1 | tee $RUN_AUD/rl_train.log" || return
  quality_run "$PY -m syncopate.train.rl_run_gate --profile $PROFILE --run-dir $RL_OUT --log $RUN_AUD/rl_train.log --out $RUN_AUD/rl_run_gate.json" || return; }
stage_rl_adapter(){ say "[stage rl-adapter] 最新 global_step → PEFT adapter（复用 verl FSDP2 分片重建，只导出 LoRA）"; need "$RL_OUT" || return
  last=$(ls -d "$RL_OUT"/global_step_* 2>/dev/null | sort -t_ -k3 -n | tail -1 || true)
  if [ -z "$last" ]; then [ "$DRY" = 1 ] && last="$RL_OUT/global_step_N" || { echo "🔴 $RL_OUT 下没有 global_step_*"; return 1; }; fi
  run "$PY -m syncopate.train.ckpt_to_adapter $last/actor --out $RL_ADAPTER/lora_adapter" || return
  run "$PY -m syncopate.train.lora_adapter_check $RL_ADAPTER/lora_adapter" || return; }
stage_rl_eval(){ say "[stage rl-eval]"; need "$MERGED" "$RL_ADAPTER/lora_adapter" || return
  if [ "$PROFILE" = smoke ]; then
    run "$PY -m syncopate.train.eval_local --model $MERGED --adapter $RL_ADAPTER/lora_adapter --limit 8 --samples-per-case 2 --out $RUN_AUD/eval_rl.json" || return
  else
    run "$PY -m syncopate.train.eval_local --model $MERGED --adapter $RL_ADAPTER/lora_adapter --samples-per-case 8 --out $RUN_AUD/eval_rl.json" || return
  fi; }
stage_opd_train(){ say "[stage opd-train] 必须读取本轮 RL adapter；学生@GPU0 · 教师+锚@GPU1"
  need "$MERGED" "$RL_ADAPTER/lora_adapter" || return
  [ "$DRY" = 1 ] || mkdir -p "$RUN_AUD"
  if [ "$PROFILE" = smoke ]; then
    local real_steps="${OPD_SMOKE_REAL_STEPS:-1}" batch="${OPD_SMOKE_BATCH:-2}"
    local attempts="${OPD_SMOKE_MAX_ATTEMPTS:-$((real_steps * 8))}"
    extra="--max-steps $real_steps --max-attempts $attempts --batch $batch --save-every 1 --probe-every 1 --no-wandb"
  else extra=""; fi
  quality_run "CUDA_VISIBLE_DEVICES=0,1 OPD_AUX_GPUS=1 $PY -m torch.distributed.run --nproc_per_node=1 --master_port 29517 -m syncopate.train.opd --base $MERGED --adapter $RL_ADAPTER/lora_adapter --out $OPD_OUT $extra 2>&1 | tee $RUN_AUD/opd_train.log" || return
  if [ "$PROFILE" = smoke ]; then expected_steps="$real_steps"; else expected_steps=1; fi
  quality_run "$PY -m syncopate.train.opd_run_gate --log $RUN_AUD/opd_train.log --out-dir $OPD_OUT --expected-real-steps $expected_steps --out $RUN_AUD/opd_run_gate.json" || return
  [ "$DRY" = 1 ] || [ -d "$OPD_OUT/final" ] || return 10; }
stage_opd_eval(){ say "[stage opd-eval]"; need "$MERGED" "$OPD_OUT/final" "$OPD_OUT/completion.json" || return
  if [ "$PROFILE" = smoke ]; then
    run "$PY -m syncopate.train.eval_local --model $MERGED --adapter $OPD_OUT/final --limit 8 --samples-per-case 2 --out $RUN_AUD/eval_opd.json" || return
  else
    run "$PY -m syncopate.train.eval_local --model $MERGED --adapter $OPD_OUT/final --samples-per-case 8 --out $RUN_AUD/eval_opd.json" || return
  fi; }

ALL=(cases menus split gates supply rl-data teacher sft-data teacher-stop sft-train sft-eval sft-select merge exam rl-train rl-adapter rl-eval opd-train opd-eval)
TRAIN_ALL=(sft-train sft-eval sft-select merge exam rl-train rl-adapter rl-eval opd-train opd-eval)
run_stage(){
  local s="$1"; local fn="stage_${s//-/_}"; local rc=0 status=pass
  declare -F "$fn" >/dev/null || { echo "🔴 未知 stage：$s（可选：${ALL[*]} train-all all）"; return 2; }
  if [ "$RESUME" = 1 ] && [ "$DRY" = 0 ]; then
    if "$PY" -m syncopate.pipeline.run_state --manifest "$RUN_MANIFEST" \
         --run-id "$RUN_ID" --profile "$PROFILE" --gate-mode "$GATE_MODE" \
         --stage "$s" --check-resumable; then
      say "[stage $s] resume：本轮账本已完成（PASS，或 observe 下保留 WARN），复用原产物"
      return 0
    else
      rc=$?
      [ "$rc" = 1 ] || return "$rc"
      rc=0
    fi
  fi
  STAGE_WARN=0
  "$fn" || rc=$?
  case "$rc" in
    0) [ "$STAGE_WARN" = 1 ] && status=warn; record_stage "$s" "$status" 0 || return;;
    10) record_stage "$s" warn "$rc" || return; return 0;;
    20) record_stage "$s" block_next "$rc" || return; return 20;;
    *) record_stage "$s" fatal "$rc" || true; return "$rc";;
  esac
}
if [ "$STAGE" = all ]; then
  trap cleanup_teacher EXIT
  for s in "${ALL[@]}"; do run_stage "$s" || exit $?; done
elif [ "$STAGE" = train-all ]; then
  for s in "${TRAIN_ALL[@]}"; do run_stage "$s" || exit $?; done
else
  run_stage "$STAGE" || exit $?
fi
say "done ($STAGE, profile=$PROFILE, gate=$GATE_MODE, run=$RUN_ID, dry=$DRY)"
