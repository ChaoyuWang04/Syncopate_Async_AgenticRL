#!/usr/bin/env bash
# v16 考场链：端点 → API/worker → 考场 → 判卷 → 本轮门禁。
# smoke/observe：质量缺口返回 10，由总管线记 WARN 后继续；candidate/strict：返回 20，阻止 RL。
set -uo pipefail
export SYNCOPATE_CONTRACT="${SYNCOPATE_CONTRACT:-v15}" SYNCOPATE_THINK="${SYNCOPATE_THINK:-1}"
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
[ -f /workspace/.env ] && { set -a; . /workspace/.env; set +a; }
PY="${PY:-$( [ -x /env/.venv/bin/python ] && echo /env/.venv/bin/python || echo .venv/bin/python )}"
export PATH="$(dirname "$PY"):$PATH"

MERGED="${1:?合并模型路径}"
ARM="${2:-v16_sft}"
EXAM="${3:-context_v4}"
PASSES="${EXAM_PASSES:-4}"
GATE_MODE="${EXAM_GATE_MODE:-strict}"
PROFILE="${EXAM_PROFILE:-smoke}"
[ "$GATE_MODE" = observe ] || [ "$GATE_MODE" = strict ] || { echo "🔴 EXAM_GATE_MODE=$GATE_MODE 非法"; exit 2; }
[ "$PROFILE" = smoke ] || [ "$PROFILE" = candidate ] || { echo "🔴 EXAM_PROFILE=$PROFILE 非法"; exit 2; }
read -r DV MAX_MODEL_LEN < <("$PY" - <<'PY' | sed -n 's/^__SYNCOPATE_EXAM_CONSTANTS__ //p'
from syncopate.pipeline.split import DATA_VERSION
from syncopate.train.rollout_budget import MAX_PROMPT_LENGTH, MAX_RESPONSE_LENGTH
print("__SYNCOPATE_EXAM_CONSTANTS__", DATA_VERSION, MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH)
PY
)
[[ "${DV:-}" =~ ^v[0-9]+$ ]] || { echo "🔴 无法从唯一常量源读取 DATA_VERSION：${DV:-<空>}"; exit 1; }
[[ "${MAX_MODEL_LEN:-}" =~ ^[0-9]+$ ]] || { echo "🔴 无法从唯一常量源读取 MAX_MODEL_LEN：${MAX_MODEL_LEN:-<空>}"; exit 1; }
AUD="${EXAM_AUDIT_DIR:-_audit/$DV/exam/$ARM}"
mkdir -p "$AUD" logs/u_route
say(){ echo "[V16-EXAM $(date +%H:%M:%S)] $*"; }

V=""; A=""; W=""
cleanup(){
  local pid
  for pid in "$W" "$A" "$V"; do
    [ -n "$pid" ] || continue
    kill "$pid" 2>/dev/null || true
  done
  for _ in $(seq 1 20); do
    local alive=0
    for pid in "$W" "$A" "$V"; do [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null && alive=1; done
    [ "$alive" = 0 ] && break
    sleep 1
  done
  for pid in "$W" "$A" "$V"; do
    [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null && kill -9 "$pid" 2>/dev/null || true
  done
  V=""; A=""; W=""
}
trap cleanup EXIT INT TERM

if pgrep -f "syncopate.runtime.worker" >/dev/null; then
  echo "🔴 已有 worker 在跑；先精确停止它，避免新旧模型抢同一队列："
  ps -eo pid,etime,cmd | grep "[s]yncopate.runtime.worker"
  exit 1
fi

SERVED="v16_${ARM}"
say "① 起端点（GPU0，:8100；max_model_len=$MAX_MODEL_LEN 来自 rollout_budget）"
CUDA_VISIBLE_DEVICES=0 nohup "$PY" -m vllm.entrypoints.openai.api_server --model "$MERGED" \
  --served-model-name "$SERVED" --max-model-len "$MAX_MODEL_LEN" --host 127.0.0.1 --port 8100 \
  --gpu-memory-utilization 0.85 > "$AUD/vllm.log" 2>&1 &
V=$!
waited=0
until curl -sf http://127.0.0.1:8100/health >/dev/null 2>&1; do
  sleep 5; waited=$((waited + 5))
  kill -0 "$V" 2>/dev/null || { echo "🔴 VLLM-DEAD"; tail -20 "$AUD/vllm.log"; exit 1; }
  [ "$waited" -lt 1500 ] || { echo "🔴 vLLM 1500 秒仍未就绪"; exit 1; }
done
curl -sf http://127.0.0.1:8100/v1/models > "$AUD/models.json" || {
  echo "🔴 端点已健康但 /v1/models 读不到，无法证明实际服务的模型身份"
  exit 1
}

say "② 起 API + worker，并核对数据库种子"
SYNCOPATE_CONTRACT=v15 SYNCOPATE_API_DB_POOL=12 nohup "$PY" -m uvicorn syncopate.runtime.api:app \
  --host 127.0.0.1 --port 8000 --workers 2 > "$AUD/api.log" 2>&1 &
A=$!
waited=0
until curl -sf http://127.0.0.1:8000/healthz >/dev/null 2>&1; do
  sleep 2; waited=$((waited + 2))
  kill -0 "$A" 2>/dev/null || { echo "🔴 API-DEAD"; tail -20 "$AUD/api.log"; exit 1; }
  [ "$waited" -lt 120 ] || { echo "🔴 API 120 秒仍未就绪"; exit 1; }
done
"$PY" -m syncopate.runtime.seed_demo_data > "$AUD/seed.log" 2>&1 || { tail -20 "$AUD/seed.log"; exit 1; }
"$PY" -m syncopate.runtime.seed_demo_data --check >> "$AUD/seed.log" 2>&1 || { tail -20 "$AUD/seed.log"; exit 1; }
SYNCOPATE_CONTRACT=v15 SYNCOPATE_DECIDER_URL=http://127.0.0.1:8100 \
  SYNCOPATE_DECIDER_TOKENIZER="$MERGED" SYNCOPATE_DECIDER_MODEL="$SERVED" \
  nohup "$PY" -m syncopate.runtime.worker --org-id org_demo --worker-id v16-exam \
  --daily-cost-cap-micros 10000000000 > "$AUD/worker.log" 2>&1 &
W=$!
sleep 5
kill -0 "$W" 2>/dev/null || { echo "🔴 WORKER-DEAD"; tail -20 "$AUD/worker.log"; exit 1; }

say "③ 考场 $EXAM 跑 $PASSES 遍"
LIMIT_ARGS=()
[ -n "${EXAM_LIMIT:-}" ] && LIMIT_ARGS=(--limit "$EXAM_LIMIT")
for i in $(seq 1 "$PASSES"); do
  "$PY" -m syncopate.evaluation.exam_run --exam "$EXAM" --arm "${ARM}_r$i" \
    --concurrency 4 "${LIMIT_ARGS[@]}" > "$AUD/exam_r$i.log" 2>&1 || {
      echo "🔴 第 $i 遍失败"; tail -30 "$AUD/exam_r$i.log"; exit 1;
    }
  say "  第 $i 遍完成"
done

say "④ 收尾：只停本脚本启动的三个 PID"
cleanup

say "⑤ 判卷 + 本轮门禁"
JUDGED=()
for i in $(seq 1 "$PASSES"); do
  run_file="logs/u_route/run_${ARM}_r${i}_${EXAM}.jsonl"
  [ -f "$run_file" ] || { echo "🔴 缺本轮考场产物：$run_file"; exit 1; }
  JUDGED+=("$run_file")
done
"$PY" -m syncopate.evaluation.exam_judge --context "${JUDGED[@]}" > "$AUD/judge.log" 2>&1 || {
  echo "🔴 判卷失败"; tail -30 "$AUD/judge.log"; exit 1;
}

# 原始答卷和判卷结果必须跟随 run audit 持久化；logs/u_route 是容器临时目录，不能充当证据。
for i in $(seq 1 "$PASSES"); do
  run_file="logs/u_route/run_${ARM}_r${i}_${EXAM}.jsonl"
  judged_file="logs/u_route/judged_${ARM}_r${i}_${EXAM}.jsonl"
  [ -f "$judged_file" ] || { echo "🔴 缺本轮判卷产物：$judged_file"; exit 1; }
  cp "$run_file" "$AUD/$(basename "$run_file")"
  cp "$judged_file" "$AUD/$(basename "$judged_file")"
done

GATE_ARGS=(
  --raw "$AUD/run_${ARM}_r*_${EXAM}.jsonl"
  --judged "$AUD/judged_${ARM}_r*_${EXAM}.jsonl"
  --exam-spec "data/u_route/${EXAM}_exam.jsonl"
  --models-json "$AUD/models.json"
  --served-model "$SERVED"
  --model-path "$MERGED"
  --profile "$PROFILE"
  --expected-passes "$PASSES"
  --limit "${EXAM_LIMIT:-0}"
  --out "$AUD/exam_run_gate.json"
)
[ -n "${EXAM_CANDIDATE_POLICY:-}" ] && GATE_ARGS+=(--candidate-policy "$EXAM_CANDIDATE_POLICY")
"$PY" -m syncopate.evaluation.exam_run_gate "${GATE_ARGS[@]}" > "$AUD/gate.log" 2>&1
TRC=$?
case "$TRC" in
  0) say "✅ 本轮门禁通过";;
  2)
    tail -20 "$AUD/gate.log"
    if [ "$GATE_MODE" = observe ]; then
      echo "🟡 本轮 Exam 链路完整，但有质量缺口；smoke 记录 WARN 后继续，不得晋级 candidate"
      exit 10
    fi
    echo "🔴 本轮 Exam 有质量/门槛缺口；candidate 在这里停止，不启动 RL"
    exit 20;;
  1) echo "🔴 Exam 证据或链路不完整"; tail -30 "$AUD/gate.log"; exit 1;;
  *) echo "🔴 Exam 门禁程序失败 rc=$TRC"; tail -30 "$AUD/gate.log"; exit 1;;
esac
say "✅ $PASSES 遍完成；本轮证据在 $AUD"
