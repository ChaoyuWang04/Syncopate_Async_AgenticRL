#!/usr/bin/env bash
# S2.4 · goodput@SLO before/after 同尺全栈阶梯 + loadtest 服务面子集（lat/comp/conc）。
set -u
cd /workspace/Syncopate_Async_AgenticRL
set -a; . /workspace/.env; set +a
source .venv/bin/activate
MODEL=models/Qwen3-4B-sft-v13r2-e1
ADAPTER=checkpoints/grpo/cand_v13r2_e1/adapter_global_step_25
D=logs/b4; say() { echo "[GP $(date +%H:%M:%S)] $*"; }

until GPUS=0,1,2,3 bash scripts/infra/gpu_gate.sh >/dev/null 2>&1; do echo "[GP] gate 等 60s" >&2; sleep 60; done

say "== BEFORE: 单卡生产配置 =="
CUDA_VISIBLE_DEVICES=0 vllm serve "$MODEL" --served-model-name sft-base \
  --enable-lora --lora-modules candidate="$ADAPTER" \
  --max-lora-rank 32 --max-model-len 14336 --kv-cache-dtype fp8 \
  --host 127.0.0.1 --port 8100 > "$D/gp_vllm_before.log" 2>&1 &
SRV=$!
for _ in $(seq 1 90); do sleep 7; curl -sf http://127.0.0.1:8100/health >/dev/null && break; done
curl -sf http://127.0.0.1:8100/health >/dev/null || { say "🔴 before 引擎没起来"; exit 1; }
bash scripts/serving/b4_stack.sh start 64 || { say "🔴 stack 失败"; exit 1; }
say "loadtest 子集(before)"
timeout 1800 .venv/bin/python scripts/serving/runtime_loadtest.py --skip crash,model_down,cost,sse,idem \
  > "$D/loadtest_before.log" 2>&1 || say "⚠️ loadtest before 非零退出"
say "goodput 阶梯(before)"
timeout 3600 .venv/bin/python scripts/serving/b4_goodput.py --tag before --levels 8,16,24,32,48,64 \
  --out "$D/goodput_before.json" > "$D/goodput_before.log" 2>&1 || say "⚠️ goodput before 非零退出"
bash scripts/serving/b4_stack.sh stop
kill $SRV 2>/dev/null; wait $SRV 2>/dev/null
for _ in $(seq 1 30); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0); [ "$u" -lt 2000 ] && break; sleep 5; done

say "== AFTER: 4×DP + 亲和 router + fleet 旗子 =="
bash scripts/serving/b4_serve_4x.sh start 4 affinity -- --max-num-batched-tokens 16384 || { say "🔴 after 拓扑失败"; exit 1; }
bash scripts/serving/b4_stack.sh start 64 || { say "🔴 stack 失败"; exit 1; }
say "loadtest 子集(after)"
timeout 1800 .venv/bin/python scripts/serving/runtime_loadtest.py --skip crash,model_down,cost,sse,idem \
  > "$D/loadtest_after.log" 2>&1 || say "⚠️ loadtest after 非零退出"
say "goodput 阶梯(after)"
timeout 3600 .venv/bin/python scripts/serving/b4_goodput.py --tag after --levels 8,16,24,32,48,64 \
  --out "$D/goodput_after.json" > "$D/goodput_after.log" 2>&1 || say "⚠️ goodput after 非零退出"
curl -s http://127.0.0.1:8100/router/stats > "$D/goodput_after_router_stats.json" 2>/dev/null
bash scripts/serving/b4_stack.sh stop
bash scripts/serving/b4_serve_4x.sh stop
echo GOODPUT-CHAIN-DONE
