#!/bin/bash
# O-1 探针用：**裸底座** Qwen3-4B（OPD 的老师候选）。与 :8100 的候选同源同模板。
cd /workspace/Syncopate_Async_AgenticRL
set -a; . /workspace/.env; set +a
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=1
exec vllm serve models/Qwen3-4B --served-model-name base \
  --max-model-len 18432 --host 127.0.0.1 --port 8101 \
  --gpu-memory-utilization 0.85
