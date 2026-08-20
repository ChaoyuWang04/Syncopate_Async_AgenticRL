#!/bin/bash
# B-4 模型端点（部署侧上限 14336，刻意 > 训练契约 7168 —— 理由见 decider.py 的长注释）
cd /workspace/Syncopate_Async_AgenticRL
set -a; . /workspace/.env; set +a
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=0
exec vllm serve models/Qwen3-4B-sft-v13r2-e1 \
  --served-model-name sft-base \
  --enable-lora --lora-modules candidate=checkpoints/grpo/cand_v13r2_e1/adapter_global_step_25 \
  --max-lora-rank 32 --max-model-len 14336 \
  --host 127.0.0.1 --port 8100
