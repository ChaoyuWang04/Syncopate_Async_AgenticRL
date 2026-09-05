#!/bin/bash
# ⚠️ 单卡=让卡/后备模式（要给训练/实验腾卡时用）。
# ★ 生产默认是四卡：bash logs/runtime/start_serving.sh（Chaoyu 08-28 裁定：serving 期整机全上；
#   曾被误留成单卡默认——E33 收尾时的保守判断，已纠正）
# B-4 模型端点（部署侧上限 14336，刻意 > 训练契约 7168 —— 理由见 decider.py 的长注释）
cd /workspace/Syncopate_Async_AgenticRL
set -a; . /workspace/.env; set +a
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=0
# ★ 2026-08-21：--kv-cache-dtype fp8 默认（E19 §8：KV 池 ×2·并发 +50%·质量 −0.009 在 MDE 界）
# ★ 2026-08-28（E32 收官两项，均实测采纳+冒烟验证）：
#   --max-num-batched-tokens 16384  冷 prefill 吞吐 +5.3%/TTFT −12%（双跑复现），热流量零代价
#   --speculative-config ngram      单流 TPOT 2.3×·48 并发 +41%·接受率 64%·50/50 greedy 逐字无损
# ★ 2026-08-28（B-5/E33）：--scheduling-policy priority——decider 按意图 SLO 传 priority
#   （I01 最紧先跑）；无等待队列时零作用（fcfs 等价），有队列时保紧预算意图。
# 整机四卡高吞吐模式（重生成/批式负载，扩展 3.66-3.86×）：
#   bash scripts/serving/b4_serve_4x.sh start 4 rr -- --max-num-batched-tokens 16384 --scheduling-policy priority \
#     --speculative-config '{"method":"ngram","num_speculative_tokens":4,"prompt_lookup_max":4,"prompt_lookup_min":2}'
#   （router 监听同一个 :8100 ⇒ decider/chatbox 无感；停：b4_serve_4x.sh stop）
exec vllm serve models/Qwen3-4B-sft-v13r2-e1 \
  --served-model-name sft-base \
  --enable-lora --lora-modules candidate=checkpoints/grpo/cand_v13r2_e1/adapter_global_step_25 \
  --max-lora-rank 32 --max-model-len 24576 --kv-cache-dtype fp8 \
  --max-num-batched-tokens 16384 --scheduling-policy priority \
  --speculative-config '{"method":"ngram","num_speculative_tokens":4,"prompt_lookup_max":4,"prompt_lookup_min":2}' \
  --host 127.0.0.1 --port 8100
