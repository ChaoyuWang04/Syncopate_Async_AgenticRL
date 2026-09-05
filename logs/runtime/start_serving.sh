#!/bin/bash
# ★ 生产 serving 默认入口（Chaoyu 08-28 裁定：训练/serving 分时共用整机，serving 期四卡全上）
#   = 4×独立引擎 + router v2 顶 :8100（decider/chatbox 无感）；E32/E33 全部实测采纳项已含：
#   fp8 KV · mnbt16384 · ngram 投机（单流 2.3×）· SLO 优先级 · rr 分流（重负载 3.86×·goodput 192）
#   停：bash scripts/serving/b4_serve_4x.sh stop
# 让卡模式（要给训练/实验腾 GPU1-3 时）：bash logs/runtime/start_vllm.sh（单卡 GPU0，同旗子）
# 部分让卡：把下面的 4 改成 2/3（引擎按 GPU0..N-1 排）
cd /workspace/Syncopate_Async_AgenticRL
exec bash scripts/serving/b4_serve_4x.sh start "${1:-4}" rr -- \
  --max-num-batched-tokens 16384 --scheduling-policy priority \
  --speculative-config '{"method":"ngram","num_speculative_tokens":4,"prompt_lookup_max":4,"prompt_lookup_min":2}'
