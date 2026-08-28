#!/usr/bin/env bash
# S3 · PD go/no-go：干扰两态（chunked on/off 两引擎配置）+ pinned 带宽 + 合账。
set -u
cd /workspace/Syncopate_Async_AgenticRL
set -a; . /workspace/.env; set +a
source .venv/bin/activate
MODEL=models/Qwen3-4B-sft-v13r2-e1
ADAPTER=checkpoints/grpo/cand_v13r2_e1/adapter_global_step_25
D=logs/b4; mkdir -p "$D"
say() { echo "[S3 $(date +%H:%M:%S)] $*"; }

serve() { local tag=$1; shift
  CUDA_VISIBLE_DEVICES=0 vllm serve "$MODEL" --served-model-name sft-base \
    --enable-lora --lora-modules candidate="$ADAPTER" \
    --max-lora-rank 32 --max-model-len 14336 --kv-cache-dtype fp8 \
    --max-num-batched-tokens 16384 \
    --host 127.0.0.1 --port 8100 "$@" > "$D/s3_vllm_$tag.log" 2>&1 &
  SRV=$!
  for _ in $(seq 1 90); do sleep 7; curl -sf http://127.0.0.1:8100/health >/dev/null && return 0; kill -0 $SRV 2>/dev/null || break; done
  say "🔴 $tag 没起来"; tail -8 "$D/s3_vllm_$tag.log"; return 1
}
teardown() { kill $SRV 2>/dev/null; wait $SRV 2>/dev/null
  for _ in $(seq 1 30); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0); [ "$u" -lt 2000 ] && break; sleep 5; done; }

until GPUS=0 bash scripts/gpu_gate.sh >/dev/null 2>&1; do echo "[S3] gate 等 60s" >&2; sleep 60; done

say "臂① chunked on（生产默认）"
serve chunkon || exit 1
.venv/bin/python scripts/b4_pd_probe.py interfere --duration 120 --out "$D/pd_interfere_chunkon.json" || say "🔴 interfere① 失败"
teardown

say "臂② chunked off"
serve chunkoff --no-enable-chunked-prefill || { say "chunked-off 不支持=记档"; SRV=; }
if [ -n "${SRV:-}" ]; then
  .venv/bin/python scripts/b4_pd_probe.py interfere --duration 120 --out "$D/pd_interfere_chunkoff.json" || say "🔴 interfere② 失败"
  teardown
fi

say "带宽探针"
.venv/bin/python scripts/b4_pd_probe.py bw --out "$D/pd_bw.json" || say "🔴 bw 失败"

say "合账"
.venv/bin/python scripts/b4_pd_probe.py account \
  --interfere-chunked "$D/pd_interfere_chunkon.json" \
  --interfere-nochunk "$([ -f "$D/pd_interfere_chunkoff.json" ] && echo "$D/pd_interfere_chunkoff.json" || echo "")" \
  --bw "$D/pd_bw.json" --out "$D/pd_verdict.json" || say "🔴 account 失败"
echo S3-CHAIN-DONE

# ---- S4 链（原 /tmp/b4_s4_chain.sh，同日串行）----
# S4 · ngram 投机解码探针：基线臂与 ngram 臂各测 greedy捕获 + 单流 + 并发8/48。
set -u
cd /workspace/Syncopate_Async_AgenticRL
set -a; . /workspace/.env; set +a
source .venv/bin/activate
MODEL=models/Qwen3-4B-sft-v13r2-e1
ADAPTER=checkpoints/grpo/cand_v13r2_e1/adapter_global_step_25
D=logs/b4/s4; mkdir -p "$D"
say() { echo "[S4 $(date +%H:%M:%S)] $*"; }

serve() {  # $1=tag  $@=extra flags
  local tag=$1; shift
  CUDA_VISIBLE_DEVICES=0 vllm serve "$MODEL" --served-model-name sft-base \
    --enable-lora --lora-modules candidate="$ADAPTER" \
    --max-lora-rank 32 --max-model-len 14336 --kv-cache-dtype fp8 \
    --max-num-batched-tokens 16384 \
    --host 127.0.0.1 --port 8100 "$@" > "$D/vllm_$tag.log" 2>&1 &
  SRV=$!
  for _ in $(seq 1 90); do sleep 7; curl -sf http://127.0.0.1:8100/health >/dev/null && return 0; kill -0 $SRV 2>/dev/null || break; done
  say "🔴 $tag 服务没起来"; tail -8 "$D/vllm_$tag.log"; return 1
}
teardown() {
  kill $SRV 2>/dev/null; wait $SRV 2>/dev/null
  for _ in $(seq 1 30); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0); [ "$u" -lt 2000 ] && break; sleep 5; done
}
press() {  # $1=tag
  local tag=$1
  say "$tag greedy 捕获"
  .venv/bin/python scripts/b4_greedy_diff.py capture --out "$D/greedy_$tag.json" > "$D/greedy_$tag.log" 2>&1 || say "🔴 $tag greedy 失败"
  for conc in 1 8 48; do
    say "$tag trace 并发 $conc"
    n=$([ "$conc" = 1 ] && echo 48 || echo 0)
    .venv/bin/python scripts/b4_replay.py --concurrency "$conc" --n "$n" \
      --out "$D/trace_${tag}_c${conc}.json" > /dev/null 2>&1 || say "🔴 $tag c$conc 有失败"
  done
  curl -s http://127.0.0.1:8100/metrics | grep -iE "spec_decode|num_accepted|num_draft" > "$D/specmetrics_$tag.txt" || true
}

# 门禁（等 S2/S3 让卡）
until GPUS=0 bash scripts/gpu_gate.sh >/dev/null 2>&1; do echo "[S4] gate 等 60s" >&2; sleep 60; done

say "== 臂 A: 基线（无投机）=="
serve base || exit 1
press base
teardown

say "== 臂 B: ngram 投机 =="
serve ngram --speculative-config '{"method":"ngram","num_speculative_tokens":4,"prompt_lookup_max":4,"prompt_lookup_min":2}' || { say "NGRAM-INCOMPAT（记档退场）"; exit 2; }
press ngram
teardown

say "== 无损性判据 =="
.venv/bin/python scripts/b4_greedy_diff.py diff "$D/greedy_base.json" "$D/greedy_ngram.json" | tail -3
echo S4-CHAIN-DONE
