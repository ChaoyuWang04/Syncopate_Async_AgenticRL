"""E31 第 1 步验收①（离线，§9b 方法重打）：真轨迹上四臂对拍，量"量化项对消"是否兑现。

四臂（response 段逐 token logprob，Δ = vLLM − trainer）：
  baseline_bf16   vLLM bf16 头  − trainer bf16 头     = 引擎本底（分母）
  unified_fp8     vLLM MXFP8 头 − trainer MXFP8 头    = 统一后的残差（被验对象）
  one_sided       vLLM MXFP8 头 − trainer bf16 头     = §9b 判死的毒状态（对照组，必须爆）
  patch_effect    vLLM MXFP8 头 − vLLM bf16 头        = 补丁真的生效了（防第八形态）

通过标准（E31 第 1 步①，test_e31_step1.py::test_a1 固化）：
  unified 的 token |Δ| 中位 ≤ 2× 本底 · 序列 |ΣΔ| p95 < ln2 · one_sided p95 > ln2 · patch_effect > 1e-3

数据 = 08-27 冒烟 bf16 臂 rollout_dumps 里 response 最长的 16 条（与 kl_floor 同源同模型）。
用法：.venv/bin/python scripts/e31_step1_offline.py            # 全流程（含两个 vLLM 子进程）
     （内部会以 --vllm-arm bf16/fp8 递归自调，别手动传）
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = "/workspace/hf_assets/bases/Qwen3-4B-sft-v13r2-e1"
DUMPS = ROOT / "checkpoints/grpo/smoke_newbox_0827_kvauto/rollout_dumps"
OUT = ROOT / "logs/e31/step1_offline.json"
TMP = Path("/workspace/tmp/e31_step1")
N_SEQS, MAX_LEN = 16, 7168
FLAG = "SYNCOPATE_UNIFIED_FP8"


def build_seqs():
    """挑 response 最长的 N 条真轨迹，重新分词成 (ids, prompt_len)。"""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    rows = []
    for f in sorted(DUMPS.glob("*.jsonl")):
        for line in f.read_text().splitlines():
            r = json.loads(line)
            rows.append((r["input"], r["output"]))
    seqs, seen = [], set()
    for inp, out in sorted(rows, key=lambda r: -len(r[1])):
        if out[:200] in seen:          # 去重（同题多采样的近似判重）
            continue
        p_ids = tok(inp, add_special_tokens=False)["input_ids"]
        o_ids = tok(out, add_special_tokens=False)["input_ids"]
        if len(p_ids) + len(o_ids) > MAX_LEN or len(o_ids) < 200:
            continue
        seen.add(out[:200])
        seqs.append({"ids": p_ids + o_ids, "prompt_len": len(p_ids)})
        if len(seqs) == N_SEQS:
            break
    assert len(seqs) >= 8, f"可用轨迹只有 {len(seqs)} 条"
    return seqs


def trainer_pass():
    """子进程模式：HF bf16 前向拿 hidden，一次前向喂两个头（bf16 / MXFP8），逐 token logprob。"""
    import torch
    from transformers import AutoModelForCausalLM
    from syncopate.train import unified_fp8
    seqs = json.loads((TMP / "seqs.json").read_text())
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, attn_implementation="flash_attention_2").cuda().eval()
    W = model.get_output_embeddings().weight
    qw, qw_sf = unified_fp8._weight_cache(W, "fwd")
    out = {"bf16": [], "fp8": []}
    with torch.no_grad():
        for s in seqs:
            ids = torch.tensor(s["ids"], device="cuda").unsqueeze(0)
            h = model.model(input_ids=ids).last_hidden_state[0]      # [L,K] 过了 final norm
            p = s["prompt_len"]
            hh = h[p - 1: len(s["ids"]) - 1]                          # 预测位置
            tgt = ids[0, p:]
            lg16 = (hh @ W.T).float()
            out["bf16"].append(lg16.log_softmax(-1).gather(-1, tgt[:, None]).squeeze(1).tolist())
            lg8 = unified_fp8._mxf8_logits(hh, qw, qw_sf).float()
            out["fp8"].append(lg8.log_softmax(-1).gather(-1, tgt[:, None]).squeeze(1).tolist())
    (TMP / "trainer.json").write_text(json.dumps(out))


def vllm_arm(seqs_file: Path, out_file: Path):
    """子进程模式：按当前 env 开关起 vLLM，对每条序列取 prompt 段逐 token logprob。"""
    import torch  # noqa: F401  （先 import 保证扩展 JIT 缓存路径就绪）
    from vllm import LLM, SamplingParams
    seqs = json.loads(seqs_file.read_text())
    # gpu_util 压到 0.45 + 预填充批 2048：prompt_logprobs 的 fp32 尖峰
    # （批 token × 151936 × 4B）不在 vLLM 显存预算里，0.6+8192 实测 OOM
    llm = LLM(model=MODEL, dtype="bfloat16", max_model_len=MAX_LEN,
              gpu_memory_utilization=0.55, seed=0, max_num_batched_tokens=2048,
              enable_prefix_caching=False)     # 评分臂关缓存：命中段可能不出 logprob
    sp = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=0)
    outs = llm.generate([{"prompt_token_ids": s["ids"]} for s in seqs], sp)
    res = []
    for s, o in zip(seqs, outs):
        plps = o.prompt_logprobs
        lps = []
        for j in range(s["prompt_len"], len(s["ids"])):
            d = plps[j]
            lps.append(d[s["ids"][j]].logprob)
        res.append(lps)
    out_file.write_text(json.dumps(res))


def stats(deltas):
    import torch
    tok = torch.cat([d.abs() for d in deltas])
    signed = torch.tensor([d.sum().item() for d in deltas])
    seq = signed.abs()
    return {
        "token_abs_median": tok.median().item(),
        "token_abs_mean": tok.mean().item(),
        # 签名偏置：温度偏置机理（E30 §11）的直接读数——对消看它，不看绝对值
        "token_bias": (signed.sum() / tok.numel()).item(),
        "seq_signed_pos": int((signed > 0).sum()),
        "seq_abs_sum_p95": seq.quantile(0.95).item(),
        "seq_abs_sum_max": seq.max().item(),
        "seq_abs_sums": [round(v, 4) for v in seq.tolist()],
    }


def main() -> int:
    if "--vllm-arm" in sys.argv:
        i = sys.argv.index("--vllm-arm")
        arm = sys.argv[i + 1]
        assert (os.environ.get(FLAG) == "1") == (arm == "fp8"), "臂与开关不一致"
        vllm_arm(TMP / "seqs.json", TMP / f"vllm_{arm}.json")
        return 0
    if "--trainer-arm" in sys.argv:
        trainer_pass()
        return 0

    import torch
    TMP.mkdir(parents=True, exist_ok=True)
    seqs = build_seqs()
    lens = [len(s["ids"]) - s["prompt_len"] for s in seqs]
    print(f"轨迹 {len(seqs)} 条 · response 长度 {min(lens)}–{max(lens)}（中位 {sorted(lens)[len(lens)//2]}）")
    (TMP / "seqs.json").write_text(json.dumps(seqs))

    # 三个臂全走子进程：父进程不碰 GPU（第一版 trainer 残留 8.6 GB 挤 OOM 了 vLLM 臂）
    reuse = "--reuse" in sys.argv        # 只重聚合已跑完的臂（改指标定义时用，臂数据不动）
    if not (reuse and (TMP / "trainer.json").exists()):
        print("── trainer 双头前向（子进程）──", flush=True)
        subprocess.run([sys.executable, __file__, "--trainer-arm"], check=True)
    for arm in ("bf16", "fp8"):
        if reuse and (TMP / f"vllm_{arm}.json").exists():
            continue
        print(f"── vLLM {arm} 臂（子进程）──", flush=True)
        env = dict(os.environ)
        env[FLAG] = "1" if arm == "fp8" else "0"
        subprocess.run([sys.executable, __file__, "--vllm-arm", arm], env=env, check=True)

    tr_raw = json.loads((TMP / "trainer.json").read_text())
    tr = {k: [torch.tensor(x) for x in v] for k, v in tr_raw.items()}
    v16 = [torch.tensor(x) for x in json.loads((TMP / "vllm_bf16.json").read_text())]
    v8 = [torch.tensor(x) for x in json.loads((TMP / "vllm_fp8.json").read_text())]
    report = {
        "n_seqs": len(seqs), "response_lens": lens, "model": MODEL,
        "baseline_bf16": stats([a - b.float() for a, b in zip(v16, tr["bf16"])]),
        "unified_fp8": stats([a - b.float() for a, b in zip(v8, tr["fp8"])]),
        "one_sided": stats([a - b.float() for a, b in zip(v8, tr["bf16"])]),
        "patch_effect": stats([a - b for a, b in zip(v8, v16)]),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
