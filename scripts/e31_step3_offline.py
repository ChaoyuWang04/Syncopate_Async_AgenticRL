"""E31 第 3 步验收（离线渐进）：内层逐组扩 8bit，每组四臂对拍（§9b 同尺，全臂 eager）。

组：G0'=仅 lm_head（eager 重锚）→ G1=8 层 → G2=16 → G3=24 → G4=30（末 6 层保 bf16）。
臂（每组）：trainer(fp8, N 层+头) vs vLLM(fp8, N 层+头) = unified_N；
分母 = vLLM(bf16, eager) vs trainer(bf16)。全部臂 enforce_eager=True（内层补丁只在
eager 下安全；同尺原则：分母也换 eager 重量，不混用第 1 步的 graph 版数字）。

组间门槛（E31 §1 第 3 步，跑前写死）：
  unified_N 的 token |Δlp| mean ≤ 1.5× 上一组 · 签名偏置 ≤ 2×|本底偏置| · 序列 p95 ≤ 2×本底 p95
炸的组划回 bf16 孤岛（照登，不硬闯）。

用法：.venv/bin/python scripts/e31_step3_offline.py           # 全流程（约 20–25 分钟）
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e31_step1_offline import MODEL, MAX_LEN, build_seqs, stats  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "logs/e31/step3_offline.json"
TMP = Path("/workspace/tmp/e31_step3")
GROUPS = [0, 8, 16, 24, 30]          # 0 = G0'（仅 lm_head）
FLAG, LAYERS_FLAG = "SYNCOPATE_UNIFIED_FP8", "SYNCOPATE_UNIFIED_FP8_LAYERS"


def trainer_arm(layers: int, fp8: bool, out_file: Path):
    import torch
    from transformers import AutoModelForCausalLM
    from syncopate.train import unified_fp8
    seqs = json.loads((TMP / "seqs.json").read_text())
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, attn_implementation="flash_attention_2").cuda().eval()
    model.requires_grad_(False)
    if fp8 and layers > 0:
        n = unified_fp8.patch_trainer_inner(model)
        assert n == layers * 7
    W = model.get_output_embeddings().weight
    if fp8:
        qw, qw_sf = unified_fp8._weight_cache(W, "fwd")
    res = []
    with torch.no_grad():
        for s in seqs:
            ids = torch.tensor(s["ids"], device="cuda").unsqueeze(0)
            h = model.model(input_ids=ids).last_hidden_state[0]
            hh = h[s["prompt_len"] - 1: len(s["ids"]) - 1]
            tgt = ids[0, s["prompt_len"]:]
            lg = (unified_fp8._mxf8_logits(hh, qw, qw_sf) if fp8 else hh @ W.T).float()
            res.append(lg.log_softmax(-1).gather(-1, tgt[:, None]).squeeze(1).tolist())
    out_file.write_text(json.dumps(res))


def vllm_arm(out_file: Path):
    from vllm import LLM, SamplingParams
    seqs = json.loads((TMP / "seqs.json").read_text())
    llm = LLM(model=MODEL, dtype="bfloat16", max_model_len=MAX_LEN,
              gpu_memory_utilization=0.72, seed=0, max_num_batched_tokens=2048,
              enforce_eager=True, enable_prefix_caching=False)
    sp = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=0)
    outs = llm.generate([{"prompt_token_ids": s["ids"]} for s in seqs], sp)
    res = []
    for s, o in zip(seqs, outs):
        res.append([o.prompt_logprobs[j][s["ids"][j]].logprob
                    for j in range(s["prompt_len"], len(s["ids"]))])
    out_file.write_text(json.dumps(res))


def _sub(mode: str, out: Path, fp8: bool, layers: int):
    if out.exists():
        return
    env = dict(os.environ)
    env[FLAG] = "1" if fp8 else "0"
    env[LAYERS_FLAG] = str(layers if fp8 else 0)
    subprocess.run([sys.executable, __file__, mode, str(out),
                    "1" if fp8 else "0", str(layers)], env=env, check=True)


def main() -> int:
    if sys.argv[1:] and sys.argv[1] in ("--trainer-arm", "--vllm-arm"):
        out, fp8, layers = Path(sys.argv[2]), sys.argv[3] == "1", int(sys.argv[4])
        (trainer_arm(layers, fp8, out) if sys.argv[1] == "--trainer-arm" else vllm_arm(out))
        return 0

    import torch
    TMP.mkdir(parents=True, exist_ok=True)
    if not (TMP / "seqs.json").exists():
        (TMP / "seqs.json").write_text(json.dumps(build_seqs()))

    print("── 分母臂（bf16, eager）──", flush=True)
    _sub("--trainer-arm", TMP / "tr_bf16.json", False, 0)
    _sub("--vllm-arm", TMP / "v_bf16.json", False, 0)
    for n in GROUPS:
        print(f"── G(N={n}) fp8 两臂 ──", flush=True)
        _sub("--trainer-arm", TMP / f"tr_fp8_{n}.json", True, n)
        _sub("--vllm-arm", TMP / f"v_fp8_{n}.json", True, n)

    load = lambda p: [torch.tensor(x) for x in json.loads((TMP / p).read_text())]
    tb, vb = load("tr_bf16.json"), load("v_bf16.json")
    base = stats([a - b for a, b in zip(vb, tb)])
    report = {"model": MODEL, "groups": GROUPS, "baseline_bf16_eager": base, "unified": {}}
    prev_mean, verdicts = None, {}
    for n in GROUPS:
        u = stats([a - b for a, b in zip(load(f"v_fp8_{n}.json"), load(f"tr_fp8_{n}.json"))])
        report["unified"][str(n)] = u
        ok = (abs(u["token_bias"]) <= 2 * abs(base["token_bias"])
              and u["seq_abs_sum_p95"] <= 2 * base["seq_abs_sum_p95"]
              and (prev_mean is None or u["token_abs_mean"] <= 1.5 * prev_mean))
        verdicts[str(n)] = ok
        prev_mean = u["token_abs_mean"]
    report["verdicts"] = verdicts
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    for n in GROUPS:
        u = report["unified"][str(n)]
        print(f"N={n:2d}  mean {u['token_abs_mean']:.3e}  bias {u['token_bias']:+.2e}  "
              f"p95 {u['seq_abs_sum_p95']:.2f}  {'✅' if verdicts[str(n)] else '🔴'}")
    print(f"分母  mean {base['token_abs_mean']:.3e}  bias {base['token_bias']:+.2e}  "
          f"p95 {base['seq_abs_sum_p95']:.2f}")
    return 0 if all(verdicts.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
