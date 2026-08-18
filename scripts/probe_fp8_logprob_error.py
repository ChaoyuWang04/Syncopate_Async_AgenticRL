#!/usr/bin/env python3
"""E19 §4-1 · FP8 用在 `ref` 那一遍前向上，**数值上付得起吗**。

★ 为什么问的是"KL 的相对误差"而不是"logits 的误差"：
`ref` 的输出唯一的去处是 KL 项，而它前面还乘着 `kl_loss_coef=0.001`。
⇒ 该看的是**最终那个标量**偏了多少，不是中间张量偏了多少。

做法（用**真实的 lm_head 权重和真实形状**，不是随机矩阵）：
    1. 造一批 hidden state（模拟一层输出）
    2. bf16 路径：logits = h @ W → log_softmax → 取 token 的 logprob   ← 基准
    3. fp8  路径：同样的算，但 GEMM 走 torch._scaled_mm（per-tensor scale）
    4. 比较：逐 token logprob 的绝对/相对误差，以及**由它们算出来的 KL 的相对误差**

★ 判据（跑之前写死，E19 §4-1）：**KL 的相对误差 < 5%** ⇒ ref 可以走 FP8。
  ⚠️ 若 > 5%：不是"FP8 不能用"，是"per-tensor scale 不够，要 per-block" —— 那是另一件事。
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch

def load_lm_head(model_dir: Path, hidden: int, vocab: int) -> torch.Tensor:
    """只从 safetensors 里取 lm_head 那一个张量（别加载整个模型）。"""
    from safetensors import safe_open
    idx = json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]
    key = next((k for k in idx if "lm_head" in k), None)
    if key is None:                       # 权重绑定（tie_word_embeddings）时用 embedding
        key = next(k for k in idx if "embed_tokens" in k)
    with safe_open(model_dir / idx[key], framework="pt", device="cpu") as f:
        w = f.get_tensor(key)
    print(f"  取到 {key}  shape={tuple(w.shape)}  dtype={w.dtype}")
    return w

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=Path("models/Qwen3-4B-sft-v13-e1"))
    ap.add_argument("--tokens", type=int, default=4096)
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    dev = torch.device("cuda")

    W = load_lm_head(a.model, 2560, 151936).to(dev, torch.bfloat16)      # [vocab, hidden]
    T, H = a.tokens, W.shape[1]
    torch.manual_seed(0)
    # hidden state 的尺度按真实 RMSNorm 之后的量级给（~1），不要用 randn*10 这种假分布
    h = torch.randn(T, H, device=dev, dtype=torch.bfloat16)
    tok = torch.randint(0, W.shape[0], (T,), device=dev)

    def logprobs(logits):
        return torch.log_softmax(logits.float(), dim=-1).gather(1, tok[:, None]).squeeze(1)

    # ① bf16 基准
    lp_ref = logprobs(h @ W.t())

    # ② FP8（per-tensor scale：把最大绝对值映射到 e4m3 的动态范围）
    fmax = torch.finfo(torch.float8_e4m3fn).max
    sh = (h.abs().max() / fmax).clamp(min=1e-12)
    sw = (W.abs().max() / fmax).clamp(min=1e-12)
    hf = (h / sh).to(torch.float8_e4m3fn)
    wf = (W / sw).to(torch.float8_e4m3fn)
    logits_fp8 = torch._scaled_mm(hf, wf.t().contiguous().t().t().contiguous(),
                                  scale_a=sh.float(), scale_b=sw.float(),
                                  out_dtype=torch.bfloat16)
    lp_fp8 = logprobs(logits_fp8)

    d = (lp_fp8 - lp_ref)
    # KL 的代理：verl 的 k3 估计量 exp(d) - d - 1（低方差、非负），对整批取平均
    kl_ref = torch.zeros(1, device=dev)          # 自己对自己 KL=0，所以这里量的是"引入的偏差"
    kl_bias = (torch.exp(d) - d - 1).mean()
    out = {
        "tokens": T,
        "logprob_abs_err_mean": d.abs().mean().item(),
        "logprob_abs_err_p99": d.abs().float().quantile(0.99).item(),
        "logprob_rel_err_mean": (d.abs() / lp_ref.abs().clamp(min=1e-6)).mean().item(),
        "kl_k3_bias_introduced": kl_bias.item(),
    }
    print(f"\n  逐 token logprob 误差   平均 {out['logprob_abs_err_mean']:.5f} · "
          f"p99 {out['logprob_abs_err_p99']:.5f} · 相对 {out['logprob_rel_err_mean']*100:.3f}%")
    print(f"  ★ FP8 引入的 KL 偏差（k3 估计量）= {out['kl_k3_bias_introduced']:.6f}")
    print(f"    对照：真实训练里 kl_loss 的量级是 0.007–0.028（E17 §4.1）")
    ratio = out["kl_k3_bias_introduced"] / 0.02
    out["bias_vs_real_kl"] = ratio
    out["verdict"] = ("✅ 判据通过：引入的偏差 < 真实 KL 的 5%" if ratio < 0.05 else
                      f"🔴 偏差是真实 KL 的 {ratio*100:.1f}% ⇒ per-tensor scale 不够，要 per-block")
    print(f"  ⇒ {out['verdict']}")
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    return 0

if __name__ == "__main__": raise SystemExit(main())
