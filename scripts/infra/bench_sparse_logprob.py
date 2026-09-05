"""E11 微基准：稀疏 logprob 到底能省多少时间。

★ 预测（2026-08-14 跑之前写死，不许事后改）——填 TRACK-A 的 P-A3 那一格：

  H1  lm_head+CE 在 16384 token 打包下约占「整个前向」的 **8–12%**
      依据：lm_head 参数 2560×151936=389M，模型总参 ~4.02B ⇒ FLOPs 占比 ~9.7%
  H2  按 mask 稀疏化（4.17%）后，kernel 本身的时间**近似线性下降**（省 ~20×）
      若不线性（例如只省 3–5×），说明 kernel 在小 N 下被固定开销/权重读带宽卡住
      —— 那才是真正有意思的发现：**权重矩阵 778 MB 的读取是与 N 无关的固定成本**
  H3  端到端（一次前向）收益 ≈ H1 × 95.8% ≈ **8–11%**；切 prompt 版 ≈ 8–10%
      ⇒ 若 H2 不线性，H3 会明显小于此

  如果我错了会怎样：
  - H2 若严重不线性 ⇒ **E11 自写 kernel 的意义反而变大**（可以只读一次权重、
    或换成 gather 后的小 GEMM），因为 verl 那版的固定成本正是可攻击的目标
  - H1 若远小于 8% ⇒ E11 整条线降级，如实写进报告

用法：
    .venv/bin/python scripts/infra/bench_sparse_logprob.py            # 只测 kernel
    .venv/bin/python scripts/infra/bench_sparse_logprob.py --model    # 加测真实前向做分母
"""

from __future__ import annotations

import argparse
import statistics as st
import time

import torch

HIDDEN = 2560
VOCAB = 151936
PACKED = 16384          # --max-token-len-per-gpu 的实际值
DENSITY_MASK = 0.0417   # E11 实测：助手 token 占比
DENSITY_PROMPTCUT = 0.117  # 只切 prompt（response 占比）


def timeit(fn, reps: int = 5, warmup: int = 2) -> float:
    """返回中位耗时（毫秒）。项目规矩：跑多遍取中位。"""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    xs = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        xs.append((time.perf_counter() - t0) * 1e3)
    return st.median(xs)


def bench_kernel(n: int, weight: torch.Tensor, backward: bool) -> float:
    from verl.utils.kernel.linear_cross_entropy import linear_cross_entropy

    hidden = torch.randn(n, HIDDEN, dtype=torch.bfloat16, device="cuda",
                         requires_grad=backward)
    labels = torch.randint(0, VOCAB, (n,), device="cuda")
    w = weight.detach().clone().requires_grad_(backward)

    # ⚠️ 反向必须传**显式的连续梯度**：`.sum().backward()` 的梯度是 expand 出来的，
    #    非连续，会撞上 kernels.py:1553 的 `assert dlogprobs.is_contiguous()`。
    def run():
        logprobs, entropy = linear_cross_entropy(hidden, w, labels, 1.0, "none")
        if backward:
            torch.autograd.backward(
                [logprobs, entropy],
                [torch.ones_like(logprobs).contiguous(), torch.ones_like(entropy).contiguous()],
            )

    return timeit(run)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="store_true", help="加测真实模型前向做分母")
    ap.add_argument("--packed", type=int, default=PACKED)
    args = ap.parse_args()

    torch.cuda.init()
    dev = torch.cuda.get_device_name(0)
    print(f"设备 {dev} / torch {torch.__version__}")
    print(f"形状 hidden={HIDDEN} vocab={VOCAB} packed={args.packed}")
    print(f"权重矩阵 {VOCAB * HIDDEN * 2 / 1e9:.3f} GB (bf16)\n")

    weight = torch.randn(VOCAB, HIDDEN, dtype=torch.bfloat16, device="cuda")

    rows = [
        ("稠密（verl 现状）", args.packed),
        ("切 prompt (11.7%)", max(1, int(args.packed * DENSITY_PROMPTCUT))),
        ("按 mask 筛 (4.17%)", max(1, int(args.packed * DENSITY_MASK))),
    ]

    for backward in (False, True):
        tag = "前向+反向 (update_actor 用)" if backward else "仅前向 (old_log_prob / ref 用)"
        print(f"=== {tag} ===")
        base = None
        print(f"{'配置':<22}{'N':>7}{'耗时(ms)':>11}{'相对稠密':>10}{'理想线性':>10}")
        for name, n in rows:
            ms = bench_kernel(n, weight, backward)
            if base is None:
                base = ms
            print(f"{name:<22}{n:>7}{ms:>11.2f}{base / ms:>9.2f}×{args.packed / n:>9.1f}×")
        print()

    if args.model:
        bench_model_forward(args.packed)
    return 0


def bench_model_forward(packed: int) -> None:
    """真实 Qwen3-4B 前向，给 lm_head 占比一个实测分母。"""
    from transformers import AutoModelForCausalLM

    print("=== 真实模型前向（分母）===")
    print("加载 Qwen3-4B ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        "models/Qwen3-4B-sft-v11-e1", dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).cuda().eval()

    ids = torch.randint(0, VOCAB, (1, packed), device="cuda")
    pos = torch.arange(packed, device="cuda").unsqueeze(0)

    def body_only():
        with torch.no_grad():
            model.model(input_ids=ids, position_ids=pos)

    def full():
        with torch.no_grad():
            model(input_ids=ids, position_ids=pos)

    t_body = timeit(body_only, reps=3, warmup=1)
    t_full = timeit(full, reps=3, warmup=1)
    print(f"36 层主体（不含 lm_head）  {t_body:>9.2f} ms")
    print(f"整个前向（含 lm_head）    {t_full:>9.2f} ms")
    print(f"⇒ lm_head 占前向          {(t_full - t_body) / t_full * 100:>9.2f}%")


if __name__ == "__main__":
    raise SystemExit(main())
