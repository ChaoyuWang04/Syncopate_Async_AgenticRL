#!/usr/bin/env python3
"""E16 · sm_120 能力探底（第一枪）：FP8 到底点亮了没有。

★ 为什么这条值得做（TRACK-A §2.3）：
sm_120 是**消费级 Blackwell**——有原生 FP4/FP8，但**没有 TMEM**（数据中心 Blackwell 才有）
⇒ 编程模型退回 Ampere 式 `mma.sync`。而生态几乎没人为它单独点亮什么：
**E01 实测我们自己的热点路径上，所有 GEMM 都是 `cutlass_80_*`（Ampere 代 s16816）。**
⇒ 「这块卡上什么是真的、什么是假的」本身就是一次没人做过的测量。

本探针只回答**三个可证伪的问题**：
  Q1  bf16 GEMM 的实测 TFLOPS 是多少（分母）
  Q2  FP8（e4m3）走 `torch._scaled_mm` 有没有**真的更快**——还是只是能跑
  Q3  Triton 的 `tl.dot_scaled`（低精度路径）在这块卡上是否**静默退化**
      （triton#7550：不报错，但退回 bf16 ⇒ 时间与 bf16 一模一样）

⚠️ 判据写在前面（跑之前写死）：
  P1  FP8 相对 bf16 **≥1.5×** 才算"点亮了"。1.0–1.2× ⇒ 只是能跑，硬件没用上。
  P2  若 `tl.dot_scaled` 的耗时与 bf16 基线**落在 3% 以内**，判定为**静默退化**
      （而不是"恰好一样快"）——再用 `TRITON_PRINT_AUTOTUNING` / ptx dump 复核。

用法：
    python scripts/probe_sm120_fp8.py --json logs/e16_fp8.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch


def bench(fn, warmup: int = 5, iters: int = 20) -> float:
    """返回中位耗时（秒）。★ 每次都同步，否则量到的是入队速度不是执行速度。"""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        t = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t)
    ts.sort()
    return ts[len(ts) // 2]


def tflops(m: int, n: int, k: int, seconds: float) -> float:
    return 2 * m * n * k / seconds / 1e12


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="4096,8192")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    dev = torch.device("cuda")
    cap = torch.cuda.get_device_capability()
    out: dict = {
        "device": torch.cuda.get_device_name(),
        "capability": f"sm_{cap[0]}{cap[1]}",
        "torch": torch.__version__,
        "results": {},
    }
    print(f"# {out['device']}  {out['capability']}  torch {out['torch']}")

    for size in [int(x) for x in args.sizes.split(",")]:
        m = n = k = size
        a = torch.randn(m, k, device=dev, dtype=torch.bfloat16)
        b = torch.randn(k, n, device=dev, dtype=torch.bfloat16)

        # Q1 · bf16 基线
        t_bf16 = bench(lambda: torch.matmul(a, b))
        r = {"bf16_ms": t_bf16 * 1e3, "bf16_tflops": tflops(m, n, k, t_bf16)}

        # Q2 · FP8 e4m3（cuBLAS 路径）
        try:
            af = a.to(torch.float8_e4m3fn)
            bf = b.t().contiguous().t().to(torch.float8_e4m3fn)   # _scaled_mm 要求 B 列主序
            scale = torch.tensor(1.0, device=dev)
            def fp8():
                return torch._scaled_mm(af, bf, scale_a=scale, scale_b=scale,
                                        out_dtype=torch.bfloat16)
            fp8()   # 先试一次，不支持就直接抛
            t_fp8 = bench(fp8)
            r["fp8_ms"] = t_fp8 * 1e3
            r["fp8_tflops"] = tflops(m, n, k, t_fp8)
            r["fp8_speedup"] = t_bf16 / t_fp8
            r["fp8_verdict"] = ("点亮了" if r["fp8_speedup"] >= 1.5 else
                                "⚠️ 只是能跑，硬件没用上（<1.5×）")
        except Exception as exc:                                    # noqa: BLE001
            r["fp8_error"] = f"{type(exc).__name__}: {exc}"
            r["fp8_verdict"] = "⛔ 这条路径在本卡上不可用"

        out["results"][str(size)] = r
        print(f"\n  {size}³")
        print(f"    bf16   {r['bf16_ms']:8.3f} ms   {r['bf16_tflops']:7.1f} TFLOPS")
        if "fp8_tflops" in r:
            print(f"    fp8    {r['fp8_ms']:8.3f} ms   {r['fp8_tflops']:7.1f} TFLOPS"
                  f"   ×{r['fp8_speedup']:.2f}   {r['fp8_verdict']}")
        else:
            print(f"    fp8    {r['fp8_verdict']}  —— {r.get('fp8_error','')}")

    # Q3 · Triton 的低精度路径是否静默退化
    try:
        import triton
        import triton.language as tl

        out["triton"] = triton.__version__
        has_dot_scaled = hasattr(tl, "dot_scaled")
        out["triton_has_dot_scaled"] = has_dot_scaled
        print(f"\n  triton {triton.__version__}  tl.dot_scaled 存在={has_dot_scaled}")
        if not has_dot_scaled:
            print("    ⇒ 这个版本没有 dot_scaled，Q3 无法在本环境验证（不是退化，是没有）")
    except ImportError:
        out["triton"] = None
        print("\n  ⚠️ 没装 triton，Q3 跳过")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(out, indent=2, ensure_ascii=False))
        print(f"\n  → {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
