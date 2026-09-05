#!/usr/bin/env python3
"""E16 §5-1 / A3-① · Triton `tl.dot_scaled` 在 sm_120 上是否静默退化（P2，判据 2026-08-18 写死）。

`tl.dot_scaled` 是 Triton 的块缩放 MMA 入口（MXFP8/MXFP4 的软件路径）：
元素 e4m3/e2m1 + 每 32 元素一个 e8m0 缩放——正是 Blackwell 块缩放 MMA 硬件的格式。
问题：sm_120 上它是真的发块缩放 MMA 指令，还是**反量化回 bf16 走普通 MMA**（=静默退化）。

判据（写死，E16 §1）：
    P2   dot_scaled(e4m3) 与 bf16 tl.dot 基线耗时差 < 3%  ⇒ **静默退化**（不是"恰好一样快"）
         ≥1.5× 于 bf16                                  ⇒ 硬件路径真的接上了
    锚点  torch._scaled_mm（cuBLAS FP8）同形状 TFLOPS —— "这张卡本来能到哪"
    数值  scale 全 1（e8m0=127）下与 fp32 参考对拍，max|rel| < 2e-2 才许读速度
MXFP4（e2m1）臂：能编译就测，编译不过也是结果（sm_120 支持面数据）。
"""
import statistics as st
import torch
import triton
import triton.language as tl

M = N = K = 4096
BM, BN, BK, STAGES, WARPS = 128, 128, 64, 3, 8


@triton.jit
def mm_bf16(A, B, C, M, N, K,
            sam, sak, sbk, sbn, scm, scn,
            BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pm, pn = tl.program_id(0), tl.program_id(1)
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):
        rk = k + tl.arange(0, BK)
        a = tl.load(A + rm[:, None] * sam + rk[None, :] * sak)
        b = tl.load(B + rk[:, None] * sbk + rn[None, :] * sbn)
        acc = tl.dot(a, b, acc)
    tl.store(C + rm[:, None] * scm + rn[None, :] * scn, acc.to(tl.bfloat16))


@triton.jit
def mm_scaled(A, As, B, Bs, C, M, N, K,
              sam, sak, sbk, sbn, scm, scn,
              FMT: tl.constexpr, PACK: tl.constexpr,
              BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    # A [M, K/PACK] 元素张量 · As [M, K/32] e8m0(uint8)；B 同构转置侧
    pm, pn = tl.program_id(0), tl.program_id(1)
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    KE = K // PACK                        # 元素张量的 K 维（fp4 打包后减半）
    BKE: tl.constexpr = BK // PACK
    for k in range(0, KE, BKE):
        rk = k + tl.arange(0, BKE)
        rs = (k * PACK // 32) + tl.arange(0, BK // 32)
        a = tl.load(A + rm[:, None] * sam + rk[None, :] * sak)
        b = tl.load(B + rk[:, None] * sbk + rn[None, :] * sbn)   # rhs 按 [K,N] 存（契约）
        a_s = tl.load(As + rm[:, None] * (K // 32) + rs[None, :])
        b_s = tl.load(Bs + rn[:, None] * (K // 32) + rs[None, :])
        acc = tl.dot_scaled(a, a_s, FMT, b, b_s, FMT, acc)
    tl.store(C + rm[:, None] * scm + rn[None, :] * scn, acc.to(tl.bfloat16))


def bench(fn, iters=30):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    ts = []
    import time
    for _ in range(iters):
        torch.cuda.synchronize(); t = time.time(); fn(); torch.cuda.synchronize()
        ts.append(time.time() - t)
    return st.median(ts)


def tflops(sec):
    return 2 * M * N * K / sec / 1e12


if __name__ == "__main__":
    torch.manual_seed(0)
    dev = "cuda"
    a16 = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
    b16 = torch.randn(K, N, device=dev, dtype=torch.bfloat16)
    c = torch.empty(M, N, device=dev, dtype=torch.bfloat16)
    grid = (M // BM, N // BN)

    # ---- bf16 基线（同块配置，P2 的对照臂）----
    run16 = lambda: mm_bf16[grid](a16, b16, c, M, N, K,
                                  *a16.stride(), *b16.stride(), *c.stride(),
                                  BM=BM, BN=BN, BK=BK, num_stages=STAGES, num_warps=WARPS)
    t16 = bench(run16)
    print(f"bf16 tl.dot           {tflops(t16):7.1f} TFLOPS   ({t16 * 1e3:.2f} ms)")

    # ---- 锚点：cuBLAS FP8（这张卡"本来能到"）----
    fmax = torch.finfo(torch.float8_e4m3fn).max
    a8 = (a16 / (a16.abs().max() / fmax)).to(torch.float8_e4m3fn)
    b8t = (b16.t().contiguous() / (b16.abs().max() / fmax)).to(torch.float8_e4m3fn)
    one = torch.ones((), device=dev)
    run_mm = lambda: torch._scaled_mm(a8, b8t.t(), scale_a=one, scale_b=one,
                                      out_dtype=torch.bfloat16)
    tmm = bench(run_mm)
    print(f"torch._scaled_mm fp8  {tflops(tmm):7.1f} TFLOPS   （锚点=cuBLAS）")

    # ---- 被告：tl.dot_scaled e4m3 + e8m0 全 1 缩放 ----
    a8u = a8                                     # e4m3 元素
    b8u = b8t.t().contiguous()                   # [K, N] e4m3（契约布局）
    a_sc = torch.full((M, K // 32), 127, device=dev, dtype=torch.uint8)   # e8m0 的 1.0
    b_sc = torch.full((N, K // 32), 127, device=dev, dtype=torch.uint8)
    run_sc = lambda: mm_scaled[grid](a8u, a_sc, b8u, b_sc, c, M, N, K,
                                     *a8u.stride(), *b8u.stride(), *c.stride(),
                                     FMT="e4m3", PACK=1,
                                     BM=BM, BN=BN, BK=BK, num_stages=STAGES, num_warps=WARPS)
    # 数值判据先行：scale=1 下应等于 fp8 值域内的精确乘加（对 fp32 参考）
    run_sc()
    ref = a8.to(torch.float32) @ b8t.t().to(torch.float32)
    # 分母用矩阵量级：逐元素相对误差会被相消小值放大（bf16 存储舍入），不是错
    rel = ((c.to(torch.float32) - ref).abs().max() / ref.abs().max()).item()
    print(f"数值对拍 max|Δ|/‖ref‖∞ = {rel:.3e}  （>1e-2 ⇒ 结果不可读）")
    assert rel < 1e-2, "数值不过，速度免谈"
    tsc = bench(run_sc)
    r_vs_bf16 = t16 / tsc
    print(f"tl.dot_scaled e4m3    {tflops(tsc):7.1f} TFLOPS   vs bf16 = {r_vs_bf16:.2f}×")
    if abs(tsc - t16) / t16 < 0.03:
        print("VERDICT P2：⛔ **静默退化实锤**——与 bf16 基线差 <3%（反量化走普通 MMA）")
    elif r_vs_bf16 >= 1.5:
        print("VERDICT P2：✅ 硬件块缩放路径接上了（≥1.5×）")
    else:
        print(f"VERDICT P2：🟡 介于两者（{r_vs_bf16:.2f}×）——部分兑现，读 PTX 定性")

    # ---- MXFP4（e2m1，2 元素/字节打包）：能编译就测 ----
    try:
        a4 = torch.randint(0, 256, (M, K // 2), device=dev, dtype=torch.uint8)
        b4 = torch.randint(0, 256, (K // 2, N), device=dev, dtype=torch.uint8)
        run_f4 = lambda: mm_scaled[grid](a4, a_sc, b4, b_sc, c, M, N, K,
                                         *a4.stride(), *b4.stride(), *c.stride(),
                                         FMT="e2m1", PACK=2,
                                         BM=BM, BN=BN, BK=BK, num_stages=STAGES, num_warps=WARPS)
        run_f4()
        tf4 = bench(run_f4)
        print(f"tl.dot_scaled e2m1    {tflops(tf4):7.1f} TFLOPS   vs bf16 = {t16 / tf4:.2f}×（速度臂，数值未验）")
    except Exception as e:
        print(f"tl.dot_scaled e2m1    ❌ 编译/运行失败：{type(e).__name__}: {str(e)[:160]}")
