"""E30 §12 · 反向两 GEMM（dgrad/wgrad）的 MXFP8 探针——复用前向 kernel，换收缩维。

lm_head 训练一层 = 三个 GEMM：
  前向   Z[M,V]  = H[M,K] · W[V,K]ᵀ          （已完成，§4-§10）
  dgrad  dH[M,K] = dZ[M,V] · Wt[K,V]ᵀ        收缩维 = V（15 万，超深）
  wgrad  dW[V,K] = dZt[V,M] · Ht[K,M]ᵀ       收缩维 = M（token 数，浅）
两个反向都是"操作数转置 + 沿收缩维 MXFP8 量化"后调用同一个 kernel。
dZ = softmax(Z) − onehot（CE 反向），动态范围极端——梯度量化是 FP8 训练的经典难点，
本探针用真 RL 数据（28,668 token）量它的真实代价。

读数：dH/dW 对 fp32 真值的 余弦相似度 + 相对误差（bf16 路径作本底），
     以及两个反向形状的 kernel 速度 vs torch bf16。
用法：/workspace/venvs/tilelang/bin/python scripts/infra/tl_mxfp8_backward_probe.py
"""

import time

import torch
import torch.nn.functional as F
from syncopate.train.tilelang_mxfp8 import build_kernel, quantize_mxfp8, swizzle_rows

CH = 2048


def cos_rel(x, ref, chunk=8192):
    """大矩阵余弦/相对误差：分块 fp64 累加，不整体升精度。"""
    xy = xx = yy = dd = 0.0
    for s0 in range(0, x.shape[0], chunk):
        a = x[s0:s0+chunk].double(); b = ref[s0:s0+chunk].double()
        xy += (a*b).sum().item(); xx += (a*a).sum().item(); yy += (b*b).sum().item()
        dd += ((a-b)**2).sum().item()
    import math
    return xy/math.sqrt(xx*yy), math.sqrt(dd/yy)


def main():
    d = torch.load('/workspace/tmp/e2e_rl_dump.pt')
    W = d['W'].cuda().float()                      # [V, K]
    V, K = W.shape
    H = torch.cat([h.cuda() for h in d['hidden']], 0).float()
    tgt = torch.cat([t.cuda() for t in d['targets']], 0)
    T = H.shape[0]
    Tp = (T + 127) // 128 * 128                    # pad 到 128 倍数
    print(f"真数据：{T} token（pad {Tp}）· V={V} · K={K}")

    Hp = torch.zeros(Tp, K, device='cuda'); Hp[:T] = H

    # ── 三种精度的 dZ 与反向真值（fp32 分块算，dZ 顺手量化/转置进各缓冲）──
    dH = {p: torch.zeros(Tp, K, device='cuda', dtype=torch.float32) for p in ('fp32', 'bf16')}
    dW = {p: torch.zeros(V, K, device='cuda', dtype=torch.float32) for p in ('fp32', 'bf16')}
    # mxfp8 反向的量化操作数（u8 大缓冲，按 M 块填充）
    qdZ_u8 = torch.zeros(Tp, V, device='cuda', dtype=torch.uint8)        # dgrad A：[M,V]
    qdZ_sf = torch.zeros(Tp, V // 32, device='cuda', dtype=torch.uint8)
    qdZt_u8 = torch.zeros(V, Tp, device='cuda', dtype=torch.uint8)       # wgrad A：[V,M]
    qdZt_sf = torch.zeros(V, Tp // 32, device='cuda', dtype=torch.uint8)

    Wb = W.to(torch.bfloat16)
    for s in range(0, Tp, CH):
        e = min(s + CH, Tp)
        h = Hp[s:e]
        z = h @ W.T
        dz = F.softmax(z, -1)
        del z
        if s < T:
            te = min(e, T)
            dz[: te - s].scatter_add_(1, tgt[s:te][:, None],
                                      -torch.ones(te - s, 1, device='cuda'))
        if e > T:                                   # pad 行梯度置零
            dz[T - s if s < T else 0:] = 0
        # fp32 真值
        dH['fp32'][s:e] = dz @ W
        dW['fp32'] += dz.T @ h
        # bf16 本底
        dzb = dz.to(torch.bfloat16)
        dH['bf16'][s:e] = (dzb @ Wb).float()
        dW['bf16'] += (dzb.T @ h.to(torch.bfloat16)).float()
        # mxfp8 操作数：dgrad 沿 V 量化；wgrad 沿 M 量化（转置后行=词表）
        u8, sf = quantize_mxfp8(dz)                 # [rows, V] 块沿 V ✓
        qdZ_u8[s:e], qdZ_sf[s:e] = u8, sf
        u8t, sft = quantize_mxfp8(dz.T.contiguous())  # [V, rows] 块沿 M ✓（rows%32==0）
        qdZt_u8[:, s:e] = u8t
        qdZt_sf[:, s // 32 : e // 32] = sft
        del dz, dzb, u8, sf, u8t, sft
        torch.cuda.empty_cache()

    # ── mxfp8 dgrad：dH = qdZ[M,V] @ (qWt[K,V])ᵀ ──
    Wt = W.T.contiguous()                           # [K, V]
    qWt_u8, qWt_sf = quantize_mxfp8(Wt)
    dgrad_k = build_kernel(Tp, K, V)
    C1 = torch.empty(Tp, K, device='cuda', dtype=torch.bfloat16)
    A1 = swizzle_rows(qdZ_u8); B1 = swizzle_rows(qWt_u8)
    dgrad_k(A1, B1, qdZ_sf, qWt_sf, C1)
    c, r = cos_rel(C1.float(), dH['fp32'])
    cb, rb = cos_rel(dH['bf16'], dH['fp32'])
    print(f"\ndgrad dH（收缩维 V=151936）: bf16 cos={cb:.6f} rel={rb:.3e} | mxfp8 cos={c:.6f} rel={r:.3e}")

    # 计时（形状 [Tp, 2560, 151936]）
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(10): dgrad_k(A1, B1, qdZ_sf, qWt_sf, C1)
    torch.cuda.synchronize(); t_mx = (time.perf_counter() - t0) / 10
    dZb = qdZ_u8.view(torch.float8_e4m3fn).to(torch.bfloat16)   # 仅计时用途的同形状 bf16
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(10): _ = dZb @ Wb
    torch.cuda.synchronize(); t_bf = (time.perf_counter() - t0) / 10
    fl = 2.0 * Tp * K * V
    print(f"  速度：bf16 {t_bf*1e3:.1f} ms（{fl/t_bf/1e12:.0f} TF）· mxfp8 {t_mx*1e3:.1f} ms（{fl/t_mx/1e12:.0f} TF）· {t_bf/t_mx:.2f}×")
    del A1, B1, C1, dZb, qdZ_u8, qdZ_sf, Wt, qWt_u8, qWt_sf
    torch.cuda.empty_cache()

    # ── mxfp8 wgrad：dW = qdZt[V,M] @ (qHt[K,M])ᵀ ──
    Ht = Hp.T.contiguous()                          # [K, M]
    qHt_u8, qHt_sf = quantize_mxfp8(Ht)
    wgrad_k = build_kernel(V, K, Tp, grid_m_fast=True)
    C2 = torch.empty(V, K, device='cuda', dtype=torch.bfloat16)
    A2 = swizzle_rows(qdZt_u8); B2 = swizzle_rows(qHt_u8)
    wgrad_k(A2, B2, qdZt_sf, qHt_sf, C2)
    c, r = cos_rel(C2.float(), dW['fp32'])
    cb, rb = cos_rel(dW['bf16'], dW['fp32'])
    print(f"\nwgrad dW（收缩维 M={Tp}）: bf16 cos={cb:.6f} rel={rb:.3e} | mxfp8 cos={c:.6f} rel={r:.3e}")

    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(10): wgrad_k(A2, B2, qdZt_sf, qHt_sf, C2)
    torch.cuda.synchronize(); t_mx = (time.perf_counter() - t0) / 10
    del A2, B2, C2, Ht, qHt_u8
    torch.cuda.empty_cache()
    dZtb = qdZt_u8.view(torch.float8_e4m3fn).to(torch.bfloat16)
    Hb = Hp.to(torch.bfloat16)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(10): _ = dZtb @ Hb
    torch.cuda.synchronize(); t_bf = (time.perf_counter() - t0) / 10
    fl = 2.0 * V * K * Tp
    print(f"  速度：bf16 {t_bf*1e3:.1f} ms（{fl/t_bf/1e12:.0f} TF）· mxfp8 {t_mx*1e3:.1f} ms（{fl/t_mx/1e12:.0f} TF）· {t_bf/t_mx:.2f}×")


if __name__ == '__main__':
    main()
