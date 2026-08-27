"""E30 端到端①：MXFP8 kernel 接真实 lm_head 计算，任务口径对拍（速度只作旁证）。

材料：scripts 先用生产 venv 从 v13r2 真模型 dump（/workspace/tmp/e2e_lmhead_dump.pt）：
  8 条真 SFT 验证序列的最终隐状态（过 final norm）· 真 lm_head 权重 [151936,2560] ·
  目标 token · loss_mask（监督位 4.7%）。

三条路径同算 logprob = log_softmax(h @ W^T)[target]：
  ① fp32 真值   ② bf16 现行（torch matmul = 训练器路径的近似）   ③ MXFP8（我们的 kernel）
读数（全部任务口径）：
  逐 token Δlogprob 分布（②③各对①）· 监督位 SFT loss 差 · 序列级 logprob 和的漂移
  （= RL 序列 IS 权重的扰动 exp(Δ)）· top-1 翻转率；速度：③ vs ② 同形状计时。

用法：/workspace/venvs/tilelang/bin/python scripts/tl_mxfp8_e2e_lmhead.py
"""

import sys, time
sys.path.insert(0, 'scripts')

import torch
import torch.nn.functional as F
from tl_mxfp8_gemm import build_kernel, quantize_mxfp8, swizzle_rows

DUMP = '/workspace/tmp/e2e_lmhead_dump.pt'
CHUNK_M = 4096          # kernel 形状（M 维補零到整块）


def main():
    d = torch.load(DUMP)
    W = d['W'].cuda()                              # [V, K] bf16
    V, K = W.shape
    hs = [h.cuda() for h in d['hidden']]
    tgts = [t.cuda() for t in d['targets']]
    masks = [m.cuda().bool() for m in d['loss_masks']]
    H = torch.cat(hs, 0)                           # [T, K] bf16
    T_all = H.shape[0]
    tgt = torch.cat(tgts, 0)
    msk = torch.cat(masks, 0)
    print(f"真实数据：{T_all} token · 监督位 {int(msk.sum())} · 词表 {V} · 隐层 {K}")

    # ── 量化（一次性）：权重按 K 分块 MXFP8；kernel 原生 swizzle 布局 ──
    W_u8, SFW = quantize_mxfp8(W.float())
    W_k = swizzle_rows(W_u8)
    kernel = build_kernel(CHUNK_M, V, K, grid_m_fast=True)

    lp = {p: torch.empty(T_all, dtype=torch.float64, device='cuda') for p in ('fp32', 'bf16', 'mxfp8')}
    top1 = {p: torch.empty(T_all, dtype=torch.long, device='cuda') for p in ('fp32', 'bf16', 'mxfp8')}

    W32t = W.float().T.contiguous()
    for s in range(0, T_all, CHUNK_M):
        e = min(s + CHUNK_M, T_all)
        h = H[s:e]
        t = tgt[s:e]
        # ① fp32 真值
        lg = h.float() @ W32t
        ls = F.log_softmax(lg, -1)
        lp['fp32'][s:e] = ls.gather(1, t[:, None]).squeeze(1).double()
        top1['fp32'][s:e] = lg.argmax(-1)
        # ② bf16 现行
        lg = (h @ W.T).float()
        ls = F.log_softmax(lg, -1)
        lp['bf16'][s:e] = ls.gather(1, t[:, None]).squeeze(1).double()
        top1['bf16'][s:e] = lg.argmax(-1)
        # ③ MXFP8：激活也量化（8bit 训推的完整语义），M 补零到整块
        h_pad = torch.zeros(CHUNK_M, K, device='cuda')
        h_pad[: e - s] = h.float()
        A_u8, SFA = quantize_mxfp8(h_pad)
        C = torch.empty(CHUNK_M, V, device='cuda', dtype=torch.bfloat16)
        kernel(swizzle_rows(A_u8), W_k, SFA, SFW, C)
        lg = C[: e - s].float()
        ls = F.log_softmax(lg, -1)
        lp['mxfp8'][s:e] = ls.gather(1, t[:, None]).squeeze(1).double()
        top1['mxfp8'][s:e] = lg.argmax(-1)
        del lg, ls
        torch.cuda.empty_cache()

    # ── 任务口径报表 ──
    def report(name, sel):
        n = int(sel.sum())
        print(f"\n—— {name}（{n} token）——")
        for p in ('bf16', 'mxfp8'):
            dlt = (lp[p] - lp['fp32'])[sel]
            print(f"  {p:6s} Δlogprob  mean|Δ|={dlt.abs().mean():.3e}  p50={dlt.abs().median():.3e}"
                  f"  p99={dlt.abs().quantile(0.99):.3e}  max={dlt.abs().max():.3e}"
                  f"  偏置(mean Δ)={dlt.mean():+.3e}")
        f_b = (top1['bf16'] != top1['fp32'])[sel].float().mean()
        f_m = (top1['mxfp8'] != top1['fp32'])[sel].float().mean()
        print(f"  top-1 翻转率  bf16 {f_b:.4%} · mxfp8 {f_m:.4%}")

    report("全部位置", torch.ones_like(msk))
    report("监督位（SFT loss 口径）", msk)

    loss = {p: -lp[p][msk].mean() for p in lp}
    print(f"\nSFT loss（监督位 CE）: fp32 {loss['fp32']:.6f} · bf16 {loss['bf16']:.6f}"
          f" · mxfp8 {loss['mxfp8']:.6f}")
    print(f"  Δloss: bf16 {loss['bf16']-loss['fp32']:+.2e} · mxfp8 {loss['mxfp8']-loss['fp32']:+.2e}")

    # 序列级（RL IS 权重口径）：监督位 logprob 和的漂移
    print("\n—— 序列级 logprob 和（RL 序列 IS 的扰动源；逐条）——")
    off = 0
    for i, L in enumerate(d['seqlens']):
        m = msk[off:off + L]
        s_f = lp['fp32'][off:off + L][m].sum()
        s_b = lp['bf16'][off:off + L][m].sum()
        s_m = lp['mxfp8'][off:off + L][m].sum()
        print(f"  seq{i}: 监督位 {int(m.sum()):3d} · Δ(bf16)={s_b-s_f:+.4f}"
              f" · Δ(mxfp8)={s_m-s_f:+.4f} · IS 扰动 exp(Δ)={torch.exp(s_m-s_f):.4f}")
        off += L

    # ── 速度旁证（同形状：全词表投影）──
    h = H[:CHUNK_M].contiguous() if T_all >= CHUNK_M else H
    for _ in range(3): _ = h @ W.T
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(20): _ = h @ W.T
    torch.cuda.synchronize(); t_bf16 = (time.perf_counter() - t0) / 20
    A_u8, SFA = quantize_mxfp8(torch.zeros(CHUNK_M, K, device='cuda'))
    A_k = swizzle_rows(A_u8)
    C = torch.empty(CHUNK_M, V, device='cuda', dtype=torch.bfloat16)
    for _ in range(3): kernel(A_k, W_k, SFA, SFW, C)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(20): kernel(A_k, W_k, SFA, SFW, C)
    torch.cuda.synchronize(); t_mx = (time.perf_counter() - t0) / 20
    fl = 2 * CHUNK_M * V * K
    print(f"\n速度（{CHUNK_M}×{V}×{K}）：bf16 torch {t_bf16*1e3:.2f} ms（{fl/t_bf16/1e12:.0f} TF）"
          f" · MXFP8 kernel {t_mx*1e3:.2f} ms（{fl/t_mx/1e12:.0f} TF）· 加速 {t_bf16/t_mx:.2f}×"
          f"（不含量化开销；量化可融合/预先做）")


if __name__ == '__main__':
    main()
