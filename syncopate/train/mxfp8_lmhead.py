"""A4 · lm_head 的 MXFP8 训练三件套（前向 + dgrad + wgrad），autograd 封装。

kernel = 裸 CUDA T1（627 TFLOPS，E30 §10），经 torch 扩展 JIT 编译进生产 venv
（自有源码、零第三方依赖，不触碰依赖树——守则⑧合规）。
量化标准 = OCP MXFP8（块 32 · ue8m0 幂缩放 · e4m3），与 E30 §9-§12 的全部对拍同源。

用法（SFT 稀疏投影，lm_head 冻结 ⇒ 权重量化整训缓存、wgrad 自动跳过）：
    SYNCOPATE_LMHEAD_MXFP8=1 python -m syncopate.train.sft ...
判据行：首次调用打 `[mxfp8-lmhead] 已启用 ...`；没这行 = 没接上。
"""

from __future__ import annotations

import os

import torch

_EXT = None


def _ext():
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load
        _EXT = load(
            name="mxf8_gemm_ext",
            sources=[os.path.join(os.path.dirname(__file__), "csrc", "mxf8_gemm_ext.cu")],
            extra_cuda_cflags=["-O3", "-arch=sm_120a"],
            extra_ldflags=["-lcuda"],
            verbose=False,
        )
    return _EXT


def quantize_mxfp8(x: torch.Tensor):
    """[R, K] → (u8 e4m3, ue8m0 缩放)。与 scripts/tl_mxfp8_gemm.py 同一实现。"""
    R, K = x.shape
    assert K % 32 == 0
    xb = x.float().view(R, K // 32, 32)
    amax = xb.abs().amax(dim=-1)
    e = torch.clamp(torch.ceil(torch.log2(amax / 448.0)), -127, 127)
    e = torch.where(amax == 0, torch.zeros_like(e), e)
    scale = torch.pow(2.0, e)
    q = (xb / scale.unsqueeze(-1)).to(torch.float8_e4m3fn)
    return q.view(torch.uint8).view(R, K).contiguous(), (e + 127).to(torch.uint8).contiguous()


def swizzle_rows(u8: torch.Tensor) -> torch.Tensor:
    """kernel 原生布局：行内 128B 组 16B 块 chunk^=(row&7) 置换。"""
    R, K = u8.shape
    assert K % 128 == 0
    c = u8.view(R, K // 16, 16)
    chunk = torch.arange(K // 16, device=u8.device)
    row = torch.arange(R, device=u8.device)
    perm = (chunk.unsqueeze(0) & ~7) | ((chunk.unsqueeze(0) & 7) ^ (row.unsqueeze(1) & 7))
    return torch.gather(c, 1, perm.unsqueeze(-1).expand(R, K // 16, 16)).reshape(R, K).contiguous()


def _pad128(x: torch.Tensor) -> torch.Tensor:
    n = x.shape[0]
    npad = (n + 127) // 128 * 128
    if npad == n:
        return x
    out = torch.zeros(npad, x.shape[1], device=x.device, dtype=x.dtype)
    out[:n] = x
    return out


def _quant_sw(x: torch.Tensor):
    u8, sf = quantize_mxfp8(_pad128(x))
    return swizzle_rows(u8), sf


def _weight_cache(W: torch.Tensor):
    """冻结权重的量化缓存（前向用 W[V,K] + dgrad 用 Wt[K,V]），挂在张量属性上。"""
    cache = getattr(W, "_mxf8_cache", None)
    if cache is None:
        with torch.no_grad():
            qw, qw_sf = _quant_sw(W.detach())
            qwt, qwt_sf = _quant_sw(W.detach().T.contiguous())
        cache = (qw, qw_sf, qwt, qwt_sf)
        W._mxf8_cache = cache
    return cache


class _MXF8LMHead(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h: torch.Tensor, W: torch.Tensor):
        N = h.shape[0]
        qh, qh_sf = _quant_sw(h)
        qw, qw_sf, qwt, qwt_sf = _weight_cache(W)
        logits = _ext().mxf8_gemm(qh, qw, qh_sf, qw_sf)[:N]     # [N, V] bf16
        ctx.save_for_backward(h, W)
        return logits

    @staticmethod
    def backward(ctx, dlogits: torch.Tensor):
        h, W = ctx.saved_tensors
        N = h.shape[0]
        qdz, qdz_sf = _quant_sw(dlogits)                        # 块沿 V（dgrad 收缩维）
        _, _, qwt, qwt_sf = _weight_cache(W)
        dh = _ext().mxf8_gemm(qdz, qwt, qdz_sf, qwt_sf)[:N].to(h.dtype)   # [N, K]
        dW = None
        if W.requires_grad:                                     # SFT 冻结 lm_head ⇒ 跳过
            qdzt, qdzt_sf = _quant_sw(dlogits.T.contiguous())   # [V, N] 块沿 N
            qht, qht_sf = _quant_sw(h.detach().T.contiguous())  # [K, N]
            dW = _ext().mxf8_gemm(qdzt, qht, qdzt_sf, qht_sf).to(W.dtype)
        return dh, dW


_ANNOUNCED = False


def mxf8_lm_head(h: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    """输入 [N, 2560] 任意 dtype，输出 bf16 logits [N, 151936]。"""
    global _ANNOUNCED
    if not _ANNOUNCED:
        _ANNOUNCED = True
        print(f"[mxfp8-lmhead] 已启用 · W={tuple(W.shape)} 量化缓存 · "
              f"wgrad={'开' if W.requires_grad else '关(冻结)'}", flush=True)
    return _MXF8LMHead.apply(h, W)


def enabled() -> bool:
    return os.environ.get("SYNCOPATE_LMHEAD_MXFP8") == "1"
