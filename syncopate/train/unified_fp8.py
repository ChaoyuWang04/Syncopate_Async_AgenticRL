"""E31 第 1/2 步 · 训推 lm_head 统一 MXFP8（一个开关同时切两侧）。

原理（E31 §0）：单侧量化 = 温度偏置 × 长度复利 ⇒ 序列 IS 必死（E30 §9b/§11）；
两侧用**同一份量化器 + 同一份 kernel**，量化项在 IS 比率中对消，只剩引擎本底。

两个接缝，一个开关 `SYNCOPATE_UNIFIED_FP8=1`（env 会被 spawn 的子进程继承）：
  rollout 侧  vLLM `LogitsProcessor._get_logits` —— 经 `vllm.general_plugins` 入口点注册
              （pyproject 已登记；vLLM 的每个进程含 Worker 子进程都会调 register()）
  trainer 侧  `verl_patches._pg_forward` 的投影调用点（olp / update_actor / entropy 共用）
              换成本文件的 `linear_for_ppo`（内部按开关分派；关 = 原样走 verl 融合算子）

判据行（缺任何一条 = 机制没接上，直接判负）：
  [unified-fp8] vLLM lm_head MXFP8 已生效 ...
  [unified-fp8] trainer lm_head MXFP8 已生效 ...

语义契约（与 verl `FusedLinearForPPO` 逐项对齐，tests/train/test_e31_step1.py 钉）：
  logits = (h @ Wᵀ) / temperature → fp32；log_probs = log_softmax.gather（fp32 输出）；
  entropy = logsumexp − Σ p·logits（回 h 的 dtype）。差异仅在 GEMM 本身走 MXFP8。
  entropy_coeff=0 是钉死的配置（launch_rl）⇒ backward 收到非空 dentropy 即 RuntimeError。
  lm_head 权重冻结（LoRA 不碰、tie 到 embedding）⇒ 量化一次整训缓存、无 wgrad。
"""

from __future__ import annotations

import os

import torch

FLAG = "SYNCOPATE_UNIFIED_FP8"


def enabled() -> bool:
    return os.environ.get(FLAG) == "1"


# ───────────────────────── 共用：权重量化缓存 ─────────────────────────
# 不挂张量属性（FSDP 的 .weight 是 flat_param 视图，可能每步重建）；
# 按 (data_ptr, shape) 记账 —— 权重冻结不变，同 storage 的缓存永远有效。

_WCACHE: dict = {}


def _weight_cache(W: torch.Tensor, part: str = "fwd"):
    """part="fwd"：W[V,K] 沿 K 量化（前向/推理用）；part="bwd"：Wᵀ[K,V] 沿 V 量化（dgrad 用）。

    按用途懒建 —— vLLM 推理进程只要 fwd，反向那份 0.4 GB 别白占它的显存预算
    （0.45 配额下这 0.4 GB 直接把 KV 池挤到 0，实测炸过）。
    """
    key = (W.data_ptr(), tuple(W.shape), part)
    cache = _WCACHE.get(key)
    if cache is None:
        from .mxfp8_lmhead import _quant_sw
        assert W.shape[0] % 128 == 0 and W.shape[1] % 128 == 0, \
            f"[unified-fp8] lm_head 形状 {tuple(W.shape)} 不是 128 倍数（TP 分片？）"
        with torch.no_grad():
            src = W.detach() if part == "fwd" else W.detach().T.contiguous()
            cache = _quant_sw(src)
        _WCACHE[key] = cache
        # 生产里只该有一份 lm_head（单测会喂多个随机 W，上限放宽但仍要有）
        assert len(_WCACHE) <= 64, "[unified-fp8] 权重缓存膨胀 —— data_ptr 在漂，查 FSDP 形态"
    return cache


def _mxf8_logits(h: torch.Tensor, qw: torch.Tensor, qw_sf: torch.Tensor) -> torch.Tensor:
    """[N,K] → bf16 logits [N,V]，与训推两侧共用的唯一投影实现。"""
    from .mxfp8_lmhead import _quant_sw, _ext
    N = h.shape[0]
    qh, qh_sf = _quant_sw(h)
    return _ext().mxf8_gemm(qh, qw, qh_sf, qw_sf)[:N]


# ───────────────────────── trainer 侧：分块 PPO 投影 ─────────────────────────

def _fwd_chunk(h, qw, qw_sf, ids, temperature):
    logits = (_mxf8_logits(h, qw, qw_sf) / temperature).to(torch.float32)
    probs = logits.softmax(dim=-1)
    entropy = torch.logsumexp(logits, dim=-1) - torch.sum(probs * logits, dim=-1)
    log_probs = logits.log_softmax(dim=-1)
    token_log_probs = log_probs.gather(-1, ids.unsqueeze(-1)).squeeze(-1)
    return token_log_probs, entropy.to(h.dtype)


def _bwd_chunk(dlp, h, qw, qw_sf, qwt, qwt_sf, ids, temperature):
    from .mxfp8_lmhead import _quant_sw, _ext
    logits = (_mxf8_logits(h, qw, qw_sf) / temperature).to(torch.float32)  # 确定性重算（T0.2）
    probs = logits.softmax(dim=-1)
    dlogits = probs * (-dlp).unsqueeze(-1)
    dlogits.scatter_add_(-1, ids.unsqueeze(-1), dlp.unsqueeze(-1))
    dlogits /= temperature
    qdz, qdz_sf = _quant_sw(dlogits.to(torch.bfloat16))          # 块沿 V = dgrad 收缩维
    return _ext().mxf8_gemm(qdz, qwt, qdz_sf, qwt_sf)[: h.shape[0]].to(h.dtype)


class _MXF8LinearForPPOFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, W, input_ids, temperature, chunk_size):
        ctx.set_materialize_grads(False)
        assert hidden.ndim == 2, f"只接 2D [T,D]，收到 {tuple(hidden.shape)}"
        assert not W.requires_grad, "[unified-fp8] lm_head 权重应当冻结（wgrad 未接）"
        T = hidden.shape[0]
        qw, qw_sf = _weight_cache(W, "fwd")
        log_probs = torch.empty(T, dtype=torch.float32, device=hidden.device)
        entropy = torch.empty(T, dtype=hidden.dtype, device=hidden.device)
        for s in range(0, T, chunk_size):
            e = min(s + chunk_size, T)
            lp, ent = _fwd_chunk(hidden[s:e], qw, qw_sf, input_ids[s:e], temperature)
            log_probs[s:e] = lp
            entropy[s:e] = ent
        ctx.save_for_backward(hidden, W, input_ids)
        ctx.temperature = temperature
        ctx.chunk_size = chunk_size
        return log_probs, entropy

    @staticmethod
    def backward(ctx, dlog_probs, dentropy):
        # entropy_coeff=0 时 verl 仍把 entropy 连在损失图里（0×entropy_loss），
        # 反传回来的是**全零张量**而非 None（首次冒烟实测炸过）——零梯度是合法形态，
        # 放行；真有人把 entropy 接进损失（非零梯度）才必须炸，不许静默走错公式。
        if dentropy is not None and bool((dentropy != 0).any()):
            raise RuntimeError("[unified-fp8] 收到非零 entropy 梯度 —— entropy_coeff 应为 0"
                               "（已钉），有人改了损失构成，先回来接 entropy 反向再跑")
        if dlog_probs is None:
            return None, None, None, None, None
        hidden, W, input_ids = ctx.saved_tensors
        qw, qw_sf = _weight_cache(W, "fwd")
        qwt, qwt_sf = _weight_cache(W, "bwd")
        dhidden = torch.empty_like(hidden)
        cs = ctx.chunk_size
        for s in range(0, hidden.shape[0], cs):
            e = min(s + cs, hidden.shape[0])
            dhidden[s:e] = _bwd_chunk(dlog_probs[s:e], hidden[s:e], qw, qw_sf,
                                      qwt, qwt_sf, input_ids[s:e], ctx.temperature)
        return dhidden, None, None, None, None


_ANNOUNCED = {"trainer": False, "vllm": False}


def linear_for_ppo(hidden_states: torch.Tensor, vocab_weights: torch.Tensor,
                   input_ids: torch.Tensor, temperature: float = 1.0,
                   chunk_size: int = 1024):
    """`_pg_forward` 的唯一投影入口：开关关 = 原样走 verl 融合算子（逐位不变）。"""
    if not enabled():
        from verl.utils.experimental.torch_functional import FusedLinearForPPO
        return FusedLinearForPPO()(hidden_states=hidden_states, vocab_weights=vocab_weights,
                                   input_ids=input_ids, temperature=temperature)
    if not _ANNOUNCED["trainer"]:
        _ANNOUNCED["trainer"] = True
        print(f"[unified-fp8] trainer lm_head MXFP8 已生效 · W={tuple(vocab_weights.shape)} "
              f"量化整训缓存 · T={hidden_states.shape[0]} · temp={temperature}", flush=True)
    return _MXF8LinearForPPOFn.apply(hidden_states, vocab_weights, input_ids,
                                     temperature, chunk_size)


# ───────────────────────── rollout 侧：vLLM 插件 ─────────────────────────

def patch_logits_processor(cls) -> bool:
    """把 LogitsProcessor._get_logits 换成 MXFP8 路径。返回是否新打上（幂等）。"""
    if getattr(cls, "_syncopate_unified_fp8", False):
        return False

    def _get_logits(self, hidden_states, lm_head, embedding_bias):
        if embedding_bias is not None:
            raise RuntimeError("[unified-fp8] lm_head 带 bias —— MXFP8 路径未覆盖，拒绝静默回退")
        W = lm_head.weight
        qw, qw_sf = _weight_cache(W, "fwd")
        with torch.no_grad():
            logits = _mxf8_logits(hidden_states, qw, qw_sf)
        logits = self._gather_logits(logits)
        if logits is not None:
            logits = logits[..., : self.org_vocab_size]
        if not _ANNOUNCED["vllm"]:
            _ANNOUNCED["vllm"] = True
            print(f"[unified-fp8] vLLM lm_head MXFP8 已生效 · W={tuple(W.shape)} · "
                  f"batch={tuple(hidden_states.shape)}", flush=True)
        return logits

    cls._get_logits = _get_logits
    cls._syncopate_unified_fp8 = True
    return True


def register() -> None:
    """vllm.general_plugins 入口点 —— vLLM 的每个进程（含 spawn 的 Worker）启动时调用。

    开关关的时候必须零动作零 import 开销：这是所有不开 FP8 的跑的公共路径。
    """
    if not enabled():
        return
    from vllm.model_executor.layers.logits_processor import LogitsProcessor
    if patch_logits_processor(LogitsProcessor):
        print("[unified-fp8] vLLM LogitsProcessor 补丁已注册（等待首次前向确认生效行）", flush=True)
