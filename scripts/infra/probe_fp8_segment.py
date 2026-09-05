#!/usr/bin/env python3
"""E19 续 · FP8 找新消费者（01 §1-2）：update_actor / old_log_prob 前向段的 FP8 评估。

★ 机制洞察（这是本探针存在的理由）：LoRA 训练里基座 Linear **冻结** ⇒
    ① 权重 per-tensor 量化**只做一次**（装载时），没有每步 requantize 成本；
    ② backward 只需 dgrad（对输入的梯度，链到 LoRA 与浅层），**不需要 wgrad**
    —— 通用 FP8 训练的最难两件（scale 管理 / 梯度量化）在这个形态下都退化成平凡。
    lm_head 保持 bf16（它的误差直接进 logprob，且是 E11 稀疏切片的靶子，别两头记账）。

臂（同权重同输入；LoRA B=0 ⇒ 数学上基座即全模型）：
    bf16      基准
    floor     bf16 + allow_bf16_reduced_precision_reduction 翻转 —— **数值噪声地板**
              （同一数学、不同归约路径；今天 compile 微基准没立地板挨的打，这里补上）
    fp8       q/k/v/o/gate/up/down 的基座 GEMM 走 torch._scaled_mm（per-tensor）

判据与预测（跑前写死）：
    ① 数值：|Δlogprob| (fp8 vs bf16) 与地板 |Δ| (floor vs bf16) 的**倍数**。
       预测：层内 GEMM 误差经 RMSNorm/softmax 部分归一 ⇒ per-token mean |Δ| < 0.01
       （E19 在裸 lm_head 上测 0.032；本次 lm_head 不动应显著小于它）。
       通过线：fp8 误差 ≤ 10× 地板 ⇒ 进下一关（真训练任务配对闸）；> 100× 地板 ⇒ 判死。
    ② 序列级：per-seq Σ_t Δ（这是进 seq-IS 权重的量）报告绝对值分布，与
       rollout_corr/kl 地板 3.4e-4 量级对照。
    ③ 速度：fwd-only 段（=olp 形态）与 fwd+bwd 段（=update_actor 形态，GC 开）。
       预测：GEMM 占段 ~60–70%、FP8 GEMM 1.7–2.2× ⇒ 段级 −20~35%；
       activation 每次 amax+cast 是逆风，<−10% 则「接线成本吃掉收益」要重估。
"""
import argparse, statistics as st, time, os
import torch, torch.nn.functional as F

MP = "models/Qwen3-4B-sft-v13r2-e1"
FMAX = torch.finfo(torch.float8_e4m3fn).max


def brcausal(module, q, k, v, am, *a, scaling=None, dropout=0.0, **kw):
    Lq, Lk = q.shape[-2], k.shape[-2]
    i = torch.arange(Lq, device=q.device).view(-1, 1)
    j = torch.arange(Lk, device=q.device).view(1, -1)
    o = F.scaled_dot_product_attention(q, k, v, attn_mask=(j <= i + (Lk - Lq)),
                                       scale=scaling, enable_gqa=True)
    return o.transpose(1, 2).contiguous(), None


class _FP8Linear(torch.autograd.Function):
    """前向 FP8 GEMM；反向 dgrad 用 bf16 权重（基座冻结 ⇒ 无 wgrad）。"""

    @staticmethod
    def forward(ctx, x, w_fp8, w_scale, w_bf16):
        ctx.save_for_backward(w_bf16)
        xs = (x.abs().amax() / FMAX).clamp(min=1e-12)
        xf = (x / xs).to(torch.float8_e4m3fn).reshape(-1, x.shape[-1])
        # A 行主序 [M,K]；w_fp8 是 [N,K] 行主序 ⇒ .t() 即列主序 [K,N]（E19 探针的布局坑）
        y = torch._scaled_mm(xf, w_fp8.t(), scale_a=xs.float(), scale_b=w_scale,
                             out_dtype=torch.bfloat16)
        return y.view(*x.shape[:-1], -1)

    @staticmethod
    def backward(ctx, gy):
        (w,) = ctx.saved_tensors
        return gy @ w, None, None, None


def wrap_base_linears(model):
    """把所有 lora.Linear 的 base_layer 前向换成 FP8 路径；权重量化一次。"""
    from peft.tuners.lora import Linear as LoraLinear
    n = 0
    for mod in model.modules():
        if isinstance(mod, LoraLinear):
            bl = mod.base_layer
            assert bl.bias is None, "FP8 包装假定无 bias（Qwen3 成立）"
            w = bl.weight.data                      # [N,K] bf16，冻结
            ws = (w.abs().amax() / FMAX).clamp(min=1e-12).float()
            w8 = (w / ws).to(torch.float8_e4m3fn)
            bl._fp8 = (w8, ws, w)
            bl.forward = (lambda x, _bl=bl:
                          _FP8Linear.apply(x, _bl._fp8[0], _bl._fp8[1], _bl._fp8[2]))
            n += 1
    return n


def unwrap_base_linears(model):
    from peft.tuners.lora import Linear as LoraLinear
    for mod in model.modules():
        if isinstance(mod, LoraLinear):
            bl = mod.base_layer
            if hasattr(bl, "_fp8"):
                del bl._fp8
                bl.forward = type(bl).forward.__get__(bl)


def build(gc):
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model
    m = AutoModelForCausalLM.from_pretrained(MP, dtype=torch.bfloat16,
        attn_implementation="brcausal", trust_remote_code=True).to("cuda")
    m = get_peft_model(m, LoraConfig(r=32, lora_alpha=64, lora_dropout=0.0, bias="none",
        target_modules="all-linear", task_type="CAUSAL_LM"))
    m.config.use_cache = False
    if gc:
        m.enable_input_require_grads()
        m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    m.train()
    return m


def make_batch(P, R, G, V):
    torch.manual_seed(0)
    pre = torch.randint(1, V, (1, P), device="cuda").expand(G, -1).contiguous()
    resp = torch.randint(1, V, (G, R), device="cuda")
    return dict(prompts=pre, responses=resp, uid=["g0"] * G,
                response_mask=torch.ones(G, R, dtype=torch.long, device="cuda"),
                attention_mask=torch.ones(G, P + R, dtype=torch.long, device="cuda"),
                pad_token_id=0)


def seg(pgu, m, mb, bwd):
    _, lp = pgu.forward_micro_batch_with_prefix_grouper(micro_batch=mb, model=m,
        temperature=1.0, calculate_entropy=False, device_name="cuda",
        param_dtype=torch.bfloat16)
    if bwd:
        (-lp.float().mean()).backward()
        m.zero_grad(set_to_none=True)
    return lp


def timed(fn, iters):
    ts = []
    for i in range(iters + 1):
        torch.cuda.synchronize(); t = time.time(); fn(); torch.cuda.synchronize()
        if i:
            ts.append(time.time() - t)
    return st.median(ts)


def report_delta(tag, lp, ref):
    d = (lp.float() - ref.float())
    seq = d.sum(dim=-1)                      # per-seq Σ_t Δ —— 进 seq-IS 的量
    print(f"NUMERIC {tag}: per-token |Δ| mean={d.abs().mean().item():.3e} "
          f"max={d.abs().max().item():.3e} · per-seq Σ|·| mean={seq.abs().mean().item():.3e} "
          f"max={seq.abs().max().item():.3e}")
    return d.abs().mean().item()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix-len", type=int, default=4196)
    ap.add_argument("--resp-len", type=int, default=654)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--iters", type=int, default=6)
    a = ap.parse_args()

    os.environ["SYNCOPATE_PREFIX_GROUPER"] = "1"
    from transformers import AutoConfig
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    ALL_ATTENTION_FUNCTIONS["brcausal"] = brcausal
    from syncopate.train import verl_patches as vp
    from verl.models.transformers import monkey_patch as mp_
    from verl.trainer.ppo import prefix_grouper_utils as pgu
    vp._patch_prefix_grouper(); mp_.apply_prefix_grouper_patch()
    ALL_ATTENTION_FUNCTIONS["brcausal"] = mp_._create_prefix_grouper_wrapper(brcausal)

    V = AutoConfig.from_pretrained(MP, trust_remote_code=True).vocab_size
    m = build(gc=True)
    mb = make_batch(a.prefix_len, a.resp_len, a.group, V)

    # ---------- ① 数值（fwd-only，no_grad + GC 无关） ----------
    with torch.no_grad():
        lp_ref = seg(pgu, m, mb, bwd=False).float()
        # 噪声地板：同一数学、不同 GEMM 归约路径
        old = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = not old
        lp_floor = seg(pgu, m, mb, bwd=False).float()
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = old
        floor = report_delta("floor(bf16 归约翻转)", lp_floor, lp_ref)

        nw = wrap_base_linears(m)
        print(f"[fp8] 已包 {nw} 个基座 Linear（lm_head 保持 bf16；权重量化一次性完成）")
        lp_fp8 = seg(pgu, m, mb, bwd=False).float()
        err = report_delta("fp8(层GEMM)", lp_fp8, lp_ref)
    ratio = err / max(floor, 1e-12)
    verdict = ("≤10× 地板 ⇒ 过本关" if ratio <= 10
               else (">100× 地板 ⇒ 判死" if ratio > 100 else "10–100× ⇒ 灰区，进配对闸再判"))
    print(f"VERDICT 数值：fp8/地板 = {ratio:.1f}×  （{verdict}）")

    # ---------- ② 速度：fwd-only（olp 形态） ----------
    with torch.no_grad():
        s_fp8_f = timed(lambda: seg(pgu, m, mb, bwd=False), a.iters)
    unwrap_base_linears(m)
    with torch.no_grad():
        s_bf16_f = timed(lambda: seg(pgu, m, mb, bwd=False), a.iters)
    print(f"MEASURE fwd-only(olp)   bf16 {s_bf16_f:.4f} s · fp8 {s_fp8_f:.4f} s "
          f"· Δ {100 * (s_fp8_f / s_bf16_f - 1):+.1f}%")

    # ---------- ③ 速度：fwd+bwd（update_actor 形态，GC 开） ----------
    s_bf16_t = timed(lambda: seg(pgu, m, mb, bwd=True), a.iters)
    wrap_base_linears(m)
    s_fp8_t = timed(lambda: seg(pgu, m, mb, bwd=True), a.iters)
    print(f"MEASURE fwd+bwd(ua/GC)  bf16 {s_bf16_t:.4f} s · fp8 {s_fp8_t:.4f} s "
          f"· Δ {100 * (s_fp8_t / s_bf16_t - 1):+.1f}%")
    print(f"峰值显存 {torch.cuda.max_memory_allocated() / 2**30:.2f} GB")
