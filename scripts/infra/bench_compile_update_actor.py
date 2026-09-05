#!/usr/bin/env python3
"""E14 收官③ · torch.compile 微基准 —— update_actor 前向段（PG 打包路径，脱 Ray 单进程单卡）。

段的定义 = 生产 update_actor 的一个 micro-batch：
    Qwen3-4B + LoRA r32 all-linear · GC(use_reentrant=False) · PG 打包前向（brcausal SDPA）
    · fwd + bwd（真实段两者都在计时里）；形状取 E26 口径 prefix 4196 / resp 654 / G=8。

臂（同权重同输入，torch.manual_seed 固定）：
    eager       现状
    default     torch.compile(mode 默认)
    autotune    torch.compile(mode="max-autotune")   ← E14 四采：GEMM 全是 cutlass_80（sm80 核
                跑在 sm120 上），autotune 换 Triton GEMM 是它独有的机会

预测（跑前写死）：
    ① 速度：elementwise 占 kernel 27–31%（E14 四采），compile 融合只吃得到其中一部分
       ⇒ default 臂预期 −5~15%；<−3% ⇒ 对 update_actor 判死（写进边界表）
    ② 数值：同输入 fwd 的 log_probs，max|Δ| 预期 bf16 重排量级 ~1e-3/token 以内、
       fp32 求和相对差 <1e-5；超出 ⇒ 数值红线，不进默认
    ③ 动态形状税：resp_len 654→512→800 触发的 recompile 次数与耗时；真实负载变长批，
       每换形状重编译一次的税若 > 单步收益 ⇒ 端到端负收益（这条才是 go/no-go 主判据）

判据行全部只在终态读：每臂打 MEASURE 行；对照计数 = eager 臂必须先跑通（跑不通说明段本身没搭对）。
"""
import argparse, statistics as st, time, os
import torch, torch.nn.functional as F

MP = "models/Qwen3-4B-sft-v13r2-e1"


def brcausal(module, q, k, v, am, *a, scaling=None, dropout=0.0, **kw):
    Lq, Lk = q.shape[-2], k.shape[-2]
    i = torch.arange(Lq, device=q.device).view(-1, 1)
    j = torch.arange(Lk, device=q.device).view(1, -1)
    o = F.scaled_dot_product_attention(q, k, v, attn_mask=(j <= i + (Lk - Lq)),
                                       scale=scaling, enable_gqa=True)
    return o.transpose(1, 2).contiguous(), None


def build():
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model
    m = AutoModelForCausalLM.from_pretrained(MP, dtype=torch.bfloat16,
        attn_implementation="brcausal", trust_remote_code=True).to("cuda")
    m = get_peft_model(m, LoraConfig(r=32, lora_alpha=64, lora_dropout=0.0, bias="none",
        target_modules="all-linear", task_type="CAUSAL_LM"))
    m.config.use_cache = False
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


def seg(pgu, m, mb, bwd=True):
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


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["eager", "default", "autotune"])
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
    m = build()
    P, G = a.prefix_len, a.group

    if a.arm != "eager":
        import torch._dynamo as dyn
        dyn.reset()
        mode = None if a.arm == "default" else "max-autotune"
        t0 = time.time()
        m.compile(mode=mode)

    # ② 数值判据：fwd-only 的 log_probs（eager 臂存参考，compile 臂对拍）
    mb0 = make_batch(P, a.resp_len, G, V)
    # GC 路径要求 grad 上下文 ⇒ 数值对拍用带 grad 的 fwd、不 backward
    lp0 = seg(pgu, m, mb0, bwd=False).detach().float()
    ref_path = "logs/torchprof/compile_bench_ref_lp.pt"
    if a.arm == "eager":
        os.makedirs(os.path.dirname(ref_path), exist_ok=True)
        torch.save(lp0.cpu(), ref_path)
        print(f"NUMERIC eager 参考已存：sum={lp0.sum().item():.6f}")
    elif os.path.exists(ref_path):
        ref = torch.load(ref_path).to("cuda")
        dmax = (lp0 - ref).abs().max().item()
        rsum = abs(lp0.sum().item() - ref.sum().item()) / max(abs(ref.sum().item()), 1e-9)
        print(f"NUMERIC {a.arm} vs eager：max|Δ|={dmax:.3e}  sum相对差={rsum:.3e}"
              f"  （红线：max|Δ|>1e-2 或 sum相对差>1e-4）")
    if a.arm != "eager":
        print(f"COMPILE {a.arm} 首编译+首轮 fwd 墙钟 {time.time()-t0:.1f} s")

    # ① 速度：fwd+bwd（首轮丢弃=充分 warmup/编译）
    s = timed(lambda: seg(pgu, m, mb0, bwd=True), a.iters)
    peak = torch.cuda.max_memory_allocated() / 2**30
    print(f"MEASURE {a.arm} R={a.resp_len} fwd+bwd 中位 {s:.4f} s   峰值 {peak:.2f} GB")

    # ③ 动态形状税：换两个 resp_len，各测一次首轮（含可能的重编译）与稳态
    for R2 in (512, 800):
        mb2 = make_batch(P, R2, G, V)
        torch.cuda.synchronize(); t = time.time()
        seg(pgu, m, mb2, bwd=True); torch.cuda.synchronize()
        first = time.time() - t
        s2 = timed(lambda: seg(pgu, m, mb2, bwd=True), 3)
        print(f"MEASURE {a.arm} R={R2} 首轮 {first:.3f} s（含重编译税）  稳态中位 {s2:.4f} s")

    if a.arm != "eager":
        from torch._dynamo.utils import counters
        rec = dict(counters.get("stats", {}))
        print(f"DYNAMO stats：{rec}")
        gb = sum(counters.get("graph_break", {}).values())
        print(f"DYNAMO graph_break 总数：{gb}")
