#!/usr/bin/env python3
"""★ PrefixGrouper 等价性判据（fp32 + 噪声地板对照）。

⚠️⚠️ **不要在 bf16 下判等价性。** 实测这个模型 bf16 的噪声地板（同样内容、只把两条
序列换个批组成）就有 mean 1.28e-2 / max 1.0 —— 比我们要抓的错误还大。
2026-08-19 因此在噪声里追了三轮假根因。⇒ **判据必须 fp32 + 与噪声地板比。**

判据（按顺序，前一条不过就不看后面）：
  A  [prefix-grouper] 打包前向已生效     —— 路径真的被走到
  B  隐状态差 ≤ 噪声地板                 —— 等价
  C  吞吐 / 任务尺子                     —— 本脚本不管
"""
from __future__ import annotations
import argparse, os, torch, torch.nn.functional as F


def brcausal(module, q, k, v, attention_mask, *args, scaling=None, dropout=0.0, **kw):
    """右下对齐的因果注意力（fp32 可用；HF 自带的 eager/sdpa 是左上对齐，对 suffix 子调用是错的）"""
    Lq, Lk = q.shape[-2], k.shape[-2]
    i = torch.arange(Lq, device=q.device).view(-1, 1)
    j = torch.arange(Lk, device=q.device).view(1, -1)
    o = F.scaled_dot_product_attention(q, k, v, attn_mask=(j <= i + (Lk - Lq)),
                                       scale=scaling, enable_gqa=True)
    return o.transpose(1, 2).contiguous(), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Qwen3-4B-sft-v13-e1")
    ap.add_argument("--prefix-len", type=int, default=256)
    ap.add_argument("--resp-len", type=int, default=64)
    ap.add_argument("--group", type=int, default=4)
    a = ap.parse_args()

    os.environ["SYNCOPATE_PREFIX_GROUPER"] = "1"
    from transformers import AutoModelForCausalLM, AutoConfig
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from syncopate.train import verl_patches as vp
    from verl.models.transformers import monkey_patch as mp
    from verl.trainer.ppo.prefix_grouper_utils import build_position_ids_for_prefix_grouper as bpos
    from prefix_grouper import PrefixGrouper

    ALL_ATTENTION_FUNCTIONS["brcausal"] = brcausal
    vp._patch_prefix_grouper()        # 装我们的补丁（会替换 wrapper 工厂）
    mp.apply_prefix_grouper_patch()   # 用我们的工厂把后端包上（只包硬编码名单里的）
    # ⚠️ verl 的 apply_prefix_grouper_patch 只包**硬编码名单**（flash_attention_3/2、
    #    flex_attention、sdpa）。自定义后端不在里面 ⇒ 必须手动包，否则 prefix_grouper
    #    这个 kwarg 会被当普通参数忽略、整条打包序列走普通因果注意力 ⇒ 静默算错。
    #    （2026-08-19 这个坑连着骗了我两次，两次都表现为"判据失败"而不是报错。）
    ALL_ATTENTION_FUNCTIONS["brcausal"] = mp._create_prefix_grouper_wrapper(brcausal)
    assert getattr(ALL_ATTENTION_FUNCTIONS["brcausal"], "__name__", "") == "wrapped", \
        "[判据] brcausal 没被包上 —— 后面的等价性结论无效"

    P, R, G = a.prefix_len, a.resp_len, a.group
    V = AutoConfig.from_pretrained(a.model, trust_remote_code=True).vocab_size
    m = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32,
        attn_implementation="brcausal", trust_remote_code=True).to("cuda").eval()
    torch.manual_seed(0)
    pre = torch.randint(0, V, (1, P), device="cuda")
    resp = torch.randint(0, V, (G, R), device="cuda")
    pm = torch.ones_like(pre, dtype=torch.bool); sm = torch.ones_like(resp, dtype=torch.bool)

    base = []
    with torch.no_grad():
        for i in range(G):
            ids = torch.cat([pre, resp[i:i+1]], 1)
            base.append(m(input_ids=ids, attention_mask=torch.ones_like(ids),
                          output_hidden_states=True, use_cache=False).hidden_states[-1][:, P-1:P+R-1])
    base = torch.cat(base, 0)

    # ★ 噪声地板：同样内容，只改批组成（改变归约顺序，不改变数学）
    ids2 = torch.cat([pre.expand(G, -1), resp], 1)
    with torch.no_grad():
        ctrl = m(input_ids=ids2, attention_mask=torch.ones_like(ids2),
                 output_hidden_states=True, use_cache=False).hidden_states[-1][:, P-1:P+R-1]

    pg = PrefixGrouper.from_ungrouped_masks(prefix_mask=pm, suffix_mask=sm,
                                            group_sizes=[G], device="cuda")
    with torch.no_grad():
        hp = m(input_ids=pg.concat_input(pre, pm, resp, sm), attention_mask=pg.padding_mask,
               position_ids=bpos(pg), output_hidden_states=True, use_cache=False,
               prefix_grouper=pg).hidden_states[-1]
    _, _, sh, _ = pg.split_output(hp, include_prefix_last=1)

    n = (base - ctrl).abs(); d = (base - sh[:, :-1]).abs()
    print(f"\nfp32 · P={P} R={R} G={G}")
    print(f"  噪声地板（改批组成）  max {n.max().item():.3e}  mean {n.mean().item():.3e}")
    print(f"  PrefixGrouper         max {d.max().item():.3e}  mean {d.mean().item():.3e}")
    ok = d.max().item() <= max(n.max().item() * 10, 1e-6)
    print("  ⇒", "✅ **判据B 通过：等价**" if ok
          else f"🔴 判据B 失败：大 {d.max().item()/max(n.max().item(),1e-12):.0f} 倍")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
