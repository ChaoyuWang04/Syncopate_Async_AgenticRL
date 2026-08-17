"""flash-attn 的**反向**数值判据 —— 换轮子/换机器后必跑。

2026-08-17 的教训：一个 sm_120 轮子可以**前向三项全过、反向全错**。
前向对不代表反向对，而反向错在 RL 里的表现是"训练正常跑完但什么都没学到"。

    flash_attn_func 反向         dq/dk/dv 全 nan      ⇒ verl 打 WARN 跳过 optimizer.step
    flash_attn_varlen_func 反向  有限但恒为 0          ⇒ ★ 静默，没有任何报错

用法：  python scripts/check_flash_attn_backward.py     # 退出码 0 = 可用
"""

import math
import sys

import torch


def _ref(q, k, v):
    q, k, v = (x.float().transpose(1, 2) for x in (q, k, v))
    s = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
    L = s.shape[-1]
    s = s.masked_fill(torch.ones(L, L, device=s.device, dtype=torch.bool).triu(1), float("-inf"))
    return (s.softmax(-1) @ v).transpose(1, 2)


def main() -> int:
    from flash_attn import flash_attn_func, flash_attn_varlen_func

    torch.manual_seed(0)
    dev, dt = "cuda:0", torch.bfloat16
    B, S, H, D = 2, 512, 8, 128
    ok = True

    # 参考梯度（fp32）
    base = [torch.randn(B, S, H, D, device=dev, dtype=dt) for _ in range(3)]
    ref_in = [t.detach().clone().float().requires_grad_(True) for t in base]
    _ref(*ref_in).sum().backward()
    ref_norms = [t.grad.norm().item() for t in ref_in]

    fa_in = [t.detach().clone().requires_grad_(True) for t in base]
    flash_attn_func(*fa_in, causal=True).sum().backward()
    for name, t, r in zip("qkv", fa_in, ref_norms):
        g = t.grad.float()
        got = g.norm().item()
        good = torch.isfinite(g).all().item() and r > 0 and abs(got - r) / r < 0.05
        ok &= bool(good)
        print(f"  flash_attn_func  d{name}: |g|={got:<12.3f} 参考={r:<12.3f} {'✅' if good else '🔴'}")

    # varlen（verl 的 rmpad 走这条）
    lens = [301, 77, 512, 150]
    tot = sum(lens)
    cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), device=dev, dtype=torch.int32)
    vp = [torch.randn(tot, H, D, device=dev, dtype=dt, requires_grad=True) for _ in range(3)]
    flash_attn_varlen_func(*vp, cu, cu, max(lens), max(lens), causal=True).sum().backward()
    for name, t in zip("qkv", vp):
        g = t.grad.float()
        got = g.norm().item()
        good = bool(torch.isfinite(g).all()) and got > 0     # ★ 恒 0 是静默失败，必须拦
        ok &= good
        print(f"  varlen           d{name}: |g|={got:<12.3f} {'✅' if good else '🔴 (0 或非有限)'}")

    print(("\n✅ 反向可用" if ok else
           "\n🔴 反向不可用 —— 别把 --attn-implementation 设成 flash_attention_2"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
