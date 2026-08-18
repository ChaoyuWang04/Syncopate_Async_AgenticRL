#!/usr/bin/env python3
"""E16-b · FP8 在**真实矩阵形状**上还剩多少（方阵跑分不算数）。

★ 为什么必须换形状（E16 §4.1-②）：
第一枪测的是 4096³/8192³ 的方阵，实测 1.9–2.1×。但训练里真正的形状是**又扁又长**的：

    Qwen3-4B: hidden 2560 · ffn 9728 · vocab 151936 · 36 层
    lm_head:  [T, 2560] × [2560, 151936]     ← 极扁，而且**它正是 E11 关心的那一层**
    qkv/o:    [T, 2560] × [2560, 2560/4096]
    mlp:      [T, 2560] × [2560, 9728]

**扁矩阵的算术强度低**（每读一个字节能做的乘加少）⇒ 更容易卡在访存上
⇒ **FP8 的收益可能被吃掉一大半**。这条不测清楚，"FP8 能省 23%" 就是编的。

★ 判据（跑之前写死）：
  P1  真实形状上 FP8 仍 ≥1.5×      ⇒ 值得往训练路径里接
  P2  落在 1.0–1.2×                ⇒ 这些形状是访存受限，FP8 只省显存不省时间
  P3  lm_head 那种极扁形状**单独**看 —— 它决定 E11 的稀疏化和 FP8 是不是同一块蛋糕
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import torch

H, FFN, VOCAB = 2560, 9728, 151936
SHAPES = [                       # (名字, M, N, K)
    ("lm_head      T=4096", 4096, VOCAB, H),
    ("lm_head      T=1024", 1024, VOCAB, H),
    ("mlp_up       T=4096", 4096, FFN, H),
    ("mlp_down     T=4096", 4096, H, FFN),
    ("qkv_proj     T=4096", 4096, 4096, H),
    ("o_proj       T=4096", 4096, H, H),
    ("方阵对照 4096³", 4096, 4096, 4096),
]

def bench(fn, warmup=5, iters=20):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    ts=[]
    for _ in range(iters):
        t=time.perf_counter(); fn(); torch.cuda.synchronize(); ts.append(time.perf_counter()-t)
    ts.sort(); return ts[len(ts)//2]

def main() -> int:
    ap=argparse.ArgumentParser(); ap.add_argument("--json", type=Path); a=ap.parse_args()
    dev=torch.device("cuda"); out={"device":torch.cuda.get_device_name(),"shapes":{}}
    print(f"# {out['device']}   真实形状下的 FP8（Qwen3-4B: hidden {H} / ffn {FFN} / vocab {VOCAB}）")
    print(f"  {'形状':<22}{'M×N×K':>22}{'bf16 TFLOPS':>13}{'fp8 TFLOPS':>12}{'倍数':>8}{'算术强度':>10}")
    for name,m,n,k in SHAPES:
        A=torch.randn(m,k,device=dev,dtype=torch.bfloat16)
        B=torch.randn(k,n,device=dev,dtype=torch.bfloat16)
        tb=bench(lambda: torch.matmul(A,B)); tf_b=2*m*n*k/tb/1e12
        r={"m":m,"n":n,"k":k,"bf16_tflops":round(tf_b,1)}
        # 算术强度 = FLOP / 访存字节（bf16），越低越容易访存受限
        ai = 2*m*n*k / ((m*k + k*n + m*n)*2)
        r["arith_intensity"]=round(ai,1)
        try:
            Af=A.to(torch.float8_e4m3fn); Bf=B.t().contiguous().t().to(torch.float8_e4m3fn)
            sc=torch.tensor(1.0,device=dev)
            f=lambda: torch._scaled_mm(Af,Bf,scale_a=sc,scale_b=sc,out_dtype=torch.bfloat16)
            f(); tf=bench(f); tf_f=2*m*n*k/tf/1e12
            r.update(fp8_tflops=round(tf_f,1), speedup=round(tb/tf,2))
            print(f"  {name:<22}{f'{m}×{n}×{k}':>22}{tf_b:>13.1f}{tf_f:>12.1f}{tb/tf:>7.2f}×{ai:>10.1f}")
        except Exception as exc:
            r["fp8_error"]=f"{type(exc).__name__}: {exc}"[:200]
            print(f"  {name:<22}{f'{m}×{n}×{k}':>22}{tf_b:>13.1f}{'—':>12}{'—':>8}{ai:>10.1f}   {r['fp8_error'][:60]}")
        out["shapes"][name.strip()]=r
        del A,B; torch.cuda.empty_cache()
    if a.json:
        a.json.parent.mkdir(parents=True,exist_ok=True); a.json.write_text(json.dumps(out,indent=2,ensure_ascii=False))
        print(f"  → {a.json}")
    return 0

if __name__=="__main__": raise SystemExit(main())
