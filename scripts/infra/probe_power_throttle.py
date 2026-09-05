"""E00 · 满载功耗与降频曲线 —— 「分母的分母」。

四张 5090 满载是 4×575 W = 2.3 kW。如果同时压满时核心频率掉下去，
**所有多卡对照实验里都混着这一份**，你会把「卡变慢了」错当成「协作有开销」。

方法：同一个持续 matmul 负载，先压 1 张卡、再压 N 张卡，各采样一段时间，
比较**稳态**的 SM 频率 / 功耗 / 温度 / 实测 TFLOPS。

    python scripts/infra/probe_power_throttle.py                 # 1 卡 vs 4 卡，各 240 s
    python scripts/infra/probe_power_throttle.py --seconds 120
"""

import argparse
import json
import os
import subprocess
import sys
import time

import torch
import torch.multiprocessing as mp

MATMUL_N = 8192          # bf16 8192³ ≈ 1.1 TFLOP/次，足够把 SM 压满


def _burn(dev: int, stop_at: float) -> None:
    """在一张卡上持续做大矩阵乘，直到 stop_at。"""
    torch.cuda.set_device(dev)
    a = torch.randn(MATMUL_N, MATMUL_N, device=f"cuda:{dev}", dtype=torch.bfloat16)
    b = torch.randn(MATMUL_N, MATMUL_N, device=f"cuda:{dev}", dtype=torch.bfloat16)
    n = 0
    while time.time() < stop_at:
        for _ in range(20):
            a @ b
            n += 1
        torch.cuda.synchronize(dev)
    # 把这张卡做了多少次乘法写出去，用于反算实测 TFLOPS
    with open(f"/tmp/_burn_{dev}.json", "w") as f:
        json.dump({"matmuls": n}, f)


def _sample(devs: list[int]) -> list[dict]:
    q = "index,clocks.sm,power.draw,temperature.gpu,clocks_throttle_reasons.active"
    out = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    rows = []
    for ln in out.strip().splitlines():
        p = [x.strip() for x in ln.split(",")]
        if int(p[0]) in devs:
            rows.append({"gpu": int(p[0]), "sm_mhz": float(p[1]), "watt": float(p[2]),
                         "temp": float(p[3]), "throttle": p[4]})
    return rows


def run(devs: list[int], seconds: int) -> dict:
    stop_at = time.time() + seconds
    procs = [mp.Process(target=_burn, args=(d, stop_at)) for d in devs]
    for p in procs:
        p.start()
    time.sleep(15)                      # 跳过升温阶段，只看稳态
    samples = []
    while time.time() < stop_at - 2:
        samples.extend(_sample(devs))
        time.sleep(3)
    for p in procs:
        p.join()

    matmuls = 0
    for d in devs:
        fp = f"/tmp/_burn_{d}.json"
        if os.path.exists(fp):
            matmuls += json.load(open(fp))["matmuls"]
            os.remove(fp)
    flops = matmuls * 2 * MATMUL_N ** 3
    n = len(devs)
    agg = lambda k: sum(s[k] for s in samples) / max(1, len(samples))
    thr = sorted({s["throttle"] for s in samples})
    return {"gpus": n, "sm_mhz": agg("sm_mhz"), "watt_per_gpu": agg("watt"),
            "watt_total": agg("watt") * n, "temp": agg("temp"),
            "tflops_total": flops / seconds / 1e12, "tflops_per_gpu": flops / seconds / 1e12 / n,
            "throttle_reasons": thr, "samples": len(samples)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=int, default=240)
    args = ap.parse_args()
    mp.set_start_method("spawn", force=True)

    total = torch.cuda.device_count()
    print(f"GPU {total} 张 · 每档持续 {args.seconds}s（前 15s 升温不计）· matmul {MATMUL_N}³ bf16\n")
    hdr = f"{'档位':<8}{'SM 频率':>10}{'单卡功耗':>10}{'整机功耗':>10}{'温度':>8}{'单卡TFLOPS':>12}{'整机TFLOPS':>12}"
    print(hdr); print("-" * len(hdr) * 2)

    res = {}
    for n in (1, total):
        r = run(list(range(n)), args.seconds)
        res[f"{n}gpu"] = r
        print(f"{str(n)+' 卡':<8}{r['sm_mhz']:>9.0f}M{r['watt_per_gpu']:>9.0f}W"
              f"{r['watt_total']:>9.0f}W{r['temp']:>7.0f}C"
              f"{r['tflops_per_gpu']:>12.1f}{r['tflops_total']:>12.1f}")

    one, many = res["1gpu"], res[f"{total}gpu"]
    dclk = (many["sm_mhz"] / one["sm_mhz"] - 1) * 100
    dtfl = (many["tflops_per_gpu"] / one["tflops_per_gpu"] - 1) * 100
    print(f"\n★ {total} 卡满载 vs 单卡：频率 {dclk:+.1f}% · 单卡算力 {dtfl:+.1f}%")
    print(f"  ⇒ 多卡对照里混着的「非通信」损失 ≈ {-dtfl:.1f}%"
          f"{'（可忽略）' if abs(dtfl) < 3 else ' ← ★ 必须从所有多卡加速比里扣掉'}")
    print(f"  降频原因（nvidia-smi 报告）：{many['throttle_reasons']}")

    os.makedirs("logs", exist_ok=True)
    with open("logs/e00_power_throttle.json", "w") as f:
        json.dump(res, f, indent=1)
    print("\n原始数据 → logs/e00_power_throttle.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
