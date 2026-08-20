#!/usr/bin/env python
"""单流 TPOT/TTFT 基线（压测第一件事，11 §5 / 设计文档 §19 的自反前提）。

    python scripts/measure_tpot.py                      # 对 candidate 与 sft-base 各测一轮
    python scripts/measure_tpot.py --model candidate    # 只测一个

★ 它要回答的是设计文档 §19 那个"算不过来"：TPOT 门槛反推 17.5s > I01 的 5s 预算。
  三种可能里只有 ③（真实 TPOT 远好于 25ms）能就地判定 —— 这就是判定它的那次测量。
★ 采样参数 **import 契约模块**（rollout_budget），不抄数（守则⑨）。
⚠️ 定长档用 ignore_eos 强制解码满 max_tokens：量的是**解码速率**，不受模型早停影响；
  另有一档自然停，看真实输出长度下的端到端。
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time

import httpx

sys.path.insert(0, ".")
from syncopate.train.rollout_budget import (  # noqa: E402
    SAMPLING_TEMPERATURE, SAMPLING_TOP_K, SAMPLING_TOP_P)

PROMPTS = {
    # 短 prompt：近似 I01 读意图的量级
    "short": "查一下 CMP_1 昨天的花费和转化，用一句话总结。",
    # 长 prompt：塞近 2k token 的上下文，看 prefill 对 TTFT 的影响
    "long": "以下是过去 30 天的投放日报，请总结三条要点。\n" + (
        "日期 2026-07-01：花费 1234.56 元，展示 45678 次，点击 890 次，转化 12 单，"
        "ROI 1.8，主要跑量素材为视频 A，GEO 集中在华东。\n" * 60),
}


def one_stream(client: httpx.Client, *, model: str, prompt: str,
               max_tokens: int, ignore_eos: bool) -> dict:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": SAMPLING_TEMPERATURE,
        "top_p": SAMPLING_TOP_P,
        "stream": True,
        # vLLM 扩展；top_k=-1 = 不截断（契约值）
        "top_k": SAMPLING_TOP_K,
        "ignore_eos": ignore_eos,
    }
    t0 = time.monotonic()
    ttft = None
    n = 0
    t_last = t0
    with client.stream("POST", "/v1/chat/completions", json=body, timeout=180.0) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            delta = chunk["choices"][0]["delta"].get("content") or ""
            reasoning = chunk["choices"][0]["delta"].get("reasoning_content") or ""
            if delta or reasoning:
                t_last = time.monotonic()
                if ttft is None:
                    ttft = t_last - t0
                n += 1
    # ⚠️ n 是 chunk 数 ≈ token 数（vLLM 默认逐 token 推流）
    tpot = (t_last - t0 - (ttft or 0)) / max(1, n - 1)
    return {"ttft_ms": (ttft or 0) * 1e3, "tpot_ms": tpot * 1e3,
            "tokens": n, "wall_s": t_last - t0}


def bench(client: httpx.Client, model: str, *, repeats: int) -> None:
    print(f"\n== {model} ==")
    for label, prompt in PROMPTS.items():
        # 定长档：解码速率
        runs = [one_stream(client, model=model, prompt=prompt,
                           max_tokens=256, ignore_eos=True)
                for _ in range(repeats)]
        ttft = statistics.median(r["ttft_ms"] for r in runs)
        tpot = statistics.median(r["tpot_ms"] for r in runs)
        print(f"  {label:<5} 定长256 : TTFT {ttft:7.1f} ms · TPOT {tpot:6.2f} ms/tok "
              f"(≈{1e3 / tpot:5.0f} tok/s)")
    # 自然停档：真实端到端（用 short prompt）
    nat = one_stream(client, model=model, prompt=PROMPTS["short"],
                     max_tokens=1024, ignore_eos=False)
    print(f"  short 自然停  : {nat['tokens']} tok · 墙钟 {nat['wall_s']:.2f} s · "
          f"TTFT {nat['ttft_ms']:.1f} ms · TPOT {nat['tpot_ms']:.2f} ms/tok")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base-url", default="http://127.0.0.1:8100")
    ap.add_argument("--model", default=None, help="只测这一个（默认 candidate 和 sft-base 都测）")
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    with httpx.Client(base_url=args.base_url) as client:
        served = [m["id"] for m in client.get("/v1/models").json()["data"]]
        targets = [args.model] if args.model else [m for m in ("candidate", "sft-base")
                                                   if m in served]
        if not targets:
            print(f"🔴 没有可测对象；服务端提供: {served}")
            return 1
        # 预热（首次请求含 CUDA graph 捕获等一次性开销，不计入）
        one_stream(client, model=targets[0], prompt="预热", max_tokens=8, ignore_eos=True)
        for m in targets:
            bench(client, m, repeats=args.repeats)

    print("\n判定参照（设计文档 §19）：门槛 TPOT ≤ 25 ms/tok · TTFT ≤ 800 ms；"
          "I01 反推矛盾若真实 TPOT ≪ 25 则按 ③ 收口（实测反填门槛）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
