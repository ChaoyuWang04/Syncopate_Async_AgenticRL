#!/usr/bin/env python
"""E32 · trace 轨重放客户端（cache/路由结论的唯一合法尺子；random 轨归 vllm bench serve）。

    .venv/bin/python scripts/b4_replay.py --concurrency 32 --out logs/b4/arm_x_trace.json
    .venv/bin/python scripts/b4_replay.py --base-url http://127.0.0.1:8100 --model sft-base \
        --metrics-urls http://127.0.0.1:8101,http://127.0.0.1:8102   # router 模式逐引擎抓

口径（与 E19-c 并列可比）：
- **ignore_eos + 每请求 max_tokens = 真实输出 token 数** ⇒ 解码长度分布与生产逐条一致，
  且跨臂确定（内容不影响长度）；采样参数 import 契约模块不抄数（守则⑨）。
- 闭环并发（信号量），到达顺序 = trace 顺序（保留同 case 相邻性——亲和路由吃的就是它）。
- TTFT = 首个内容 chunk；TPOT = (末 chunk − 首 chunk)/(max_tokens−1)；n_out 恒 = max_tokens。
- 压测前后各抓一次 /metrics，凡含 prefix_cache 的计数行都存 raw + 算差值命中率
  （0.12 指标实名以首跑抓到的为准——判据行没出现就明说没抓到，不静默）。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time

import httpx

sys.path.insert(0, ".")
from syncopate.train.rollout_budget import (  # noqa: E402
    SAMPLING_TEMPERATURE, SAMPLING_TOP_K, SAMPLING_TOP_P)


def pct(vals: list[float], q: float) -> float:
    if not vals:
        return float("nan")
    s = sorted(vals)
    return s[min(len(s) - 1, int(q * len(s)))]


async def scrape_metrics(urls: list[str]) -> dict[str, dict[str, float]]:
    """每个 url 一张 {metric_line_key: value}，只留 prefix_cache 相关计数。"""
    out: dict[str, dict[str, float]] = {}
    async with httpx.AsyncClient(timeout=10) as c:
        for u in urls:
            got: dict[str, float] = {}
            try:
                r = await c.get(u.rstrip("/") + "/metrics")
                for line in r.text.splitlines():
                    if "prefix_cache" in line and not line.startswith("#"):
                        m = re.match(r"^(\S+?)(\{[^}]*\})?\s+([0-9.eE+-]+)$", line)
                        if m:
                            got[m.group(1)] = got.get(m.group(1), 0.0) + float(m.group(3))
            except Exception as e:  # noqa: BLE001  探针失败要显形不中断
                got["__scrape_error__"] = 1.0
                print(f"⚠️ metrics 抓取失败 {u}: {e}")
            out[u] = got
    return out


async def one(client: httpx.AsyncClient, sem: asyncio.Semaphore, req: dict,
              model: str, results: list[dict]) -> None:
    async with sem:
        t0 = time.perf_counter()
        t_first = t_last = None
        payload = {
            "model": model, "prompt": req["prompt"], "stream": True,
            "max_tokens": req["max_tokens"], "ignore_eos": True,
            "temperature": SAMPLING_TEMPERATURE, "top_p": SAMPLING_TOP_P,
            "top_k": SAMPLING_TOP_K,
        }
        try:
            async with client.stream("POST", "/v1/completions", json=payload) as r:
                r.raise_for_status()
                async for line in r.aiter_lines():
                    if not line.startswith("data:") or line.strip() == "data: [DONE]":
                        continue
                    now = time.perf_counter()
                    if t_first is None:
                        t_first = now
                    t_last = now
        except Exception as e:  # noqa: BLE001
            results.append({"idx": req["idx"], "error": str(e)[:200]})
            return
        n = req["max_tokens"]
        results.append({
            "idx": req["idx"], "family": req["family"], "prompt_tokens": req["prompt_tokens"],
            "out_tokens": n, "ttft_s": (t_first - t0) if t_first else None,
            "tpot_ms": ((t_last - t_first) / max(1, n - 1) * 1000) if t_first and n > 1 else None,
            "wall_s": t_last - t0 if t_last else None,
        })


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8100")
    ap.add_argument("--model", default="sft-base")
    ap.add_argument("--trace", default="_audit/b4_trace.jsonl")
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--n", type=int, default=0, help="0=全量 trace")
    ap.add_argument("--metrics-urls", default="", help="逗号分隔；默认= base-url")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    reqs = [json.loads(x) for x in open(args.trace)]
    if args.n:
        reqs = reqs[: args.n]
    murls = [u for u in (args.metrics_urls.split(",") if args.metrics_urls
                         else [args.base_url]) if u]

    m_before = await scrape_metrics(murls)
    sem = asyncio.Semaphore(args.concurrency)
    results: list[dict] = []
    t0 = time.perf_counter()
    async with httpx.AsyncClient(base_url=args.base_url, timeout=600,
                                 limits=httpx.Limits(max_connections=args.concurrency + 8)) as c:
        await asyncio.gather(*[one(c, sem, r, args.model, results) for r in reqs])
    wall = time.perf_counter() - t0
    m_after = await scrape_metrics(murls)

    ok = [r for r in results if "error" not in r]
    errs = len(results) - len(ok)
    ttft = [r["ttft_s"] for r in ok if r["ttft_s"] is not None]
    tpot = [r["tpot_ms"] for r in ok if r["tpot_ms"] is not None]
    out_tok = sum(r["out_tokens"] for r in ok)

    # prefix cache 命中率：对每个引擎找 (queries, hits) 计数对，算压测窗口内差值
    cache: dict[str, dict] = {}
    for u in murls:
        b, a = m_before.get(u, {}), m_after.get(u, {})
        qk = next((k for k in a if "queries" in k), None)
        hk = next((k for k in a if "hits" in k), None)
        if qk and hk and qk in b and hk in b and a[qk] > b[qk]:
            cache[u] = {"hit_rate": round((a[hk] - b[hk]) / (a[qk] - b[qk]), 4),
                        "queries": a[qk] - b[qk]}
        else:
            cache[u] = {"hit_rate": None, "raw_after": a}   # 指标名没对上 ⇒ 显形

    summary = {
        "arm": args.out or "-", "n": len(reqs), "errors": errs,
        "concurrency": args.concurrency, "wall_s": round(wall, 1),
        "output_tok_per_s": round(out_tok / wall, 1),
        "req_per_s": round(len(ok) / wall, 3),
        "ttft_s": {"p50": round(pct(ttft, .5), 3), "p90": round(pct(ttft, .9), 3),
                   "p99": round(pct(ttft, .99), 3)},
        "tpot_ms": {"p50": round(pct(tpot, .5), 2), "p90": round(pct(tpot, .9), 2),
                    "p99": round(pct(tpot, .99), 2)},
        "prefix_cache": cache,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if errs:
        print(f"🔴 {errs}/{len(results)} 条请求失败（首条：{next(r['error'] for r in results if 'error' in r)}）")
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"summary": summary, "requests": results}, f, ensure_ascii=False)
        print(f"✅ {args.out}")
    return 1 if errs > len(results) * 0.02 else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
