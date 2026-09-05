#!/usr/bin/env python
"""E32 S3 · PD 分离 go/no-go 探针（账面收口，Chaoyu 裁定不加实跑投入）。

三个子命令，各出一份机读 json，最后 account 合账：

    b4_pd_probe.py interfere --base-url ... --out logs/b4/pd_interfere_<tag>.json
        # 对着活引擎测两态：decode-only vs decode+prefill 风暴（chunked on/off 由
        # 引擎启动旗子决定，跑两个引擎配置各一次 ⇒ 共四份读数里取三态）
    b4_pd_probe.py bw --out logs/b4/pd_bw.json
        # pinned 内存 D2H/H2D 实测带宽（PD 的 KV 必经之路；不需要引擎）
    b4_pd_probe.py account --interfere-chunked A.json --interfere-nochunk B.json \
        --bw logs/b4/pd_bw.json --out logs/b4/pd_verdict.json
        # go ⇔ chunked 残余干扰(每请求 ms) > KV 搬运附加 TTFT(每请求 ms)，判据=账对上

口径：干扰量 = 风暴态 decode TPOT P99 − 静态 TPOT P99，折算每请求 = ΔTPOT × 生产输出
p50(552 tok，_audit/b4_trace_stats.json)；搬运账 = KV p50 字节 × (1/bw_d2h + 1/bw_h2d)。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time

DECODERS = 8          # 恒定 decode 流的并发
DECODE_TOKENS = 384   # 每条 decode 请求的定长输出（ignore_eos）
STORM_CONC = 16       # prefill 风暴并发（max_tokens=1 ⇒ 纯 prefill 压力）


def pct(vals, q):
    s = sorted(vals)
    return s[min(len(s) - 1, int(q * len(s)))] if s else float("nan")


async def _decode_stream(client, model, results, stop):
    import httpx  # noqa: F401
    prompt = "请从 1 开始连续报数，每个数字之间用逗号分隔，不要输出其它内容。"
    while not stop.is_set():
        t0 = time.perf_counter(); t_first = t_last = None
        try:
            async with client.stream("POST", "/v1/completions", json={
                    "model": model, "prompt": prompt, "stream": True,
                    "max_tokens": DECODE_TOKENS, "ignore_eos": True, "temperature": 0.0}) as r:
                r.raise_for_status()
                async for line in r.aiter_lines():
                    if not line.startswith("data:") or line.strip() == "data: [DONE]":
                        continue
                    now = time.perf_counter()
                    if t_first is None:
                        t_first = now
                    t_last = now
        except Exception:  # noqa: BLE001
            continue
        if t_first and t_last:
            results.append((t_last - t_first) / (DECODE_TOKENS - 1) * 1000)


async def _storm(client, model, reqs, stop, counter):
    i = 0
    while not stop.is_set():
        req = reqs[i % len(reqs)]; i += 1
        try:
            r = await client.post("/v1/completions", json={
                "model": model, "prompt": req["prompt"], "max_tokens": 1,
                "temperature": 0.0})
            r.raise_for_status(); counter.append(req["prompt_tokens"])
        except Exception:  # noqa: BLE001
            await asyncio.sleep(0.2)


async def interfere(args) -> dict:
    import httpx
    reqs = [json.loads(x) for x in open("_audit/b4_trace.jsonl")]
    out = {}
    async with httpx.AsyncClient(base_url=args.base_url, timeout=600,
                                 limits=httpx.Limits(max_connections=64)) as c:
        for phase, with_storm in (("decode_only", False), ("decode_plus_storm", True)):
            results: list[float] = []; storm_toks: list[int] = []
            stop = asyncio.Event()
            tasks = [asyncio.create_task(_decode_stream(c, args.model, results, stop))
                     for _ in range(DECODERS)]
            if with_storm:
                tasks += [asyncio.create_task(_storm(c, args.model, reqs, stop, storm_toks))
                          for _ in range(STORM_CONC)]
            await asyncio.sleep(args.duration)
            stop.set()
            await asyncio.gather(*tasks, return_exceptions=True)
            out[phase] = {
                "n_decode_rounds": len(results),
                "tpot_ms": {"p50": round(pct(results, .5), 2), "p99": round(pct(results, .99), 2)},
                "storm_prefill_tok_per_s": round(sum(storm_toks) / args.duration, 0),
            }
            print(phase, json.dumps(out[phase], ensure_ascii=False))
    return out


def bw(args) -> dict:
    import torch
    n = 256 * 2**20
    host = torch.empty(n, dtype=torch.uint8, pin_memory=True)
    dev = torch.empty(n, dtype=torch.uint8, device="cuda:0")
    res = {}
    for name, fn in (("h2d", lambda: dev.copy_(host, non_blocking=True)),
                     ("d2h", lambda: host.copy_(dev, non_blocking=True))):
        fn(); torch.cuda.synchronize()          # 暖机
        t0 = time.perf_counter()
        for _ in range(10):
            fn()
        torch.cuda.synchronize()
        gbps = n * 10 / (time.perf_counter() - t0) / 2**30
        res[f"{name}_gib_s"] = round(gbps, 1)
    print(json.dumps(res))
    return res


def account(args) -> dict:
    stats = json.load(open("_audit/b4_trace_stats.json"))
    ic = json.load(open(args.interfere_chunked))
    inc = json.load(open(args.interfere_nochunk)) if args.interfere_nochunk else None
    bwj = json.load(open(args.bw))
    out_p50 = stats["output_tokens"]["0.5"]
    kv_mb = stats["kv_fp8_mb_per_req"]["p50"]

    def per_req_ms(j):
        d = j["decode_plus_storm"]["tpot_ms"]["p99"] - j["decode_only"]["tpot_ms"]["p99"]
        return round(max(0.0, d) * out_p50, 1)

    transfer_ms = round(kv_mb / 1024 * (1 / bwj["d2h_gib_s"] + 1 / bwj["h2d_gib_s"]) * 1000, 1)
    residual = per_req_ms(ic)
    verdict = {
        "interference_per_req_ms": {"chunked_on": residual,
                                    "chunked_off": per_req_ms(inc) if inc else None},
        "kv_transfer_per_req_ms": transfer_ms,
        "kv_mb_p50": kv_mb, "out_tokens_p50": out_p50,
        "go": bool(residual > transfer_ms),
        "note": "go ⇔ chunked-on 残余干扰 > KV 搬运附加；chunked_off 档只用来标 PD 理论上限",
    }
    print(json.dumps(verdict, ensure_ascii=False, indent=2))
    return verdict


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("interfere")
    p1.add_argument("--base-url", default="http://127.0.0.1:8100")
    p1.add_argument("--model", default="sft-base")
    p1.add_argument("--duration", type=int, default=120)
    p1.add_argument("--out", required=True)
    p2 = sub.add_parser("bw"); p2.add_argument("--out", required=True)
    p3 = sub.add_parser("account")
    p3.add_argument("--interfere-chunked", required=True)
    p3.add_argument("--interfere-nochunk", default="")
    p3.add_argument("--bw", required=True)
    p3.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.cmd == "interfere":
        res = asyncio.run(interfere(args))
    elif args.cmd == "bw":
        res = bw(args)
    else:
        res = account(args)
    with open(args.out, "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f"✅ {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
