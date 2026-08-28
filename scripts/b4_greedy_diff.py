#!/usr/bin/env python
"""E32 S4.2 · 投机解码无损性实证：50 条 trace prompt greedy 输出与基线逐字比对。

    b4_greedy_diff.py capture --base-url ... --out logs/b4/greedy_<tag>.json
    b4_greedy_diff.py diff logs/b4/greedy_base.json logs/b4/greedy_ngram.json

口径：temperature=0 · 自然停（不 ignore_eos——比的是内容）· max_tokens=1024 ·
model=candidate（生产 LoRA）。diff 输出不同条数与首个分歧位置；
判据 = 50/50 逐字一致才许谈速度（拒绝采样无损性的直接实证）。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

import httpx


async def capture(args) -> int:
    reqs = [json.loads(x) for x in open("_audit/b4_trace.jsonl")][: args.n]
    out = []
    async with httpx.AsyncClient(base_url=args.base_url, timeout=600) as c:
        for r in reqs:
            resp = await c.post("/v1/completions", json={
                "model": args.model, "prompt": r["prompt"],
                "max_tokens": 1024, "temperature": 0.0})
            resp.raise_for_status()
            out.append({"idx": r["idx"], "text": resp.json()["choices"][0]["text"]})
            print(f"  {r['idx']}: {len(out[-1]['text'])} chars", flush=True)
    with open(args.out, "w") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"✅ {args.out} ({len(out)} 条)")
    return 0


def diff(a_path: str, b_path: str) -> int:
    a = {r["idx"]: r["text"] for r in json.load(open(a_path))}
    b = {r["idx"]: r["text"] for r in json.load(open(b_path))}
    bad = 0
    for k in sorted(a):
        if a[k] != b.get(k):
            bad += 1
            ta, tb = a[k], b.get(k, "")
            pos = next((i for i, (x, y) in enumerate(zip(ta, tb)) if x != y),
                       min(len(ta), len(tb)))
            print(f"🔴 idx={k} 分歧@{pos}: base={ta[pos:pos+40]!r} vs {tb[pos:pos+40]!r}")
    n = len(a)
    print(f"{'✅' if bad == 0 else '🔴'} 逐字一致 {n - bad}/{n}")
    return 0 if bad == 0 else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("capture")
    p1.add_argument("--base-url", default="http://127.0.0.1:8100")
    p1.add_argument("--model", default="candidate")
    p1.add_argument("--n", type=int, default=50)
    p1.add_argument("--out", required=True)
    p2 = sub.add_parser("diff")
    p2.add_argument("a"); p2.add_argument("b")
    args = ap.parse_args()
    if args.cmd == "capture":
        return asyncio.run(capture(args))
    return diff(args.a, args.b)


if __name__ == "__main__":
    raise SystemExit(main())
