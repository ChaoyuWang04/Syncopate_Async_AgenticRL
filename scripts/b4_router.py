#!/usr/bin/env python
"""E32 S2 · 前缀亲和路由器（4×DP 副本的入口；监听 :8100 ⇒ decider/chatbox 零改动）。

    .venv/bin/python scripts/b4_router.py --policy affinity \
        --backends http://127.0.0.1:8101,http://127.0.0.1:8102,http://127.0.0.1:8103,http://127.0.0.1:8104

策略：
- rr        轮询（对照臂——预测 random 流量下与 affinity 无差、真实 trace 下分开）
- affinity  对 prompt[skip:skip+window] 做 md5 定副本：skip 越过全局公共前缀
            （system prompt，人人相同没有区分度），window 落在 case 上下文段 ⇒
            同 case 的多条 rollout/多轮进同一副本，前缀池不被 4 份稀释。
            副本挂了顺移到下一个活副本（后台 5s 健康巡检）。

转发是流式透传（SSE 逐 chunk），/v1/models 与 /health 代理到首个活副本；
GET /router/stats 看逐副本分发计数（路由 A/B 的判据行）。
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json

import httpx
import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

BACKENDS: list[str] = []
ALIVE: set[int] = set()
COUNTS: dict[int, int] = {}
POLICY = "affinity"
SKIP, WINDOW = 4000, 2048
_rr = 0
client: httpx.AsyncClient | None = None


def pick(body: dict) -> int:
    global _rr
    alive = sorted(ALIVE) or list(range(len(BACKENDS)))
    if POLICY == "rr":
        _rr += 1
        return alive[_rr % len(alive)]
    text = body.get("prompt") or json.dumps(body.get("messages", ""), ensure_ascii=False)
    if isinstance(text, list):
        text = "".join(map(str, text))
    key = hashlib.md5(text[SKIP:SKIP + WINDOW].encode()).digest()
    want = int.from_bytes(key[:4], "big") % len(BACKENDS)
    for off in range(len(BACKENDS)):          # 首选挂了顺移
        cand = (want + off) % len(BACKENDS)
        if cand in ALIVE or not ALIVE:
            return cand
    return want


async def proxy_post(request: Request) -> Response:
    body = await request.json()
    i = pick(body)
    COUNTS[i] = COUNTS.get(i, 0) + 1
    url = BACKENDS[i] + request.url.path
    upstream = client.stream("POST", url, json=body, timeout=600)

    async def gen():
        async with upstream as r:
            async for chunk in r.aiter_raw():
                yield chunk

    # 先拿到响应头再定状态码：stream 上下文在 gen 里开，状态码只能乐观 200；
    # 上游报错时 body 里是错误 json，压测客户端按内容判——够用且保住零拷贝流式
    ctype = "text/event-stream" if body.get("stream") else "application/json"
    return StreamingResponse(gen(), media_type=ctype)


async def proxy_get(request: Request) -> Response:
    alive = sorted(ALIVE)
    if not alive:
        return JSONResponse({"error": "no backend alive"}, status_code=503)
    r = await client.get(BACKENDS[alive[0]] + request.url.path)
    return Response(r.content, status_code=r.status_code,
                    media_type=r.headers.get("content-type"))


async def stats(_: Request) -> Response:
    return JSONResponse({"policy": POLICY, "backends": BACKENDS,
                         "alive": sorted(ALIVE), "counts": COUNTS})


async def health_loop() -> None:
    while True:
        for i, b in enumerate(BACKENDS):
            try:
                r = await client.get(b + "/health", timeout=3)
                (ALIVE.add if r.status_code == 200 else ALIVE.discard)(i)
            except Exception:  # noqa: BLE001
                ALIVE.discard(i)
        await asyncio.sleep(5)


def main() -> None:
    global BACKENDS, POLICY, SKIP, WINDOW, client
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--backends", required=True)
    ap.add_argument("--policy", choices=["rr", "affinity"], default="affinity")
    ap.add_argument("--hash-skip", type=int, default=4000)
    ap.add_argument("--hash-window", type=int, default=2048)
    args = ap.parse_args()
    BACKENDS = [b.rstrip("/") for b in args.backends.split(",") if b]
    POLICY, SKIP, WINDOW = args.policy, args.hash_skip, args.hash_window

    async def lifespan(app):
        global client
        client = httpx.AsyncClient(limits=httpx.Limits(max_connections=512))
        task = asyncio.create_task(health_loop())
        yield
        task.cancel(); await client.aclose()

    app = Starlette(routes=[
        Route("/v1/completions", proxy_post, methods=["POST"]),
        Route("/v1/chat/completions", proxy_post, methods=["POST"]),
        Route("/v1/models", proxy_get, methods=["GET"]),
        Route("/health", proxy_get, methods=["GET"]),
        Route("/router/stats", stats, methods=["GET"]),
    ], lifespan=lifespan)
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
