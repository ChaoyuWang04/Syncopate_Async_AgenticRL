#!/usr/bin/env python
"""E32/E33 · 前缀亲和路由器 v2（4×DP 副本的入口；监听 :8100 ⇒ decider/chatbox 零改动）。

    .venv/bin/python scripts/b4_router.py --policy affinity \
        --backends http://127.0.0.1:8101,http://127.0.0.1:8102,http://127.0.0.1:8103,http://127.0.0.1:8104

v2（B-5 S2 定罪后重写，E33 §7）：v1 对每个请求 request.json() 全量解析 + json= 重新序列化
（~4k token 题面 ×96 并发全挤一个 GIL）⇒ 实测 router 税 2.3 s/单。v2 三刀：
  ① 零解析转发：亲和哈希直接切**原始字节**（body[skip:skip+window]——同 case 的请求体
    字节前缀相同，散列性质与解析后等价）；上游转发 content=raw 零重序列化
  ② 非流式请求（decider 全部如此）走缓冲 Response，不走 StreamingResponse 机器；
    是否流式用字节探测 b'"stream": true'（不解析 JSON）
  ③ uvloop 事件循环
策略：rr（轮询对照）/ affinity（一致性哈希，副本挂了顺移；后台 5s 健康巡检）。
GET /router/stats 看逐副本分发计数。
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
SKIP, WINDOW = 4500, 6144       # 字节口径：JSON 包壳比裸 prompt 多 ~100B，skip 略抬
_rr = 0
client: httpx.AsyncClient | None = None
_STREAM_MARKERS = (b'"stream": true', b'"stream":true')


def pick_bytes(body: bytes) -> int:
    global _rr
    alive = sorted(ALIVE) or list(range(len(BACKENDS)))
    if POLICY == "rr":
        _rr += 1
        return alive[_rr % len(alive)]
    # ⚠️ 自适应窗（08-28 学费 ×2）：固定 skip 对短请求体越界成空切片 ⇒ 全部请求
    #   同哈希塌到一台引擎（E32 goodput after 臂 1713/1713 全进 backend 0，
    #   "四卡臂实际是单卡臂"——分发计数这行判据必须每跑必看）。
    #   短体退到尾部窗（区分度在 user 消息/会话尾部），长体维持越过公共前缀的原窗。
    skip = SKIP if len(body) >= SKIP + 512 else max(0, len(body) - WINDOW)
    key = hashlib.md5(body[skip:skip + WINDOW]).digest()
    want = int.from_bytes(key[:4], "big") % len(BACKENDS)
    for off in range(len(BACKENDS)):
        cand = (want + off) % len(BACKENDS)
        if cand in ALIVE or not ALIVE:
            return cand
    return want


async def proxy_post(request: Request) -> Response:
    raw = await request.body()
    i = pick_bytes(raw)
    COUNTS[i] = COUNTS.get(i, 0) + 1
    url = BACKENDS[i] + request.url.path
    headers = {"content-type": request.headers.get("content-type", "application/json")}
    if any(m in raw for m in _STREAM_MARKERS):
        upstream = client.stream("POST", url, content=raw, headers=headers, timeout=600)

        async def gen():
            async with upstream as r:
                async for chunk in r.aiter_raw():
                    yield chunk

        return StreamingResponse(gen(), media_type="text/event-stream")
    r = await client.post(url, content=raw, headers=headers, timeout=600)
    return Response(r.content, status_code=r.status_code,
                    media_type=r.headers.get("content-type", "application/json"))


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
    ap.add_argument("--hash-skip", type=int, default=4500)
    ap.add_argument("--hash-window", type=int, default=6144)
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
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning",
                loop="uvloop", http="httptools")


if __name__ == "__main__":
    main()
