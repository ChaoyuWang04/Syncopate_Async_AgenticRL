"""K3-3 · Outbox dispatcher：纯搬运工，不碰业务（课件 CH3 §6.3/§6.8）。

    扫 outbox_jobs（pending ∧ 到期，LIMIT 100）→ publish 到 queue → **同一事务**标 dispatched
    + 写 run.enqueued → 失败退避（cap 300s）→ 超限死信。

顺序铁律：先 publish 后标记。"投了没记"下轮重投无害（worker 五道闸兜住）；
"记了没投"任务永久消失。⇒ 把不可恢复的一步放前面。
门铃：LISTEN outbox_jobs（0003 的触发器），收到就立刻扫；收不到 2s 兜底轮询。
**nudge 是优化，扫表是正确性**——判据行 `[dispatcher] listener 就位` 没打也不影响正确性。

    python -m syncopate.runtime.dispatcher            # 常驻；publish = Celery apply_async
"""
from __future__ import annotations

import asyncio
import contextlib
import os
import time
from typing import Any, Awaitable, Callable

from syncopate.runtime.db import (Database, fetch_due_outbox, mark_outbox_dispatched,
                                  mark_outbox_retry)

Publish = Callable[[dict[str, Any]], Awaitable[None]]


def celery_publish_sync(job: dict[str, Any]) -> None:
    """生产 publish：交给 Celery（K3-4）。task 体只放 run_id/org_id（H23）。"""
    from syncopate.runtime.celery_app import execute_run
    payload = job["payload"]
    execute_run.apply_async(args=[payload["run_id"], payload["org_id"]],
                            queue=job["queue"], task_id=f"outbox-{job['id']}")


async def celery_publish(job: dict[str, Any]) -> None:
    # Celery 客户端是同步的；丢到线程里，别卡住事件循环
    await asyncio.to_thread(celery_publish_sync, job)


class Dispatcher:
    def __init__(self, db: Database, publish: Publish | None = None, *,
                 batch: int = 100, poll_seconds: float = 2.0, org_id: str | None = None,
                 _unsafe_mark_first: bool = False) -> None:
        self.db = db
        self.org_id = org_id
        self.publish = publish or celery_publish
        self.batch = batch
        self.poll_seconds = poll_seconds
        # ⛔ 只给 K3 门槛④ 的负向认证用：先标记后 publish ⇒ 「任务消失」测试必红
        self._unsafe_mark_first = _unsafe_mark_first
        self.stats = {"dispatched": 0, "retried": 0, "dead": 0}

    async def _one(self, job: dict[str, Any]) -> None:
        rid = job["payload"].get("run_id")
        if self._unsafe_mark_first:                 # 负向认证专用路径
            await mark_outbox_dispatched(self.db, job_id=job["id"])
            await self.publish(job)
            self.stats["dispatched"] += 1
            return
        try:
            await self.publish(job)                 # ① 不可恢复的一步放前面
        except Exception as exc:                    # noqa: BLE001 —— 失败不丢，退避
            outcome = await mark_outbox_retry(self.db, job_id=job["id"], error=repr(exc)[:500])
            self.stats["retried" if outcome == "retry" else "dead"] += 1
            print(f"[outbox] 🔴 job={job['id']} run={rid} publish 失败 ⇒ {outcome}: {exc!r}",
                  flush=True)
            return
        if await mark_outbox_dispatched(self.db, job_id=job["id"]):   # ② 同事务标记 + run.enqueued
            self.stats["dispatched"] += 1
            print(f"[outbox] dispatched job={job['id']} run={rid} queue={job['queue']}", flush=True)

    async def dispatch_once(self) -> int:
        jobs = await fetch_due_outbox(self.db, limit=self.batch, org_id=self.org_id)
        for job in jobs:
            await self._one(job)
        return len(jobs)

    async def serve(self, *, stop: asyncio.Event) -> None:
        bell = asyncio.Event()
        listener = asyncio.create_task(self._listen(bell))
        try:
            while not stop.is_set():
                n = await self.dispatch_once()
                if n >= self.batch:
                    continue                        # 积压时连扫，不等铃
                bell.clear()
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(bell.wait(), timeout=self.poll_seconds)
        finally:
            listener.cancel()

    async def _listen(self, bell: asyncio.Event) -> None:
        import asyncpg
        while True:
            try:
                conn = await asyncpg.connect(self.db.dsn)
                await conn.add_listener("outbox_jobs", lambda *_: bell.set())
                print("[dispatcher] listener 就位（门铃只是加速，2s 轮询保正确性）", flush=True)
                while not conn.is_closed():
                    await asyncio.sleep(5)
            except asyncio.CancelledError:
                raise
            except Exception as exc:                # noqa: BLE001
                print(f"[dispatcher] listener 异常 {exc!r}，5s 重连（轮询在扛）", flush=True)
                await asyncio.sleep(5)


async def _serve(poll_seconds: float, org_id: str | None = None) -> None:
    import signal

    db = Database()
    await db.connect(max_size=int(os.environ.get("SYNCOPATE_DISPATCHER_DB_POOL", "4")))
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)
    from syncopate.runtime.celery_app import BROKER_URL_REDACTED
    print(f"[dispatcher] mode=celery broker={BROKER_URL_REDACTED} poll={poll_seconds}s org={org_id or '*'}", flush=True)
    t0 = time.monotonic()
    try:
        await Dispatcher(db, poll_seconds=poll_seconds, org_id=org_id).serve(stop=stop)
    finally:
        await db.close()
        print(f"[dispatcher] 退出，运行 {time.monotonic() - t0:.0f}s", flush=True)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Syncopate outbox dispatcher")
    ap.add_argument("--poll-seconds", type=float, default=2.0)
    ap.add_argument("--org-id", default=None, help="只投一个租户（默认全局）")
    a = ap.parse_args()
    asyncio.run(_serve(a.poll_seconds, a.org_id))


if __name__ == "__main__":
    main()
