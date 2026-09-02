"""K3 进程级验收（真 Redis + 真 celery worker 子进程）：

  ① 端到端：create_run（outbox 同事务）→ dispatcher publish → celery worker 定向 claim → 执行
     → 事件流 run.created → run.enqueued → run.started → 终态/等审批
  ② 门槛③「不重」：终态已写、ack 前 worker 子进程 os._exit ⇒ Celery（acks_late +
     reject_on_worker_lost）重投 ⇒ 新子进程读到非 queued 直接跳过并 ack：run 不重跑、不卡死、
     attempts 仍是 1、队列最终为空

⚠️ 没有 Redis / PG 时整文件 skip（不是通过）。每个测试用独立队列名，不和常驻 worker 抢。
"""
from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

import pytest

from syncopate.runtime.db import Database, create_run
from syncopate.runtime.dispatcher import Dispatcher
from tests.runtime.test_api import _pg_available

REPO = Path(__file__).resolve().parents[2]
REDIS_URL = os.environ.get("SYNCOPATE_REDIS_URL", "redis://:syncopate-dev@127.0.0.1:6379/0")


def _redis_available() -> bool:
    try:
        import redis
        return bool(redis.Redis.from_url(REDIS_URL, socket_connect_timeout=1).ping())
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not (_pg_available() and _redis_available()),
                                reason="需要 PostgreSQL + Redis：pg_bootstrap.sh / redis_bootstrap.sh")


class CeleryWorker:
    """起一个真 celery worker（prefork，-c 1），只订阅一个测试专用队列。"""

    def __init__(self, queue: str, extra_env: dict[str, str] | None = None) -> None:
        self.queue = queue
        self.log = REPO / "logs" / "runtime" / f"celery-test-{queue}.log"
        self.log.parent.mkdir(parents=True, exist_ok=True)
        self.name = f"t{uuid.uuid4().hex[:6]}@%h"
        env = {**os.environ, "SYNCOPATE_REDIS_URL": REDIS_URL, **(extra_env or {})}
        self._fh = open(self.log, "w")
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "celery", "-A", "syncopate.runtime.celery_app", "worker",
             "-Q", queue, "-c", "1", "-P", "prefork", "-n", self.name, "--loglevel=INFO",
             "--without-gossip", "--without-mingle"],
            cwd=REPO, env=env, stdout=self._fh, stderr=subprocess.STDOUT)

    def wait_ready(self, timeout: float = 30.0) -> None:
        from syncopate.runtime.celery_app import app
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.proc.poll() is not None:
                raise RuntimeError(f"celery worker 提前退出，看 {self.log}")
            if "[worker-init]" in self.log.read_text(errors="ignore"):
                if app.control.ping(timeout=1.0):
                    return
            time.sleep(0.5)
        raise RuntimeError(f"celery worker {timeout}s 内没就绪，看 {self.log}")

    def stop(self) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self._fh.close()

    def text(self) -> str:
        return self.log.read_text(errors="ignore")


async def _wait_status(db: Database, org: str, run_id: str, *, leave: set[str], timeout: float = 30.0) -> str:
    t0 = time.time()
    while True:
        async with db.tx() as conn:
            st = await conn.fetchval("SELECT status FROM agent_runs WHERE org_id=$1 AND run_id=$2", org, run_id)
        if st not in leave:
            return st
        if time.time() - t0 > timeout:
            return st
        await asyncio.sleep(0.25)


async def _kinds(db, org, run_id):
    async with db.tx() as conn:
        return [r["kind"] for r in await conn.fetch(
            "SELECT kind FROM run_events WHERE org_id=$1 AND run_id=$2 ORDER BY seq", org, run_id)]


def _drive(queue: str, extra_env: dict[str, str] | None = None):
    """create → route outbox to test queue → dispatch → wait。返回 (状态, 事件, attempts, worker 日志)。"""
    org, run_id = f"org_{uuid.uuid4().hex[:8]}", f"run_{uuid.uuid4().hex[:12]}"
    w = CeleryWorker(queue, extra_env)
    try:
        w.wait_ready()

        async def go():
            db = Database()
            await db.connect(max_size=3)
            try:
                await create_run(db, org_id=org, run_id=run_id, user_message="集成测试")
                async with db.tx() as conn:
                    await conn.execute("UPDATE outbox_jobs SET queue=$1 WHERE org_id=$2", queue, org)
                n = await Dispatcher(db, org_id=org).dispatch_once()
                assert n == 1, f"dispatcher 应投 1 条，实投 {n}"
                st = await _wait_status(db, org, run_id, leave={"queued", "running"})
                async with db.tx() as conn:
                    attempts = await conn.fetchval("SELECT attempts FROM agent_runs WHERE org_id=$1 AND run_id=$2", org, run_id)
                    ob = await conn.fetchval("SELECT status FROM outbox_jobs WHERE org_id=$1", org)
                return st, await _kinds(db, org, run_id), attempts, ob
            finally:
                await db.close()

        st, kinds, attempts, ob = asyncio.run(go())
        # 给 Celery 一点时间完成 ack / 重投 / 跳过
        time.sleep(3.0)
        import redis
        qlen = redis.Redis.from_url(REDIS_URL).llen(queue)
        return st, kinds, attempts, ob, qlen, w.text()
    finally:
        w.stop()


def test_end_to_end_outbox_dispatch_celery_execute() -> None:
    queue = f"test-{uuid.uuid4().hex[:8]}"
    st, kinds, attempts, ob, qlen, log = _drive(queue)
    assert st in ("succeeded", "failed", "cancelled", "waiting_for_user"), (st, kinds, log[-2000:])
    # ⚠️ run.enqueued 可能排在 run.started **之后**：dispatcher 先 publish 再标记（顺序铁律），
    #   worker 比标记事务快时就先写了 run.started。这是设计的必然（28 S-16），不是 bug。
    assert kinds[0] == "run.created" and {"run.enqueued", "run.started"} <= set(kinds), kinds
    assert attempts == 1 and ob == "dispatched"
    assert qlen == 0, "消息没被 ack 掉"
    assert "[worker-init]" in log, "子进程没跑 worker_process_init（连接池不在本进程）"


def test_crash_after_terminal_before_ack_is_skipped_on_redelivery() -> None:
    """K3 门槛③：ack 前 kill worker ⇒ 重投后新 worker 读到终态直接跳过并 ack。"""
    queue = f"test-{uuid.uuid4().hex[:8]}"
    st, kinds, attempts, ob, qlen, log = _drive(queue, {"SYNCOPATE_TEST_CRASH_AFTER_FINISH": "1"})
    assert st in ("succeeded", "failed", "cancelled", "waiting_for_user"), (st, kinds)
    assert "ack 前 os._exit" in log, "测试钩子没触发：这一跑没有在 ack 前崩"
    assert "不可领取" in log, ("重投后新子进程没有走到'跳过并 ack'——要么没重投（reject_on_worker_lost 没生效），"
                              f"要么重跑了。日志尾：{log[-1500:]}")
    assert attempts == 1, f"run 被重跑了：attempts={attempts}"
    assert kinds.count("run.started") == 1, kinds
    assert qlen == 0, "重投的消息没被 ack 掉 ⇒ 会无限重投"
