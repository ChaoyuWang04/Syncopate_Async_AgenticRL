"""K3-4 · Celery + Redis：只承担"投递/消费/ack/分队列"，⛔ 不承担 run 状态、事务、幂等、lease
（课件 CH3 §4："装个 Celery ≠ 有了 Agent 后端"）。

每个配置项对应 28 号坑表一条，别"顺手"改回默认：
    task_acks_late=True                 C-01  先执行后 ack；反了 = 先 ack 再写库（H29，run 永久卡死）
    worker_prefetch_multiplier=1        C-03  长任务不预取；进程死了那批不用等 visibility_timeout
    task_reject_on_worker_lost=True     C-04  子进程被 kill -9 也重投（安全前提 = 五道闸）
    task_ignore_result=True             C-05  事实在 PG，不往 Redis 写第二份
    json 序列化                          C-07  pickle + Redis 密码泄漏 = RCE
    visibility_timeout                  C-02  ≥ 2×lease TTL；重投后定向 claim 只认 queued ⇒ 重复无害
    broker_connection_retry_on_startup  C-15  worker 起在 Redis 之前不退出
    ⛔ 不设 task_time_limit（硬杀 = H105）；取消/超时全部协作式（ActionGate 安全点 + K9 max_duration）
    ⛔ 不用 countdown/eta 做退避（C-14）；退避在 outbox.next_attempt_at

worker 进程模型（C-09）：prefork 子进程各自一个长活事件循环 + 自己的 asyncpg 池
（worker_process_init 里建，⛔ 不能在父进程建再 fork）。任务里 run_until_complete。

    celery -A syncopate.runtime.celery_app worker -Q interactive -c 4 -n w1@%h
"""
from __future__ import annotations

import asyncio
import os
import re
import socket
from typing import Any

from celery import Celery, signals
from kombu import Queue

from syncopate.runtime.db import QUEUE_BATCH, QUEUE_INTERACTIVE, QUEUE_MAINTENANCE

BROKER_URL = os.environ.get("SYNCOPATE_REDIS_URL", "redis://:syncopate-dev@127.0.0.1:6379/0")
BROKER_URL_REDACTED = re.sub(r"://:[^@]*@", "://:<pass>@", BROKER_URL)
LEASE_TTL_S = int(os.environ.get("SYNCOPATE_LEASE_TTL_S", "60"))          # K3-7：TTL = 3×心跳
HEARTBEAT_S = max(1, LEASE_TTL_S // 3)
VISIBILITY_TIMEOUT_S = int(os.environ.get("SYNCOPATE_VISIBILITY_TIMEOUT_S", str(max(2 * LEASE_TTL_S, 900))))
INFRA_RETRY_COUNTDOWN_S = (5, 15, 45)   # 基础设施错误：短退避重投（业务退避走 outbox）

app = Celery("syncopate", broker=BROKER_URL)
app.conf.update(
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    task_reject_on_worker_lost=True,
    task_ignore_result=True,
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    broker_transport_options={"visibility_timeout": VISIBILITY_TIMEOUT_S},
    broker_connection_retry_on_startup=True,
    task_default_queue=QUEUE_INTERACTIVE,
    task_queues=(Queue(QUEUE_INTERACTIVE), Queue(QUEUE_BATCH), Queue(QUEUE_MAINTENANCE)),
    task_create_missing_queues=True,
    worker_hijack_root_logger=False,
    worker_send_task_events=False,
)


class _State:
    """每个 prefork 子进程一份：事件循环 + 连接池 + Worker。"""
    loop: asyncio.AbstractEventLoop | None = None
    db: Any = None
    worker: Any = None
    worker_id: str = ""


_state = _State()


@signals.worker_process_init.connect
def _on_process_init(**_: Any) -> None:
    from syncopate.runtime.worker import WorkerConfig, build_worker
    _state.loop = asyncio.new_event_loop()
    asyncio.set_event_loop(_state.loop)
    _state.worker_id = f"celery-{socket.gethostname()}-{os.getpid()}"
    cfg = WorkerConfig(worker_id=_state.worker_id, lease_seconds=LEASE_TTL_S,
                       org_id=os.environ.get("SYNCOPATE_WORKER_ORG_ID") or None)
    _state.db, _state.worker = _state.loop.run_until_complete(
        build_worker(cfg, pool_size=int(os.environ.get("SYNCOPATE_WORKER_DB_POOL", "4"))))
    print(f"[worker-init] pid={os.getpid()} worker_id={_state.worker_id} "
          f"pool={os.environ.get('SYNCOPATE_WORKER_DB_POOL', '4')} lease_ttl={LEASE_TTL_S}s "
          f"heartbeat={HEARTBEAT_S}s visibility_timeout={VISIBILITY_TIMEOUT_S}s", flush=True)


@signals.worker_process_shutdown.connect
def _on_process_shutdown(**_: Any) -> None:
    if _state.loop is not None and _state.db is not None:
        _state.loop.run_until_complete(_state.db.close())
        _state.loop.close()


def _is_infrastructure_error(exc: BaseException) -> bool:
    """基础设施错误（claim 失败/DB 断连/broker 不可用）归 worker 层重投；业务错误在 execute 内消化（§14.3）。"""
    import asyncpg
    return isinstance(exc, (asyncpg.PostgresConnectionError, asyncpg.InterfaceError,
                            ConnectionError, OSError))


async def _execute(run_id: str, org_id: str) -> str:
    from syncopate.runtime.db import claim_run
    from syncopate.runtime.worker import LeaseHeartbeat

    db, worker = _state.db, _state.worker
    claimed = await claim_run(db, worker_id=_state.worker_id, lease_seconds=LEASE_TTL_S,
                              org_id=org_id, run_id=run_id)
    if claimed is None:
        # 五道闸第 1/2 道：终态或别人正持有 ⇒ 不执行、ack 掉这条消息（重投无害的物理形态）
        print(f"[celery] run={run_id} 不可领取（终态/非 queued/他人持有）⇒ 跳过并 ack", flush=True)
        return "skipped"
    hb = LeaseHeartbeat(db, org_id=org_id, run_id=run_id, worker_id=_state.worker_id,
                        ttl_seconds=LEASE_TTL_S, interval_seconds=HEARTBEAT_S)
    hb.start()
    try:
        await worker.execute_claimed(claimed, heartbeat=hb)
    finally:
        await hb.stop()
    if os.environ.get("SYNCOPATE_TEST_CRASH_AFTER_FINISH") == "1":
        # 测试钩子（K3 门槛③）：库已写终态、还没 ack 就死 ⇒ 重投后新 worker 必须"读到终态直接跳过"
        print(f"[celery] TEST 钩子：run={run_id} 终态已写，ack 前 os._exit", flush=True)
        os._exit(1)
    return "done"


@app.task(name="syncopate.execute_run", bind=True, max_retries=len(INFRA_RETRY_COUNTDOWN_S))
def execute_run(self, run_id: str, org_id: str) -> str:
    """worker 层很薄：claim → 执行 → ack。业务错误不到这里；只有基础设施错误才重投。"""
    assert _state.loop is not None, "worker_process_init 没跑：连接池不在本进程"
    try:
        return _state.loop.run_until_complete(_execute(run_id, org_id))
    except Exception as exc:                        # noqa: BLE001
        if _is_infrastructure_error(exc):
            n = self.request.retries
            from syncopate.runtime.log import log_event
            log_event("celery", "infrastructure_error", level="error", run_id=run_id, org_id=org_id,
                      retries=n, error=repr(exc)[:300], worker_id=_state.worker_id)
            if n < len(INFRA_RETRY_COUNTDOWN_S):
                print(f"[celery] 🔴 run={run_id} 基础设施错误 {exc!r} ⇒ {INFRA_RETRY_COUNTDOWN_S[n]}s 后重投"
                      f"（第 {n + 1} 次）", flush=True)
                raise self.retry(exc=exc, countdown=INFRA_RETRY_COUNTDOWN_S[n])
            # 超限：死信（病历）。run 状态留给 sweeper（K8）判定，worker 层不改 run 状态（H102）
            from syncopate.runtime.db import dead_letter
            async def _dl():
                async with _state.db.tx() as conn:
                    await dead_letter(conn, org_id=org_id, source="worker", job_type="execute_run",
                                      payload={"run_id": run_id, "org_id": org_id},
                                      attempts=n + 1, error={"error": repr(exc)[:500]})
            _state.loop.run_until_complete(_dl())
            print(f"[dlq] run={run_id} 基础设施错误重投 {n + 1} 次仍失败 ⇒ dead_letter_jobs", flush=True)
            return "dead_letter"
        raise
