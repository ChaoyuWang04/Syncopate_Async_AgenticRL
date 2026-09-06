"""运行自己创建的进程组；超时只回收这一组，不按名字杀进程。"""
from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path


def _stop_group(proc: subprocess.Popen, grace: float) -> tuple[str, str]:
    # 即使 shell 已退出，子进程仍可能持有输出管道，进程组仍须回收。
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        return proc.communicate(timeout=grace)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        return proc.communicate(timeout=grace)


def run_guarded(command: str | list[str], *, cwd=None, env=None, timeout: float | None = None,
                completed_path: Path | None = None, exit_grace: float = 60,
                stop_grace: float = 10, poll_interval: float = 1) -> dict:
    """完成文件出现后只等待正常退出，不把质量 WARN 当作杀进程条件。"""
    if any(value <= 0 for value in (exit_grace, stop_grace, poll_interval)) or (timeout is not None and timeout <= 0):
        raise ValueError("超时和检查间隔必须大于零")
    if completed_path is not None and completed_path.exists():
        raise FileExistsError("完成文件已存在，拒绝把旧结果当成本次完成")
    started = time.monotonic()
    deadline = started + timeout if timeout is not None else float("inf")
    completed_at = None
    reason = None
    proc = subprocess.Popen(command, shell=isinstance(command, str), cwd=cwd, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                            start_new_session=True)
    try:
        while True:
            now = time.monotonic()
            if completed_path is not None and completed_at is None and completed_path.is_file():
                completed_at = now
            exit_deadline = completed_at + exit_grace if completed_at is not None else float("inf")
            if now >= min(deadline, exit_deadline):
                reason = "exit_after_result" if exit_deadline <= deadline else "runtime"
                stdout, stderr = _stop_group(proc, stop_grace)
                break
            try:
                stdout, stderr = proc.communicate(timeout=min(poll_interval, deadline - now, exit_deadline - now))
                break
            except subprocess.TimeoutExpired:
                continue
    except BaseException:
        _stop_group(proc, stop_grace)
        raise
    return {"rc": proc.returncode if reason is None else -1,
            "process_returncode": proc.returncode, "out": (stdout + stderr)[-8000:],
            "secs": round(time.monotonic() - started, 1), "timed_out": reason is not None,
            "timeout_reason": reason}
