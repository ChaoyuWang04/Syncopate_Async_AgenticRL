"""vLLM Python 入口的进程启动前置；本模块不导入 CUDA。"""
from __future__ import annotations

import os
import logging
import sys


def prepare_vllm_process() -> str:
    # CLI 会设置 spawn，Python API 不保证设置。项目入口均有 __main__ 保护。
    method = os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    if method != "spawn":
        raise ValueError("当前 vLLM Python 入口要求 VLLM_WORKER_MULTIPROC_METHOD=spawn，拒绝 fork CUDA")
    return method


def redirect_stdout_logging(logger: logging.Logger) -> None:
    """只处理普通 stdout handler，不改测试、宿主应用或文件的日志收集器。"""
    for handler in logger.handlers:
        if type(handler) is logging.StreamHandler and handler.stream is sys.stdout:
            handler.setStream(sys.stderr)


class VLLMEngineLifecycle:
    """共享推理入口显式管理引擎；不依赖解释器退出时的析构顺序。"""

    def close(self) -> None:
        if self.engine is not None:
            self.engine.shutdown(timeout=30)
            self.engine = None
            print("[vllm-process] shutdown=complete", file=sys.stderr, flush=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False


async def run_with_engine(engine: VLLMEngineLifecycle, operation):
    """在事件循环关闭前回收引擎，包括生成或写结果时抛异常的路径。"""
    with engine:
        return await operation()
