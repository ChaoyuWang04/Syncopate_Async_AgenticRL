"""只监视本容器可见 GPU，保留采样文件并回收自己创建的监视进程。"""
from __future__ import annotations

import subprocess
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def gpu_telemetry(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = "timestamp,uuid,pci.bus_id,name,utilization.gpu,memory.used,power.draw,clocks.sm,clocks.mem"
    with path.open("w") as log:
        process = subprocess.Popen(["nvidia-smi", f"--query-gpu={fields}",
                                    "--format=csv", "--loop-ms=1000"], stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            yield
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
