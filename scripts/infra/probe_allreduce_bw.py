"""卡间 all-reduce 带宽探针（E00 的全局常量之一）。

口径**严格复刻** `docs/infra_exp/README.md` §6 当初测出 6.4 GB/s 的那把尺子：
all-reduce · `NCCL_CUMEM_ENABLE=0` · 1/8/64/256 MB 扫描 · 报 **bus bandwidth**。

    algbw = size / time
    busbw = algbw * 2*(n-1)/n        # nccl-tests 的 all-reduce 定义

⚠️ 旧机器四卡同 NUMA（PHB 全对称），这台是 **2+2 跨 socket**（GPU0/1@node0、
GPU2/3@node1，跨组是 SYS 走 UPI）。所以「6.4 GB/s」不再是单一常数，
必须按 **组内 / 跨组 / 四卡** 分别记。

用法：
    python scripts/infra/probe_allreduce_bw.py                    # 默认全套
    python scripts/infra/probe_allreduce_bw.py --devices 0,2      # 只测一组
    python scripts/infra/probe_allreduce_bw.py --bind             # 把进程绑到 GPU 所属 NUMA 节点
"""

import argparse
import json
import os
import sys
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

SIZES_MB = [1, 8, 64, 256]
WARMUP = 5


def _iters(size_mb: int) -> int:
    """小消息多跑几次，让计时不被单次抖动主导。"""
    return 100 if size_mb <= 8 else (30 if size_mb <= 64 else 20)


def _numa_of(dev: int) -> int | None:
    """从 sysfs 读 GPU 的 NUMA 节点（topo 里的 NUMA Affinity）。

    ⚠️ torch 的 `pci_bus_id` 是 **int**（如 33 = 0x21），不是字符串 —— 第一版按字符串
    拼路径，静默返回 None，输出里的「组内/跨组」标签因此全错。
    """
    try:
        p = torch.cuda.get_device_properties(dev)
        pci = f"{p.pci_domain_id:04x}:{p.pci_bus_id:02x}:{p.pci_device_id:02x}.0"
    except Exception:
        return None
    path = f"/sys/bus/pci/devices/{pci}/numa_node"
    try:
        with open(path) as f:
            n = int(f.read().strip())
        return n if n >= 0 else None
    except OSError:
        return None


def _bind_to_numa(node: int) -> None:
    """把本进程的 CPU 亲和性限制到该 NUMA 节点（不装 numactl，直接用 sched_setaffinity）。"""
    try:
        with open(f"/sys/devices/system/node/node{node}/cpulist") as f:
            spec = f.read().strip()
    except OSError:
        return
    cpus: set[int] = set()
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            cpus.update(range(int(a), int(b) + 1))
        elif part:
            cpus.add(int(part))
    if cpus:
        os.sched_setaffinity(0, cpus)


def _worker(rank: int, devices: list[int], port: int, bind: bool, out_path: str) -> None:
    dev = devices[rank]
    if bind:
        node = _numa_of(dev)
        if node is not None:
            _bind_to_numa(node)

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(dev)
    dist.init_process_group("nccl", rank=rank, world_size=len(devices))

    n = len(devices)
    factor = 2.0 * (n - 1) / n
    results = {}

    for size_mb in SIZES_MB:
        numel = size_mb * 1024 * 1024 // 4  # float32
        buf = torch.ones(numel, dtype=torch.float32, device=dev)

        for _ in range(WARMUP):
            dist.all_reduce(buf)
        torch.cuda.synchronize(dev)
        dist.barrier()

        iters = _iters(size_mb)
        t0 = time.perf_counter()
        for _ in range(iters):
            dist.all_reduce(buf)
        torch.cuda.synchronize(dev)
        elapsed = (time.perf_counter() - t0) / iters

        nbytes = numel * 4
        algbw = nbytes / elapsed / 1e9
        results[size_mb] = {
            "ms": elapsed * 1e3,
            "algbw_GBps": algbw,
            "busbw_GBps": algbw * factor,
        }
        del buf
        torch.cuda.empty_cache()
        dist.barrier()

    if rank == 0:
        with open(out_path, "w") as f:
            json.dump(results, f)
    dist.destroy_process_group()


def run_group(devices: list[int], port: int, bind: bool) -> dict:
    out_path = f"/tmp/_bw_{'-'.join(map(str, devices))}.json"
    mp.spawn(_worker, args=(devices, port, bind, out_path), nprocs=len(devices), join=True)
    with open(out_path) as f:
        return json.load(f)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--devices", default=None, help="逗号分隔，如 0,2；不给则跑全套")
    ap.add_argument("--bind", action="store_true", help="进程绑到 GPU 所属 NUMA 节点")
    ap.add_argument("--port", type=int, default=29511)
    args = ap.parse_args()

    # 复刻基线口径：这一项决定 6.44 vs 2.09 GB/s（README §6 / 08 文档 §5）
    os.environ.setdefault("NCCL_CUMEM_ENABLE", "0")

    numa = {d: _numa_of(d) for d in range(torch.cuda.device_count())}
    print(f"GPU → NUMA: {numa}    NCCL_CUMEM_ENABLE={os.environ['NCCL_CUMEM_ENABLE']}"
          f"    bind={args.bind}\n")

    if args.devices:
        groups = [[int(x) for x in args.devices.split(",")]]
    else:
        groups = [[0, 1], [2, 3], [0, 2], [1, 3], [0, 1, 2, 3]]

    header = f"{'组':<14}{'NUMA':<10}" + "".join(f"{s:>8} MB" for s in SIZES_MB)
    print(header)
    print("-" * len(header))

    all_results = {}
    for i, devs in enumerate(groups):
        nodes = sorted({numa.get(d) for d in devs})
        tag = "组内" if len(nodes) == 1 else "跨 socket"
        res = run_group(devs, args.port + i, args.bind)
        all_results["-".join(map(str, devs))] = res
        cells = "".join(f"{res[str(s)]['busbw_GBps']:>8.2f}   " for s in SIZES_MB)
        print(f"{'GPU ' + ','.join(map(str, devs)):<14}{tag:<8}{cells}")

    print("\n(单位 GB/s，bus bandwidth；对照基线：旧机器 2 卡同 NUMA "
          "1MB 5.20 / 8MB 5.60 / 64MB 5.56 / 256MB 6.44)")

    stamp = os.environ.get("PROBE_TAG", "run")
    dest = f"logs/e00_allreduce_{stamp}{'_bind' if args.bind else ''}.json"
    os.makedirs("logs", exist_ok=True)
    with open(dest, "w") as f:
        json.dump({"numa": numa, "bind": args.bind, "results": all_results}, f, indent=1)
    print(f"原始数据 → {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
