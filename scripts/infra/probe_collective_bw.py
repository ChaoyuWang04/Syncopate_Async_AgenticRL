"""A8 · E00 · 集合通信带宽的**分算子**口径。

为什么必须分算子：2026-08-17 用 all-reduce 的 25.6 GB/s 去推算 ZeRO-3 的
all-gather 代价，**错了 8 倍**。⇒ 「卡间带宽」不是一个数。

口径同 nccl-tests：algbw = 参与字节 / 时间；busbw = algbw × 系数（各算子不同）。
"""
import json, os, sys, time
import torch, torch.distributed as dist, torch.multiprocessing as mp

SIZE_MB = 256
ITERS, WARMUP = 20, 5
OPS = ["all_reduce", "all_gather", "reduce_scatter", "broadcast"]
FACTOR = {"all_reduce": lambda n: 2*(n-1)/n, "all_gather": lambda n: (n-1)/n,
          "reduce_scatter": lambda n: (n-1)/n, "broadcast": lambda n: 1.0}


def _w(rank, devs, port, out):
    n = len(devs); dev = devs[rank]
    os.environ["MASTER_ADDR"]="127.0.0.1"; os.environ["MASTER_PORT"]=str(port)
    torch.cuda.set_device(dev); dist.init_process_group("nccl", rank=rank, world_size=n)
    per = SIZE_MB*(1<<20)//4//n          # 每 rank 的份额，保证整除
    numel = per*n
    res = {}
    for op in OPS:
        if op == "all_gather":
            src = torch.ones(per, dtype=torch.float32, device=dev)
            dst = torch.zeros(numel, dtype=torch.float32, device=dev)
            call = lambda: dist.all_gather_into_tensor(dst, src)
        elif op == "reduce_scatter":
            src = torch.ones(numel, dtype=torch.float32, device=dev)
            dst = torch.zeros(per, dtype=torch.float32, device=dev)
            call = lambda: dist.reduce_scatter_tensor(dst, src)
        else:
            buf = torch.ones(numel, dtype=torch.float32, device=dev)
            call = (lambda: dist.all_reduce(buf)) if op == "all_reduce" else (lambda: dist.broadcast(buf, 0))
        for _ in range(WARMUP): call()
        torch.cuda.synchronize(dev); dist.barrier()
        t0 = time.perf_counter()
        for _ in range(ITERS): call()
        torch.cuda.synchronize(dev)
        dt = (time.perf_counter()-t0)/ITERS
        nbytes = numel*4                      # 各算子统一按「参与的总缓冲字节」计
        algbw = nbytes/dt/1e9
        res[op] = {"ms": dt*1e3, "algbw": algbw, "busbw": algbw*FACTOR[op](n)}
        del call; torch.cuda.empty_cache(); dist.barrier()
    if rank == 0: json.dump(res, open(out, "w"))
    dist.destroy_process_group()


def run(devs, port):
    out = f"/tmp/_cbw_{len(devs)}.json"
    mp.spawn(_w, args=(devs, port, out), nprocs=len(devs), join=True)
    return json.load(open(out))


if __name__ == "__main__":
    os.environ.setdefault("NCCL_CUMEM_ENABLE", "0")
    groups = {"3卡(trainer)": [0,1,2], "4卡": [0,1,2,3], "2卡组内": [0,1], "2卡跨socket": [0,2]}
    allr = {}
    print(f"消息 {SIZE_MB} MB · {ITERS} 次取平均 · NCCL_CUMEM_ENABLE=0\n")
    print(f"{'组':<14}" + "".join(f"{o:>17}" for o in OPS))
    print("-"*(14+17*len(OPS)))
    for i,(name, devs) in enumerate(groups.items()):
        r = run(devs, 29701+i); allr[name] = r
        print(f"{name:<14}" + "".join(f"{r[o]['algbw']:>8.1f}/{r[o]['busbw']:<8.1f}" for o in OPS))
    print("\n(每格 = algbw / busbw，GB/s)")
    t = allr["3卡(trainer)"]
    print(f"\n★ 3 卡上 all_reduce busbw {t['all_reduce']['busbw']:.1f} GB/s "
          f"vs all_gather algbw {t['all_gather']['algbw']:.1f} GB/s "
          f"= **{t['all_reduce']['busbw']/t['all_gather']['algbw']:.1f}×**")
    print("  ⇒ 推算 FSDP/ZeRO 的代价必须用 all_gather 那一列，不能用 all-reduce 的数")
    os.makedirs("logs", exist_ok=True); json.dump(allr, open("logs/e00_collective_bw.json","w"), indent=1)
