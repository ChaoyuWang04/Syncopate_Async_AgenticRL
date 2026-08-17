"""A6 追问 · 分片惩罚是「次数的代价」还是「字节的代价」？

判据：**固定总字节数，只改集合通信的次数**。
  字节受限 ⇒ 三档耗时应该差不多（都搬同样多的数据）
  次数受限 ⇒ 切得越碎越慢（每次都要付一份固定开销）
"""
import os, sys, time, json
import torch, torch.distributed as dist, torch.multiprocessing as mp

TOTAL_GB = 4.0
SPLITS = [1, 8, 36, 144, 576]      # Qwen3-4B 有 36 层；FSDP 可能按层或更细包


def _w(rank, world, port, out):
    os.environ["MASTER_ADDR"]="127.0.0.1"; os.environ["MASTER_PORT"]=str(port)
    torch.cuda.set_device(rank); dist.init_process_group("nccl", rank=rank, world_size=world)
    res={}
    total = int(TOTAL_GB*(1<<30))
    for k in SPLITS:
        n = total//k//world//2*2                       # 每次 all_gather 每 rank 出 n 字节
        shard = torch.zeros(n, dtype=torch.uint8, device=rank)
        buf   = torch.zeros(n*world, dtype=torch.uint8, device=rank)
        for _ in range(2):                              # warmup
            dist.all_gather_into_tensor(buf, shard)
        torch.cuda.synchronize(rank); dist.barrier()
        t0=time.perf_counter()
        for _ in range(k):
            dist.all_gather_into_tensor(buf, shard)
        torch.cuda.synchronize(rank)
        dt=time.perf_counter()-t0
        moved = n*world*k
        res[k]={"ms":dt*1e3,"moved_GB":moved/(1<<30),"eff_GBps":moved/dt/1e9,
                "per_call_ms":dt*1e3/k}
        del shard, buf; torch.cuda.empty_cache(); dist.barrier()
    if rank==0: json.dump(res, open(out,"w"))
    dist.destroy_process_group()


if __name__ == "__main__":
    os.environ.setdefault("NCCL_CUMEM_ENABLE","0")
    world = int(sys.argv[1]) if len(sys.argv)>1 else 3
    out="/tmp/_gran.json"
    mp.spawn(_w, args=(world,29631,out), nprocs=world, join=True)
    r=json.load(open(out))
    print(f"\n{world} 卡 all_gather · **固定搬运总量 ≈ {TOTAL_GB} GB**，只改切分次数\n")
    print(f"{'次数':>6}{'每次大小':>12}{'总耗时':>11}{'单次耗时':>11}{'有效带宽':>12}")
    print("-"*54)
    for k in SPLITS:
        d=r[str(k)]
        print(f"{k:>6}{d['moved_GB']/k*1024:>10.1f}M{d['ms']:>10.1f}ms"
              f"{d['per_call_ms']:>10.2f}ms{d['eff_GBps']:>11.1f}GB/s")
    a,b=r[str(SPLITS[0])],r[str(SPLITS[-1])]
    print(f"\n★ 切成 {SPLITS[-1]} 次 vs {SPLITS[0]} 次：总耗时 {b['ms']/a['ms']:.1f}× "
          f"（搬运总量完全相同）")
    print(f"  ⇒ {'**次数**的代价占主导' if b['ms']/a['ms']>1.5 else '字节的代价占主导，次数影响小'}")
    os.makedirs("logs",exist_ok=True); json.dump(r,open("logs/e02_granularity.json","w"),indent=1)
