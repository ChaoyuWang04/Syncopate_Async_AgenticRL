import os,time,json,torch,torch.distributed as dist,torch.multiprocessing as mp
BASE=24*(1<<20)
OFFS=[0,4,8,12,16,32,48,64,80,96,112,124]      # 每 rank 分块 = BASE + off 字节
def w(rank,n,port,out):
    os.environ["MASTER_ADDR"]="127.0.0.1"; os.environ["MASTER_PORT"]=str(port)
    torch.cuda.set_device(rank); dist.init_process_group("nccl",rank=rank,world_size=n)
    res={}
    for off in OFFS:
        nb=BASE+off; per=nb//4
        s=torch.ones(per,dtype=torch.float32,device=rank); g=torch.zeros(per*n,dtype=torch.float32,device=rank)
        rs_in=torch.ones(per*n,dtype=torch.float32,device=rank); rs_out=torch.zeros(per,dtype=torch.float32,device=rank)
        for _ in range(3): dist.all_gather_into_tensor(g,s)
        torch.cuda.synchronize(); dist.barrier()
        t=time.perf_counter()
        for _ in range(15): dist.all_gather_into_tensor(g,s)
        torch.cuda.synchronize(); ag=per*4/((time.perf_counter()-t)/15)/1e9
        for _ in range(3): dist.reduce_scatter_tensor(rs_out,rs_in)
        torch.cuda.synchronize(); dist.barrier()
        t=time.perf_counter()
        for _ in range(15): dist.reduce_scatter_tensor(rs_out,rs_in)
        torch.cuda.synchronize(); rs=per*n*4/((time.perf_counter()-t)/15)/1e9
        res[off]={"bytes":per*4,"m16":(per*4)%16,"ag":ag,"rs":rs}
        del s,g,rs_in,rs_out; torch.cuda.empty_cache(); dist.barrier()
    if rank==0: json.dump(res,open(out,"w"))
    dist.destroy_process_group()
if __name__=="__main__":
    mp.set_start_method("spawn",force=True); out="/tmp/_cl.json"
    mp.spawn(w,args=(3,30011,out),nprocs=3,join=True)
    r=json.load(open(out))
    print(f"\n  {'每rank字节':>12}{'%16':>5}{'%128':>6}{'all_gather':>13}{'reduce_scatter':>16}")
    print("  "+"-"*54)
    for o in OFFS:
        d=r[str(o)]; print(f"  {d['bytes']:>12}{d['m16']:>5}{d['bytes']%128:>6}{d['ag']:>11.1f}GB/s{d['rs']:>13.1f}GB/s")
