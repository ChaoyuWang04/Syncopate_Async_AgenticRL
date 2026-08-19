import os, time, torch, torch.nn as nn, torch.distributed as dist, torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy, fully_shard
from torch.distributed.tensor import DTensor
from torch.nn.parallel import DistributedDataParallel as DDP

def worker(rank, world):
    os.environ["MASTER_ADDR"] = "127.0.0.1"; os.environ["MASTER_PORT"] = "29531"
    torch.cuda.set_device(rank); dist.init_process_group("nccl", rank=rank, world_size=world)

    def run(tag, wrap):
        torch.manual_seed(0)
        model = wrap(nn.Linear(256, 256, bias=False).cuda(rank))
        x = torch.randn(4, 256, device=rank) * (rank + 1)      # different data per rank
        model(x).square().mean().backward()
        g = next(p.grad for p in model.parameters() if p.grad is not None)
        # FSDP2 exposes DTensor grads; read the LOCAL shard -- a global .norm()
        # would reduce across ranks and mask the very difference we test for.
        g = g.to_local() if isinstance(g, DTensor) else g
        buf = [None] * world; dist.all_gather_object(buf, g.detach().norm().item())
        if rank == 0:
            same = max(buf) - min(buf) < 1e-9 * max(buf)
            print(f"{tag:<46} {[f'{b:.8f}' for b in buf]}  {'OK' if same else 'NOT SYNCED'}")

    mesh = init_device_mesh("cuda", (world, 1), mesh_dim_names=["ddp", "fsdp"])
    run("(N,1) + HYBRID_SHARD (= fsdp_size=1 today)",
        lambda m: FSDP(m, device_mesh=mesh, sharding_strategy=ShardingStrategy.HYBRID_SHARD,
                       use_orig_params=True, sync_module_states=True, device_id=rank))
    mesh2 = init_device_mesh("cuda", (world, 1), mesh_dim_names=["ddp2", "fsdp2"])
    run("(N,1) + NO_SHARD (= proposed fix, same mesh)",
        lambda m: FSDP(m, device_mesh=mesh2, sharding_strategy=ShardingStrategy.NO_SHARD,
                       use_orig_params=True, sync_module_states=True, device_id=rank))
    mesh3 = init_device_mesh("cuda", (world, 1), mesh_dim_names=["ddp3", "fsdp3"])
    def as_fsdp2(m):
        fully_shard(m, mesh=mesh3); return m
    run("(N,1) + FSDP2 fully_shard (same mesh)", as_fsdp2)
    run("plain DDP (control)", lambda m: DDP(m, device_ids=[rank]))
    dist.destroy_process_group()

if __name__ == "__main__":
    mp.spawn(worker, args=(min(torch.cuda.device_count(), 3),),
             nprocs=min(torch.cuda.device_count(), 3), join=True)
