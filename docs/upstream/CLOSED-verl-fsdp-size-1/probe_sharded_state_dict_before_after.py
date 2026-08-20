"""补做：修复**前**（HYBRID_SHARD + (N,1) 被钳）的 SHARDED_STATE_DICT 探针。
矩阵那次只探了 E/G 两个修复形态 ⇒ 'both configurations' 那句话当时没有证据。"""
import os, json, torch, torch.nn as nn, torch.distributed as dist, torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import (FullyShardedDataParallel as FSDP, ShardedStateDictConfig,
                                    ShardingStrategy, StateDictType)

def probe(model):
    try:
        with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT,
                                  ShardedStateDictConfig(offload_to_cpu=False)):
            sd = model.state_dict()
        return {"ok": True, "n_entries": len(sd),
                "value_types": sorted({type(v).__name__ for v in sd.values()}),
                "shapes": [list(v.shape) for v in sd.values()][:3]}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"[:300]}

def worker(rank, world):
    os.environ["MASTER_ADDR"] = "127.0.0.1"; os.environ["MASTER_PORT"] = "29541"
    torch.cuda.set_device(rank); dist.init_process_group("nccl", rank=rank, world_size=world)
    out = {}
    for tag, strat, dims in [("A_broken_HYBRID", ShardingStrategy.HYBRID_SHARD, ["ddp","fsdp"]),
                             ("G_fixed_NOSHARD_same_mesh", ShardingStrategy.NO_SHARD, ["ddp2","fsdp2"])]:
        torch.manual_seed(0)
        mesh = init_device_mesh("cuda", (world, 1), mesh_dim_names=dims)
        m = FSDP(nn.Linear(64, 64, bias=False).cuda(rank), device_mesh=mesh,
                 sharding_strategy=strat, use_orig_params=True,
                 sync_module_states=True, device_id=rank)
        out[tag] = probe(m)
    if rank == 0:
        print(json.dumps(out, indent=2))
        same = out["A_broken_HYBRID"].get("value_types") == out["G_fixed_NOSHARD_same_mesh"].get("value_types")
        print("\n修复前后 state_dict 值类型相同 =", same)
    dist.destroy_process_group()

if __name__ == "__main__":
    mp.spawn(worker, args=(3,), nprocs=3, join=True)
