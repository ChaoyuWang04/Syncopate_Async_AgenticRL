# tests/workers/engine/fsdp/test_degenerate_mesh_grad_sync.py
import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn


def _worker(rank, world, port, q):
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    from verl.workers.engine.fsdp.utils import create_device_mesh, get_sharding_strategy

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world)
    torch.manual_seed(0)
    mesh = create_device_mesh(world_size=world, fsdp_size=1)     # (N, 1)
    strategy = get_sharding_strategy(mesh)                        # NO_SHARD after the fix
    model = FSDP(nn.Linear(64, 64, bias=False).cuda(rank), device_mesh=mesh,
                 sharding_strategy=strategy, use_orig_params=True,
                 sync_module_states=True, device_id=rank)
    x = torch.randn(2, 64, device=rank) * (rank + 1)              # different data per rank
    model(x).square().mean().backward()
    g = next(p.grad for p in model.parameters() if p.grad is not None)
    buf = [None] * world
    dist.all_gather_object(buf, g.detach().norm().item())
    if rank == 0:
        q.put(buf)
    dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs >= 2 GPUs")
def test_fsdp_size_1_gradients_synchronized():
    world = 2
    ctx = mp.get_context("spawn")
    q = ctx.SimpleQueue()
    mp.spawn(_worker, args=(world, 29532, q), nprocs=world, join=True)
    norms = q.get()
    assert max(norms) - min(norms) < 1e-9 * max(norms), (
        f"gradients differ across ranks: {norms} — fsdp_size=1 is not synchronizing"
    )
