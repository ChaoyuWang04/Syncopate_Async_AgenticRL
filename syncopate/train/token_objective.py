"""自有 SFT/OPD 数据并行循环的 token 均值目标；不能直接用于 TP/FSDP 分片。"""
from __future__ import annotations

import hashlib
import json
import math


def finish_token_window(parameters, *, local_tokens: int, local_loss_sum: float, device) -> dict:
    """先累积 token loss 的和；这里跨 DP rank 求和，再除以全窗口的 token 数。

    必须在 clip/optimizer 之前调用。各 rank 都参与，包括本地零监督的 rank；
    最后的短窗口使用实际 token 数，没有固定 grad_accum 分母。
    """
    import torch
    import torch.distributed as dist

    params = [p for p in parameters if p.requires_grad]
    if any(hasattr(p, "placements") for p in params):
        raise ValueError("token 窗口归约只适用于完整 DP 副本，不能归约 TP/FSDP 分片")
    distributed = dist.is_available() and dist.is_initialized()
    totals = torch.tensor([local_loss_sum, local_tokens], dtype=torch.float64, device=device)
    if distributed:
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    loss_sum, token_count = totals.tolist()
    if not math.isfinite(loss_sum) or not math.isfinite(token_count) or token_count <= 0:
        raise ValueError("更新窗口没有有效 token 或 loss 非有限")
    for param in params:
        if param.grad is None:
            if not distributed:
                continue
            param.grad = torch.zeros_like(param)
        if distributed:
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
        param.grad.div_(token_count)
    return {"loss": loss_sum / token_count, "supervised_tokens": int(token_count)}


def assert_replicated_parameters(named_parameters) -> dict:
    """只对可训练完整副本做逐字节指纹；相同范数不等于相同权重。"""
    import torch
    import torch.distributed as dist

    digest = hashlib.sha256()
    count = elements = 0
    for name, param in named_parameters:
        if not param.requires_grad:
            continue
        if hasattr(param, "placements"):
            raise ValueError("完整副本指纹不能用于 TP/FSDP 分片")
        tensor = param.detach().to("cpu").contiguous()
        digest.update(json.dumps([name, list(tensor.shape), str(tensor.dtype)]).encode() + b"\0")
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        count += 1
        elements += tensor.numel()
    if count == 0:
        raise ValueError("没有可训练参数，不能验证副本")
    result = {"sha256": digest.hexdigest(), "tensors": count, "elements": elements}
    if dist.is_available() and dist.is_initialized():
        copies = [None] * dist.get_world_size()
        dist.all_gather_object(copies, result)
        if any(copy != result for copy in copies):
            raise ValueError("各 DP rank 的完整可训练参数指纹不同")
    return result
