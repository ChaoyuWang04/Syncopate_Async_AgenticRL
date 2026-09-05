"""负向测试：让 rank1 的梯度在同步后被污染（权重静默发散的本体形态），
验证 sft.py 的权重一致断言**会红**。

[blank-thresholds-are-not-passes] 家规：判据必须被证明能对自己失败。
注意不能直接跳过 all_reduce —— 那会先死锁（集合通信次数错位），到不了断言；
要验的是「通信都在跑、但各 rank 权重仍在悄悄分叉」这类静默降级。
预期：epoch1 末尾断言触发「rank 间权重发散」，进程崩溃 —— 崩溃 = 测试通过。
"""
import os
import sys

sys.path.insert(0, "/workspace/Syncopate_Async_AgenticRL")
from syncopate.train import sft  # noqa: E402

rank = int(os.environ.get("RANK", "0"))
_orig_all_reduce = sft.dist.all_reduce


def broken_all_reduce(tensor, op=None, **kw):
    if op is not None:
        kw["op"] = op
    out = _orig_all_reduce(tensor, **kw)
    if rank == 1 and op == sft.dist.ReduceOp.AVG:
        tensor.mul_(1.01)  # 同步照常完成，但 rank1 的梯度悄悄多 1% —— 权重开始发散
    return out


sft.dist.all_reduce = broken_all_reduce
sys.exit(sft.main([
    "--model", "models/Qwen3-0.6B",
    "--train-file", "data/sft/v14/val.parquet",
    "--val-file", "data/sft/v14/val.parquet",
    "--out", "/tmp/sft_smoke_broken",
    "--epochs", "1", "--batch-size", "2", "--grad-accum", "4",
    "--no-wandb",
]))
