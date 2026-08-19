# 提交包 · FSDP 分片无 16 字节对齐 ⇒ NCCL all_gather 掉 12×（E18）

> **状态：A17 端到端双臂在跑，其余材料齐备；正文待 Chaoyu 过目。**
> 两个目标：**pytorch/pytorch**（issue，修法主推 FSDP2）+ **NVIDIA/nccl**（#413 重开评论，不提新 issue）。
> ⚠️ 与包①②不同：这条已**不影响我们自己的训练**（生产路径定了 DDP 不分片）——纯社区贡献件。

## 一句话

FSDP1/FSDP2 切分只保证「每 rank 元素数相等」，不保证「每 rank 字节数是 16 的倍数」；
NCCL Simple 内核对齐检查是全员投票 + **整段**回退（无头尾剥离）⇒ 非 2 幂次 rank 数下
all_gather 掉 12×。**+4 字节（FSDP1）/ +12 字节（FSDP2）恢复全部带宽。**
上游 6 年前已把修法指派给调用方（#413 "padding is a solution that would always work well"）；
DeepSpeed 私下照做了、veScale 论文明文记载了，PyTorch tracker 上是空白。

## 文件清单

| 文件 | 是什么 |
|---|---|
| [`submission-EN.md`](submission-EN.md) | **PyTorch issue + NCCL #413 重开评论**（英文，直接粘贴） |
| [`analysis.md`](analysis.md) | 中文分析与证据链（悬崖表 / 99.9% 现场 / Broadcast 对照组 / 六条核查情报） |

## 证据（仓库内产物）

```
_audit/infra/e18_fsdp2_alignment.json        ★ FSDP2 实测（2026-08-19）：真实 Qwen3-4B 层，
                                             per-rank 67,319,300 B（%16=4，预测=实测逐字节吻合），
                                             错位钉在 4 个 1-D 参数；该尺寸 2.15→27.03 GB/s = 12.56×（+12 B）
scripts/probe_fsdp2_alignment.py             上面的探针（fully_shard + unshard + 捕获真实 all_gather）
scripts/probe_alignment_cliff.py             悬崖扫描（%16 是唯一开关；FSDP1 真实尺寸 12.2×）
_audit/infra/a14_zero3_*_align.json          真实 FSDP1 训练 99.9% 字节错位（Broadcast 0% 对照组）
logs/a17_zero3_{simple,aligned}_20260819.log 〔跑完回填〕A17 端到端双臂
docs/infra_exp/E18-rank3-allgather-collapse.md  完整实验记录
```

## 提交前最后一眼

- [ ] A17 双臂收尾 → 回填 submission-EN 的两处〔A17〕
- [ ] PyTorch issue 先发 → 链接填进 NCCL 评论
- [ ] FSDP2 的 PR 意向写成"方向确认后再发"（padded size 牵动 DTensor 元数据，先听 maintainer）
