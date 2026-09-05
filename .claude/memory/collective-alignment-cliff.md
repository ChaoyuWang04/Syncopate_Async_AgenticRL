---
name: collective-alignment-cliff
description: 集合通信有个 16 字节对齐悬崖——分块字节数不被 16 整除，NCCL 的 Simple kernel 整段放弃向量化，all_gather 掉 12×
metadata:
  node_type: memory
  type: project
  modified: 2026-08-17T12:00:00.000Z
---

**2026-08-17 在 4×5090 上刨到底的一条**（完整叙事见 `docs/archive/infra_exp/legacy-4x5090/E18-rank3-allgather-collapse.md`）。

## 结论

```
每 rank 分块字节数 % 16 == 0   →  all_gather 13.2 GB/s
                   % 16 != 0   →  all_gather  1.1 GB/s      ★ 差 12×
% 128 完全不相干（16/32/48/64/80/96/112 全都快）
```

**16 字节 = 128 位 = GPU 向量化访存宽度（`float4`/`LDG.128`）。**
分块字节数不是 16 的倍数 ⇒ NCCL 的 Simple kernel **整段**退化成标量路径 ——
实证：24 MB **只多 4 个字节**，整段从 13.2 掉到 1.1（不是只有尾巴慢）。

**为什么表现成「3 卡受诅咒」**：常见张量尺寸是 2 的幂次，**÷3 破坏对齐、÷4 天然保持**。
⇒ 3 卡 ZeRO-3 的 `update_actor` 是 DDP 的 **6.02×**，4 卡只有 **1.54×**。

**为什么 all_gather 最惨**：它是纯数据搬运、全靠向量化；`reduce_scatter` 带规约计算，
同样错位只掉 1.3–1.6×；`all_reduce`/`broadcast` 在整块对齐缓冲上跑，不中招。

## How to apply

1. **引用「卡间带宽」之前先问是哪个算子、几个 rank。** 用 all-reduce 的数推算 all_gather
   会错 8–12 倍 —— 我就是这么把预测做错的。尺子：`scripts/infra/probe_collective_bw.py`、
   `probe_alignment_cliff.py`。
2. **应急手段** `NCCL_PROTO=LL128`（3 卡 ZeRO-3 47.94→14.40 s，3.33×），
   ⚠️ **代价 all_reduce −30% / broadcast −41%** ⇒ **不能全局开**；
   已写进 `launch_rl`：只在 `fsdp_size>1` 时自动带上 + 判据行。**DDP 走 all_reduce，不要开。**
3. **治本是把分片补齐到 16 字节**，不是换协议。
4. ⚠️ **未闭合**：机制已证实，但**没验证 verl 的 ZeRO-3 确实产生了错位分片**（队列 A14，
   要按**字节加权**统计，不是按调用次数）。在那之前别把「6.02× 由对齐造成」当定论。

## ★ 方法学：这条链上我错了三次，每次都是「停得太早」

```
错① 「带宽 ×4 ⇒ 惩罚减半」        用错尺子（25.6 是 all-reduce 的数）
错② 「那就是次数的代价」          切 576 次反而更快 ——「不是 A」不等于「就是 B」
错③ 「NCCL 成本模型选错了协议」    实测四算子在 3/4 卡全部选 RING+SIMPLE，选择根本没变
对④ 「按机制预测 4 卡该好」        实测 1.54×，误差 6% —— **唯一一个基于测量而非推理的预测**
```

⇒ ★★ **停在③就会得到一个「看起来完整、实际是错的」根因，而且导出错误建议（换协议）。
多问一句「那它为什么选错」，才碰到真正的地板。**

相关：[[feedback-measure-dont-infer]] [[machine-4x5090-constraints]] [[infra-line-state]]
