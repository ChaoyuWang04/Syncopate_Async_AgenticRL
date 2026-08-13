---
name: machine-4x5090-constraints
description: 这台 4×5090 机器无 PCIe P2P、卡间只有 6.4 GB/s，因此只能用 DDP，FSDP/TP/序列并行是净亏损
metadata: 
  node_type: memory
  type: project
  originSessionId: 254d8707-7512-4e9b-bd89-6e1eeec39011
  modified: 2026-08-13T17:23:29.602Z
---

2026-08-13 搬到 4×RTX 5090（RunPod）后实测：

- **P2P 全关**（`can_device_access_peer` 4×4 全 0）。GeForce 从 4090 起被驱动关掉，
  **所有 4×5090 机器都这样**，不是这台坏了。
- **卡间带宽上限 6.4 GB/s**（2 卡 all-reduce bus bandwidth，经主机内存中转）。
  对照 NVLink 的 300–450 GB/s，差约 50 倍。
- 四卡完全对称（PHB / 单 NUMA），RAM 944 GB，`/workspace` 是网络盘。

**因此**：训练侧 `--fsdp-size 1`（DDP）是**必选项**，不是优化。
实测 3 卡 FULL_SHARD 每步 1182 s，1 卡不切分 198 s —— **多给卡慢 6 倍**；
换 DDP 后 3.00× 完美线性扩展。FSDP / TP / 序列并行在这台机器上**不是慢一点，是不能用**。

⇒ 并行不该发生在模型内部，该发生在外部：rollout 副本、时间上的异步、实验级并行 ——
这三种通信量都极小。（PP 理论上对慢网络友好，但 verl 的 megatron/torchtitan 后端没装。）

多卡还必须设 `NCCL_CUMEM_ENABLE=0`（已在 launch_rl 自动处理）：触发条件是
**P2P 缺失 × Ray 只给每个 worker 开放一张卡**两个条件叠加，`NCCL_P2P_DISABLE` 无效。

**flash-attn（2026-08-13 晚起）**：真轮子 2.8.3 已装（预编译，sm_120 kernel 验证过），
存档 `/workspace/wheels/`，`launch_rl` 默认 `flash_attention_2`。**垫片已退役**——
它满足 import 不满足契约：sdpa 路径恒物化 mask，dynamic_bsz 打包 2.19× 倒退（FA2 后待重测）。

细节见 `docs/syncopate/08-machine-and-environment.md`。相关：[[feedback-measure-dont-infer]] [[infra-line-state]]
