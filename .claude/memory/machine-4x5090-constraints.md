---
name: machine-4x5090-constraints
description: 这台 4×5090 机器无 PCIe P2P、卡间只有 6.4 GB/s，因此只能用 DDP，FSDP/TP/序列并行是净亏损
metadata: 
  node_type: memory
  type: project
  originSessionId: 254d8707-7512-4e9b-bd89-6e1eeec39011
  modified: 2026-08-16T13:52:23.285Z
---

⚠️ **容器可能被重建**（临时换卡、重启等）。若重建后 `.venv/bin/python` 断链
（它指向 `/usr/local/bin/python`，新镜像里可能没有），**site-packages 是完好的、别重装**，
重新 `ln -sf /workspace/tools/uv-python/cpython-3.12-linux-x86_64-gnu/bin/python3.12 .venv/bin/python`
并同步改 `.venv/pyvenv.cfg` 的 `home` 即可。git 凭据同样会丢
⇒ **凭据/私钥放 `/workspace/tools/` 下才活得过重建。**

2026-08-13 搬到 4×RTX 5090（RunPod）后实测：

- **P2P 全关**（`can_device_access_peer` 4×4 全 0）。GeForce 从 4090 起被驱动关掉，
  **所有 4×5090 机器都这样**，不是这台坏了。
- **卡间带宽上限 6.4 GB/s**（2 卡 all-reduce bus bandwidth，经主机内存中转）。
  对照 NVLink 的 300–450 GB/s，差约 50 倍。
- 四卡完全对称（PHB / 单 NUMA），RAM 944 GB，`/workspace` 是网络盘。
- ⚠️ **`/workspace` 有卷配额，而 `df` 看不到它**（`df` 报 732 T 可用，实际 100 G 就写不动了；
  2026-08-14 已扩到 200 G）。超限是**静默**的：`cp` 产出 0 字节文件不报错，
  M7 收尾的 27 GB ckpt 被写到一半掐断且训练日志无任何提示。
  ⇒ **判断空间要用写入探针（真写几百 MB 再删），不能信 `df`。**
  ⇒ fully_async 一个 ckpt 27 GB（3 个 rank 全量 state_dict），200 G 也只放得下 7 个。
- 🔴 **`/workspace` 还不支持权限位**（`chmod 700` → 仍是 777，FUSE），
  且容器 **`!cap_sys_admin`** ⇒ 不能 mount，"放个 ext4 镜像再 loop 挂"这条路也堵死。
  ⇒ **PostgreSQL 的 PGDATA 物理上放不进 `/workspace`**（PG 硬性要求 0700）。
  M9 的处理：二进制和 deb 离线包放 `/workspace/tools/postgres`、schema 在仓库，
  **数据库降级成派生产物**，`bash scripts/pg_bootstrap.sh` 一条命令重建。
  ⚠️⚠️ **陷阱：`--save-freq 999` 挡不住收尾时那一次保存。** 每个短实验跑（哪怕只有 12 步、
  纯为计时）结束时照样落一个 **27 GB** 的 ckpt。2026-08-14 跑了三四个计时实验，
  差点把 61 GB 的 MoE 下载挤爆（当时已用 145 G / 配额 200 G）。
  ⇒ **计时/探针类的短跑，跑完就删 `checkpoints/grpo/<exp>/global_step_*`**
  （`dispatched.jsonl` 55K 和 `rollout_dumps` 12M 要留，分析靠它们）。
  ⇒ **下大模型之前先 `du -sh /workspace/*`**（会慢，网络盘；`df` 仍然不可信）。

**因此**：训练侧 `--fsdp-size 1`（DDP）是**必选项**，不是优化。
实测 3 卡 FULL_SHARD 每步 1182 s，1 卡不切分 198 s —— **多给卡慢 6 倍**（5.97×）；
换 DDP 后 3 卡稳态 91.6 s。FSDP / TP / 序列并行在这台机器上**不是慢一点，是不能用**。

⚠️ **这个招牌数字的口径（2026-08-14 核实，见 `docs/infra_exp/E02-data-parallel.md` §6）**：
1182 vs 198 **两边都是首步**，口径一致、比较公平；**但 FULL_SHARD 那次跑第一步之后就崩了，
稳态数据不存在**。单卡/DDP 的首步:稳态是 1.71×/1.90× ⇒ 稳态差距估计仍有 5–6×，方向不变，
**但引用「6 倍」时要说明它只有首步支撑**。另：DDP 那组同时开了 dynamic_bsz（混杂已用 2×2 拆开）。

⇒ 并行不该发生在模型内部，该发生在外部：rollout 副本、时间上的异步、实验级并行 ——
这三种通信量都极小。（PP 理论上对慢网络友好，但 verl 的 megatron/torchtitan 后端没装。）

多卡还必须设 `NCCL_CUMEM_ENABLE=0`（已在 launch_rl 自动处理）：触发条件是
**P2P 缺失 × Ray 只给每个 worker 开放一张卡**两个条件叠加，`NCCL_P2P_DISABLE` 无效。

**flash-attn（2026-08-13 晚起）**：真轮子 2.8.3 已装（预编译，sm_120 kernel 验证过），
存档 `/workspace/wheels/`，`launch_rl` 默认 `flash_attention_2`。**垫片已退役**——
它满足 import 不满足契约：sdpa 路径恒物化 mask，dynamic_bsz 打包 2.19× 倒退（FA2 后待重测）。

细节见 `docs/syncopate/08-machine-and-environment.md`。相关：[[feedback-measure-dont-infer]] [[infra-line-state]]
