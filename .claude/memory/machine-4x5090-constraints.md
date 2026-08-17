---
name: machine-4x5090-constraints
description: 4×5090 无 P2P 所以只能 DDP；★2026-08-17 换了新机器（2+2 跨 socket / PCIe Gen5 / 本地 XFS），旧机器的带宽与存储约束大半已失效
metadata:
  node_type: memory
  type: project
  originSessionId: 254d8707-7512-4e9b-bd89-6e1eeec39011
  modified: 2026-08-17T07:10:00.000Z
---

## ★★★ 2026-08-17：换机器了。分清哪些还成立

**没变的（这条线的支柱，仍然成立）**：
- **P2P 全关**（`can_device_access_peer` 4×4 全 0，实测复验）。GeForce 从 4090 起被驱动关掉，
  **所有 4×5090 都这样**。⇒ 卡间一律经主机内存中转（NCCL 走 `SHM/direct/direct`）。
- **训练侧 `--fsdp-size 1`（DDP）仍是必选**。⚠️ 但支撑它的那个数字变了，见下。
- 多卡必须 `NCCL_CUMEM_ENABLE=0`（launch_rl 已自动处理）。

**变了的 —— 引用旧数字之前先看这里**：

| | 旧机器 | 🆕 这台 |
|---|---|---|
| CPU/拓扑 | 四卡对称，**单 NUMA**（PHB） | **2 socket EPYC 9V74，2+2**：GPU0/1@node0、GPU2/3@node1，跨组 `SYS` 走 UPI |
| PCIe | max **Gen4**（空闲报 Gen1） | **Gen5 x16**（32 GT/s；空闲仍报 Gen1，属正常） |
| all-reduce busbw @256MB | **6.44 GB/s** | 组内 **28.8** · 跨 socket **22.2** · **四卡 25.6**（= DDP 实际走的） |
| `/workspace` | mfs 网络盘，FUSE | **本地 XFS**（`/dev/nvme0n1p1`），300 G |
| 权限位 | ❌ `chmod 700` → 仍 777 | ✅ 生效 ⇒ **PGDATA 可以放 `/workspace`**，数据库不再是"重启即丢" |
| 隐形卷配额 | 🔴 `df` 看不到，超限**静默截断**（丢过 27 GB ckpt） | 未复现，`df` 与写入探针一致 |
| `/`（overlay） | 不是瓶颈 | 🔴 **只有 16 G**，`/root` `/tmp` `/var` 全在上面 |

⇒ **跨 socket 只掉 22%，而且 NUMA 绑定救不回来**（`sched_setaffinity` 实测 22.23→22.34，噪声内）
—— 但换算到负载上：DDP 梯度 260 MB 从 40.4 ms 变 10.2 ms，跨 socket 的净代价 **1.2 ms/步**，
而一步 32–91 s ⇒ **占 0.004%，不值得为它换机器**。尺子在 `scripts/probe_allreduce_bw.py`，
数据 `logs/e00_allreduce_*.json`。（torch 2.9/NCCL 2.27.5 与 torch 2.11/NCCL 2.28.9 量出来一致。）

🔴 **但基线整体作废**：`6.4 GB/s` 是 README §6 的全局常量，E02/E07/E11/E12 都拿它当分母；
所有 before 数字（117.8 / 74.1 s/步、1.59×、占空比 31%、权重同步 59.8 s）都是旧机器的。
**换机器救不了这个 —— 旧机器已经没了，任何新机器都不是它。** ⇒ 基线必须在这台重测。
最可能被推翻的是 **E02「FSDP 慢 6 倍」**（因果就是 6.4 GB/s，带宽 ×4 后可能只剩 ~2×）；
最稳的是 E11 监督密度 4.17%（结构性特征，与硬件无关）。

## 16 G overlay：所有缓存都要改道

`/` 是 16 G overlay 且 recycle 即丢，`/workspace` 才是持久卷（`workspace_is_volume=true`）。
镜像 `bootstrap.sh` 只管了 uv/npm/HF。**训练栈这些默认全落 overlay**，写在 `/workspace/.env`：
`RAY_TMPDIR`（★ Ray 溢写，最容易撑爆）· `TRITON_CACHE_DIR` · `TORCH_EXTENSIONS_DIR` ·
`VLLM_CACHE_ROOT` · `CUDA_CACHE_PATH` · `XDG_CACHE_HOME` · `TMPDIR` · `PIP_CACHE_DIR` ·
`PGDATA` · `LD_LIBRARY_PATH`（PG 的 libpq）。
⇒ **每个 shell 先 `set -a; . /workspace/.env; set +a`。** 实测重建全程 overlay 稳在 48 M。

⚠️ **容器重建后 git 凭据会丢** ⇒ 凭据/私钥放 `/workspace/tools/` 下才活得过重建。
⚠️ **`--save-freq 999` 挡不住收尾那次保存**：fully_async 一个 ckpt **27 GB**，
计时/探针类短跑**跑完就删** `checkpoints/grpo/<exp>/global_step_*`
（`dispatched.jsonl` 和 `rollout_dumps` 要留，分析靠它们）。

---

## 仍然有效的原始结论（2026-08-13 实测，方向不变）

- 实测 3 卡 FULL_SHARD 每步 1182 s，1 卡不切分 198 s —— **多给卡慢 6 倍**（5.97×）；
  换 DDP 后 3 卡稳态 91.6 s。⚠️ **口径**（E02 §6）：1182 vs 198 **两边都是首步**，
  比较公平；但 FULL_SHARD 跑完第一步就崩了，**稳态数据不存在**。
  ⚠️⚠️ 这个数是**旧机器**的，新机器带宽 ×4 后必须重测（队列 A6）。
- ⇒ 并行不该发生在模型内部，该发生在外部：rollout 副本、时间上的异步、实验级并行 ——
  这三种通信量都极小。（PP 对慢网络友好，但 verl 的 megatron/torchtitan 后端没装。）
- **flash-attn**：垫片**已退役**（满足 import 不满足契约：sdpa 恒物化 mask，打包 2.19× 倒退）。
  真轮子由 pyproject 的 `[tool.uv.sources]` 装，见 [[clean-machine-only-gaps]]。

细节见 `docs/syncopate/08-machine-and-environment.md`。
相关：[[feedback-measure-dont-infer]] [[infra-line-state]] [[clean-machine-only-gaps]]
