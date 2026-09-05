---
name: machine-4x5090-constraints
description: 4×5090 历史机器画像；现行机器只看 docs/syncopate/05-COMPUTE.md
metadata:
  node_type: memory
  type: project
  originSessionId: 254d8707-7512-4e9b-bd89-6e1eeec39011
  modified: 2026-08-17T07:10:00.000Z
---

> **历史机器记忆。** 当前主场已经迁到 Modal 2×B200；现行环境只看
> `docs/syncopate/05-COMPUTE.md`，不要把下面的 5090 数字套到 B200。

## 当时机器画像（2026-08-17 实测）

- **P2P 全关**（`can_device_access_peer` 4×4 全 0）。GeForce 从 4090 起被驱动关掉，
  **所有 4×5090 都这样** ⇒ 卡间经主机内存中转，NCCL 走 `SHM/direct/direct`。
- **2 socket EPYC 9V74，2+2**：GPU0/1@node0、GPU2/3@node1；**PCIe Gen5 x16**。
- **all-reduce busbw @256MB**：组内 **28.8** · 跨 socket **22.2** · **四卡 25.6** GB/s。
  跨 socket 掉 22%，**NUMA 绑定救不回来**（22.23→22.34，噪声内）＝ UPI 跳的物理代价。
  换算：DDP 梯度 260 MB ≈ 10.2 ms/步，跨 socket 净代价 **1.2 ms/步 ＝ 一步的 0.004%**。
  尺子 `scripts/infra/probe_allreduce_bw.py`，数据 `logs/e00_allreduce_*.json`。
- **`/workspace` = 本地 XFS 300 G 持久卷**，权限位生效（PGDATA 可放这里）、无隐形配额。
- 🔴 **`/` 只有 16 G overlay**（`/root` `/tmp` `/var` 都在上面）。

★ **2+2 让「放置」第一次有了意义**：默认 trainer=GPU0/1/2 跨了 socket ⇒ DDP 每步走 UPI。
候选摆法已并入 **B11**。⚠️ **`E04 TP/PP` 的停放理由（「带宽上 TP 大概率净负」）已过期** ——
组内 28.8 GB/s 下 TP=2 限制在同一 socket 内是另一个命题，已复活成 30 min 探针。

**因此**：训练侧 `--fsdp-size 1`（DDP）是必选项。⚠️ FULL_SHARD / ZeRO-2 的稳态对照
**当前没有数据** ⇒ 队列 **A6**（1 小时，决定「模型内部并行是净亏损」还剩多少力气）。
⇒ 并行不该发生在模型内部，该发生在外部：rollout 副本、时间上的异步、实验级并行。

## 16 G overlay：所有缓存都要改道

`/` 是 16 G overlay 且 recycle 即丢，`/workspace` 才是持久卷（`workspace_is_volume=true`）。
镜像 `bootstrap.sh` 只管了 uv/npm/HF。**训练栈这些默认全落 overlay**，写在 `/workspace/.env`：
`RAY_TMPDIR`（★ Ray 溢写，最容易撑爆）· `TRITON_CACHE_DIR` · `TORCH_EXTENSIONS_DIR` ·
`VLLM_CACHE_ROOT` · `CUDA_CACHE_PATH` · `XDG_CACHE_HOME` · `TMPDIR` · `PIP_CACHE_DIR` ·
`PGDATA` · `LD_LIBRARY_PATH`（PG 的 libpq）。
⇒ **每个 shell 先 `set -a; . /workspace/.env; set +a`。** 实测重建全程 overlay 稳在 48 M。

## 其它固定约束

⚠️ **容器重建后 git 凭据与 `user.name/email` 都会丢** ⇒ SSH key 放 `/workspace/tools/ssh/`
（`~/.ssh` 是软链）才活得过重建。
⚠️ **`--save-freq 999` 挡不住收尾那次保存**：fully_async 一个 ckpt **27 GB**，
计时/探针类短跑**跑完就删** `checkpoints/grpo/<exp>/global_step_*`
（`dispatched.jsonl` 和 `rollout_dumps` 要留）。
⚠️ **flash-attn 必须用官方 cu13 轮子 + CUDA13 运行时**，换轮子先跑
`scripts/infra/check_flash_attn_backward.py` —— 见 [[clean-machine-only-gaps]]。

历史细节见 `docs/archive/syncopate/pre-consolidation-v16/08-machine-and-environment.md`；现行机器看 `docs/syncopate/05-COMPUTE.md`，焦点迁移见 `docs/archive/infra_exp/legacy-4x5090/focus-migration-2026-08.md`。
相关：[[feedback-measure-dont-infer]] [[infra-line-state]] [[clean-machine-only-gaps]]

---

## ⚠️ 2026-08-27 搬家：拓扑变了，本文件的带宽/拓扑数字全部待重测

新机（vast.ai）：仍是 4×5090 / sm_120 / P2P 全关（这些结论迁移），但——
**单 socket EPYC 9B14 96 核 · 4 NUMA（NPS4）**：GPU0/1 同桥 PHB@node3 ·
GPU2@node2 · GPU3@node0；RAM 566G；驱动 595.71.05；/workspace = 本地盘 300G 持久卷。
✅ 08-27 重画像七探针全绿（README §6 权威）：同桥对 16.3 · 跨 NUMA 14.4–15.1 ·
**四卡 17.9 GB/s——比旧机 25.6 低 30%**（同 socket 挤一份内存子系统，"方向未知"已裁决为反变差）；
对间差距只剩 8% ⇒ 摆位（B11）基本不重要；NUMA 绑定仍无效。
★ 三个"换机可能翻"全复现没翻：3-rank all_gather 塌陷 4.3 GB/s（LL128 治到 14.2，×3.3 同旧机；
LL128 仍伤其它算子 ⇒ 只在分片路径开）· 16B 对齐悬崖 4.7×（verl 真实分块实测）·
满载降频 −0.9% 可忽略 ⇒ 上游两 issue 拿到跨机验证。
★ param_sync 三段账：13.3s（旧机 08-13）→ ~3.3s（旧机 E14 修理）→ 1.02s（新机）；
最后一段**不是带宽红利**（带宽反而低 30%），在 CPU 侧取参/拷贝路径，未归因。
✅ 判据③ kl 地板已重标（08-27 冒烟）：bf16 KV 3.6–4.8e-4，旧口径沿用；
   新机步速 9.3–9.7 s/gstep（比旧机 11.33 快 ~14%）· param_sync 稳态 1.02s
   （旧机 update_weights 之谜 13.3s 在新机大幅缩小，未归因）。
