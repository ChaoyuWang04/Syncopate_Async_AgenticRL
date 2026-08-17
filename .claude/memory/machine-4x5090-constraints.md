---
name: machine-4x5090-constraints
description: 4×5090 无 P2P 只能 DDP；2+2 跨 socket / PCIe Gen5 / 本地 XFS；带宽 组内28.8 跨22.2 四卡25.6 GB/s
metadata:
  node_type: memory
  type: project
  originSessionId: 254d8707-7512-4e9b-bd89-6e1eeec39011
  modified: 2026-08-17T07:10:00.000Z
---

## 当前机器画像（2026-08-17 实测）

- **P2P 全关**（`can_device_access_peer` 4×4 全 0）。GeForce 从 4090 起被驱动关掉，
  **所有 4×5090 都这样** ⇒ 卡间经主机内存中转，NCCL 走 `SHM/direct/direct`。
- **2 socket EPYC 9V74，2+2**：GPU0/1@node0、GPU2/3@node1；**PCIe Gen5 x16**。
- **all-reduce busbw @256MB**：组内 **28.8** · 跨 socket **22.2** · **四卡 25.6** GB/s。
  跨 socket 掉 22%，**NUMA 绑定救不回来**（22.23→22.34，噪声内）＝ UPI 跳的物理代价。
  换算：DDP 梯度 260 MB ≈ 10.2 ms/步，跨 socket 净代价 **1.2 ms/步 ＝ 一步的 0.004%**。
  尺子 `scripts/probe_allreduce_bw.py`，数据 `logs/e00_allreduce_*.json`。
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
`scripts/check_flash_attn_backward.py` —— 见 [[clean-machine-only-gaps]]。

细节见 `docs/syncopate/08-machine-and-environment.md`；焦点迁移见 `docs/focus-migration-2026-08.md`。
相关：[[feedback-measure-dont-infer]] [[infra-line-state]] [[clean-machine-only-gaps]]
