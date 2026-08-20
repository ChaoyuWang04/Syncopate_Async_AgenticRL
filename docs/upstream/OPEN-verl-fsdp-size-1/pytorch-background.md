# 上游 issue 草稿 · FSDP 的 `HYBRID_SHARD` 在分片维为 1 时**静默停止同步梯度**

> 状态：**草稿完成，待 Chaoyu 决定是否提交**　建于 2026-08-18
> 归属：独立线（`docs/upstream/`）。完整实验记录：[`../infra_exp/E21-ddp-not-syncing.md`](../../infra_exp/E21-ddp-not-syncing.md)
> 姊妹文档：[`../READY-fsdp-shard-alignment/analysis.md`](../READY-fsdp-shard-alignment/analysis.md)（同一批调查里的另一个上游问题）

---

> ⛔ **2026-08-18/19 调查后，本文的定位整个换了**（原「issue 一直没被提」是**错的**，保留下面时间线示错）：
>
> ```
> 2025-05-31  论坛 #220486  报告 (n,1) mesh 回退 NO_SHARD 后梯度不同步
> 2025-06-02  H-Huang 回「looks like a bug」，请对方提 issue —— **当天就提了**：pytorch#154888
> 2025-06-03  triage：cc 整个 distributed oncall，assign 给 mori360
> 2025-06-05  FSDP 维护者 weifengpy 确认 "this is a bug ... we might be slow in fixing
>             fsdp1, are you interested in fsdp2?"
> 2025-06-09  被 assignee 关成 **not_planned**。至今零个 PR 引用它。
> ```
>
> ⇒ **bug 已被 PyTorch 承认过一次，然后因 FSDP1 维护模式被主动放弃**
> （`_init_utils.py` 自 2025-06 的 17 次提交里 16 次是 lint/typing；`NO_SHARD` 自身已挂
> FutureWarning「deprecated, use DDP」）。**空着的不是 issue 位，是修复。**
> ⇒ **主战场移到 verl**（[`analysis.md`](analysis.md)）；
> 本文降级为它的**上游证据链**：PyTorch 侧的动作改为
> ① 在 #154888 评论补真实训练证据、请求 reopen（几分钟，不指望结果）
> ② 可选：提一个只加 `raise` 的小 PR（§6-①）——「不修行为可以，但不该继续静默」。
> 同族：[pytorch#90050](https://github.com/pytorch/pytorch/issues/90050) ·
> [pytorch#152710](https://github.com/pytorch/pytorch/issues/152710)（FSDP2 侧讨论，
> 维护者确认 `(Replicate, Shard=1)` 在 FSDP2 **开箱即用** —— 已被我们实测证实，见 §3.1-F）

## 0 · 一句话

**当 `HYBRID_SHARD` 的分片维大小为 1 时，FSDP 会自动降级成 `NO_SHARD`（只打一行 `UserWarning`），
而降级之后的梯度归约走的是那个只有 1 个成员的进程组 ⇒ 归约变成空操作
⇒ 复制维上的 N 个 rank 各训各的、永不同步，而训练照常跑完、所有指标都正常。**

---

## 1 · 为什么这个 bug 会活这么久（成立范围要老实写）

★ **触发它需要一个"看起来很合理、实际上退化"的配置**：

| 条件 | 主流用法 | 触发本 bug 的用法 |
|---|---|---|
| 分片维大小 | 2 / 4 / 8（真的要分片） | **1**（"我不想分片，只想数据并行"） |
| 表达"不分片"的方式 | 直接用 `DDP` 或 `NO_SHARD` | **用 `HYBRID_SHARD` + `(N, 1)` 的二维 mesh** |
| 谁会这么写 | —— | **训练框架**：为了让"分片/不分片"共用一条代码路径，把 `fsdp_size` 当成旋钮，`1` 表示不分片 |

⇒ 也就是说：**这是"框架把 FSDP 参数化"时最自然的写法**，而不是用户手写 FSDP 会犯的错。
我们是在 [verl](https://github.com/volcengine/verl) 上撞到的（它的 `create_device_mesh(world_size, fsdp_size)`
在 `fsdp_size=1` 时会造出 `(world_size, 1)` 的 mesh），但**根因在 FSDP 这一侧**：
**同样的 mesh 交给 FSDP，它自己选择了降级，然后把归约留在了错误的组上。**

⚠️ **而"少见"恰恰让它更危险**：

> 它**不报错、不崩、loss 会降、所有指标都正常**。
> 唯一的信号是一行 `UserWarning`，而那行 warning 说的是"我换了个策略"，
> **没说"顺便，你的梯度不再同步了"**。

---

## 2 · 环境指纹

```
GPU        4 × NVIDIA GeForce RTX 5090（sm_120, 32 GB）；本复现只用 3 张
           ⚠️ PCIe P2P 全关（GeForce 常态）⇒ NCCL 走 SHM/direct
驱动/CUDA  595.58.03 / CUDA 12.8
torch      **2.9.0+cu128**
NCCL       2.27.5（torch 自带）
关键环境   NCCL_CUMEM_ENABLE=0
```

✅ **主干已复核（2026-08-18）**：main 与 2.9.0 在全部关键位置**逐字相同**——
`_init_utils.py:127`（`world_size = process_group.size()`）、`:152-153`（hybrid 的两个组）、
`_runtime_utils.py:936`（`all_reduce(grad, group=state.process_group)`）。**未修。**

---

## 3 · 最小复现（3 卡，纯 PyTorch，不依赖任何训练框架）

完整脚本：本仓库 `scripts/repro_fsdp_hybrid_nosync.py`。核心就这几行：

```python
mesh = init_device_mesh("cuda", (world, 1), mesh_dim_names=["ddp", "fsdp"])   # ← 分片维 = 1
model = FSDP(module, device_mesh=mesh,
             sharding_strategy=ShardingStrategy.HYBRID_SHARD,
             use_orig_params=True, sync_module_states=True, device_id=rank)

x = torch.randn(4, 256, device=rank) * (rank + 1)      # ★ 每个 rank 喂**不同的数据**
loss = (model(x) - target).square().mean()
loss.backward()
# 反向之后收集各 rank 的梯度范数 —— 若同步，三个数必须逐位相同
```

**实测（world_size=3）**：

| 变体 | 三个 rank 的梯度范数 | 判定 |
|---|---|---|
| **`mesh(3,1)` + `HYBRID_SHARD` + 部分参数可训** | `[0.0376, 0.0673, 0.1294]` | 🔴 **没同步** |
| **`mesh(3,1)` + `HYBRID_SHARD` + 全部参数可训** | `[0.0457, 0.0913, 0.1370]` | 🔴 **没同步** |
| `NO_SHARD` + 不传 `device_mesh` | `[0.27395082, 0.27395082, 0.27395082]` | ✅ 同步 |
| **纯 `DDP`（对照组）** | `[0.27395082, 0.27395082, 0.27395082]` | ✅ 同步 |

★ **最后两行打出逐位相同的数值** —— 说明测试装置本身是对的，
且 `FSDP(NO_SHARD)` + 默认进程组的行为与 `DDP` 完全一致。
★ **"部分参数可训"不是触发条件**（第二行也坏）—— 这一条排除了 LoRA/PEFT 之类的干扰。

### 3.1 🆕 2026-08-19 扩成七变体确定性矩阵（提交时用这张表）

脚本已改造：每个模型创建前固定 seed ⇒ **全部正确实现打出同一个数，任何人可逐位复验**。
产物 `_audit/infra/e21_grad_sync_matrix.json`（环境指纹/逐变体范数/warning 原文/内部状态取证）。

| 变体 | 三 rank 梯度范数 | 判定 |
|---|---|---|
| A `mesh(3,1)`+`HYBRID_SHARD`·部分可训 | `[0.04565846, 0.09131692, 0.13697541]` | 🔴 没同步，一步 SGD 后已发散 |
| B 同 A·全部可训 | 同 A | 🔴 |
| C `mesh(3,)`+`FULL_SHARD` | 梯度按 rank 分片 | ⚠️ 无可比值 |
| E `NO_SHARD`·不传 mesh | `0.27395082` ×3 | ✅ |
| **G `mesh(3,1)`+`NO_SHARD`（显式）** | `0.27395082` ×3 | ✅ **网格不动也对** |
| **F `mesh(3,1)`+FSDP2 `fully_shard`** | `0.27395082` ×3 | ✅ **同一网格 FSDP2 正确** |
| D 纯 DDP | `0.27395082` ×3 | ✅ |

★ **F 行是给 PyTorch 看的核心一行**：同一个 mesh，FSDP2 对、FSDP1 静默错
⇒ 唯一解释是 FSDP1 的降级路径坏了，不是用户配置问题。
★ **算术闭环**：A 的三个数恰为 `[g, 2g, 3g]`（各 rank 只剩自己数据的梯度，
且被 FSDP 预除以 3 —— 它以为会有一个从未发生的 all-reduce）；
均值 `2g=0.27395077` 与同步变体实测差 4.5e-8。
★ **内部状态取证**（写进 JSON）：`state.world_size=1`、归约组 size=1、
**复制组（size=3）建了但被遗弃** —— `orphaned_inter_node_pg_size: 3`。

复现时 PyTorch 自己打出的唯一提示（`torch/distributed/fsdp/_init_utils.py:430`）：

```
UserWarning: FSDP is switching to use `NO_SHARD` instead of ShardingStrategy.HYBRID_SHARD
             since the world size is 1
```

---

## 4 · 真实训练里的表现（不是合成场景）

在 verl 0.8.0 + Qwen3-4B + LoRA r32 + 3 张训练卡的**真实 RL 训练**上：

**运行时探针**（挂在 `optimizer_step` 之前，此时反向的归约钩子本应已经执行）：

```
step=1  rank=0  lora_B  权重范数=0.000000   梯度范数=2.209380e-05
step=1  rank=1  lora_B  权重范数=0.000000   梯度范数=2.565470e-05
                        ↑ 起点完全相同         ↑ **梯度差 16%**
step=4  rank=0  lora_B  权重=0.013482       梯度=2.906737e-04
step=4  rank=2  lora_B  权重=0.013628       梯度=8.142963e-05      ← 梯度差 3.6 倍
```

★ **判据为什么干净**：LoRA 的 `B` 矩阵是**零初始化**的 —— step 1 时三个 rank 的权重
**都是 `0.000000`**（起点完全一致），而梯度不同 ⇒ **只可能是没有 all-reduce**。
★ 旁证：同一步 `lora_A` 的梯度**恰好是 0**（因为 `B=0` ⇒ `dL/dA = Bᵀ(...) = 0`），
**数学上必须为 0，探针也确实读到 0** ⇒ 探针读的是真梯度。

**保存下来的状态**（训练 15 次更新之后）：

| 检查项 | 结果 |
|---|---|
| 基座（冻结）权重跨 rank | 相对差 **0.0** ⇒ 是复制不是分片 |
| `lora_B`（零初始化）跨 rank | 范数 0.0931 / 0.0787 / 0.0754，相对差 **1.3** |
| Adam `exp_avg_sq` 跨 rank | 相对差 **99%** ⇒ 梯度历史完全不同 |

**⇒ 后果**：`world_size` 张卡上跑的是 **N 份独立的模型**，
而最终被保存/被部署的那一份**只学到了 1/N 的数据**。
在我们这里 N=3 ⇒ 每次更新的有效数据从 48 条序列变成 **16 条**，
而这一点**在任何指标上都看不出来**。

---

## 5 · 责任在哪一层

| 层 | 它做了什么 | 对不对 | 问题 |
|---|---|---|---|
| **① 训练框架**（本例是 verl） | `fsdp_size=1` ⇒ 造出 `(N, 1)` 的 mesh，交给 `HYBRID_SHARD` | 意图正确（"不分片"） | 🟠 用了一个会退化的表达方式 |
| **② FSDP：策略降级** | 分片维只有 1 个 rank ⇒ 降级为 `NO_SHARD` | 逻辑正确 | 🟠 只是 `UserWarning` |
| **③ FSDP：降级后的归约** | 在**分片进程组**（大小 1）上做梯度归约 | 🔴 **这里错了** | **降级之后，复制维上的 N 个 rank 再也不同步，而没有任何提示** |

★ 机制已在源码坐实（2026-08-18，main 分支同）：

```
_init_utils.py:152-153  hybrid 分支：_inter_node_pg = mesh.get_group(0)   ← 复制维（N 个 rank）
                                     process_group  = mesh.get_group(1)   ← 分片维（1 个 rank）
_init_utils.py:127      state.world_size = state.process_group.size()  ⇒ 1
_init_core_state        world_size==1 ⇒ UserWarning + 钳成 NO_SHARD
_runtime_utils.py:936   dist.all_reduce(flat_param.grad, group=state.process_group)
                        ⇒ 在那个 size-1 的组上 ⇒ 空操作
                        （_inter_node_pg 只在 hybrid 分支 :868 使用 —— 策略已被钳走，永不到达）
对照 FSDP2：分派只看 mesh.ndim（_fully_shard.py:194-203），没有"降级"动作；
复制维 all_reduce 显式存在（_fsdp_param_group.py:554）且 ReduceOp.AVG 分母含两维。
```

★ **最该修的是 ③**：它是唯一一个**用户无法察觉**的环节。
①②都还留有痕迹（一行 warning、一个可读的配置），而 ③ 的后果**只能通过逐 rank 比较梯度才能发现**。

---

## 6 · 建议的修法（按侵入性从小到大）

**① 最小改动 —— 把 warning 升级成 error**

```python
if sharding_strategy in (ShardingStrategy.HYBRID_SHARD, ShardingStrategy._HYBRID_SHARD_ZERO2) \
        and shard_group_size == 1:
    raise ValueError(
        "HYBRID_SHARD with a shard-group size of 1 degrades to NO_SHARD, and gradients would "
        "only be reduced within that size-1 group (i.e. not at all). "
        "Use NO_SHARD with the default process group (or DDP) instead."
    )
```
⇒ **静默失效变成显式失败。** 这一条就足以让所有踩坑的人当场发现。

**② 更好的行为 —— 降级之后仍在完整的 mesh 上归约**

若判定 `HYBRID_SHARD(shard=1)` 语义上等价于"在复制维上做 DDP"，
那么降级成 `NO_SHARD` 之后，梯度归约应当走**复制维的进程组**（或整个 mesh），
而不是那个大小为 1 的分片组。

**③ 文档 —— 在 `HYBRID_SHARD` 的说明里写明"分片维不能为 1"**

---

## 7 · ⚠️ 我们还没做的（提交前要么补做、要么写明）

1. ~~没有读 FSDP 源码定位~~ ✅ **已坐实**（§5 的四行源码，2026-08-18）。
2. **只测了 `world_size=3`、分片维 =1** 这一种退化形状。
   `(N, 1)` 之外（比如 `(1, N)`）没测。
3. ~~只在 torch 2.9.0 上测过，主干未查~~ ✅ **主干已查，逐字相同、未修**（§2）。
   另 FSDP2 同网格对照 ✅ **已做**（§3.1-F）。
4. `FULL_SHARD`（一维 mesh）那一档在我们的复现里**读不到梯度**，因此**没有结论** ——
   不是"它也坏"，是"没测出来"。

---

## 8 · 提交清单（真要提的时候照做）

- [x] 复核 torch 主干 / 最新 release 是否已修 ——**未修**（§2；#154888 closed not_planned）
- [ ] 把 `scripts/repro_fsdp_hybrid_nosync.py` 精简成**不依赖本仓库**的单文件（去掉四变体，只留 A + D 对照）
- [ ] 附 §3 的表（**必须含 DDP 对照组** —— 它证明测试装置本身是对的）
- [ ] 附 §4 的真实训练证据（LoRA `B` 零初始化那条判据尤其有说服力）
- [ ] 明确写出 §7 的四条"还没做的"
- [ ] 同时提 verl 那条（[`analysis.md`](analysis.md)），互相引用
