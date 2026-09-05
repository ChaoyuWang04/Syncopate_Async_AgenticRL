# E21 · 三个 trainer rank **没有同步梯度** —— 我们一直在训三份不同的 LoRA

> ⛔⛔ **本文含已作废的实测数字**（2026-08-18）—— 查出两个基石级 bug：
> **三个 trainer rank 的梯度没有同步** · **trainer 的权重从没推给 rollout engine**。
> ⇒ 2026-08-14 至 08-18 之间**所有 RL 训练的实测数字都不可引用**。
> **引用之前必须先读 [`21-invalidated-numbers.md`](../archive/syncopate/pre-consolidation-v16/21-invalidated-numbers.md)** —— 那里也列了**仍然有效**的部分（SFT / 数据 / 静态代码事实 / 硬件测量）。

> 状态：✅ **已确证 → 已定根因 → 已修 → 已复验**　建于 2026-08-18　最后更新：2026-08-18 11:0x
> ⚠️ **但受影响的实验结论尚未重测**（清单见 `01-TASKS` 的「重测队列」）
> ⚠️⚠️⚠️ **这是正确性 bug，优先级高于 E20 与其余一切。**
> 在它修好之前，**本项目所有关于位移 / ESS / lr / 更新次数的结论都建立在一个坏掉的基线上。**

---

## 0 · 结论卡片

| | |
|---|---|
| **问题** | fully_async 的 3 个 trainer rank，**梯度没有 all-reduce**，各自在训一份**独立的** LoRA |
| **怎么发现的** | 写「ckpt → PEFT adapter」转换脚本时，随手加的一条断言（"DDP 下各 rank 的 LoRA 应该相同"）**当场炸了** |
| **静态证据** | 基座权重跨 rank **完全相同**（⇒ 是复制不是分片）；而 `lora_B`（**零初始化**）跨 rank 相对差 **1.3**；Adam 的 `exp_avg_sq` 相对差 **99%** |
| **★ 运行时证据** | 探针在 **step 1** 打出：三个 rank 的 `lora_B` **权重都是 0.000000**（起点相同），**而梯度范数各不相同**（2.209e-05 / 2.565e-05，差 16%）⇒ **梯度没有被 all-reduce** |
| **后果** | 每个 rank 只用 **1/3 的数据**（16 条序列，不是 48 条）；最终推给 vLLM / 存进 ckpt 的那一份，**只学到了三分之一** |
| **根因** | ✅ **已定**（§4.5）：`fsdp_size=1` ⇒ 网格 `(3,1)` ⇒ `HYBRID_SHARD` ⇒ **PyTorch 见分片维=1 自动降级成 `NO_SHARD`，而归约留在那个大小为 1 的组上** ⇒ 空操作。脱离 verl 已复现 |
| **修复** | ✅ **已落地并复验**（§4.6）：退化网格下改用 `NO_SHARD` + 默认进程组。复验后三个 rank 的梯度**逐位相同** |
| **追责** | §5.5：bug 是上游的，但**四处信号我们一处都没接住**（warning 就在自己日志里，每跑 2 次） |
| **下一步** | §5：先定根因，再修，然后**所有基线重测** |

---

## 1 · 怎么发现的（值得记：它不是"查出来"的，是**断言炸出来**的）

任务级尺子（B5）需要把 RL ckpt 转成 PEFT adapter。写 `syncopate/train/ckpt_to_adapter.py` 时，
因为 verl 存的是**每 rank 一个文件**，我加了一条防御性断言：

```python
# ⚠️ DDP 下每个 rank 是完整副本；分片下这个假设不成立
assert same, "★ rank_0 与 rank_1 的 LoRA 权重不同 ⇒ 这是分片存的，只读 rank_0 会漏权重"
```

**它当场炸了。** 而后续排查表明：不是分片，是**三份不同的权重**。

★ **教训**：这条断言原本只是为了"别读错分片"，**顺手写下的一个前提检查，抓到了一个更大的 bug**。
⇒ **凡是"我假设 X 成立"的地方，把它写成断言，成本几乎为零。**

---

## 2 · 静态证据（读存下来的 ckpt，`b16_ref_off_60/global_step_15`）

| 检查项 | 结果 | 说明 |
|---|---|---|
| **基座权重**跨 rank | 相对差 **0.0** | ⇒ 是**复制**不是分片 —— 排除"读错了分片"这个解释 |
| **`lora_A`** | 三 rank 范数几乎一样（3.2681 / 3.2636 / 3.2630），**相对差 1.415** | `√2` ≈ 两个无关向量 ⇒ 方向完全不同 |
| **`lora_B`** | 范数**各不相同**（0.0931 / 0.0787 / 0.0754），相对差 1.29–1.31 | ★ **B 是零初始化的**，若梯度同步，三份必须逐位相同 |
| **Adam `exp_avg_sq`** | 相对差 **99%** | ⇒ 三个 rank 的**梯度历史**完全不同 |

⚠️ 静态证据有一个洞：**它只能说明"存下来的状态不同"**，
有可能训练本身是同步的、只是保存路径各存各的。⇒ 所以必须有运行时证据。

---

## 3 · ★★★ 运行时证据（`SYNCOPATE_DDP_PROBE=1`，`logs/ddp_probe.log`）

探针挂在 `FSDPEngine.optimizer_step` **之前**（此时 FSDP 的反向归约钩子应当已经跑完），
打印本 rank 的权重范数与**梯度范数**：

```
step=1  rank=0  lora_A  权重范数=3.255484  梯度范数=0.000000e+00   ← A 的梯度为 0，符合预期（B=0 ⇒ dL/dA=0）
step=1  rank=0  lora_B  权重范数=0.000000  **梯度范数=2.209380e-05**
step=1  rank=1  lora_B  权重范数=0.000000  **梯度范数=2.565470e-05**
                        ↑ 起点完全相同        ↑ **梯度差 16%**
step=3  rank=0  lora_B  权重范数=0.009759   梯度范数=1.072568e-04
step=3  rank=1  lora_B  权重范数=0.009753   梯度范数=9.757336e-05
step=4  rank=0  lora_B  权重范数=0.013482   梯度范数=2.906737e-04
step=4  rank=2  lora_B  权重范数=0.013628   梯度范数=8.142963e-05   ← **梯度差 3.6 倍**
```

**⇒ 判据成立**：

```
step 1 时三个 rank 的 lora_B 权重**都是 0.000000**（零初始化，起点完全一致）
而它们的梯度范数**不同**
⇒ 梯度**没有**被 all-reduce。若同步，三个 rank 会看到同一个平均梯度。
⇒ 之后权重逐步分叉（step3 差 0.06% → step15 差 130%）
```

★ 顺带一个"探针没坏"的旁证：**`lora_A` 在 step 1 的梯度恰好是 0** ——
因为 `B=0` 时 `dL/dA = Bᵀ(...) = 0`。**数学上必须为 0，探针也确实读到 0。**

---

## 4 · 这意味着什么

```
我们以为   每次更新用 6 条 prompt × 8 采样 = **48 条序列**
实际上     每个 rank 只用自己那份 ≈ **16 条**，而且三份各训各的
⇒ 最终被保存 / 被推给 vLLM 的那一份，**只学到了 1/3 的数据**
```

**它把已知的每一个问题都放大三倍：**

| 已有结论 | 在这条 bug 下要怎么重读 |
|---|---|
| 主线 17 §1.2「batch 48 × ESS 0.3 = 有效 14 条」 | 实际是 **16 × ESS ≈ 5 条** |
| E20 §3.6「一个 epoch 只更新 109 次」 | 次数没错，但**每次只看了 1/3 的数据** |
| 位移 0.0487% / 0.10% | 是**「1/3 数据」训出来的位移**；而且 `rl_ckpt_drift.py` **只读 rank_0**，量的是那一份 |
| E20-a「token 级 IS 让 ESS 回到 1.000」 | **结论本身不受影响**（两臂同样受这条 bug 影响，比值可信），但绝对值都在坏基线上 |

⇒ ⛔ **在修好之前，不要用任何位移/ESS 的绝对值去定 lr、mini_batch 或停机线。**

---

## 4.5 ★★★ 最小复现 + 根因 + 修法（2026-08-18，脱离 verl）

尺子：`scripts/infra/repro_fsdp_hybrid_nosync.py`（3 卡、纯 PyTorch、**不依赖 verl**）。
配置照抄 verl（`workers/engine/fsdp/utils.py:40`）：`fsdp_size=1, world_size=3`
⇒ `mesh_shape=(3, 1)`，维名 `["ddp", "fsdp"]` ⇒ 二维网格 ⇒ `HYBRID_SHARD`。
每个 rank **喂不同的数据**（正是 DDP 的场景），反向之后收集各 rank 的梯度范数。

| 变体 | 三个 rank 的梯度范数 | 判定 |
|---|---|---|
| **A · `mesh(3,1)` HYBRID + 部分可训（我们的配置）** | `[0.0376, 0.0673, 0.1294]` | 🔴 **没同步** |
| **B · 同 A，但全部参数可训** | `[0.0457, 0.0913, 0.1370]` | 🔴 **没同步** |
| **E · `NO_SHARD` + 不传 device_mesh（候选修法）** | `[0.27395082, 0.27395082, 0.27395082]` | ✅ **同步** |
| **D · 纯 DDP（对照组）** | `[0.27395082, 0.27395082, 0.27395082]` | ✅ 同步 |

⇒ **三条结论：**

1. ✅ **在 verl 之外复现了** ⇒ 这是 **PyTorch FSDP 的行为**，不是 verl 的接线错误。
2. ✅ **和 LoRA / "只有部分参数可训" 无关**（B 也一样坏）⇒ 触发条件更普遍。
3. ★ **E 与 D 打出逐位相同的数值**（0.27395082）⇒ **`FSDP(NO_SHARD)` + 默认进程组 = DDP**，
   这就是修法。

### 4.5.1 根因：**FSDP 自己把 HYBRID_SHARD 降级成了 NO_SHARD，然后在一个大小为 1 的组里做归约**

复现时 PyTorch 自己打了这行警告（`_init_utils.py:430`）：

```
UserWarning: FSDP is switching to use `NO_SHARD` instead of ShardingStrategy.HYBRID_SHARD
             since the world size is 1
```

把它和 verl 的网格构造放在一起，链条就完整了：

```
verl:  fsdp_size=1, world_size=3
       ⇒ mesh_shape = (world_size // fsdp_size, fsdp_size) = **(3, 1)**   ["ddp", "fsdp"]
       ⇒ 二维网格 ⇒ get_sharding_strategy 返回 **HYBRID_SHARD**
PyTorch: HYBRID_SHARD 看到**分片维大小 = 1** ⇒ 降级成 NO_SHARD（并打了上面那行警告）
       ⇒ 而降级后的梯度归约走的是**那个大小为 1 的组** ⇒ **空操作** ⇒ 三个副本各训各的
```

★ **两层各自"合理"，缝里掉了东西**（本项目第 N 次遇到这个形状）：
- **verl**：想表达"不分片"，于是把 `fsdp_size` 设成 1 —— 意图没错
- **PyTorch**：分片维只有 1 个 rank，退化成 NO_SHARD —— 也没错
- 🔴 **但退化之后，"复制维的 3 个 rank 该不该同步梯度"这件事没人管** ——
  **而且只打了一行 UserWarning，训练照常跑完，指标全都正常。**

⇒ 这是**静默失效**的教科书样本：不报错、不崩、曲线好看，只是**每个副本各学各的**。

### 4.5.2 修法（已验证）

```python
# 当 fsdp_size == 1（本意就是"不分片"）时：
#   ❌ 不要构造 (world_size, 1) 的二维网格 + HYBRID_SHARD
#   ✅ 用 NO_SHARD + **默认进程组**（不传 device_mesh）
FSDP(module, sharding_strategy=ShardingStrategy.NO_SHARD, use_orig_params=True, ...)
```
⇒ 实测与纯 DDP **逐位相同**。

### 4.5.2.1 🆕 上游情报（2026-08-18 检索）：**已经有人报过，而且维护者点名要我们手上这个东西**

[PyTorch 论坛 #220486](https://discuss.pytorch.org/t/potential-bug-with-hybrid-shard-and-n-1-device-mesh-falling-back-to-no-shard/220486)
描述的现象与本文**完全一致**：`(n,1)` mesh 回退成 `NO_SHARD` 后 world size 被当成 1 ⇒
**梯度不跨复制维归约，一步之后参数就发散**；报告者也做了同样的对照
（纯 DDP / FSDP+`NO_SHARD` 都正常，只有 `HYBRID_SHARD` 发散）。

```
维护者 H-Huang 回复：「这看起来是个 bug」，请报告者到 GitHub 提 issue 并**附可复现脚本**
⇒ 而那个 issue **一直没有被提**。
```

> ⛔ **2026-08-18 晚更正（上面最后一句是错的，按纪律保留原文示错）**：那个 issue **当天就提了** ——
> [pytorch#154888](https://github.com/pytorch/pytorch/issues/154888)（同一报告者，GitHub 账号 origin-bio）。
> 时间线：2025-06-02 提 → 06-03 triage、cc 整个 distributed oncall、assign 给 mori360 →
> 06-05 FSDP 维护者 weifengpy 确认 *"this is a bug ... **we might be slow in fixing fsdp1**,
> are you interested in fsdp2?"* → **06-09 被 assignee 关成 `not_planned`**。至今零个 PR 引用它。
> ⇒ **空着的不是"issue 位"，是"修复"** —— FSDP1 维护模式下 PyTorch 明确不修
> （`_init_utils.py` 自 2025-06 起 17 次提交里 16 次是 lint/typing，`NO_SHARD` 自己已挂 FutureWarning）。
> ⇒ **论点随之要换**：不是"替他们补一个没提的 issue"，而是
> 「**上游已确认、已放弃 ⇒ 框架层（verl）是唯一能兜住的地方**」——主战场移到 verl（§4.8）。
> ⚠️ 教训同 §5.5-②：「搜不到 ⇒ 没人提」是一句夹在事实中间的**推断** ——
> 当时只查了论坛帖有没有回链，没搜 GitHub issue 标题。

⇒ ★ **我们手上正好就是他们要的那个东西**：`scripts/infra/repro_fsdp_hybrid_nosync.py`
（纯 PyTorch · 3 卡 · 带 DDP 对照组 · `REPRO_APPLY_FIX=1` 还能兼作修复验证）。
⇒ **这把上游那份文档从"可提可不提"变成了"该提"** —— 是一个维护者明确要过、且一直空着的位置。

同族的已有 issue（都指向"size-1 组上的策略退化"这一类）：
[pytorch#90050](https://github.com/pytorch/pytorch/issues/90050)（`ShardingStrategy` 在传入 size-1 进程组时被忽略）·
[pytorch#152710](https://github.com/pytorch/pytorch/issues/152710)。

⚠️ **这条同时是对我们自己的一记**：这个现象**两个月前就能搜到**。
⇒ 纪律补一条：**撞到"框架行为不符合预期"时，先花五分钟搜上游 issue/论坛** ——
成本几乎为零，而它这次能省下的是两个月。

### 4.5.3 这能不能提上游

**两边都可以提，而且我倾向都提**：
- **PyTorch**：`HYBRID_SHARD` 在分片维为 1 时降级成 `NO_SHARD`，**却把梯度归约留在了大小为 1 的组里**
  ⇒ 建议要么在整个 mesh 上归约、要么直接**报错**而不是 UserWarning。
  **静默地不同步梯度，是最坏的一类失败。**
- **verl**：`create_device_mesh(world_size, fsdp_size=1)` 会造出退化的 `(N, 1)` 网格
  ⇒ 建议 `fsdp_size == 1` 时走 `NO_SHARD` + 默认进程组。

⚠️ 提之前要做的：① 确认 torch 主干是否已改（本文基于 **2.9.0+cu128**）
② 把复现脚本精简成不依赖本仓库的单文件。**流程照抄 `docs/upstream/` 那份 16 字节对齐的清单。**

## 4.6 ✅ 修复与复验（2026-08-18）

**补丁**（`verl_patches._patch_fsdp_degenerate_mesh`，`SYNCOPATE_FSDP_DDP_FIX` **默认开启** ——
这是正确性修复不是可选优化）：拦住 FSDP 的构造，**只在「`HYBRID_SHARD` + 分片维=1」这一种情况下**
把 `sharding_strategy` 改成 `NO_SHARD`、`device_mesh` 置空，其余一律原样放行。

⚠️ **为什么不改 verl 的 `create_device_mesh`**：`self.device_mesh` 在 verl 里还被别处用
（fsdp2 路径、state_dict 加载）⇒ 整个置空风险大。**改动面要最小。**

### 4.6.1 最小复现上的验证

| 变体 | 修复前 | **修复后** |
|---|---|---|
| A · `mesh(3,1)` HYBRID + 部分可训 | `[0.043, 0.076, 0.132]` 🔴 | **`[0.198, 0.198, 0.198]` ✅** |
| B · 同 A，全部可训 | `[0.046, 0.091, 0.137]` 🔴 | **`[0.274, 0.274, 0.274]` ✅** |

★ 复现脚本加了 `REPRO_APPLY_FIX=1` 开关 ⇒ **同一个脚本既是复现又是验证**。
⚠️ 踩过的坑：`mp.spawn` 的子进程**不继承**父进程打的补丁，必须在子进程内装。

### 4.6.2 ★ 真实训练上的复验（`SYNCOPATE_DDP_PROBE=1`）

| | 修复前 | **修复后** |
|---|---|---|
| step1 `lora_B` 梯度 | rank0 `2.209380e-05` · rank1 `2.565470e-05`（差 16%） | rank0 `1.004384e-02` · rank2 **`1.004384e-02`**（逐位相同） |
| step3 | `1.072568e-04` vs `9.757336e-05` | 两 rank 都是 **`7.648559e-03`** |
| step4 | `2.906737e-04` vs `8.142963e-05`（差 3.6×） | 两 rank 都是 **`2.674036e-03`** |
| step3/4 的**权重** | 0.009759 vs 0.009753；0.013482 vs 0.013628（在分叉） | **0.016778 / 0.020890 完全一致** |

★ 旁证：修复后日志出现 `[repeated 2x across cluster]` ——
**Ray 只在多个进程打出完全相同的行时才折叠**，这本身就是一致性的证据。

⚠️⚠️ **一个我不下结论的观察**：修复后梯度范数大了约 450×（2.2e-05 → 1.0e-02）。
**这不能归功于修复** —— RL 每批的优势值方差差异极大，两次跑抽到的数据不同就够解释。
**决定性证据是「跨 rank 逐位相同」，不是数值变大。**

## 4.7 ★ 0-A · 修复之后的下一个问题：梯度**合对了**吗？（2026-08-18，主线 I1/I2 的答复）

E21 修好的是「三个 rank 的梯度**有没有**碰面」。**但"碰了面"不等于"合对了"。**
主线（2026-08-18 交接，原信已删）把这条提成 I1：
「逐位相同只证明 all-reduce 发生了，没证明它是**按 48 条求平均** —— 若退化成求和，梯度会系统性大 3 倍。」

### 4.7.1 先写预测（读码得出，标明是推断）

```
torch/distributed/fsdp/_runtime_utils.py:925  _reduce_grad_no_shard
    _div_if_needed(grad, predivide) → all_reduce(SUM) → _div_if_needed(grad, postdivide)
_init_utils.py:130                predivide × postdivide = data_parallel_world_size
⇒ [推断] 总共正好除以 world_size = **求平均**（拆成两半除只是 fp16 防溢出）
⇒ 会推翻这个推断的唯一分支：verl 若注册了 comm_hook 就整段绕过 —— grep 过，**没有注册**
```

### 4.7.2 实测（`scripts/infra/repro_ddp_reduce_convention.py`，3 卡纯 PyTorch，`logs/0a_reduce_convention.log`）

判据是数据并行的定义本身：**3 卡合起来的梯度必须等于 1 卡在拼接后全部数据上的梯度。**
参考值由同进程内的非 FSDP 模型算出（同一 seed ⇒ 权重逐位相同）。
比 **‖g‖（幅度）与 g·u（方向）两项** —— 只比范数会漏掉方向错误。
⚠️ 构造方式**照抄 verl**（mesh(3,1) + HYBRID_SHARD）**再装上本文的补丁**，量的是真实路径。

| 变体 | 各卡样本数 | 3 卡 ‖g‖ | 1 卡 ‖g‖ | 比值 | 判定 |
|---|---|---|---|---|---|
| ① 等量 + 本地平均 | [4,4,4] | 178.72511292 | 178.72511292 | **1.000000** | ✅ |
| ② 等量 + **verl 口径** | [4,4,4] | 178.72511292 | 178.72511292 | **1.000000** | ✅ |
| ③ **不等量** + 本地平均 | [2,4,6] | 158.26535034 | 215.13304138 | **0.735663** | 🔴 方向也不一致 |
| ④ **不等量** + **verl 口径** | [2,4,6] | 215.13304138 | 215.13304138 | **1.000000** | ✅ |

⇒ **I1 结案：归约是求平均，口径正确**（1.000000，不是 3.000）。
⇒ 顺带钉死 §4.6.2 那个悬着的观察：**「修复后梯度大了 450×」与口径无关**，只能是数据差异。

### 4.7.3 ★ 白捡的一条：verl 那套"绕圈"的归一化**确实在保护我们**

「平均还是求和」**不是自由选项**，它由「每张卡本地怎么归一化」决定，两者必须配套：

```
甲 · 本地平均（最常见）  loss_r = Σ_{i∈r} l_i / n_r
     ⇒ 归约求平均才对。**隐含前提：每张卡的 token 数一样多**
乙 · verl 的口径         loss_r = Σ_{i∈r} l_i / N_global × dp_size      （core_algos.py:1172）
     ⇒ 预先乘了 dp_size ⇒ 与"除以 dp_size"抵消，**各卡数量不等也成立**
```

③ 与 ④ 的对照就是这句话的实测：**同一份不等量数据，甲错 26% 且方向都不对，乙逐位正确。**
⇒ 我们是变长序列负载，各 rank 的 token 数天然不等 ⇒ **甲的写法会长期、静默地训错**，
而这正是 2024 年底那轮著名的 "gradient accumulation bug" 的同一个形状。
⇒ **verl 这段设计比通行做法更讲究，值得在对外叙述里点名。**

### 4.7.4 ✅ I2 结案：补丁在 actor 和 ref 两路都生效（用了比"数 print"更强的判据）

```
修复前 logs/ddp_probe.log   PyTorch 自己的降级 UserWarning 出现在 **2 个 FSDP 构造点**
修复后 logs/ddp_fixed.log   该 warning **归零**；我们的判据行在 2 个构造点各打一次
该跑 use_kl_loss=True ⇒ ref policy 是活的 ⇒ 两个构造点 = actor + ref
```
★ 判据的强度来自**反向**：只要有任何一个退化网格漏网，**PyTorch 会替我们喊** —— 现在是 0。
⇒ 这是 §5.5.3 第 1 条纪律（「框架打出的 warning 要当判据读」）第一次被**主动**用上。

### 4.7.5 🆕 补丁引入的新前提，已写成常驻断言

把 `device_mesh` 置空 ⇒ FSDP 改用**默认进程组**归约 ⇒
**「FSDP 除以谁」的来源从 mesh 变成了默认进程组**。它必须覆盖同一批 rank，否则分母和
verl 乘的 `dp_size` 对不上。⇒ 已在补丁里写成断言（`verl_patches.py`，不成立直接 `RuntimeError`），
判据行里也带上了 `world=`。

⚠️ **仍然成立的前提，写在这里备查**：`ulysses_sequence_parallel_size` 必须为 **1**
（verl 默认值，我们不设它；实跑日志确认为 1）。因为修复后
**FSDP 按 `world` 除、verl 按 `dp_size = world // sp` 乘 ⇒ 只有 sp=1 时两者抵消。**
⇒ 已写进 `launch_rl` 注释；**开 Ulysses SP 之前必须重做 0-A。**

### 4.7.6 ⚠️ 原计划的"第二级"（真实 verl，1 卡 vs 3 卡）为什么没做

原设计是「同一份固定数据，`--trainer-gpus 1` vs `3`，比第一次更新的 `grad_norm`」。
**设计到执行时发现它有方法论问题**：RL 管线里 rollout 是 vLLM 随机生成的，
改 trainer 卡数会同时改变批次构成与 vLLM 的并行度 ⇒ **两边拿不到同一份数据**
⇒ `grad_norm` 不同时，分不清是"归约错了"还是"数据不同" —— 正是本项目记过的**判据量错对象**。

⇒ 改成上面这条能干净判定的路径（单进程参考 + 幅度与方向双探针 + 真实构造路径）。
⇒ 若日后要做真正的端到端级验证，可行的做法是**离线回放**：
从 `rollout_dumps` 取一个真实批次，喂给 1 卡 / 3 卡的 actor 更新 —— 数据同一性有保证。
**成本约 1–2 小时写harness，建议并进 B5 那一轮做，不单独占卡。**

## 4.8 ★★★ 修法×实现矩阵（2026-08-19，七变体一次跑完）——「同一个网格，FSDP2 对、FSDP1 静默错」

> 提上游前要回答三个问题：① 同一个 `(N,1)` 网格交给 **FSDP2** 对不对（决定 PyTorch 侧论点）
> ② 拟提给 verl 的**最小修法**（网格原样保留、只把策略换成 `NO_SHARD`）真的有效吗
> ③ 修法会不会改变 ckpt 格式（verl 的 fsdp ckpt 走 `SHARDED_STATE_DICT`）。
> 脚本：`scripts/infra/repro_fsdp_hybrid_nosync.py`（2026-08-19 改造：**全脚本确定性** + 七变体 +
> 一步 SGD 后果 + A 的内部状态取证 + ckpt 探针）
> 产物：`_audit/infra/e21_grad_sync_matrix.json`（+ `_fixmode`）·
> `logs/e21_grad_sync_matrix{,_fixmode}_20260819.log`

```
变体（确定性 seed，任何人可逐位复验）                三 rank 梯度范数                          判定
A  mesh(3,1) HYBRID_SHARD·部分可训（我们的）  [0.04565846, 0.09131692, 0.13697541]   🔴 没同步，一步 SGD 后权重已发散
B  同 A·全部可训                             同 A                                   🔴（排除「部分可训」变量）
C  mesh(3,)  FULL_SHARD 真分片               [None, None, 0.27395082]               ⚠️ 梯度按 rank 分片，无可比值（历来无结论）
E  NO_SHARD·不传 mesh（已上线的补丁形态）      [0.27395082, 0.27395082, 0.27395082]   ✅
G  mesh(3,1)·NO_SHARD（拟提 verl 的修法）     [0.27395082, 0.27395082, 0.27395082]   ✅ ★ 网格原样保留也对
F  mesh(3,1)·FSDP2 fully_shard               [0.27395082, 0.27395082, 0.27395082]   ✅ ★ 同一网格 FSDP2 正确
D  纯 DDP（对照组）                           [0.27395082, 0.27395082, 0.27395082]   ✅ 装置本身是对的
```

**五条结论**：

1. **★ 同一个 `(3,1)` 网格：FSDP2 正确、FSDP1 静默错误** ⇒ 「你用法不对」这个反驳被排除了。
   机制差异在源码上坐实：FSDP2 **没有"降级"这个动作**（mesh 即策略，`_fully_shard.py:194-203`
   只看 `ndim`），复制维的 all_reduce 显式存在（`_fsdp_param_group.py:554`）且 `ReduceOp.AVG`
   分母含两维（1×3=3，`_fsdp_collectives.py:693-707`）—— 顺带按同一口径回答了主线 I1。
2. **★ G 成立 ⇒ verl 侧的 3 行修法验证通过**：`get_sharding_strategy` 在 `mesh.size(1)==1` 时
   返回 `NO_SHARD` 即可，**mesh 一个字不动** —— FSDP1 的非 hybrid 路径自己会取
   `mesh_dim=0`（复制维，N 个 rank）当归约组（`_init_utils.py:119`）。
3. **ckpt 探针：修法不改变 checkpoint 行为**。E / G / 当前坏形态三者的 `SHARDED_STATE_DICT`
   **全部**被 `NO_SHARD` 短路成 full tensor（torch 自打 `When using NO_SHARD ... full_state_dict
   will be returned`，值类型全是 `Tensor`）⇒ resume 兼容不受影响。
   ⚠️ 我此前读码推断「sharded hook 对 NO_SHARD 不短路、G 可能撞 DTensor chunking」——**推断错了，
   探针纠正**。又一条「读码会漏、判据要实测」（§5.5-②同族）。
4. **算术闭环（坏变体连"错的方式"都完全可解释）**：A 的三个数恰为 `[g, 2g, 3g]` ——
   各 rank 手里只有**自己那份数据**的梯度（数据按 rank+1 缩放），且已被 FSDP 预除以 3
   （它以为会有一个**从未发生**的 all-reduce）。均值 `(1+2+3)g/3 = 0.27395077`，
   与全部同步变体的实测 `0.27395082` 差 4.5e-8。
5. **fix-mode 复验**：`REPRO_APPLY_FIX=1` 装上我们的补丁后 A/B 变绿，A 的内部状态
   `world_size` 1→3。
6. 🆕 **G 的确切 diff 在 verl 源码树里真跑通过（2026-08-19）**：把 3 行改动直接打进
   `verl/workers/engine/fsdp/utils.py`、**关掉我们的 monkeypatch**（`SYNCOPATE_FSDP_DDP_FIX=0`），
   fully_async 3+1 真实训练 4 步 ⇒ `assert_ranks_identical` 三 rank **504/504 张量逐位相同** +
   优化器一致（指纹 `_audit/infra/e21_verl_gdiff_rank_fingerprint.json`）。跑完已还原 stock verl。
   ⚠️ 过程中的一条判据教训：**Ray 日志去重忽略数字差异** —— `[repeated 6x]` 折叠的探针行
   **不能**当"逐位相同"读，要靠逃过去重的显式行 + ckpt 比对收口。

**A 的内部状态取证（bug 的结构性证据，写进了 JSON）**：

```
sharding_strategy_after_init = NO_SHARD（被钳）
state_world_size = 1 · process_group_size = 1        ← 归约用的组
orphaned_inter_node_pg_size = 3                       ← 复制组**建了、从没被用**
构造期唯一提示 = "FSDP is switching to use `NO_SHARD` ... since the world size is 1."
```

---

## 5 · 下一步（按"先定根因、再修、最后重测"）

| # | 动作 | 说明 |
|---|---|---|
| **1** | **定根因** | 已知：`HYBRID_SHARD` + `fsdp_size=1` ⇒ 网格 `(3,1)`；`sync_module_states=True`。<br>**待查**：① 归约钩子是否注册在复制维上 ② `use_orig_params=True` + 「只有 LoRA 可训」是否走了没有钩子的路径 ③ 换 `fsdp_size=-1`（真分片）或 `fsdp2` 后是否同步 |
| **2** | **最小复现** | 脱离 verl，用 3 卡 + FSDP(HYBRID_SHARD, mesh(3,1)) + 一个只有部分参数可训的模块，看梯度会不会同步。**这也是能不能提上游的前提** |
| **3** | **修** | 视根因而定：换 sharding 策略 / 显式 all-reduce 梯度 / 换 fsdp2 |
| **4** | **重测所有基线** | ⚠️ **E20 的全部数字、E17 的 A/B、B10/B19 的代价曲线，都要在修好之后重跑一遍** |

⚠️ **不要跳过第 2 步直接改配置** —— 本项目为"没查因就动手"付过两次钱（E12 §6）。

---

## 5.5 ★★★ 追责：这**不是纯上游问题**，我们至少有四处该拦住它

> Chaoyu 2026-08-18 问：「这是纯粹上游的问题吗？有没有我们自己的原因？」
> **答：bug 是上游的，但"两个月里没人发现"是我们的。** 逐条摆证据。

### 5.5.1 上游的份

| 谁 | 问题 | 严重度 |
|---|---|---|
| **PyTorch** | `HYBRID_SHARD` 分片维为 1 时降级成 `NO_SHARD`，**归约留在大小为 1 的组上** ⇒ 梯度不再同步。**而这件事只用一行 `UserWarning` 通知**，warning 的内容还是「我换了个策略」，**没说"你的梯度不再同步了"** | 🔴 **核心缺陷**：正确性级别的行为变化，用告警级别的方式通知 |
| **verl** | `create_device_mesh(world_size, fsdp_size=1)` 造出退化的 `(N,1)` 网格 | 🟠 触发条件 |

⚠️ **但要说句公道话**：verl 的配置文档里，`fsdp_size` 的说明只有一句
「`FSDP group size. -1 means use all available GPUs.`」——
**它从没说过"1 = DDP"。默认值是 `-1`。**
⇒ **「fsdp_size=1 就是 DDP」是我们自己的推断，不是它承诺的契约。**

### 5.5.2 我们的份（四处，按"本该多容易拦住"排）

**① ★★★ 那行 warning 一直在我们自己的日志里，每跑两次，没人读**

```
实测：logs/rl_v13e1.log（主线正式跑）    2 次
     logs/b16_ref_on_60.log             2 次
     每一次训练都有
内容：UserWarning: FSDP is switching to use `NO_SHARD` instead of
      ShardingStrategy.HYBRID_SHARD since the world size is 1
```
⇒ **信号一直在，成本为零，我们从没读过。**
⚠️ 而这条正是本项目反复记录的形状的**又一个变种**：
「**日志里有，但没人把它当判据**」——
我们给自己定的纪律是"每个机制都要打判据行"，却**没有反过来读框架已经打出来的告警**。

**② ★★★ 我们读了同一段源码，描述全对，然后加了一句没验证的推断 —— 而错就错在那一句**

`launch_rl` 的注释（我们自己写的）：

```
create_device_mesh(world_size, fsdp_size)（fsdp/utils.py:40）：
    fsdp_size <= 0 或 >= world_size  ⇒ 一维 mesh，全部参数切分（FULL_SHARD）   ← 对
    否则 ⇒ (world_size//fsdp_size, fsdp_size)，维度 ["ddp","fsdp"]            ← 对
取 1 ⇒ fsdp 维长度为 1 = **不切分**                                          ← 对
    **只在 ddp 维 all-reduce 梯度 = DDP**                                    ← 🔴 **错，且从没验证**
```

⇒ **前三句都是读码读出来的事实，第四句是推的。** 而 bug 恰好就在第四句。
★ 这是记忆里 `feedback-measure-dont-infer`（用推理代替测量）**最锋利的一次兑现**：
不是"懒得测"，是**推断混在一串正确的事实中间，看起来像是同一条链上的结论**。

**③ ★★ 我们把那个推断变成了一个"看起来是实测"的数字，并在 5 份文档 6 处引用**

```
「DDP 梯度 260 MB/步」出现在：
  focus-migration §1        用它算出「跨 socket 净代价 1.2 ms/步 = 一步的 0.004%」
  distributed-training §555 用它论证「三个数量级」
  E08 §4.9 / handoff §1.1 / handoff §2   …
```
⇒ **而这 260 MB 是算出来的**（66M 参数 × 4 字节），**不是量出来的**。
⇒ 如果梯度从来没同步过，**这段流量根本不存在** ——
**B11（拓扑放置）那条"差 1.6%，因为通信量小"的结论，它的前提整个是空的。**
（结论本身可能还对，但**理由要重写**。）

**④ ★ 我们有一次天然的机会拦住它，而我们量错了对象**

E00 用独立探针量了 all-reduce 带宽（组内 28.8 / 跨 socket 22.2 GB/s），
然后**直接换算成"DDP 每步 10.2 ms"**。
⇒ 探针量的是「**这台机器能不能做 all-reduce**」，
我们当成了「**这次训练确实在做 all-reduce**」。
★ **能力 ≠ 发生。** 而两者之间隔着的，正是这个 bug。

### 5.5.3 ⇒ 结论与该改的纪律

**责任划分**：
```
bug 本身          上游（PyTorch 的静默降级是不可辩护的）
两个月没被发现     我们（四处信号，一处都没接住）
```

**该加的三条纪律**（都不花钱）：

1. **框架打出的 warning 要当判据读。** 我们只盯自己打的判据行，却把上游的告警当噪声。
   ⇒ 训练启动后 grep 一遍 `UserWarning`，**新出现的 warning 要有人看过**。
2. **「读码得出的事实」和「据此做的推断」要在注释里显式分开。**
   ②那段注释里三句事实 + 一句推断，排版上完全一样 ⇒ 没人会去质疑第四句。
   ⇒ 推断句要标 `[推断，未验证]`（主线 17 那份文档已经在这么做了，值得抄）。
3. **凡是"每步同步 X MB"这种数字，要有一次实测**（哪怕只量一次 NCCL 流量）。
   ⇒ 算出来的数字不许直接进结论表，除非标明是算的。

★ **最后一条公道话**：我们最终**是靠一条随手写的防御性断言抓到它的**
（`rl_ckpt_to_adapter.py` 里那句"DDP 下各 rank 的 LoRA 应该相同"）。
⇒ **这个做法要保留并推广**：写工具时把"我假设 X 成立"写成断言，成本几乎为零，
而它抓到的是**四处显式信号都没抓住的东西**。

## 5.6 ★ 这条 bug 交付到主线产物的**确切路径**（主线 18 §0-P2 查实）

主线在同一天把 E21 的方法施加到管线上（`docs/archive/syncopate/pre-consolidation-v16/18-pipeline-assumption-probes.md`），
查到了这条 bug 是**怎么走到我们评测的模型上**的：

```
P2  models/Qwen3-4B-rl-v13-s110/lora_adapter **就是 rank_0 那一份**
    q_proj ‖ΔW_eff‖：adapter 0.041528 · rank0 **0.041530** · rank1 0.046643 · rank2 0.043280
    ⇒ **E21 的「1/3」通过 merger 原样交付**到了主线用来评测的模型上
```

⇒ **也就是说：我们此前评测过的每一个 RL 模型，都只学到了 1/3 的数据。**

同一份排查还查到两条相关的：
```
P1  models/Qwen3-4B-rl-v13-s110 的**主权重与 SFT 模型逐位相同** ——
    RL 学到的东西只在 lora_adapter/ 里。评测是对的（显式传了 --adapter），
    但**下一轮 RL 拿它当起点会静默丢掉整轮 RL**（launch_rl 没有加载 adapter 的入口）
    ⇒ 已加断言 `_assert_model_is_merged`（给错宁可报错）
P3  prune_rl_ckpts.py 用 `next(glob(...))` **非确定性地**留一份 LoRA、删掉其余
    ⇒ global_step_5..25 的 rank1/2 **已永久丢失**，事后无法重建正确的 DDP 平均
    ⇒ 只有未瘦身的 global_step_27 还留着三份
```

⚠️ **P3 是这次事故里唯一不可逆的损失** —— 而它恰恰是"读一个 rank 就代表全部"这个
前提的下游。⇒ 三个读单 rank 的工具（drift / 瘦身 / merger）**都已加上跨 rank 一致性断言**。

## 6 · 这条与 E20 的关系

**E20（序列级 IS 崩塌）和 E21（梯度不同步）是两个独立的 bug**，但它们叠在一起：

```
E21  每次更新只用 1/3 的数据      ⇒ 信号本来就少
E20  序列级 IS 把 67% 的样本压平  ⇒ 剩下的信号又被砍掉三分之二
⇒ 合起来：**每次更新真正起作用的样本 ≈ 16 × 0.33 ≈ 5 条**
```

⇒ 「跑完一整遍数据集、能力几乎没变化」这个现象，**两条都有份**。
⇒ **E20-a（token 级 IS）的修复仍然有效且应该保留** —— 它把 0.33 那一项修回 1.0；
但只有 E21 修好，那 16 才会变回 48。
