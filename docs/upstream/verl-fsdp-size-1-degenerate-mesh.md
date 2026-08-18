# 上游 issue 草稿 · verl：`fsdp_size=1` 会造出退化的 `(N, 1)` mesh，导致**梯度不再同步**

> 状态：**草稿完成，待 Chaoyu 决定是否提交**　建于 2026-08-18
> 归属：独立线（`docs/upstream/`）。完整实验记录：[`../infra_exp/E21-ddp-not-syncing.md`](../infra_exp/E21-ddp-not-syncing.md)
> 配套：[`pytorch-fsdp-hybrid-shard-no-grad-sync.md`](pytorch-fsdp-hybrid-shard-no-grad-sync.md)（同一根因的上游那一半）

---

## 0 · 一句话

**`actor_rollout_ref.actor.fsdp_config.fsdp_size=1` 的本意是"不分片、纯数据并行"，
但它会让 verl 构造出 `(world_size, 1)` 的二维 device mesh 并选中 `HYBRID_SHARD`；
FSDP 随后把它降级成 `NO_SHARD`，而梯度归约落在那个只有 1 个成员的分片组上
⇒ `world_size` 个 rank 各训各自的模型，永不同步，训练照常跑完且所有指标正常。**

---

## 1 · 触发条件（很容易撞上）

```yaml
actor_rollout_ref.actor.fsdp_config.fsdp_size: 1      # ← 「不要分片」
trainer.n_gpus_per_node: 3                            # ← 多卡
actor_rollout_ref.actor.strategy: fsdp                # FSDP1
```

⇒ 任何"**多卡 + 不想分片**"的场景。对**小模型 / LoRA** 尤其自然：
模型单卡放得下，分片只会白付通信，所以 `fsdp_size=1` 是**推荐做法**。

⚠️ 我们正是这么用的，而且 `launch_rl` 的注释里还写着
「本机默认 1 = 不切分 = DDP，`-1` 全切分在这台没有 P2P 的机器上实测慢 6 倍，别用」——
**配置的意图完全正确，坏在实现路径上。**

---

## 2 · 代码路径（verl 0.8.0）

```python
# verl/workers/engine/fsdp/utils.py:40
def create_device_mesh(world_size, fsdp_size):
    if fsdp_size < 0 or fsdp_size >= world_size:
        device_mesh = init_device_mesh(device_name, mesh_shape=(world_size,), mesh_dim_names=["fsdp"])
    else:
        device_mesh = init_device_mesh(
            device_name,
            mesh_shape=(world_size // fsdp_size, fsdp_size),   # ← fsdp_size=1 ⇒ **(N, 1)**
            mesh_dim_names=["ddp", "fsdp"],
        )
    return device_mesh

# verl/workers/engine/fsdp/utils.py:61
def get_sharding_strategy(device_mesh, ...):
    # 二维 mesh ⇒ HYBRID_SHARD
```

⇒ `fsdp_size=1` 落进 `else` 分支 ⇒ mesh `(N, 1)` ⇒ 二维 ⇒ `HYBRID_SHARD`
⇒ **分片维大小是 1**（等于不分片，意图达成），
**但 FSDP 随后把归约留在了这个大小为 1 的组上**（详见配套文档）。

日志里唯一的痕迹：
```
ShardingStrategy.HYBRID_SHARD
UserWarning: FSDP is switching to use `NO_SHARD` instead of ShardingStrategy.HYBRID_SHARD
             since the world size is 1
```

---

## 3 · 证据

### 3.1 运行时（挂在 `FSDPEngine.optimizer_step` 之前）

```
step=1  rank=0  lora_B  权重范数=0.000000   梯度范数=2.209380e-05
step=1  rank=1  lora_B  权重范数=0.000000   梯度范数=2.565470e-05
                        ↑ 起点完全相同         ↑ **梯度差 16%**
step=4  rank=0  lora_B  权重=0.013482       梯度=2.906737e-04
step=4  rank=2  lora_B  权重=0.013628       梯度=8.142963e-05     ← 差 3.6 倍
```

★ **判据为什么无可辩驳**：LoRA 的 `B` 是**零初始化**的 ——
step 1 时三个 rank 的权重**都是 `0.000000`**（起点完全一致），
而梯度不同 ⇒ **只可能是没有 all-reduce**。

### 3.2 保存下来的 checkpoint（训练 15 次更新后）

| 检查项 | 结果 | 说明 |
|---|---|---|
| 基座（冻结）权重跨 rank | 相对差 **0.0** | ⇒ 是复制不是分片（排除"读错分片"） |
| `lora_A` 跨 rank | 范数几乎相同、**相对差 1.415**（≈√2） | 方向完全不同 |
| `lora_B` 跨 rank | 范数 0.0931 / 0.0787 / 0.0754，相对差 **1.3** | 零初始化 ⇒ 差异全部来自更新 |
| Adam `exp_avg_sq` 跨 rank | 相对差 **99%** | 梯度历史完全不同 |

### 3.3 脱离 verl 的最小复现

见配套文档 §3：纯 PyTorch、3 卡，`mesh(3,1)+HYBRID_SHARD` 不同步，
而 `NO_SHARD`（不传 mesh）与 `DDP` **打出逐位相同的梯度**。
⇒ **根因在 FSDP，但 verl 的 mesh 构造是触发条件。**

---

## 4 · 后果

```
配置意图    3 卡数据并行，每次更新 6 prompt × 8 采样 = **48 条序列**
实际发生    每个 rank 只用自己那份 ≈ **16 条**，且三份模型各自演化
⇒ 最终被 checkpoint / 被推给 vLLM 的那一份，**只学到了 1/3 的数据**
```

⚠️ **而且它在任何指标上都看不出来**：loss 会降、reward 会动、grad_norm 正常、
熵正常、ESS 正常。我们是在写"ckpt → PEFT adapter"转换脚本时，
被一条随手加的断言（"DDP 下各 rank 的 LoRA 应该相同"）**炸出来的**。

---

## 5 · 建议的修法

**① 直接的修法**：`fsdp_size == 1` 时不要构造二维 mesh

```python
def create_device_mesh(world_size, fsdp_size):
    if fsdp_size < 0 or fsdp_size >= world_size:
        return init_device_mesh(device_name, mesh_shape=(world_size,), mesh_dim_names=["fsdp"])
    if fsdp_size == 1:
        # 「不分片」应当用 NO_SHARD + 默认进程组表达；
        # (N, 1) 的 mesh 会让 FSDP 降级成 NO_SHARD 却把梯度归约留在大小为 1 的组上
        return None            # 并让 get_sharding_strategy 返回 NO_SHARD
    return init_device_mesh(device_name, mesh_shape=(world_size // fsdp_size, fsdp_size),
                            mesh_dim_names=["ddp", "fsdp"])
```

**② 防御性的修法（建议同时做）**：构造完之后**断言梯度确实会同步**

```python
# 训练第一步之后，对一个可训练参数的梯度做一次 all_gather 比较；不一致就直接失败。
# 成本：一次 all_gather，只在第一步做。收益：这类静默失效再也跑不过第一步。
```

★ **我们更希望有②** —— 因为①只修了这一种退化形状，
而"梯度没同步"这一类失败**在任何配置下都应该是硬失败，不是靠人去比对 checkpoint 才发现**。

---

## 6 · ⚠️ 还没做的

1. **没有验证修法在 verl 里跑通** —— 本文提交时我们正在打补丁并复验；
   若复验通过会补上「修复前后梯度逐位相同」的对照。
2. **只在 FSDP1（`strategy: fsdp`）上测过**；`fsdp2` / megatron 路径未测。
3. **只测了 `world_size=3`**。
4. 只在 **verl 0.8.0 / torch 2.9.0** 上测过。

---

## 7 · 提交清单

- [ ] 复核 verl 主干是否已改（本文基于 0.8.0）
- [ ] 附 §3.1 的运行时探针输出（**LoRA `B` 零初始化那条判据是核心**）
- [ ] 附 §3.3 的纯 PyTorch 复现（**含 DDP 对照组**）
- [ ] 附 §4 的后果说明（"在任何指标上都看不出来"这一句要突出）
- [ ] 提出 §5-② 的防御性断言（比只修 mesh 更有价值）
- [ ] 与 PyTorch 那条互相引用
