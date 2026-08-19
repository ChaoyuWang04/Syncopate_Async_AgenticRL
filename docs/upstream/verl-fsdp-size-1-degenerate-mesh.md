# 上游 issue 草稿 · verl：`fsdp_size=1` 会造出退化的 `(N, 1)` mesh，导致**梯度不再同步**

> 状态：**草稿完成，待 Chaoyu 决定是否提交**　建于 2026-08-18　**★ 2026-08-19 升级为主战场**
> 归属：独立线（`docs/upstream/`）。完整实验记录：[`../infra_exp/E21-ddp-not-syncing.md`](../infra_exp/E21-ddp-not-syncing.md)
> 配套：[`pytorch-fsdp-hybrid-shard-no-grad-sync.md`](pytorch-fsdp-hybrid-shard-no-grad-sync.md)（同一根因的上游那一半）

> 🆕 **2026-08-19 · 提交前调查完成，五条结论**：
>
> 1. **仓库已迁移 `volcengine/verl` → `verl-project/verl`**（旧名下 GitHub 搜索 API 直接报
>    「resources do not exist」——先前"搜不到"有一半是这个原因）。提交与引用一律用新名。
> 2. **空白确认**：新仓库 issues+PRs 全历史 —— `HYBRID_SHARD` **0 命中**；
>    `NO_SHARD`(9)/`create_device_mesh`(2)/`fsdp_size`(90) 逐条看过，无一是本问题。
>    **没有人报过、没有 PR、没有官方回复。**
> 3. 最形似的 [#2478](https://github.com/verl-project/verl/issues/2478)（2025-07）：贴的是**同一行
>    UserWarning**，但成因是集群没配好（FULL_SHARD、world_size 真的是 1），维护者答「去查 ray status」，
>    半年后被清理机器人关闭。⇒ ★ **同一行 warning 的常见成因是良性的** —— 看到它的人第一反应是查集群，
>    没有人会想到"梯度不再同步了"。这行 warning 无法承担报警职责，**必须在框架层堵**。
> 4. **PyTorch 侧确定不修**：[pytorch#154888](https://github.com/pytorch/pytorch/issues/154888)
>    维护者确认是 bug（"this is a bug ... we might be slow in fixing fsdp1"）后被关成
>    `not_planned`（FSDP1 维护模式）⇒ **verl 是唯一能兜住的地方** —— 这句要写进 issue，
>    把"该谁修"提前回答死。
> 5. ★ **修法已换成更小的形态并实测通过**（§5-①，七变体矩阵的 G 行）；ckpt 探针证明它
>    **不改变 checkpoint 行为**（§5-①-c）。产物：`_audit/infra/e21_grad_sync_matrix.json`。

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

## 2 · 代码路径（verl 0.8.0；✅ 2026-08-18 复核：main 分支 `create_device_mesh` / `get_sharding_strategy` **逐字未变**，今天写 `fsdp_size=1` 的人仍在踩）

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

### 3.3 脱离 verl 的最小复现（2026-08-19 扩成七变体确定性矩阵）

见配套文档 §3.1：纯 PyTorch、3 卡、全脚本确定性（任何人可逐位复验）。给 verl 看的三行：

```
A  mesh(3,1)+HYBRID_SHARD（= fsdp_size=1 现状）   [0.0457, 0.0913, 0.1370]   🔴 没同步，一步后已发散
G  mesh(3,1)+NO_SHARD（本文提议的修法）            [0.27395082] ×3            ✅ 网格一个字不动
D  纯 DDP（对照组）                                [0.27395082] ×3            ✅ 装置是对的
```

⇒ **根因在 FSDP，但 verl 的 mesh 构造是触发条件；而 FSDP1 上游已确认不修（#154888）。**
产物：`_audit/infra/e21_grad_sync_matrix.json` · 脚本 `scripts/repro_fsdp_hybrid_nosync.py`。

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

**① 直接的修法（★ 2026-08-19 已实测验证 —— 矩阵 G 行）**：
`get_sharding_strategy` 遇分片维为 1 时返回 `NO_SHARD`，**mesh 一个字不动**：

```python
# verl/workers/engine/fsdp/utils.py :: get_sharding_strategy
elif device_mesh.ndim == 2:
    if device_mesh.size(1) == 1:
        # fsdp_size=1 ⇒ 分片维退化。HYBRID_SHARD 会被 FSDP 钳成 NO_SHARD，
        # 但梯度归约仍留在那个 size-1 的分片组上 ⇒ 空操作，跨 rank 永不同步且不报错
        # （PyTorch 侧已确认是 bug 且不修：pytorch#154888, closed not_planned）。
        # 显式 NO_SHARD：FSDP 的非 hybrid 路径取 mesh_dim=0（复制维）当归约组
        # （torch/distributed/fsdp/_init_utils.py:119）⇒ 归约落在正确的 N 个 rank 上。
        return ShardingStrategy.NO_SHARD
    sharding_strategy = hsdp_strategy
```

为什么它是影响面最小的：

```
a. get_sharding_strategy 全仓库只有 1 个调用点（transformer_impl.py:375）；签名/配置项零变化
b. mesh 形状与 mesh_dim_names 完全不变 ⇒ model_merger 的
   assert mesh_dim_names in (("fsdp",), ("ddp","fsdp")) 不受影响；state._device_mesh 照设
c. ★ ckpt 行为实测不变：NO_SHARD 下 SHARDED_STATE_DICT 本来就短路成 full tensor
   （torch 自打 "When using NO_SHARD ... full_state_dict will be returned"）——
   当前坏形态（被钳的 NO_SHARD）、本修法、我们线上补丁三者的探针输出**完全相同**
   ⇒ resume 兼容，不需要动 checkpoint 路径
d. 非退化配置（size(1)>1 / 一维 mesh）走原路，一个字节都没动
e. 唯一改变的语义 = 梯度归约的进程组：size-1 分片组 → 复制维组。正是坏掉的那件事
```

> ⛔ **被否掉的旧方案（2026-08-18 首版，保留示错）**：`create_device_mesh` 在 `fsdp_size==1` 时
> 返回 `None`/一维 mesh。否决理由：要同时改两个函数的契约；`mesh_dim_names` 变成 `("fsdp",)`
> 会撞 `model_merger` 的断言、影响旧 ckpt 恢复；下游所有拿 mesh 的地方要加 None 判断。
> **改动面数倍于 ①，收益相同。**

**② 防御性的修法（建议同时做）**：构造完之后**断言梯度确实会同步**

```python
# 训练第一步之后，对一个可训练参数的梯度做一次 all_gather 比较；不一致就直接失败。
# 成本：一次 all_gather，只在第一步做。收益：这类静默失效再也跑不过第一步。
```

★ **我们更希望有②** —— 因为①只修了这一种退化形状，
而"梯度没同步"这一类失败**在任何配置下都应该是硬失败，不是靠人去比对 checkpoint 才发现**。

**PR 必须带的测试**：③ 3 rank 玩具模型（照矩阵脚本缩）——修前梯度各异、修后逐位相同；
④ `SHARDED_STATE_DICT` save→load→resume 一轮（钉住 c 那条「格式不变」）。

---

## 6 · ⚠️ 还没做的

1. ~~没有验证修法在 verl 里跑通~~ ✅ **全闭合（2026-08-19）**：
   ①的**确切 diff** 已打进 verl 源码树（`utils.py::get_sharding_strategy`）真跑 4 步
   （fully_async 3+1、Qwen3-4B+LoRA r32、**我们的 monkeypatch 关闭** ⇒ verl 侧修法是唯一机制）：
   - 判据行 `[verl-G-fix] degenerate (N,1) mesh -> NO_SHARD` 在 worker 里打出
   - 运行时：step=3 逃过 Ray 去重的显式对照 —— rank0 与 rank2 权重范数 0.016867、
     梯度范数 2.804719e-03 **完全相同**（⚠️ Ray 日志去重忽略数字差异，被折叠的行不算证据）
   - 落盘：`assert_ranks_identical` 三 rank **504/504 张量逐位相同** + 优化器一致，
     指纹 `_audit/infra/e21_verl_gdiff_rank_fingerprint.json` · 日志 `logs/e21_verl_gdiff_20260819.log`
2. ~~`fsdp2` 路径未测~~ ✅ **已测**：同一个 `(3,1)` mesh 交给 `fully_shard` 梯度同步正确（矩阵 F 行）
   ⇒ `strategy: fsdp2` 不受此 bug 影响，issue 里可写「workaround: 切 fsdp2 或等本修法」。
3. **只测了 `world_size=3`**。
4. 只在 **verl 0.8.0 / torch 2.9.0** 上测过（⚠️ main 的两个函数逐字未变，见 §2）。

---

## 7 · 提交清单

- [x] 复核 verl 主干是否已改 ——**未改，逐字相同**（2026-08-18；仓库已迁 verl-project/verl）
- [x] 全库搜索确认空白（issues+PRs：HYBRID_SHARD 0 命中；#2478 是同 warning 的良性成因，反成论据）
- [x] 修法实测（矩阵 G 行）+ ckpt 探针（格式不变）——`_audit/infra/e21_grad_sync_matrix.json`
- [x] ①的确切 diff 在 verl 源码树里跑短训练 ——**504/504 张量三 rank 逐位相同**（§6-1，2026-08-19）
- [ ] 附 §3.1 的运行时探针输出（**LoRA `B` 零初始化那条判据是核心**）
- [ ] 附 §3.3 的纯 PyTorch 复现（**含 DDP 对照组**）
- [ ] 附 §4 的后果说明（"在任何指标上都看不出来"这一句要突出）
- [ ] 提出 §5-② 的防御性断言（比只修 mesh 更有价值）
- [ ] 与 PyTorch 那条互相引用
