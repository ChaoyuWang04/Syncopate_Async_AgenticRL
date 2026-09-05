# 03 · colocate / split / hybrid：时间复用 vs 空间划分

> 调查日期：2026-07-28
> 代码基准：`reference/industrial_posttrain_training_release/verl/upstream/`（verl `0.8.0.dev` 快照，下简称 `UP/`）
> 承接 [01-why-verl](01-why-verl.md) §2 的"两次解耦"——这篇把它落到具体代码行

---

## 0. 三种资源模式的坐标系

先把词统一了，后面才不会乱（verl 自己的命名也有点混乱，见 §3.1）：

| 模式 | 空间：卡怎么分 | 时间：谁在什么时候用卡 | verl 里的名字 |
|---|---|---|---|
| **colocate（共卡）** | 所有角色共用同一批卡 | **分时复用**：rollout 完 sleep 掉，训练完再 wake up | `hybrid_engine=True` → `RolloutMode.HYBRID` |
| **split（分卡）** | trainer 一批卡，rollouter 另一批 | 各自常驻，同时跑 | `hybrid_engine=False` → `RolloutMode.STANDALONE` |
| **hybrid（混合）** | 一部分 rollout 副本贴着 trainer，一部分独占卡 | 两种并存，还能弹性增减 | fully_async 的 `hybrid_replicas` + standalone 混编 |

**核心权衡一句话**：colocate 用**时间**换空间（显存省了，但 sleep/wakeup + 权重同步的切换开销要付），split 用**空间**换时间（无切换开销，但 rollout 卡在训练时闲着、训练卡在 rollout 时闲着，且引入 staleness）。

**老师的配置是 full colocate。** 下面逐条验证。

---

## 1. `ResourcePoolManager`：角色 → 资源池的映射表

### 1.1 定义（`UP/verl/single_controller/ray/base.py:182`）

```python
@dataclass
class ResourcePoolManager:
    resource_pool_spec: dict[str, list[int]]   # pool_id → 每个节点几张卡
    mapping: dict[int, str]                    # Role → pool_id
    max_colocate_count: int = 3
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def get_resource_pool(self, role) -> RayResourcePool:
        return self.resource_pool_dict[self.mapping[role]]
```

**整个资源调度的全部信息就是这两个 dict。** `resource_pool_spec` 声明有哪些池、每池多大；`mapping` 声明每个角色去哪个池。改部署形态 = 改这两个 dict，算法代码一行不动——这就是 [01-why-verl](01-why-verl.md) 说的"解耦①"的物理实现。

`max_colocate_count` 的注释（`:200-204`）说明了它的含义：一个 RayResourcePool 里允许放几个 WorkerGroup（进程）。FSDP 后端用 3（actor_critic_ref / rollout / reward model），Megatron 建议 >1 以便不同模型用不同 WorkerGroup。

### 1.2 默认构造：**所有角色 → `global_pool`**（`UP/verl/trainer/main_ppo.py:157-190`）

```python
def init_resource_pool_mgr(self, config):
    global_pool_id = "global_pool"
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    if config.reward.reward_model.enable_resource_pool:          # 可选：reward model 独立池
        resource_pool_spec["reward_pool"] = [...]
    if is_distillation_enabled(distillation_config):             # 可选：蒸馏 teacher 独立池
        resource_pool_spec["teacher_pool"] = [...]
    return ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=self.mapping)
```

而 `mapping` 在注册 worker 时被硬编码成 `global_pool`：

```python
self.mapping[role] = "global_pool"          # :145  actor/rollout/ref
self.mapping[Role.Critic] = "global_pool"   # :156  critic
```

### 1.3 老师的配置：**确认是 full colocate**

老师的 override（`REL/scripts/train_grpo_verl.py`）里：
- `critic.enable=False`、`reward_model.enable=False`、无 distillation
- 没有传 `enable_resource_pool`
- 没有传 `actor_rollout_ref.hybrid_engine`（默认 `true`，见 `UP/verl/trainer/config/ppo_trainer.yaml:53`）

所以实际展开是：

```python
resource_pool_spec = {"global_pool": [64]}        # trainer.n_gpus_per_node=64, nnodes=1
mapping = {Role.ActorRolloutRef: "global_pool"}   # 就这一条
```

**只有一个池、只有一个角色条目。** 而且注意 `Role.ActorRolloutRef`（`UP/verl/trainer/ppo/utils.py:33`）——actor、rollout、ref **三个角色被融合进同一个 worker 类** `ActorRolloutRefWorker`（`main_ppo.py:138-146`）。这比"共卡"更进一步，是**同进程融合**：

```python
if need_reference_policy(config) and not ref_in_actor:
    role = Role.ActorRolloutRef      # ← 老师走这条（未开 LoRA）
else:
    role = Role.ActorRollout
```

> 补充：如果开了 LoRA（`lora_rank > 0`），ref 会退化成"actor 关掉 adapter"（`ref_in_actor=True`），连 ref 的那份权重都省了——`_compute_ref_log_prob` 里传 `no_lora_adapter=True`（`ray_trainer.py:1241`）。**Phase 1 用 7B+LoRA 时会自动吃到这个红利**，值得记一笔。

### 1.4 标准 trainer 只支持 colocate

`UP/verl/trainer/ppo/ray_trainer.py:333-334`：

```python
self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
assert self.hybrid_engine, "Currently, only support hybrid engine"
```

**`RayPPOTrainer` 硬性只接受 colocate。** 想做 split 就必须换 trainer——这正是 `experimental/fully_async_policy/` 和 `one_step_off_policy/` 存在的原因（它们都反过来 `assert not self.hybrid_engine`）。

---

## 2. split / hybrid 池的官方示例

### 2.1 训练包里没有 `examples/`，但 `fully_async_policy/shell/` 有三个

老师的 release 把 verl 的 `examples/`、`recipe/`、`docs/` 都裁掉了，唯一幸存的示例是：

```
UP/verl/experimental/fully_async_policy/shell/
├── dapo_7b_async_retool.sh
├── grpo_qwen35_35b_megatron_async.sh
└── geo3k_qwen3vl_30b_megatron_6_2_npu_async.sh
```

### 2.2 关键片段：单机内切分（`dapo_7b_async_retool.sh`）

```bash
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}
n_gpus_rollout=4                                    # :57
n_gpus_training=$((NGPUS_PER_NODE - n_gpus_rollout))  # :58  → 4
...
    actor_rollout_ref.hybrid_engine=False \         # :88  ★ 关掉共卡
    actor_rollout_ref.rollout.mode=async \          # :109
    trainer.nnodes=$NNODES \                        # :130
    trainer.n_gpus_per_node=$n_gpus_training \      # :131  ← trainer 池 4 卡
    rollout.nnodes=$NNODES \                        # :132
    rollout.n_gpus_per_node=$n_gpus_rollout \       # :133  ← rollout 池 4 卡
```

**声明多个池的方式不是写一个 dict，而是新增一个顶层 `rollout:` 配置节。** 定义在 `UP/verl/experimental/fully_async_policy/config/fully_async_ppo_trainer.yaml:29-41`：

```yaml
# Rollout config
rollout:
  nnodes: 1              # rollout 池节点数
  n_gpus_per_node: 8     # rollout 池每节点卡数
  n: 4                   # GRPO 采样数
  total_rollout_steps: 100
```

注意**它和 `actor_rollout_ref.rollout` 是两个不同的东西**：顶层 `rollout:` 管**资源池划分**，`actor_rollout_ref.rollout:` 管**引擎参数**（vLLM 的 TP、gpu_memory_utilization 等）。这个命名很容易看错。

### 2.3 跨节点切分（`grpo_qwen35_35b_megatron_async.sh:241-244`）

```bash
    trainer.nnodes=${NNODES_TRAIN} \
    trainer.n_gpus_per_node=${NGPUS_PER_NODE} \
    rollout.nnodes=${NNODES_ROLLOUT} \
    rollout.n_gpus_per_node=${NGPUS_PER_NODE} \
```

整节点粒度分给 trainer / rollouter。**对我们 Phase 2 的 3+1 / 2+2 实验，用的是 §2.2 的单机内切法。**

---

## 3. colocate 的时间复用机制

### 3.1 `RolloutMode` 三态（`UP/verl/workers/rollout/replica.py:54-67`）

代码注释非常清楚，直接抄：

```python
class RolloutMode(Enum):
    # Rollout engine and training engine(fsdp/megatron) fused in same process.
    # Rollout and trainer share GPUs, switch context with weight synchronization.
    # Usage scenarios: on-policy training.
    HYBRID = "hybrid"

    # Rollout engine colocated with hybrid engine in same ray placement group but in separate process.
    # Rollout and hybrid processes share GPUs, switch context without weight synchronization.
    # Usage scenarios: GRM (LLM as a judge).
    COLOCATED = "colocated"

    # Standalone rollout server with separate GPU resource, disaggregated architecture.
    # Usage scenarios: off-policy training.
    STANDALONE = "standalone"
```

> ⚠️ **命名陷阱**：verl 的 `COLOCATED` **不是**我们平时说的"共卡训练"！我们说的共卡对应的是 **`HYBRID`**。`COLOCATED` 特指"同一个 placement group 但**独立进程**、**不做权重同步**"的场景（比如 LLM-as-judge 的奖励模型——它权重是冻结的，本来就不需要同步）。读代码时极易搞反。

模式不由 `rollout.mode` 直接决定，而由调用哪个 `init_*` 决定（`UP/verl/workers/rollout/llm_server.py:314-325`）：

```python
if self.worker_group and self.rollout_config.name != "trtllm":
    await asyncio.gather(*[server.init_hybrid(self.worker_group) ...])       # → HYBRID
elif self.worker_group and self.rollout_config.name == "trtllm":
    ... init_hybrid_colocated(...)                                            # → HYBRID
else:
    await asyncio.gather(*[server.init_standalone() ...])                     # → STANDALONE
```

**判据是 `worker_group` 是否存在**——`hybrid_engine=True` 时 trainer 已经建好了融合的 worker group 并传进来，走 HYBRID；`hybrid_engine=False` 时没有 worker_group，走 STANDALONE 自建资源池。老师的配置 → **HYBRID**。

### 3.2 sleep / wake_up 的调用点与 level

**入口是 `update_weights`**（`UP/verl/workers/engine_workers.py:665-745`），一次完整的"训练态 → rollout 态"切换，四步：

```python
# 1. resume rollout memory (weights were released during sleep)
if self.config.rollout.free_cache_engine:
    await self.rollout.resume(tags=["weights"])           # :706-707

# 2-3. 拿训练侧权重，推给 rollout
per_tensor_param, peft_config = self.actor.engine.get_per_tensor_param(...)   # :711
await self.rollout.update_weights(per_tensor_param, ...)                       # :729

# 3. offload model to cpu   ← 训练侧权重让位
if self.actor.engine.is_param_offload_enabled:
    self.actor.engine.to("cpu", model=True, optimizer=False, grad=False)       # :737
aggressive_empty_cache(force_sync=True)                                        # :738

# 4. resume kv_cache        ← 最后才给 KV cache 分配显存
if self.config.rollout.free_cache_engine:
    await self.rollout.resume(tags=["kv_cache"])                               # :741-742
```

**注意 resume 被拆成了 `weights` 和 `kv_cache` 两个 tag，中间插入训练侧 offload。** 这是精心设计的显存腾挪顺序：先只恢复 rollout 权重（小），同步完权重后立刻把训练侧参数甩到 CPU（腾出大块），最后才让 KV cache（最大）来填这块空地。**如果一次性 wake_up 全部，峰值会同时包含训练参数和 KV cache，直接 OOM。** 这个顺序是 colocate 能跑起来的关键。

**sleep level**（`UP/verl/workers/rollout/vllm_rollout/vllm_async_server.py:626-635`）：

```python
async def sleep(self):
    if self.node_rank != 0 or not self.config.free_cache_engine:
        return
    if self.rollout_mode == RolloutMode.HYBRID:
        await self._sleep_hybrid()                    # ← 老师走这条
    elif self.rollout_mode == RolloutMode.COLOCATED:
        await self.engine.sleep(level=1)
    elif self.rollout_mode == RolloutMode.STANDALONE:
        logger.info("skip sleep in standalone mode")  # ← split 模式根本不 sleep
```

`_sleep_hybrid`（`:933-950`）：

```python
if self.lora_as_adapter or is_torch_npu_available(...):
    sleep_level = 1      # LoRA 只更新 adapter 权重
else:
    sleep_level = 2      # ← 老师（full-param）走这条
await self.engine.sleep(level=sleep_level)
```

| level | 释放什么 | 谁用 |
|---|---|---|
| **1** | 只放 KV cache，**保留权重** | LoRA（`lora_as_adapter`）、NPU、`RolloutMode.COLOCATED` |
| **2** | 权重 + KV cache **全放** | **full-param HYBRID（老师的情况）** |

> **对 Phase 1 的直接影响**：Phase 1 计划用 7B + LoRA。**开 LoRA 会让 sleep level 从 2 降到 1**，即 vLLM 权重常驻不释放——显存峰值行为和 full-param **不一样**。测"colocate 切换开销"时必须注明是哪种，否则和 Phase 0（0.6B full-param，level=2）的数字不可比。这是个容易埋雷的对照组污染点。

对应的 wake_up 默认两个 tag 都要（`:929-931`）：`return ["kv_cache", "weights"]`。另外 wake_up 后会 `reset_prefix_cache(reset_connector=True)`（`:614, 622`）——**权重变了，prefix cache 里按旧权重算出的 KV 必须作废**，否则是静默的正确性 bug。

### 3.3 权重同步：**没有 sharding_manager 了**，改成 per-tensor 流式

> ⚠️ 又一个"网上教程会带偏你"的点：老版 verl 有 `verl/workers/sharding_manager/fsdp_vllm.py` 专门做 FSDP→vLLM 的 reshard。**这个模块在 0.8 快照里已经不存在**（全仓库搜 `sharding_manager` 只剩 `verl/utils/ulysses.py`，那是 Ulysses SP 的，另一回事）。

现在的路径是 `get_per_tensor_param()`（`UP/verl/workers/engine/fsdp/transformer_impl.py:794-871`），核心是一个**生成器**：

```python
load_fsdp_model_to_gpu(self.module)                    # :801  先把参数拉回 GPU
params = self.module.state_dict()                      # :824
params = convert_weight_keys(params, ...)              # :822  FSDP 键名 → HF 键名

if self._is_offload_param:
    offload_fsdp_model_to_cpu(self.module)             # :830  ★ 立刻又甩回 CPU

per_tensor_param = (                                   # :836-845  ← 生成器，惰性求值
    (
        name,
        param.to(device, non_blocking=True).full_tensor().to(torch.bfloat16, non_blocking=True)
        if isinstance(param, DTensor) else param,
    )
    for name, param in params.items()
)
return per_tensor_param, peft_config_dict
```

**回答"是 all-gather 到完整权重再切 TP，还是有更省的路径"：是逐张量 all-gather 的流式路径，不是整模型 all-gather。**

- `param.full_tensor()` 是 **DTensor 的 per-tensor all-gather**——一次只还原**一个**张量的完整形状；
- 因为包在**生成器表达式**里，只有当 `rollout.update_weights()` 迭代到某个张量时才真正 all-gather，用完即释放；
- 峰值额外显存 ≈ **单个最大张量**（通常是 embedding / lm_head），而不是整个模型；
- 顺手做了 `.to(torch.bfloat16)`——fp32 master weight 在传输前降到 bf16，**同步流量减半**（`:834` 的 TODO 说想做更细粒度控制，比如 MoE gate 保持 fp32）。

TP 切分不在这里做：verl 交出的是 **HF 键名的完整张量**，由 vLLM 自己在 `update_weights` 里按它的 TP 布局切。**职责边界很干净**——这也是为什么换 rollout 引擎（vLLM ↔ SGLang）对训练侧几乎无感。

另有 `layered_summon` 参数（LoRA 场景逐层 gather）和 `_qat_enabled` 分支（量化感知训练时顺带量化）。

### 3.4 param_offload / optimizer_offload：**verl 自己实现的（FSDP1），FSDP2 才用原生**

`UP/verl/utils/fsdp_utils.py:167-215`：

```python
def offload_fsdp_model_to_cpu(model: FSDP, empty_cache: bool = True):
    if fsdp_version(model) == 2 or fsdp_version(model) == 0:
        offload_fsdp2_model_to_cpu(model, empty_cache)      # → 就是 model.cpu()
        return

    assert isinstance(model, FSDP)
    _lazy_init(model, model)
    assert model._is_root, "Only support root model offloading to CPU"
    for handle in model._all_handles:                        # ← 手动遍历 FSDP 内部 handle
        if handle._offload_params:
            continue
        flat_param = handle.flat_param
        assert (flat_param.data.data_ptr() == flat_param._local_shard.data_ptr() and ...)
        handle.flat_param_to(torch.device("cpu"), non_blocking=True)
        flat_param._local_shard = flat_param.data            # ← 手动维护内部不变量
    if empty_cache:
        get_torch_device().empty_cache()
```

分两条路：

| 后端 | 实现 | 性质 |
|---|---|---|
| **FSDP1** | 手动遍历 `model._all_handles`，操作 `handle.flat_param` 和 `flat_param._local_shard` | **verl 自己写的，且深度依赖 FSDP 私有属性**（`_all_handles`、`_local_shard`、`_lazy_init`、`_is_root` 全是下划线开头） |
| **FSDP2 / 无 FSDP** | `offload_fsdp2_model_to_cpu()` = `model.cpu()` | **PyTorch 原生**（FSDP2 的 DTensor 天然支持） |

那三行 `assert` 是在校验 FSDP1 的内部不变量（`data` 和 `_local_shard` 必须指向同一块内存但是不同 Python 对象）——**典型的"贴着私有 API 写"的代码，PyTorch 小版本升级就可能碎**。这也解释了为什么 verl 在推 FSDP2（`fsdp_config.strategy=fsdp2`，fully_async 示例里就是用的 fsdp2）。

优化器 offload 同理，在同文件里有 `offload_fsdp_optimizer` / `load_fsdp_optimizer`（遍历 `optimizer.state` 逐个 `.to()`）。

**老师的配置全开**：`actor.fsdp_config.param_offload=True`、`optimizer_offload=True`、`ref.fsdp_config.param_offload=True`。代价是每步都要 CPU↔GPU 搬一遍 —— **Phase 1 的 nsys timeline 里这段会很显眼，是"colocate 时间复用成本"的主要构成，务必单独标注计时**。

---

## 4. `fully_async_policy` 的资源划分（概览，Phase 2 再深入）

### 4.1 结构上最大的差异：**两个独立 Ray actor，而不是一个 `fit()` 循环**

`UP/verl/experimental/fully_async_policy/fully_async_main.py`：

```python
self._create_trainer(config)          # :78   FullyAsyncTrainer.remote(...)
self._setup_hybrid_worker_group(config)  # :81
self._create_rollouter(config)        # :84   FullyAsyncRollouter.remote(...)
...
message_queue = MessageQueue.remote(config, max_queue_size)   # :98
ray.get(rollouter.set_message_queue_client.remote(client))    # :103
ray.get(trainer.set_message_queue_client.remote(client))      # :104
...
def _run_training_loop(self):         # :176
    # 同时启动 Rollouter 和 Trainer，等 futures
```

对照 colocate 的 `RayPPOTrainer.fit()`（一段自上而下的顺序代码，generate → reward → logprob → advantage → update），fully_async 是：

| 维度 | colocate `ray_trainer.fit()` | fully_async |
|---|---|---|
| 控制流 | **一个** while 循环，顺序执行 | **两个** Ray actor 各跑各的循环，并发 |
| 通信 | 函数返回值 / DataProto 直传 | **`MessageQueue` Ray actor**（样本队列，有 `max_queue_size` 背压） |
| 权重同步 | 每步必同步（`update_weights` 在循环内） | 按 `trigger_parameter_sync_step` 触发（默认 4 步一次） |
| 断言 | `assert self.hybrid_engine` | `assert not self.hybrid_engine`（trainer:75、rollouter:413） |

### 4.2 资源池：trainer 池 + rollout 池 + 可选 teacher 池

- trainer 侧：沿用 `ResourcePoolManager`，但规模来自 `trainer.n_gpus_per_node`（示例里 = 总卡数 − rollout 卡数）
- rollout 侧：`init_standalone()` **自建独立池**（`replica.py:191+`），规模来自顶层 `rollout.n_gpus_per_node × rollout.nnodes`
- teacher（蒸馏）：`fully_async_rollouter.py:738-748` 单独建 `teacher_pool`

### 4.3 ★ 意外发现：它支持 **hybrid + standalone 混编**，而且能弹性伸缩

`UP/verl/experimental/fully_async_policy/fully_async_rollouter.py:177-215` 的 `_initialize_llm_servers` 分两步：

```python
# ── Step 1: hybrid replicas first (replica_rank 0 … N_e-1) ──
#   "Starting from rank 0 gives hybrid actors the lowest-numbered placement-group
#    bundles which are co-located with the training engine, maximising GPU affinity"
if self.worker_group is not None:
    await super()._initialize_llm_servers(start_rank=0)
    for i, replica in enumerate(self.rollout_replicas):
        self.hybrid_replicas[f"hybrid_{i}"] = replica       # 迁到 hybrid_replicas，先睡着
    ...
# ── Step 2: standalone replicas via parent class ──
#   临时把 worker_group 置 None，逼父类走 init_standalone 分支
```

加上成员变量 `last_hybrid_add_time` / `last_hybrid_remove_time`（`:174-175`），说明**它能在训练过程中动态增删贴着 trainer 的 rollout 副本**——训练闲时把 trainer 的卡临时借给 rollout，忙时收回。

**这就是真正的 "hybrid"（第三种模式）**：不是二选一，而是 split 为主 + colocate 副本作弹性缓冲。

### 4.4 fully_async 的默认配置（`config/fully_async_ppo_trainer.yaml`）

```yaml
async_training:
  staleness_threshold: 0.1          # 样本陈旧度上限
  trigger_parameter_sync_step: 4    # 4 步同步一次参数  ← Phase 2 的 E3 旋钮
  require_batches: 1
  partial_rollout: True             # 参数同步时中断的 rollout 自动续跑  ← E4 旋钮
  use_trainer_do_validate: False

rollout:
  nnodes: 1
  n_gpus_per_node: 8
  n: 4
  total_rollout_steps: 100

actor_rollout_ref:
  actor:
    use_rollout_log_probs: True     # 必须
  rollout:
    calculate_log_probs: True       # "Must be enabled! Otherwise, log_probs cannot be calculated."
    checkpoint_engine:
      backend: "nccl"               # 权重走 NCCL 而非 checkpoint 文件

algorithm:
  rollout_correction:
    bypass_mode: True               # ★
```

**注意最后一条**：fully-async **默认开 `bypass_mode=True`**，即 `old_log_probs := rollout_log_probs`，只有两个策略。这和 [02-train-inference-mismatch](02-train-inference-mismatch.md) §3 完全对上——异步模式下 trainer 算力紧张，省掉重算 old_log_prob 的那次 forward 特别划算，而且数据本来就是 off-policy 的，用 rollout logprob 当锚点在语义上更自洽。

**老师的 colocate 配置是 `bypass_mode=false`（Decoupled 三策略）。** 所以 Phase 2 的 E1 vs E2 对比里，**这两组的 loss 计算路径本身就不同**——如果直接跑默认配置，对比就混入了"三策略 vs 两策略"这个额外变量。**必须显式控制**：要么两边都强制 `bypass_mode=false`，要么承认这是异步方案的一部分并单独拆一格实验。这是我目前看到的 Phase 2 最大的实验设计陷阱。

`checkpoint_engine.backend: "nccl"` 也印证了 §3.3 的 `update_weights` 里 `effective_mode != "naive"` 那条分支（`engine_workers.py:695-699`）——split 模式走 `checkpoint_engine.send_weights()` 跨池传权重，而不是 colocate 的进程内直传。

---

## 5. 仍不确定 / 待验证

1. **切换开销的绝对量级**：sleep(level=2) → 权重同步 → wake_up 一整轮要多久？`update_weights` 里已经埋了 4 个 `log_gpu_memory_usage` 打点（"Before resume weights" / "After resume weights" / "After update_weights" / "After resume kv_cache"），Phase 1 直接读日志就能拆出各段耗时，**不用自己加埋点**。
2. `max_colocate_count=3` 的默认值对 FSDP 是否合适——注释说 FSDP 用 3（actor_critic_ref / rollout / reward），但老师关了 critic 和 RM，实际只需要 2。是否浪费了 placement group bundle 未确认。
3. fully_async 的弹性伸缩（§4.3）触发条件是什么？`last_hybrid_add_time` 的使用逻辑没读，Phase 2 要看。
4. `staleness_threshold: 0.1` 的确切语义（是版本差比例还是样本比例？）——Phase 2 读 rollouter 主循环时确认。
5. STANDALONE 模式下 `sleep()` 直接 return（`vllm_async_server.py:634-635`），所以 split 模式的 rollout **权重和 KV cache 全程常驻**。那么它的显存预算模型和 colocate 完全不同，`gpu_memory_utilization` 该怎么设，需要实测。

---

## 6. 对 Syncopate 的直接影响

1. **Phase 2 的 3+1 / 2+2 切分，照 §2.2 抄**：`hybrid_engine=False` + `trainer.n_gpus_per_node=3` + `rollout.n_gpus_per_node=1`（单机 4 卡）。不需要自己写资源池代码。
2. **实验设计陷阱（最重要）**：fully_async 默认 `bypass_mode=True` 而 colocate 是 `False`，直接对比会混入 loss 路径差异。见 §4.4。
3. **LoRA 会改变 sleep level（2→1）**，Phase 0（full-param）和 Phase 1（LoRA）的 colocate 切换开销数字不可直接比较。见 §3.2。
4. **§3.2 的四步腾挪顺序（resume weights → offload trainer → resume kv_cache）本身就是一张很好的图**，做同步 vs 异步数据流对照图时，colocate 这一侧的核心就是它——而 split 侧这一整段**完全消失**（STANDALONE 不 sleep）。这就是"时间复用 vs 空间划分"最直观的可视化差异。
5. **§3.3 的 per-tensor 流式 all-gather 是个反直觉的好素材**：直觉上"FSDP 分片 → vLLM 需要完整权重"意味着要 all-gather 整个模型，实际是逐张量惰性 gather + 即时 bf16 降精度，峰值只有单个最大张量。写 blog 时值得单独讲。
