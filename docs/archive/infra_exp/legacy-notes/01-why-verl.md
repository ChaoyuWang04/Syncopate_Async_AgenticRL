# 01 · Why verl？——从 TRL 的痛点到两次解耦，再到五层抽象

> 代码锚点全部来自 `reference/industrial_posttrain_training_release/verl/upstream/`（verl `0.8.0.dev` 快照），下文简称 `UP/`。
> 老师包的接线层锚点来自训练包根目录，简称 `REL/`。

---

## 1. 起点：TRL 式实现的痛点 —— 五个角色同时常驻

一次 PPO/GRPO 迭代需要五种"角色"（role），每个都是一份模型或一段计算：

| 角色 | 干什么 | 需要梯度？ |
|---|---|---|
| Rollout（生成器） | 采样轨迹，追求吞吐，要 PagedAttention/连续批处理 | 否 |
| Actor（策略） | 被优化的模型，要 FSDP/Megatron + 优化器状态 | **是** |
| Critic（价值） | 估计 V(s)（GRPO 里去掉了） | **是** |
| Reference（参考） | 算 KL 惩罚的锚点，权重冻结 | 否 |
| Reward Model（奖励） | 打分（本项目用 verifier + LLM judge 替代） | 否 |

朴素实现（早期 TRL 风格）把它们**同时常驻在同一批 GPU 上**，于是有三个绕不开的问题：

1. **显存被角色数量线性放大**：7B 模型 ×5 份权重，还没算 AdamW 的 fp32 master+m+v；
2. **推理和训练抢同一块显存，却需要完全不同的最优配置**：rollout 想要大 KV cache，训练想要大激活空间，两者静态共存必然互相挤压；
3. **时间上严重浪费**：rollout 阶段 actor 的优化器状态白占显存，update 阶段 vLLM 的 KV cache 白占显存——**任一时刻总有一半的驻留内存在闲置**。

第 3 点是关键洞察：这五个角色在一次迭代里**根本不是同时活跃的**，它们有明确的时间先后。既然如此，为什么要同时常驻？

---

## 2. verl 的两次解耦

### 解耦 ①：角色 ↔ 卡（谁在哪张卡上跑，是配置不是代码）

verl 把"角色"和"物理资源"拆成两个独立概念，再用一张映射表连起来：

- 角色枚举：`UP/verl/trainer/ppo/utils.py:27` `class Role(Enum)`
- 资源池管理：`UP/verl/single_controller/ray/base.py:182` `class ResourcePoolManager`
- 融合成一个 worker 类：`UP/verl/single_controller/ray/base.py` 的 `create_colocated_worker_cls`，在 `UP/verl/trainer/ppo/ray_trainer.py:864` 被调用

于是 **colocate（多个角色挤一张卡）和 disaggregate（各占各的卡）只是映射表的差别，算法代码一行不改**。这正是 Syncopate 能"同一份任务代码跑同步 colocate 和 fully-async 分离"的根本原因——`UP/verl/experimental/fully_async_policy/` 换的是映射和调度，不是 loss。

### 解耦 ②：时间复用（同一块显存，不同阶段服务不同角色）

既然角色在时间上错开，就让它们**分时复用**同一块显存：

- 参数/优化器 offload：`UP/verl/utils/fsdp_utils.py:167 offload_fsdp_model_to_cpu` / `:201 load_fsdp_model_to_gpu`
  （老师脚本开了 `actor.fsdp_config.param_offload=True`、`optimizer_offload=True`、`ref.fsdp_config.param_offload=True`）
- rollout 引擎 KV cache 释放：`actor_rollout_ref.rollout.free_cache_engine=True`（老师脚本 `REL/scripts/train_grpo_verl.py:107`）

代价是 offload/reload 的搬运开销和 vLLM 反复重建 KV cache 的开销——**这个开销有多大，正是 Phase 1 要用 nsys 量出来的东西**，也是判断"异步分离是否值得"的天平另一端。

> 一句话总结：**解耦①解决"显存被角色数放大"，解耦②解决"任一时刻总有一半内存闲置"。** fully-async 则是把解耦①推到极致（trainer/rollouter 彻底分家），代价是引入 staleness——见 [02-train-inference-mismatch](02-train-inference-mismatch.md)。

---

## 3. 五层抽象（每层都锚定到本仓库真实位置）

```
L5  算法核函数     advantage / policy loss —— 纯数学，不知道 GPU 的存在
L4  Worker 实现    FSDP actor / vLLM rollout / ref —— 知道自己在哪张卡上
L3  数据协议       DataProto —— 层与层之间唯一的通货
L2  资源与角色映射  Role ↔ ResourcePool ↔ WorkerGroup —— 决定"谁在哪"
L1  单控制器       fit() 主循环 —— 一段可以从上往下读的普通 Python
```

| 层 | 是什么 | 锚点（文件:行） |
|---|---|---|
| **L1 单控制器** | 整个训练循环写成**一段单线程顺序代码**：generate → reward → old_logprob → correction → advantage → update。这是 HybridFlow 的"single-controller"——读起来像单机代码，实际每一行背后是全集群的 SPMD 调用 | `UP/verl/trainer/ppo/ray_trainer.py:1362 RayPPOTrainer.fit()`<br>入口 `UP/verl/trainer/main_ppo.py` |
| **L2 角色↔资源映射** | 把 Role 绑到 ResourcePool，决定 colocate 还是分离 | `UP/verl/trainer/ppo/utils.py:27 Role`<br>`UP/verl/single_controller/ray/base.py:182 ResourcePoolManager`<br>`UP/verl/trainer/ppo/ray_trainer.py:775 init_workers()` |
| **L3 数据协议** | `DataProto`：batch（tensor）+ non_tensor_batch（numpy/object）+ meta_info，带 union/repeat/chunk。**所有跨层数据都长这样**，是读懂 verl 的钥匙 | `UP/verl/protocol.py:318 class DataProto` |
| **L4 Worker 实现** | 真正碰硬件的地方：FSDP 训练 worker、vLLM rollout server、ref worker | `UP/verl/workers/`（loss 入口 `UP/verl/workers/utils/losses.py:88-112`）<br>`UP/verl/utils/fsdp_utils.py`（offload） |
| **L5 算法核函数** | 纯张量函数，无副作用、无分布式概念，最好测也最好读 | `UP/verl/trainer/ppo/core_algos.py:268 compute_grpo_outcome_advantage`<br>`:1279 compute_policy_loss_vanilla`<br>`:70 get_policy_loss_fn`（注册表） |

### 横向扩展点：AgentLoop（L1 与 L4 之间开的一个口子）

多轮 agentic 训练不适合塞进 L1 的顺序循环（每条轨迹的轮数不同、要等工具返回）。verl 的做法是在 rollout 环节开一个**协程级扩展点**，三层结构：

| 组件 | 职责 | 锚点 |
|---|---|---|
| `AgentLoopManager` | 切分 batch 分发给多个 worker，汇总结果与耗时指标 | `UP/verl/experimental/agent_loop/agent_loop.py:1149` |
| `AgentLoopWorker` | Ray actor，每条样本起一个协程，pad 成定长张量 | `:462`（后处理 `:762 _agent_loop_postprocess`） |
| `AgentLoopBase` | 抽象基类，`run()` 是 abstractmethod；只提供 `server_manager.generate()`、`apply_chat_template()` 等公共零件 | `:242` |

**契约是 `AgentLoopOutput`（`:120`）**：不管 loop 内部怎么循环，最终必须交出 `prompt_ids / response_ids / response_mask / reward_score`。训练侧只认这四样，完全不关心工具怎么调的。

老师的实现就是第三个 `run()`：`REL/train/verl_agent_loop_adapter.py:39 IndustrialPosttrainAgentLoop`，通过 `REL/configs/verl_agent_loop.yaml` 把注册名 `industrial_posttrain_agent` 解析到该类。

> **这是整个课程项目最值得学的一处设计**：verl 划的边界是"token 生成留在框架内（才能复用 vLLM 的批处理与 KV 管理），业务循环交给用户"。用户只需实现一个 async 函数，就免费获得了分布式调度、张量 padding、指标聚合。

---

## 4. 代价：这套抽象贵在哪

诚实记录，Phase 4 对比 slime 时要用：

1. **配置面爆炸**：Hydra 多层 override，老师的启动器要拼 60+ 个 override 才能跑起来（`REL/scripts/train_grpo_verl.py:65-138`）。配置项的真实语义常常只能读源码确认。
2. **调用栈深**：一次 `_update_actor()` 要穿过 Ray RPC → WorkerGroup dispatch → FSDP → loss fn 四层，栈里全是框架帧，业务报错定位困难。
3. **PPO-first 的历史包袱**：GRPO 也走 `main_ppo`，`reward_model.enable=False` 却仍有 reward 流转（见 [00-project-overview](../../syncopate/legacy-notes/00-project-overview.md) 的 reward 链路），`critic.enable=False` 只是把 L2 里的一个角色摘掉。命名和实际语义已经漂移。
4. **experimental/ 里躺着四套并行方案**：`agent_loop`、`fully_async_policy`、`one_step_off_policy`、`separation`、`reward_loop`——说明设计空间尚未收敛，也意味着我们 Phase 2 踩的是活跃变动区。

---

## 5. 我们自己要回答的问题（待填充）

- [ ] 在 1×/4×5090 规模下，哪些复杂度是**本质的**（正确性所需），哪些是为 64+ 卡集群设计的**冗余**？
- [ ] slime 的 native pass-through 哲学在同一个任务上要写多少代码？出 bug 时哪个更好查？（Phase 4）
- [ ] AgentLoop 的协程模型在**长尾轨迹**下的实际调度行为：单个 worker 内 N 条协程，最慢的那条会不会拖住整个 worker 的 batch 提交？（Phase 1 用 nsys 验证——这直接决定 Syncopate 的核心论点）
