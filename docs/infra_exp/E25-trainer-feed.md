# E25 · trainer 是「活重」还是「没喂饱」——两个旋钮的证伪

> 状态：✅ **完成（否定结果）**　建于 **2026-08-19**
> 尺子 `scripts/probe_trainer_feed.py` · 原始数据 `logs/e25/` · 预测 `logs/e25/PREDICTION.md`

---

## 0 · 结论卡片

| | |
|---|---|
| **Track / 兑现物** | **B** · 占空比成因③（训练侧三次前向）—— 本报告负责**排除**两个候选解法 |
| **需求从哪来** | E08 §5.5 之后靶子只剩三次前向（占步 88.9%）。动手之前先问：它慢是**活多**还是**喂不饱**？ |
| **答案** | 🔴 **活多。「没喂饱」被证伪。** 拉高 `micro_batch` 是**负收益**，关 `gradient_checkpointing` **显存不够** |
| **省下的** | 两条本来排在队列里的优化路线，以及它们后面所有的调参时间 |
| **推走的** | 唯一有量级的方向只剩「**让它少算**」：prefix grouper（上界 4.1×）· 砍 ref（12.7%）· ref 走 FP8 |

---

## 1 · 问题与预测

**问题**：`update_actor` 占一步 55.5%，`old_log_prob` + `ref` 再占 31.9%。
这些时间里，有多少是「GPU 真在算」，有多少是「GPU 在等着被喂」？

**★ 预测（跑之前写死，原文在 `logs/e25/PREDICTION.md`）**

```
P1  关掉 gradient_checkpointing ⇒ fwd+bwd 快 20–30%，显存峰值 25–30 GB
P2  micro_batch 1 → 2 ⇒ 提升 < 10%（一条序列已有 4850 token，不算小）
P3  gc=off + micro_batch ≥ 2 会 OOM；gc=on 时 micro_batch 能到 4 或 8
P4  fwd_only 不受 GC 影响（no_grad 下 GC 不生效），只受 micro_batch 影响
```

---

## 2 · 方法：把训练侧**隔离**出来

生产里一步被三件事混在一起，且必须 4 卡 + Ray + vLLM，每档 15 分钟且**不能并行**。
⇒ 本探针不起 Ray、不起 vLLM，只加载模型 + LoRA，喂一批与生产**形状相同**的假数据：

```
每卡 16 条序列（= 生产 48 ÷ 3 卡）· 每条 4196 题面 + 654 回答（= 实测的 prompt/response 均值）
量：fwd+bwd 秒数（update_actor 的代理）· fwd_only 秒数（old_log_prob/ref 的代理）· 显存峰值
```

⇒ **四个臂同时跑在四张卡上**。并行的合法性依据是 A7：4 卡满载时单卡算力只掉 **2.0%**，
且四臂条件相同。

### 2.1 ★ 保真度自检（这份探针的判据，先于任何相对结论）

| | 本探针（gc=on, mb=1） | 生产实测 | 差 |
|---|---|---|---|
| fwd+bwd / `update_actor` | **20.44 s** | 17.56 s | +16% |
| fwd_only / `old_log_prob` | **6.16 s** | 5.45 s | +13% |
| 显存峰值 | **12.74 GB** | 15.55 GB | −18% |

⇒ ✅ **落在事先写死的 ±30% 判据带内 ⇒ 探针保真，相对结论可用。**
偏高 15% 的方向也说得通：没有 rmpad 打包、物化了完整 logits、四卡同时在烧。

⚠️ **已知的三个保真度缺口**（读数时必须记得）：
① 没有 FSDP/DDP 包装；② 用等长序列代替 rmpad 打包；
③ **每次前向物化完整 logits**（`4850 × 151936` ≈ 1.47 GB），而生产 `use_fused_kernels=True` 不物化
⇒ **本探针的显存偏高**，生产的真实余量比这里量到的好一些。

### 2.2 ⚠️ 写探针时当场撞到一次静默降级（值得单列）

第一版冒烟打出：

```
UserWarning: None of the inputs have requires_grad=True. Gradients will be None
```

⇒ 梯度检查点在默认的 **reentrant** 模式下，若输入不 `requires_grad`，**整段反向会被跳过** ——
那样 `gc=on` 这一臂量的是一个**空操作**，而只有一行警告。

⇒ 修法：非 reentrant + `enable_input_require_grads()`，并把假设写成**断言**：

```
[判据] 梯度检查 ✓ 可训练张量 504 个，非零梯度 252 个
```

★ 252 是对的：LoRA 的 B 初始化为 0 ⇒ 第一步 A 的梯度必然为 0，只有 B 有。
★ **504 这个数正好和 E22 量到的 adapter 张量数一致**——白捡的交叉验证。

---

## 3 · 数据

```
gc=on  定长   mb=1   fwd+bwd 20.437 s   fwd_only 6.158 s   12.74 GB
              mb=2   fwd+bwd 20.636 s   fwd_only 6.208 s   16.92 GB    ← 慢 1.0%
              mb=4   OOM
gc=on  变长   mb=1   fwd+bwd 20.735 s   fwd_only 6.261 s   13.25 GB
              mb=2   fwd+bwd 22.035 s   fwd_only 6.588 s   17.92 GB    ← 慢 6.3%
              mb=4   OOM
gc=off        mb=1   **OOM**（定长、变长两个臂都是）
```

---

## 4 · 结论

### 4.1 🔴 拉高 `micro_batch` 是**负收益**，不是「收益小」

```
定长  1 → 2   20.437 → 20.636 s   **慢 1.0%**，多花 4.2 GB
变长  1 → 2   20.735 → 22.035 s   **慢 6.3%**，多花 4.7 GB
```

**为什么**：喂饱 GPU 的是 **token 数，不是序列条数**。一条序列已经 **4850 token**，
一次前向就是 `[4850 × 2560]` 量级的 GEMM——早就吃饱了。
⇒ ★ **`micro_batch = 1` 在 LLM 训练里不等于「batch 很小」，因为序列本身就是一个大批次。**

变长臂更慢的原因也是干净的：`mb=2` 要把两条不等长的序列 pad 到 `max(lens)`，**产生 padding 浪费**；
而 `mb=1` 没有 padding。⇒ **在变长负载上，`micro_batch=1` 等价于完美打包，本来就是最优的。**

⇒ 这同时解释了 **B20**（FA2 下复测 dynamic_bsz 只有 +4~5%）：
**打包的收益在我们这个 token 长度下本来就接近零。**

### 4.2 🔴 关掉 `gradient_checkpointing`：显存不够，`micro_batch=1` 就 OOM

⇒ 「峰值 15.55 / 32 GB ⇒ 还有 16 GB 可用」这个读法是错的：
**那 15.55 GB 正是梯度检查点省出来的结果。** 关掉它要存 36 层 × 4850 token 的全部激活值。

### 4.3 ⇒ 「trainer 没喂饱」这个假设被证伪

trainer 是**真的在算**。它慢是因为**活多**（要算的 token 是 rollout 的 ~11.6 倍），
不是因为算得不够满。E01 §4.6 那 22–25% 的空档也**不是喂不饱造成的**——
否则加大 `micro_batch` 会把它填上，实测没有。

⇒ **省时间的唯一方向是「让它少算」：**

| 方向 | 量级 | 状态 |
|---|---|---|
| **prefix grouper**（8 条样本共享的题面只算一次） | **上界 4.1×** | 🔴 见 §5，不是即插即用 |
| 砍 `ref` 那遍前向 | 12.7%（E17 实测） | 🟡 吞吐已测，**精度侧从没测过** |
| `ref` 走 FP8 | ~7.8% | 🟡 数值对拍已过，接线未做 |
| ~~拉高 micro_batch~~ | ⛔ 负收益 | **本报告证伪** |
| ~~关梯度检查点~~ | ⛔ 显存不够 | **本报告证伪** |

---

## 5 · ⛔⛔ prefix grouper 在 verl 0.8.0 里**根本没接上**（2026-08-19 查实）

★ **这是「机制在但没接上」的又一例，而且形状和 E22 一模一样**：
两端的能力都在，断的是**中间没有传参的那一栏**。

装上 `prefix-grouper==0.0.1.post1` 之后逐处核对（`grep` 全包）：

```
apply_monkey_patch() 在整个 verl 里只有 **1 个调用点**
    verl/workers/engine/fsdp/transformer_impl.py:292
    实参：model / use_remove_padding / ulysses_sp_size / use_fused_kernels / fused_kernels_backend
    ⇒ 🔴 **没有 use_prefix_grouper** ⇒ 默认 False ⇒ `apply_prefix_grouper_patch()` **永不执行**

forward_micro_batch_with_prefix_grouper()  调用点数量 = **0**
    ⇒ 🔴 那条共享前缀的前向路径**没有任何人调用**

use_prefix_grouper=True 实际生效的地方只剩：
    ray_trainer.py:1163 / 1204  —— **只是让同一个 GRPO 组的样本别被拆到不同卡上**
```

⇒ **打开这个开关既不会加速，也不会出错 —— 它只改变批次划分。**
⇒ ⚠️ 如果只测「打开前后快了多少」，会得到「没有收益」这个**看起来干净、实际是空的**结论。
   判据必须先验「这条路径真的被走到了」（例如在 `pg_forward` 里打一行），**再**谈快慢。

### 5.1 ⇒ 要用上它，我们要接三处（前两处是 verl 欠的线，第三处是多轮特有的）

```
① transformer_impl.py:292      把 use_prefix_grouper 传进 apply_monkey_patch（否则注意力没被 patch）
② actor 的前向路径             改调 forward_micro_batch_with_prefix_grouper
③ prefix_grouper_utils.py:56   **mask 语义修正**（下节）—— 只有多轮场景需要
```

### 5.2 🔴 第三处：`suffix_mask` 的语义被用错了（这条对上游也成立）

包的源码里语义没有歧义：

```python
prefix_grouper/__init__.py:77   suffix_lens = suffix_mask.sum(dim=1)          # 每条响应有多长
prefix_grouper/__init__.py:196  suffix_mask.nonzero(as_tuple=False)           # 哪些位置进入拼接
⇒ suffix_mask = 「这个 token **存不存在**」
```

而 verl 传的是 `response_mask`：

```python
verl/trainer/ppo/prefix_grouper_utils.py:79
    PrefixGrouper.from_ungrouped_masks(prefix_mask=prefix_ids.ne(pad_token_id),
                                       suffix_mask=response_mask, ...)
```

**而在多轮工具场景下，`response_mask` 是「梯度掩码」，工具返回的 token 是 0** ——
⚠️ **verl 自己的 agent loop 也是这个语义**，不是我们特有的：

```python
verl/experimental/agent_loop/tool_agent_loop.py:262   response_mask += [1] * len(response_ids)   # 模型生成
verl/experimental/agent_loop/tool_agent_loop.py:400   response_mask += [0] * len(response_ids)   # ← 工具返回
```

⇒ 🔴 **一旦 ①② 接上，工具 observation 的 token 会被从模型输入里静默删掉。**
⇒ 正确的修法需要**两个掩码**（这也是为什么它不是"换个参数"）：

```
打包（进模型的输入）   用「存在掩码」= attention_mask[:, prompt_len:]   ← 含工具返回
算损失（哪些给梯度）   用「梯度掩码」= response_mask                    ← 不含工具返回
```
而 `pg_forward` 目前把 `split_output` 回来的**同一个掩码**当成了两者。

⇒ ★ **立项判据**：开/关两条路的 `log_probs` **逐位相同**（同一批数据、同一权重）。
不是"快了多少"。

---

## 5.9 ⛔ 原始记录（2026-08-19 上午的第一版判断，保留以便回溯）

verl **已经内置** `actor_rollout_ref.actor.use_prefix_grouper`（默认 `false`），
接在 `fully_async_trainer` / `one_step_off` / `ray_trainer` 三处，并配了按 GRPO 组对齐的批次划分。
**但我们打不开，而且直接打开是错的：**

```
① 依赖的 pip 包 `prefix_grouper` **没装**（ModuleNotFoundError）
② verl/trainer/ppo/prefix_grouper_utils.py:79
      PrefixGrouper.from_ungrouped_masks(prefix_mask=prefix_ids.ne(pad_token_id),
                                         suffix_mask=response_mask, ...)
   ⇒ 它把 `response_mask` 当成「这个 token 存不存在」
   ⇒ 而我们的 `response_mask` 是**梯度掩码**：1=模型生成，**0=工具 observation**
   ⇒ 🔴 直接开会把**工具返回从模型输入里删掉**。跑得起来、出得了数、不报错。
```

⇒ 单轮任务上 verl 那个等式成立，**多轮 agent 上不成立**。
⇒ 立项时的判据必须是 **「开/关两条路的 logprob 逐位相同」**，不是「变快了多少」。

---

## 6 · ⛔ 推翻了什么（四条预测错三条）

| | 预测 | 实测 | 推翻后 |
|---|---|---|---|
| **P1** | 关 GC 快 20–30%、显存 25–30 GB | **mb=1 就 OOM** | 显存余量是**紧的**，不是宽裕的 |
| **P2** | micro_batch 1→2 提升 < 10% | **慢 1.0% / 6.3%** | 方向就错了：不是收益小，是**负收益** |
| **P3** | gc=off + mb≥2 才 OOM | gc=off 在 **mb=1** 就爆 | — |
| **P4** | fwd_only 不受 GC、受 mb 影响 | 不受 GC ✅；**也几乎不受 mb**（+0.8%） | 对了一半 |

### ★ 教训：**拿 A 条件下量到的数，去推 B 条件下的余量**

「峰值 15.55 GB ⇒ 还剩 16 GB」——**那个 15.55 正是开着 GC 的结果**。
这是 [[feedback-measure-dont-infer]] 的第三次兑现，形状和前两次一样：
**一个指标换个前提就不是同一件事。**

### ★ 教训二：**「batch=1」这个字面值极具误导性**

它触发的直觉是「太小了，肯定没喂饱」——而真正的喂饱单位是 token。
⇒ 一般化：**看一个参数大不大，要先问它的单位是什么。**
