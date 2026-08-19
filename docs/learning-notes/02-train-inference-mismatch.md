# 02 · 训推不一致与 Rollout Correction（TIS 专题）

> 调查日期：2026-07-27
> 代码基准：`reference/industrial_posttrain_training_release/verl/upstream/`（verl `0.8.0.dev` 快照，下文简称 `UP/`）
> 老师的 GRPO 默认**开启**这条链路：`rollout_is=sequence, threshold=2.0, batch_normalize=false, bypass_mode=false`

---

## 0. 问题：为什么需要它

RL post-training 里有**三个策略**，很容易被误当成一个：

| 记号 | 是谁 | 在哪算 |
|---|---|---|
| π_rollout | 采样时真正生成 token 的策略 | vLLM，BF16，PagedAttention kernel |
| π_old | 训练侧对同一批数据重算一遍 logprob 的策略 | FSDP，FP32/BF16，PyTorch/FlashAttention kernel |
| π_θ | 当前正在被优化的策略（mini-batch 更新中不断变） | FSDP |

**理想情况 π_rollout = π_old**（同一份权重），但实际不等：数值精度不同、算子实现不同、kernel 融合方式不同。这就是 **training-inference mismatch**。它带来的偏差在两个场景被放大：

1. **实现失配**：vLLM BF16 vs FSDP FP32，同一权重同一 token 算出的 logprob 有差；
2. **权重陈旧（staleness）**：异步/off-policy 训练中，rollout 用的是 k 步之前的权重——**这正是 Syncopate Phase 2 fully-async 的核心矛盾**。

不做修正，PPO 的 ratio `π_θ/π_old` 就默认了"数据是 π_old 采的"这个前提，而实际数据是 π_rollout 采的 → 梯度估计有偏 → 训练崩塌。

`UP/verl/trainer/ppo/rollout_corr_helper.py` 模块 docstring 直接给了两个出处（不是我猜的）：

- "When Speed Kills Stability: Demystifying RL Collapse from the Training-Inference Mismatch" — https://richardli.xyz/rl-collapse
- Off-policy RL（IS 的理论基础）— https://fengyao.notion.site/off-policy-rl
- 代码归属：函数 docstring 注明 `The implementation is copied from szrlee <szrlee@gmail.com>`（`rollout_corr_helper.py:1032, 1120`）
- 另有两处指向 `docs/algo/rollout_corr.md` 和 `docs/algo/rollout_corr_math.md`，但**训练包没有随包 docs/**，正式版本需去 verl 官方仓库查

---

## 1. 配置项全集

配置 schema：`UP/verl/trainer/config/algorithm/rollout_correction.yaml`（默认全关）

| 配置项 | 取值 | 语义 |
|---|---|---|
| `rollout_is` | `null` / `"token"` / `"sequence"` | IS 权重聚合粒度。null = 关闭 |
| `rollout_is_threshold` | float 或 `"lower_upper"` 字符串 | float → **TIS**（只有上界）；`"0.5_2.0"` 这种 → **IcePop**（双边，越界置 0） |
| `rollout_is_batch_normalize` | bool | 是否把权重归一化到 batch 均值 1.0 |
| `rollout_rs` | `null` / `token_k{1,2,3}` / `seq_{sum,mean,max}_k{1,2,3}`，逗号可串联 | **拒绝采样**（RS）：硬 trust region，超阈值的 token/序列直接从 `response_mask` 里剔除 |
| `rollout_rs_threshold` | 字符串/数值，每个 option 一个 | RS 阈值；k1 类需 `lower_upper` 双边 |
| `bypass_mode` | bool | false = **Decoupled 三策略**；true = **Bypass 两策略** |
| `loss_type` | `ppo_clip` / `reinforce` | 仅 bypass_mode 下生效 |

**关键区分**：IS（软加权，降方差）和 RS（硬剔除，去离群）是**两条独立的路**，可单开可同开。`compute_rollout_correction_and_add_to_batch` 的注释写得很明确：`response_mask` **永远**被 RS 更新，`rollout_is_weights` **只在 `rollout_is` 非 null 时**加进 batch（`rollout_corr_helper.py:1015-1021`）。这个设计是为了"先只看指标、不动训练"的渐进上线。

---

## 2. 数学公式（从代码反推，非凭印象）

### 2.1 log ratio

```
log_ratio_t = log π_old(y_t) − log π_rollout(y_t)
```
`rollout_corr_helper.py:842`，即 `old_log_prob - rollout_log_prob`。注意**分子是训练侧、分母是 rollout 侧**，方向别记反。

### 2.2 权重聚合（两种粒度）

**token 级**（`:570-574`）：
```
w_t = exp( clamp(log_ratio_t, −20, +20) )
```

**sequence 级**（`:576-584`）——老师用的就是这个：
```
S = Σ_{t: response_mask_t=1} log_ratio_t          # masked_sum，先在 log 空间求和
w_seq = exp( clamp(S, −20, +20) )
w_t = w_seq  ∀t                                    # broadcast 回该序列每个 token
```

回答"是 token logp 求和后取 exp，还是 token 权重连乘"——**代码是前者**（`masked_sum` 后 `exp`），数学上等价于后者（∏exp = exp∑），但在 log 空间做避免了溢出。这就是 docstring 里 "Log-space computations to avoid overflow" 的含义。`SAFETY_BOUND=20.0`，exp(±20) ≈ 4.85e8 / 2e-9。

然后 padding 位清零：`w = w * response_mask`（`:590`）。

### 2.3 截断

**TIS（单上界 C，老师用的路径，C=2.0）**（`:593-594`）：
```
w̄_t = min(w_t, C)
```
**只有上界，没有下界**（下界只出现在 metrics 里，用 1/C 作诊断参考，见 `:693-694`）。这是"Truncated IS"的标准形式：只压制爆炸的权重，不管趋零的权重——因为大权重才是方差爆炸的元凶。

**IcePop（`"lower_upper"` 字符串）**（`:596-602`）：
```
w̄_t = w_t   if  lower ≤ w_t ≤ upper
      0     otherwise
```
双边，越界直接**置零**（不是 clamp），等价于把该 token 逐出本次更新。

### 2.4 detach 与可选归一化

```python
rollout_is_weights = rollout_is_weights.detach()   # :623
```
注释写明理由："IS weights change the measure, not the objective"——IS 权重换的是**测度**（在哪个分布上求期望），不是目标函数本身，所以绝不能对它求梯度。这是最容易写错的一行。

可选 batch normalize（`:626-653`）：把权重除以 batch 均值使其均值为 1，减少梯度尺度漂移。老师关掉了。

### 2.5 权重乘在哪里 —— **乘在 loss 上，不是 advantage 上**

`UP/verl/trainer/ppo/core_algos.py:1279 compute_policy_loss_vanilla`：

```python
pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)   # :1354  PPO 双 clip 完成
if rollout_is_weights is not None:
    pg_losses = pg_losses * rollout_is_weights                              # :1357-1358  ★
pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, ...)        # :1360
```

**顺序至关重要：先 PPO clip，再乘 IS 权重，最后聚合。** 同样的 pattern 出现在 `compute_policy_loss_dppo_tv:1435`、`dppo_kl:1519`、`gspo:1593`。

另有一条**不同**的用法：`compute_optimal_token_baseline_advantage`（`:876, 995`）里 `w_per_timestep *= rollout_is_weights**2`——那是给 optimal baseline 估计器做 MSE 最小化的**平方**权重，和 GRPO 默认路径无关，别混淆。

完整的一步梯度（GRPO + TIS，advantage>0 且未触发 clip 时）：
```
L = − (1/|T|) Σ_t  w̄_t · ρ_t · Â_t ,     ρ_t = π_θ(y_t)/π_old(y_t)，  w̄_t = min(π_old/π_rollout, C)
```

---

## 3. 和 PPO clip / GRPO advantage 怎么共存（会不会"双重重要性权重"？）

**不会——这是刻意的因式分解，不是重复计数。** 三策略各司其职：

```
        π_θ            π_θ        π_old
      ───────   =   ───────  ×  ─────────
     π_rollout       π_old      π_rollout
     （真正需要的）   PPO ratio    TIS 权重
                    带梯度、被clip  detach、被truncate
```

- **PPO ratio `π_θ/π_old`**：带梯度，用 `clip_ratio` 做**软**信任域，控制"这一步更新别走太远"（on-policy 的近端约束）。
- **TIS 权重 `π_old/π_rollout`**：`detach()`，用 `threshold` 做**截断**，修正"数据其实不是 π_old 采的"这个测度偏差。

两者相乘正好还原成真正需要的 `π_θ/π_rollout`。之所以拆开，是因为两者需要**不同的处理方式**：一个要梯度+clip，一个要 detach+truncate。这就是配置里叫 **Decoupled（3 policies）** 的原因。

**Bypass mode（`bypass_mode=true`）是另一条路**：直接令 `old_log_probs := rollout_log_probs`（`rollout_corr_helper.py:1131`），省掉训练侧重算 old_log_prob 的那次 forward（省算力！）。此时只有 2 个策略，ratio 本身就是 `π_θ/π_rollout`，**修正已经内含在 ratio 里**，所以 trainer 会**跳过** IS 权重计算——条件见 `ray_trainer.py:1610-1613` 的 `and not bypass_recomputing_logprobs`。loss 换成 `compute_policy_loss_bypass_mode`（`core_algos.py:2352`）。

> 对 Syncopate 的意义：bypass_mode 用一次 forward 的代价换取修正，在**分离式异步架构**里格外划算（trainer 侧算力紧张）。Phase 2 对比 E2/E3 时这是个值得单独拉一格的旋钮。

GRPO advantage 完全不受影响：`compute_grpo_outcome_advantage`（`core_algos.py:268`）只吃 `token_level_rewards` 和分组 index，做组内均值/标准差归一化，和 IS 权重是**正交**的两件事——一个管"这条轨迹好不好"，一个管"这条轨迹该按多大权重计入期望"。

---

## 4. vLLM logprob 的完整数据通路

老师脚本用 `actor_rollout_ref.rollout.calculate_log_probs=True` 打开源头。之后：

| # | 位置 | 发生了什么 |
|---|---|---|
| 1 | vLLM 生成 → `TokenOutput.log_probs` | rollout 引擎返回每个生成 token 的 logprob |
| 2 | `train/verl_agent_loop_adapter.py:105,151` | adapter 把 `output.log_probs` 累进 `response_logprobs`；**工具/反馈 token 补 0.0**（`:363-364`，与 mask=0 对齐） |
| 3 | `train/verl_agent_loop_adapter.py:307-311` | 写进 `AgentLoopOutput.response_logprobs` |
| 4 | `UP/verl/experimental/agent_loop/agent_loop.py:758-761` | `_agent_loop_postprocess` 右 pad 到 `response_length` |
| 5 | `UP/.../agent_loop.py:1039-1040` | batch 拼接 → **改名为 `rollout_log_probs`** 进 DataProto（★ 字段改名点，grep 时容易断线） |
| 6 | `UP/verl/trainer/ppo/ray_trainer.py:1541` | Decoupled 模式下训练侧重算 `old_log_probs` 并 union 进 batch |
| 7 | `UP/verl/trainer/ppo/ray_trainer.py:1610-1616` | `compute_rollout_correction_and_add_to_batch(batch, cfg)` → 读 `old_log_probs`/`rollout_log_probs`/`response_mask`，写回 `rollout_is_weights` 和修正后的 `response_mask` |
| 8 | `UP/verl/workers/utils/losses.py:88-112` | actor 从 data 取 `rollout_is_weights` 传给 policy loss fn |
| 9 | `UP/verl/trainer/ppo/core_algos.py:1357-1358` | `pg_losses *= rollout_is_weights` |

**注意第 7 步的位置**：它在 `marked_timer("adv")` 块**内部**、`compute_advantage()` **之前**（`ray_trainer.py:1616` vs `:1624`）。所以 RS 剔除的 token 会影响 advantage 的 mask，但 IS 权重不参与 advantage 计算。

---

## 5. 附带的诊断指标（我们 Phase 1/2 要盯的）

`compute_offpolicy_metrics`（`:897`）无条件计算，全部带 `rollout_corr/` 前缀，直接进 W&B：

| 指标 | 公式（代码位置） | 读法 |
|---|---|---|
| `kl` | `E[log π_rollout − log π_old]`（`:952`） | 正值 = rollout 比训练侧更自信 |
| `k3_kl` | `E[exp(log_ratio) − log_ratio − 1]`（`:957-959`） | 小 KL 时更稳的 K3 估计量 |
| `chi2_token` | `E[ρ²] − 1`（`:993`） | **IS 权重方差的直接度量** |
| `chi2_seq` | `E[(∏ρ_t)²] − 1`（`:999-1000`） | 序列级方差；长序列上会爆炸得很快 |
| `log_ppl_diff` | 训练 vs rollout 的 log 困惑度差（`:972`） | 最直观的"两个引擎有多不一样" |
| `rollout_is_eff_sample_size` | `1/E[(w/w̄)²]`（`:752-757`） | **ESS**：有效样本量。掉得厉害说明少数样本主导了梯度 |
| `rollout_is_ratio_fraction_high` | 超上界的比例 | 截断触发得多不多 |

**Syncopate 的用法**：Phase 2 对比同步 vs fully-async 时，`chi2_seq` 和 ESS 就是 **staleness 的定量代理**——同步模式下它们应该只反映 BF16/FP32 实现差，异步模式下会额外叠加权重陈旧。这两条曲线的差值，就是"异步引入了多少 off-policy"的直接证据。比单看 reward 曲线有说服力得多。

---

## 6. 仍不确定 / 待验证

1. ~~`docs/algo/rollout_corr_math.md` 未随包~~ → **已补**，见 §7 对照笔记。
2. `rollout_is=sequence` 在 **multi-turn** 场景的语义：`masked_sum` 会把一条轨迹里**所有轮次**的模型 token 的 log_ratio 全加起来。轮次越多、序列越长，S 的方差越大，w 越容易撞上截断上界 C=2.0。**已做纸面量纲分析，见 §8**——结论是 ESS 随 T 指数衰减，这是 Syncopate 的核心研究问题（[[../syncopate/23-research-question]]）。
3. RS（`rollout_rs`）老师完全没开。它和 partial rollout 在"丢弃 off-policy 样本"这件事上是竞争关系，Phase 2 可以对比。
4. bypass_mode 省掉的那次 forward 在 8B/64 卡下占比多少，未测。

---

## 7. 官方文档对照笔记（2026-07-28 补）

来源：`docs/algo/rollout_corr.md` 和 `rollout_corr_math.md`（verl **main 分支**，训练包未随包）。
**代码基准是 0.8.0.dev 快照，冲突一律以代码为准**；差异单独记为版本演进线索。

### 7.1 一致的部分（代码 ≡ 文档）

| 项 | 文档 | 代码 | |
|---|---|---|---|
| sequence 级公式 | `w_seq = min(∏ρ_t, C) = min(exp(Σlog ρ_t), C)` | `masked_sum` → `exp` → `clamp(max=C)`（`:578-594`） | ✅ |
| token 级公式 | `w_t = min(ρ_t, C)` | `exp(clamp(δ_t))` → `clamp(max=C)`（`:573-594`） | ✅ |
| IcePop | 越界**置零**而非截断，用于"尾部是垃圾/对抗样本"时 | `torch.where(kept_mask, w, 0)`（`:596-602`） | ✅ |
| RS vs IS 相互独立 | "operate independently on separate structures"，可单开可同开 | 两个独立分支（`:845-871`），RS 改 `response_mask`，IS 加 `rollout_is_weights` | ✅ |
| bypass + ppo_clip 不另算 IS | "no separate IS weights are typically computed"，否则双重计数 | `ray_trainer.py:1610-1613` 的 `and not bypass_recomputing_logprobs` | ✅ |
| Decoupled 模式权重乘在 clip 后 | `L = -E[w_t · min(r_t A, clip(r_t) A)]` | `core_algos.py:1357-58` 在 `torch.where(...)` 之后 | ✅ |

### 7.2 ★ 差异与冲突（逐条）

**① `geometric` 聚合方式：文档提到，代码里不存在**

`rollout_corr.md` 的运维建议写着"若 ESS < 0.3，consider switching to **geometric** aggregation"。但快照代码的白名单是死的：

```python
valid_is_levels = {"token", "sequence"}          # :563
if rollout_is not in valid_is_levels:
    raise ValueError(...)
```

**这是最明确的版本演进线索**：main 分支已经加了第三种聚合（几何平均，大概是 `exp(mean(δ_t))` 而非 `exp(sum(δ_t))`——那样 T 就不会进指数，正好治 §8 的病）。**0.8.0.dev 上照文档配 `geometric` 会直接 ValueError。**
→ 对 Syncopate 的意义：如果 §8 的预测成立、sequence 级在长序列上崩掉，**升级到 main 拿 geometric 是一条现成的退路**，而不必自己写。Phase 2 值得先确认 main 的实现。

**② detach 的必要性：文档说 Decoupled 模式不需要，代码无条件 detach**

文档 §3.2.1 说 *"Since π_old is frozen, w_t requires no stopgrad"*；代码 `:623` 无条件 `.detach()`。
**不算真冲突**（decoupled 下 w 本来就在 no_grad 下算出、没有计算图，detach 是 no-op），但代码更防御性——bypass 模式下 `old_log_probs := rollout_log_probs` 时这行才真正起作用。**保留它是对的。**

**③ 文档的损失公式是 REINFORCE 形态，代码默认是 PPO-clip 形态**

文档 §3.2.2 反复用 `loss = -advantages * log_prob * rollout_is_weights.detach()` 讲解，容易让人以为 IS 权重乘在 REINFORCE 目标上。实际那是 **bypass_mode + `loss_type="reinforce"`** 的专用路径（`core_algos.py:2352 compute_policy_loss_bypass_mode`）。老师走的 Decoupled 路径乘在 **PPO 双 clip 之后的 `pg_losses`** 上（`:1357-58`）。
**读文档时要先认准自己在哪条分支**，否则会以为代码写错了。

**④ 阈值 2.0 在推荐区间里的位置：文档暗示它偏紧**

文档给的经验区间：**token 级 1.5–5.0，sequence 级 2.0–10.0**。
verl 的默认值 `2.0` 对两者都用同一个数，**但它是 sequence 级推荐区间的最下限**。老师用 `sequence + 2.0` = 推荐范围内**最激进的截断**。
→ 直接可用的动作：Phase 1 先看 `rollout_is_ratio_fraction_high`，若显著 >0（比如 >5%），说明 2.0 对这个任务偏紧，应往 3–5 调。

### 7.3 (b) 文档推荐 sequence 还是 token？—— 推荐 sequence，理由是**无偏**

文档的表述很明确：

- **sequence 级**："unbiased, suitable for most cases"、"Recommended for most general cases"，缺点是"more sensitive to outliers"；
- **token 级**："lower variance"，但引入 **`O(T²·Δ_max)` 的偏差**（T = 序列长度，Δ_max = 单 token 最大策略散度），"Use when data is clean and mismatch is moderate"、"works well when rollout policy stays within the trust region"。

配套的 blog《When Speed Kills Stability》立场更强：明确指出 **token 级 IS 在长序列上会引入严重偏差**，并提出 sequence 级 **MIS（masked IS，超阈值直接丢弃整条序列）**，主张 sequence 级方法"should be considered a default for any serious LLM-RL training stack"。

> **注意这里有个漂亮的对偶，是我们研究问题的起点**：
> - **token 级：偏差随 T 增长**（`O(T²Δ)`），方差不随 T 增长；
> - **sequence 级：无偏，但方差随 T 指数增长**（§8 推导）。
>
> 文档和 blog 只强调了前者（所以推荐 sequence），**没有量化后者**。而 agentic 长轨迹恰好落在后者的危险区。**这就是 Syncopate 可以补上的空白。**

### 7.4 (c) C=2.0 的来源：**纯经验值，两份文档都没有推导**

我逐节读了 `rollout_corr_math.md`，**没有任何从 KL 界、方差界或 ESS 目标推导出 2.0 的段落**。文档只给了：

- 经验区间（token 1.5–5.0 / sequence 2.0–10.0）；
- 运维判据：`rollout_is_mean` 应 ≈1.0、`rollout_is_eff_sample_size` 应 **>0.3**、`chi2_*` >1.0 视为严重偏移、mean IS weight <0.5 或 >2.0 时告警。

**结论：C=2.0 是经验默认值，不是推导结果。** 但 **ESS>0.3 这个判据是可以反推出阈值约束的**——§8 就是这么做的。这也意味着：**我们完全可以自己给 C 一个基于 T 和 staleness 的推导，那本身就是可发表的贡献。**

### 7.5 文档给的运维阈值（代码里没有，值得抄进监控）

| 指标 | 健康值 | 告警 |
|---|---|---|
| `rollout_corr/rollout_is_mean` | ≈ 1.0 | <0.5 或 >2.0 |
| `rollout_corr/rollout_is_eff_sample_size` | **> 0.3** | < 0.3 |
| `rollout_corr/chi2_token` / `chi2_seq` | < 1.0 | > 1.0 |
| `rollout_corr/rollout_rs_masked_fraction` | 越低越好 | — |

---

## 8. sequence-level TIS 的长序列量纲分析（纸面推导）

### 8.1 本任务的 T 有多大

`response_mask=1` 的 token 数（即真正参与 IS 的模型生成 token，**不含**工具 observation）：

从 121 条 SFT gold 轨迹实测（含 `tool_calls` 的 JSON，那是模型真要生成的）：

| | assistant 轮数 | 模型生成字符 | **估计 token 数 T** |
|---|---|---|---|
| P50 | 5 | 614 | **≈ 280** |
| P90 | 8 | 1141 | ≈ 510 |
| max | 10 | 1549 | ≈ 700 |

配置上限：`max_assistant_turns=8`、`max_response_length=4096`（但 4096 是**含工具 observation** 的预算，模型自己生成的部分远小于它）。

**结论：本任务 T ~ 10²–10³ 量级，典型值几百。** 注意 RL rollout 会比 gold 轨迹更长更乱（parse_error 重试、无效探索），取 **T ≈ 300–800** 作为工作估计。作为对照，Phase 3 的 ALFWorld（50 步 horizon）会把 T 推到 10³–10⁴。

### 8.2 核心推导：ESS 随 T 指数衰减

记单 token log ratio 为 δ_t = log π_old(y_t) − log π_rollout(y_t)，
S = Σ_{t=1..T} δ_t，未截断权重 w = exp(S)。

**假设**（明确标注，后面要验证）：δ_t 近似 i.i.d.，均值 μ、方差 σ²。则

```
S ~ N(Tμ, Tσ²)          w = exp(S) 服从对数正态
```

有效样本量比例（ESS/N = 1/E[w̃²]，w̃ = w/E[w]）对对数正态有闭式解：

```
E[w²]/E[w]² = exp(Tσ²)
⟹  ESS/N ≈ exp(−T σ²)          ← ★ 核心结论
```

**对比 token 级**：每个 token 独立持有自己的权重 exp(δ_t)，同样推导给出 `ESS/N ≈ exp(−σ²)`——**与 T 无关**。

> **一句话**：sequence 级的有效样本量随序列长度**指数衰减**，token 级不衰减。这就是"无偏"要付的代价，而文档只字未提。

把文档的运维红线 **ESS/N > 0.3** 代入：

```
exp(−Tσ²) > 0.3   ⟹   T·σ² < 1.20   ⟹   σ < √(1.20 / T)
```

| T | σ 的上限 |
|---|---|
| 300 | 0.063 |
| 800 | 0.039 |
| 4096 | 0.017 |
| 10000 | 0.011 |

**解读**：本任务 T≈300–800 时，只要每 token 的 log-prob 偏差标准差 σ < 0.04–0.06 就安全。BF16-vs-FP32 的纯实现失配通常在 1e-3~1e-2 量级 → **同步 colocate 下 sequence 级完全够用**（这解释了为什么老师的配置能稳定跑）。

**但 staleness 会把 σ 顶上去**。异步训练下 δ_t 里除了实现失配还叠加"策略漂移了 k 个版本"的成分。定性地 σ(k) 随 k 单调增（若漂移近似随机游走则 σ ∝ √k）。于是安全条件变成 **T·σ²(k) < 1.2** —— **T 和 k 是乘性耦合的**，这正是研究问题的形式。

粗略代入感受一下（σ=0.05 时）：

| T | T·σ² | ESS/N |
|---|---|---|
| 300 | 0.75 | 0.47 ✅ |
| 800 | 2.0 | 0.135 ⚠️ |
| 4096 | 10.2 | 4×10⁻⁵ ❌ 完全崩 |

### 8.3 截断如何救场（以及救不了什么）

C=2.0 的截断把权重压到 ≤2，**方差被硬性限住了，所以不会数值爆炸**。但代价是**偏差**：当 `P(exp(S) > C)` 很高时，大量序列被压到同一个值 2.0，权重之间失去区分度——**退化成"给所有 off-policy 序列一个常数权重"，IS 修正实际失效**。

超阈值比例（对数正态）：
```
P(w > C) = 1 − Φ( (ln C − Tμ) / (σ√T) )
```
`ln 2.0 = 0.693`。若 μ≈0、σ√T ≫ 0.693，这个概率趋近 50%——**一半序列被截断到同一个数**。σ=0.05、T=800 时 σ√T=1.41，P ≈ 31%。

**所以真正的失效信号不是 NaN，而是三个指标同时走坏**：`rollout_is_ratio_fraction_high` 升高 + `rollout_is_eff_sample_size` 下降 + `rollout_is_mean` 偏离 1.0。**这是安静的失效，不看指标根本发现不了。**

### 8.4 (b) 代码里有没有 per-token log_ratio 的均值/标准差？—— **没有直接的，但可以反解**

我把 `compute_offpolicy_metrics`（`:897-1003`）和 `compute_is_metrics`（`:658-776`）逐个指标核对过：

- `rollout_is_std`（`:747`）是**截断后权重**的标准差，不是 δ 的；
- sequence 模式下 `rollout_is_max/min`（`:706-707`）是 **exp(S)** 的极值，也不是逐 token 的；
- **没有任何一个指标直接是 `std(δ_t)`。**

**但可以从两个现有指标反解出来**（这就是我们要的抓手）：

```
metrics["kl"]    = E[log π_rollout − log π_old] = E[−δ] = −μ          (:952)
metrics["k3_kl"] = E[exp(δ) − δ − 1] ≈ E[δ²]/2 = (σ² + μ²)/2          (:957-959)

⟹   μ ≈ −kl                                    
⟹   σ² ≈ 2·k3_kl − kl²          ← ★ 从 W&B 现有指标直接算 per-token 方差
```

（Taylor 展开 `exp(δ) ≈ 1 + δ + δ²/2`，在 |δ|≲0.1 时误差可忽略——正是我们关心的区间。）

**验证抓手（Phase 1 就能做，零成本）**：
1. 从 W&B 取 `rollout_corr/kl` 和 `rollout_corr/k3_kl` → 算 σ²；
2. 用 `T · σ²` 预测 `ESS/N = exp(−Tσ²)`；
3. 和实测的 `rollout_corr/rollout_is_eff_sample_size` 对比。
**若两者吻合 → i.i.d. 假设成立，§8.2 的模型可用；若实测 ESS 明显更低 → 说明 δ_t 正自相关**（`Var[S] = Tσ²(1 + 2Σρ_k) > Tσ²`），那本身就是个有意思的发现（多轮 agentic 的 token 失配在轮内聚集）。

另一条交叉验证：`chi2_seq = E[exp(2S)] − 1`（`:996-1001`），对数正态下 `= exp(2Tμ + 2Tσ²) − 1`。**和 ESS 是同一个量的两种写法**，可以互相印证。

### 8.5 (c) IcePop（双边置零）vs TIS（单边截断）：处理系统性偏移的差异

关键区别在**对"整体偏移"的响应**：

| | TIS `min(w, C)` | IcePop `w·1[l ≤ w ≤ u]` |
|---|---|---|
| w 偏大（系统性正偏移） | 压到 C，**样本保留**，梯度方向不变、幅度受限 | **整条置零 = 丢样本** |
| w 偏小（系统性负偏移） | **不处理**（无下界！） | 低于 l 也置零，**同样丢样本** |
| 系统性偏移变大时 | 越来越多样本被压到同一个 C → **退化成常数权重，修正失效但仍有梯度** | 越来越多样本被丢 → **有效 batch 缩水，最坏全空** |

**回答"staleness 大时哪种更稳"：TIS 更稳，但"稳"的含义不同。**

- **staleness 是系统性偏移**（整个分布平移），不是个别离群点。IcePop 的设计意图（文档明说）是对付 *"toxic tails / garbage or adversarial samples"* —— **离群点**。用它去处理系统性偏移，等于把大半个 batch 判为离群点丢掉，**batch 有效样本数直接坍缩**，训练会因梯度噪声暴涨而不稳。
- TIS 在同样情况下退化成"所有 off-policy 样本权重都是 C"——**修正失效了，但至少还在用全部样本做无修正的梯度更新**，行为可预测（等价于 vanilla policy gradient + 常数缩放）。**失效是软的、可观测的**（fraction_high 会先涨上去报警）。

> **一句话**：IcePop 治的是"少数样本坏得离谱"，TIS 治的是"所有样本都偏了一点"。**staleness 属于后者，所以 fully-async 场景应该用 TIS，不是 IcePop。**
>
> 另有一层：TIS 没有下界，意味着 **w 很小的样本会被静默地近乎忽略**（权重趋 0 但不置零）。系统性负偏移时这是隐性的样本浪费，而 `rollout_is_ratio_fraction_low` 这个指标能看见它（`:716, 729`）——**这个指标平时没人看，但在异步场景下应该重点盯**。

### 8.6 (d) `rollout_rs` 拒绝采样：判据、与 IS 的关系

**判据**（`compute_rollout_rejection_mask:195-411`）基于 **KL 散度估计量**，不是权重本身：

```python
log_ratio_safe = clamp(log_ratio, ±20)
token_k1 = -log_ratio_safe                       # :261  k1 估计量
token_k2 = 0.5 * log_ratio_safe**2               # :262  k2
token_k3 = exp(log_ratio_safe) - 1 - log_ratio_safe   # :263  k3（同 metrics 里的 k3_kl）
```

- **k1 类**（`token_k1` / `seq_*_k1`）需要**双边**阈值 `"lower_upper"`，判据 `lower_log ≤ stat ≤ upper_log`（`:314`）；
- **k2/k3 类**只需单个上界，判据 `stat ≤ upper`（`:317, 320`）——因为 k2/k3 天然非负，理想值 0；
- **聚合粒度**：`token_*`（逐 token 判）、`seq_sum_*` / `seq_mean_*` / `seq_max_*`（整条序列一起判，不合格则该序列**全部 token** 置 0，`:338`）；
- **多条件用逗号串联，逻辑 AND**（`:242, 378`）：`"token_k1,seq_mean_k3"` 表示两个都得过。

**与 IS 的关系：两条完全独立的路，可以同时开。**

`compute_rollout_correction_and_rejection_mask` 里是两个平行的 if（`:845-871`）：Step 2 算 IS 权重、Step 3 算拒绝 mask，**共用同一个 `log_ratio`，但互不读取对方结果**。作用对象也不同：

| | 作用对象 | 下游影响 |
|---|---|---|
| **IS** | 新增 `rollout_is_weights` 字段 | 只乘在 policy loss 上 |
| **RS** | **改写 `response_mask`** | 影响 loss 聚合**和 advantage 计算**（`response_mask` 在 `compute_advantage` 之前就被改了，`ray_trainer.py:1616` vs `:1624`） |

注意一个不对称：**`response_mask` 永远被更新**（`:1054` 无条件），而 IS 权重只在 `rollout_is` 非 null 时才加。文档解释了这个设计意图——**允许"先只看指标、不动训练"的渐进上线**。

还有个细节：**IS 权重是用原始 `response_mask` 算的**（`:848-853` 传的是未修改的 mask），RS 之后才改 mask。所以被拒绝的 token 仍有非零 IS 权重，只是在 loss 聚合时被 mask 掉了——数值上无害，但读代码时容易困惑。

**对 Syncopate 的意义**：RS 和 partial rollout 在"丢弃 off-policy 样本"这件事上是**竞争关系**（一个在 token/序列级丢，一个在轨迹级中断续跑），Phase 2 的 E4 可以顺带对比。
