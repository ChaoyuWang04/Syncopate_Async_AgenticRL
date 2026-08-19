# Syncopate · 23 — 核心研究问题

> 立项日期：2026-07-28
> 状态：**假设已成型，待实证**。理论推导见 [[../learning-notes/02-train-inference-mismatch]] §8。

---

## 1. 核心假设

> **序列长度 T 与 policy staleness k 共同决定 sequence-level TIS 的可用区间；超过阈值必须降级到 token-level。**

展开成可证伪的三段：

**H1（衰减律）** sequence-level 重要性采样的有效样本量随序列长度**指数衰减**：
```
ESS/N ≈ exp(−T·σ²(k))
```
其中 σ²(k) 是单 token log-ratio 的方差，随 staleness k 单调增。token-level 的 ESS 则**与 T 无关**。

**H2（耦合）** T 和 k 通过乘积 `T·σ²(k)` 耦合，存在一条**等值线**而非两条独立阈值。即：短序列可以容忍大 staleness，长序列即使小 staleness 也会崩。可用区间的边界近似为 `T·σ²(k) < 1.2`（由文档给出的运维红线 ESS/N > 0.3 反推）。

**H3（降级点）** 越过该边界后，sequence-level 会**安静地失效**——不报错、不出 NaN，而是大量序列的权重被压到同一个截断上界 C，IS 修正退化成常数缩放。此时 token-level 虽然有 `O(T²Δ)` 的偏差，但**方差可控**，是更好的选择。存在一个 T·σ² 的交叉点，两侧最优选择不同。

---

## 2. 为什么这是个真问题（而不是已知结论）

现有工作的立场是**一边倒推荐 sequence-level**：

- verl 官方文档：sequence-level "unbiased, suitable for most cases"、"Recommended for most general cases"；
- 《When Speed Kills Stability》：明确指出 **token-level IS 在长序列上引入严重偏差**，主张 sequence-level 方法"should be considered a default for any serious LLM-RL training stack"；
- 老师的工业训练包：默认 `rollout_is=sequence, threshold=2.0`。

**但这些论证都只覆盖了偏差侧，没有量化方差侧。** 存在一个漂亮的对偶：

| | 偏差 | 方差 |
|---|---|---|
| **token-level** | `O(T²·Δ_max)`，**随 T 增长** | 与 T 无关 |
| **sequence-level** | 无偏 | `ESS/N ≈ exp(−Tσ²)`，**随 T 指数衰减** |

**没有一方在所有 T 上占优。** 现有文献只看到了左下角（token 的偏差问题），推荐了 sequence；而 **agentic 长轨迹 + 异步 staleness 恰好把工作点推到了右上角**（sequence 的方差问题）。这是个真空。

而且这两个条件正是 Syncopate 项目本身的设定：多轮工具调用 → T 大；fully-async → k 大。**研究问题和项目场景天然重合，不是硬凑的。**

---

## 3. 理论预测（待验证）

推导过程见 [[../learning-notes/02-train-inference-mismatch]] §8.2。假设 δ_t i.i.d.、均值 μ 方差 σ²：

```
S = Σ_{t=1..T} δ_t  ~  N(Tμ, Tσ²)
w = exp(S)          服从对数正态
ESS/N = 1/E[w̃²]   = exp(−T σ²)
```

**安全边界**（代入 ESS/N > 0.3）：

| T | σ 上限 | 对应场景 |
|---|---|---|
| **222** | **0.074** | 本任务 **P50 实测**（gold 轨迹，5 轮）|
| **381** | **0.056** | 本任务 **P90 实测**（8 轮）|
| **534** | **0.047** | 本任务 **max 实测** |
| 4096 | 0.017 | `max_response_length` 打满（RL 探索 + thinking 可能逼近）|
| 10000 | 0.011 | ALFWorld 50 步 horizon（Phase 3） |

> T 已用真实 Qwen3 tokenizer 在 121 条 gold 轨迹上实测（见 [[../learning-notes/05-anatomy-of-a-trajectory]] §2.1），
> **只计 `response_mask=1` 的模型生成 token**——它只占 response 区的 39%，其余 61% 是工具返回。
> ⚠️ 但 gold 轨迹**不含 thinking 块**（SFT 侧 `enable_thinking=False`），而 RL rollout 侧模板默认允许模型输出真实 thinking 且计入 mask=1
> （见 00 §4.3）。**所以真实 RL 的 T 分布可能显著宽于上表**，Phase 0 必须用 `token_trace` 实测校准。

**σ 的两个来源**：
- **实现失配**（BF16 vs FP32 等）：文献量级 1e-3 ~ 1e-2，**同步 colocate 下只有这一项** → 本任务 T≈300–800 时 `Tσ² ≈ 0.003–0.08`，ESS/N > 0.92，**完全安全**。这解释了为什么老师的工业配置能稳定跑。
- **staleness**：异步下额外叠加策略漂移。若漂移近似随机游走则 `σ ∝ √k`，于是 `Tσ² ∝ T·k` —— **H2 的乘性耦合**。

代入感受（σ=0.05）：

| T | T·σ² | ESS/N | 判定 |
|---|---|---|---|
| 300 | 0.75 | 0.47 | ✅ 安全 |
| 800 | 2.0 | 0.135 | ⚠️ 越线 |
| 4096 | 10.2 | 4×10⁻⁵ | ❌ 完全崩 |

---

## 4. 度量方案

### 4.1 主曲线：ESS/N 和 chi2_seq 随 k 的衰减

**两条线并排画**（这是核心图）：

| 曲线 | 配置 | 预期 |
|---|---|---|
| sequence-level | `rollout_is=sequence` | 随 k 增大**指数下坠** |
| token-level | `rollout_is=token` | 随 k 增大**缓慢下降或基本持平** |

- **x 轴**：staleness k（用 `trigger_parameter_sync_step` 调，fully_async 默认 4；同步 colocate 是 k=0 的对照点）
- **y 轴 1**：`rollout_corr/rollout_is_eff_sample_size`（ESS/N）
- **y 轴 2**：`rollout_corr/chi2_seq`（对数正态下 `= exp(2Tμ+2Tσ²)−1`，和 ESS 是同一量的两种写法，互为交叉验证）
- **第三条线**：把同一组 run 按 T 分桶（P50 短轨迹 vs P90 长轨迹），验证 H2 的 T×k 耦合

### 4.2 关键：σ² 可以从现有指标反解，**零成本**

代码里**没有** per-token log-ratio 标准差的指标（已逐个核对 `compute_offpolicy_metrics` 和 `compute_is_metrics`）。但可以从两个现有指标反解：

```
μ  ≈ −rollout_corr/kl                        # kl = E[log π_rollout − log π_old] = −E[δ]
σ² ≈ 2·rollout_corr/k3_kl − (rollout_corr/kl)²    # k3_kl = E[e^δ−δ−1] ≈ (σ²+μ²)/2
```

（Taylor 展开在 |δ|≲0.1 时误差可忽略，正是关心的区间。）

**这给出了一个免费的模型自检**：

1. 从 W&B 取 `kl` 和 `k3_kl` → 算 σ²；
2. 用实测 T（从 `token_trace` 数 `response_mask=1` 的 token 数）预测 `ESS/N = exp(−Tσ²)`；
3. 和实测 `rollout_is_eff_sample_size` 对比。

**两种结果都有价值**：
- **吻合** → i.i.d. 假设成立，§3 的解析模型可用，可以直接给出"给定 T 和 k，该用哪种聚合"的**闭式判据**；
- **实测更低** → δ_t 存在正自相关（`Var[S] = Tσ²(1+2Σρ_k) > Tσ²`），说明**多轮 agentic 的 token 失配在轮内聚集**——这本身是个更有意思的发现，而且意味着 sequence-level 比理论预测**更早**失效。

### 4.3 辅助指标（失效的早期信号）

按"先亮起来"的顺序：

1. `rollout_corr/rollout_is_ratio_fraction_high` —— 超上界比例。**最早的信号**，在 ESS 明显下坠前就会涨。
2. `rollout_corr/rollout_is_mean` —— 应 ≈1.0，偏离说明修正在失效（文档告警线：<0.5 或 >2.0）。
3. `rollout_corr/rollout_is_ratio_fraction_low` —— 低于 1/C 的比例。**平时没人看，但 TIS 没有下界**，系统性负偏移时这里是唯一的可见性。
4. `rollout_corr/rollout_is_eff_sample_size` —— 文档红线 0.3。
5. 训练侧后果：`actor/pg_clipfrac`、`actor/grad_norm`、reward 曲线斜率。

### 4.4 实验矩阵（挂在 Phase 2 的 E3 上）

| 格 | rollout_is | sync 频率 k | T 分桶 | 看什么 |
|---|---|---|---|---|
| A0 | sequence | 0（同步 colocate 基线） | 全部 | σ² 的实现失配基线值 |
| A1–A3 | sequence | 1 / 4 / 8 | 全部 | ESS 衰减曲线 |
| B1–B3 | token | 1 / 4 / 8 | 全部 | 对照线（预期持平） |
| C | 二者 | 固定 4 | 按 T 的 P50/P90 分桶 | **H2 的 T×k 耦合** |

**成本控制**：这几格**不需要跑到收敛**——ESS 和 chi2 在前 20–30 step 就稳定了，看的是稳态值不是终点性能。可以复用 E2/E3 已有的 run，只是多记指标 + 多拉一条 `rollout_is=token` 的对照。**边际成本很低。**

---

## 5. 两个必须先排除的混淆变量

**① `bypass_mode` 的默认值不一致（最危险）**

`fully_async_ppo_trainer.yaml` 默认 `bypass_mode: True`（两策略），而老师的 colocate 是 `false`（三策略）。**bypass 模式下 trainer 根本不计算 IS 权重**（`ray_trainer.py:1610-1613` 会跳过），本研究问题的主指标直接消失。

→ **本实验必须全程强制 `bypass_mode=false`**，否则 sequence vs token 的对比无从谈起。详见 [[../learning-notes/03-resource-scheduling]] §4.4。

**② `rollout_rs` 必须保持关闭**

拒绝采样会改写 `response_mask`，进而改变 T 本身（被剔除的 token 不计入 `masked_sum`）。开着它，T 就成了一个随 k 变化的内生变量。老师默认关闭，**保持关闭**。

---

## 6. 可能的产出

按确定性从高到低：

1. **一条实证的 `T×k` 可用区间等值线**——"给定轨迹长度和同步频率，sequence-level TIS 还能不能用"的定量判据。这是工程上直接能用的东西。
2. **σ² 反解法的验证**（§4.2）——如果成立，任何 verl 用户都能从现有 W&B 指标零成本诊断自己的 TIS 是否处于安全区，不需要加任何埋点。
3. **δ_t 自相关的实证**——如果多轮 agentic 的 token 失配确实在轮内聚集，那么 `Var[S]` 的正确估计需要考虑自相关，现有的 i.i.d. 直觉全部偏乐观。
4. **一个上游 issue / PR**——最小版本是给 `rollout_corr_helper.py` 加一个 `rollout_corr/log_ratio_std` 指标（现在只能反解）；进阶版本是提出基于 T 的自适应阈值 `C(T)`。

**关于 C=2.0**：已确认它在 verl 文档里**没有任何推导，是纯经验默认值**（只给了经验区间 token 1.5–5.0 / sequence 2.0–10.0）。**这意味着"从 ESS 目标反推 C(T, k)"是一片空地**——如果 §3 的模型成立，可以直接给出闭式的阈值建议。这可能是本项目理论贡献最大的一块。

---

## 7. 已知的退路

**verl main 分支已经有 `geometric` 聚合选项**（文档提到"若 ESS < 0.3, consider switching to geometric aggregation"），而 0.8.0.dev 快照的白名单只有 `{token, sequence}`，配 geometric 会直接 ValueError。

几何平均大概是 `exp(mean(δ_t))` 而非 `exp(sum(δ_t))`——**T 不进指数，正好治 §3 的病**。

→ 这既是**退路**（如果我们的假设成立，升级到 main 就有现成解法），也是**佐证**（说明上游已经意识到了这个问题，只是还没有人量化它）。**Phase 2 应先确认 main 分支 geometric 的具体实现**，若它确实是 mean 聚合，我们的分析就从"发现问题"变成"解释上游为什么要加这个选项 + 给出何时该切换的定量判据"——**后者的价值反而更高，也更好发表。**

---

## 8. 下一步

- [ ] Phase 0：在 smoke run 里确认 `rollout_corr/*` 指标正常输出，记录 k=0 时的 σ² 基线
- [ ] Phase 1：从 `token_trace` 统计真实 T 分布（`response_mask=1` 的 token 数），替换 §3 表格里的估计值
- [ ] Phase 1：跑 §4.2 的反解自检，判定 i.i.d. 假设是否成立
- [ ] Phase 2：铺 §4.4 的实验矩阵
- [ ] 确认 verl main 的 `geometric` 实现（§7）

---

## 9. staleness 的定义与控制

> 调查日期：2026-07-28
> 代码：`UP/verl/experimental/fully_async_policy/`（`rollouter` 1167 行、`trainer` 767 行、`detach_utils` 374 行、`message_queue` 234 行）
> 该目录**自带 `README.md` / `README_zh.md`**，是训练包里唯一幸存的官方文档，写得相当详细。

### 9.1 `staleness_threshold: 0.1` —— 是**比例**，不是步数

**相对于"一个参数同步周期所需的样本数"的超额生成比例。** 0.1 = 允许最多多生成 10%。

代码（`fully_async_rollouter.py:530-534`）：

```python
self.required_samples = config.actor_rollout_ref.actor.ppo_mini_batch_size * self.require_batches
self.max_required_samples = int(
    self.required_samples
    * (self.staleness_threshold + 1)
    * self.config.async_training.trigger_parameter_sync_step
)
```

README 给的等价公式：
```
rollout_num = (1 + staleness_threshold) × (trigger_parameter_sync_step × require_batches × ppo_mini_batch_size) − num_staleness_sample
```

语义锚点（README §Parameter Description）：

| 取值 | 含义 |
|---|---|
| `0` | **同步训练**——两次参数更新之间只生成刚好够用的样本 |
| `0 < x < 1` | 异步，允许 x 比例的超额生成。**官方建议 < 1** |
| `1` | rollout 足够快时，**基本等价于 one-step-off policy** |

### 9.2 超阈值的行为：**阻塞生成**，既不丢弃也不降权

消费点是 `_should_pause_generation()`（`:1077-1099`）：

```python
if queue_size >= self.max_queue_size:          # 条件一：队列满
    return True
if self.staleness_samples >= self.max_required_samples:   # 条件二：陈旧样本超额
    return True
```

返回 True → `self.paused = True` → rollouter **停止提交新样本**，等 trainer 消费掉一批、触发参数同步后由 `reset_staleness()`（`:564`）唤醒。**背压（back-pressure）机制，不是过滤器。**

**唯一的丢弃路径**是队列溢出（`:947-949`）：

```python
success = await self.message_queue_client.put_sample(...)
if success:  self.total_generated_samples += 1
else:        self.dropped_stale_samples += 1
```

而 `max_queue_size = max_required_samples`，所以正常情况下 pause 会先触发、队列不该溢出。**`dropped_stale_samples > 0` 是"背压没兜住"的告警信号**，Phase 2 要盯。

`staleness_samples` 本身的定义（`:575`）——参数更新时**已生成但尚未被训练消费**的样本数：

```python
self.staleness_samples = len(self.active_tasks) + await self.message_queue_client.get_queue_size()
```

之后每生成一条 `+1`（`:897`）。

### 9.3 staleness 怎么被追踪：三跳

| 跳 | 位置 | 动作 |
|---|---|---|
| ① 服务端打戳 | `UP/verl/workers/rollout/vllm_rollout/vllm_async_server.py:137, 675, 559` | `self.global_steps` 在 `update_weights` 时更新（`:675`），每次 generate 输出挂 `extra_fields = {"global_steps": self.global_steps}`（`:559`） |
| ② 客户端收集 | `fully_async_rollouter.py:96, 130-132, 147-149` | `FullyAsyncLLMServerClient.generate` 在续跑循环里记 `min_global_steps` / `max_global_steps`，写进 `final_output.extra_fields` |
| ③ 进 batch | `detach_utils.py:151-152, 161` | `non_tensor_batch["min_global_steps"]` / `["max_global_steps"]`；`meta_info["trajectory_param_versions"] = max_global_steps` |

**字段名就是 `min_global_steps` / `max_global_steps`**，在 `non_tensor_batch` 里（不是 tensor）。

**训练侧算 k**（`fully_async_trainer.py:756-758`）：

```python
trajectory_param_versions = batch.meta_info["trajectory_param_versions"]   # == max_global_steps
stale_traj_count = sum(1 for v in trajectory_param_versions if self.current_param_version - v >= 1)
```

即 **`k = current_param_version − max_global_steps`**。`current_param_version` 在 trainer 每次参数同步时 `+= 1`（`:498`）。

⚠️ **注意它用的是 `max`（最新的那段），不是 `min`** —— 见 §9.5，这对 partial rollout 是个系统性低估。

### 9.4 `trigger_parameter_sync_step=4` 与 k 的理论上界

两个参数管的是**不同的东西**，别混：

- `trigger_parameter_sync_step` = trainer 做多少次**本地更新**才推一次参数（时间轴上的同步频率）
- `staleness_threshold` = 允许**超额生成多少样本**（队列深度）

**理论上界推算**（`staleness_threshold=0.1, trigger=4, require_batches=1`）：

```
max_required_samples = mini_bsz × 1.1 × 4 = 4.4 × mini_bsz
trainer 每次本地更新消费 = mini_bsz × require_batches = 1 × mini_bsz
⟹ 队列最多积压 4.4 次本地更新 = 1.1 个参数同步周期
⟹ k 的设计上界 ≈ 1~2
```

**能不能出现 k 远大于设计值？能，有两条路：**

1. **长尾 rollout 在飞**：`_should_pause_generation` 只在**提交新样本前**检查，已经 in-flight 的 `active_tasks` 不受背压约束。一条特别慢的轨迹可以跨越多次参数同步一直在飞。
   - `partial_rollout=False` 时，README（mode 3）明说参数同步会**等 active tasks 跑完**——这限制了 k，代价是 trainer 空转（正是 Syncopate 关心的长尾问题换了个位置出现）；
   - `partial_rollout=True` 时会中断它，k 受限，但引入 §9.5 的分段问题。
2. **队列积压 + 消费慢**：若 trainer 比 rollouter 慢，队列一直是满的，队尾样本的 k 会顶到上界。

> **对我们的意义**：k 不是一个可以直接设定的量，是**长尾分布 + 背压参数共同决定的涌现量**。所以 Phase 2 扫 staleness 不能只调 `trigger_parameter_sync_step`，**必须同时记录实测的 k 分布**（§9.6 给了零埋点方案）。

### 9.5 ★ `partial_rollout=True`：一条轨迹一个 staleness 的假设**确实失效**

实现在 `FullyAsyncLLMServerClient.generate`（`fully_async_rollouter.py:56-151`），是个 `while True` 续跑循环：

```python
while True:
    output = await super().generate(
        request_id=request_id,
        prompt_ids=prompt_ids + final_output.token_ids,     # ← 把已生成的接回去当 prompt
        sampling_params=sampling_params, ...
    )
    final_output.token_ids.extend(output.token_ids)
    if output.log_probs is not None:
        final_output.log_probs.extend(output.log_probs)      # ← ★ 直接拼接，无任何标记
    ...
    global_steps = output.extra_fields.get("global_steps", None)
    if min_global_steps is None: min_global_steps = global_steps
    max_global_steps = global_steps
    ...
    if output.stop_reason not in ("aborted", "abort") or not self.config.async_training.partial_rollout:
        break
    await asyncio.sleep(1)                                   # ← 等参数同步完成
```

**逐条回答：**

**(a) 恢复时用新权重还是旧权重 → 新权重。** 被 abort 后 `sleep(1)` 再重新调 `super().generate()`，此时 rollout server 已经完成 `update_weights`，续跑段由**新策略**生成。所以一条 partial 轨迹的前半段是 π_v、后半段是 π_{v+1}（甚至更多段）。

**(b) staleness 按开始还是结束算 → 按结束（`max_global_steps`）。**
`detach_utils.py:161` `trajectory_param_versions = final_batch.non_tensor_batch["max_global_steps"]`，trainer 用它算 k。
**这会系统性低估早期 token 的真实 staleness** —— 早段其实是 `current − min_global_steps` 那么陈旧，却被按最新段记账。

**(c) rollout_log_probs 怎么拼 → 裸拼接，没有任何版本标记。**
`final_output.log_probs.extend(output.log_probs)` 一行，段与段之间无分隔、无 mask、无版本 id。**一条轨迹的 `rollout_log_probs` 向量是多个策略版本的混合。**

> **一个强力佐证**：紧挨着的 `routed_experts` 分支（`:114-123`）有明确的版本感知处理，注释写着 *"On partial rollout resume **the model version may differ**, so keep existing routing and only append routing for newly generated tokens"*。
> **作者清楚知道版本会混合，但只在 MoE 路由上做了处理，logprobs 上什么都没做。**

**这对 TIS 的直接影响（§8 的假设在此失效）：**

```
log_ratio_t = old_log_prob_t − rollout_log_prob_t
              └─ 训练侧单一 π_old 重算    └─ 分段混合，段间分布突变
```

于是 δ_t 在轨迹内部有**分段的分布跳变**：早段是 `π_v vs π_old`，晚段是 `π_{v+1} vs π_old`。sequence-level 的 `masked_sum` 把这些**来自不同分布**的 δ 直接求和：

- §8.2 的 i.i.d. 假设在 partial 轨迹上**直接失效**；
- S 不再是单一对数正态，而是**分段混合**；
- 更糟的是段间不独立（后段以前段为 prompt），`Var[S]` 无法用 `Tσ²` 估计。

**对实验设计的三条硬约束：**

1. **E3（staleness 扫描）必须先用 `partial_rollout=False` 做干净对照**——否则"一条轨迹一个 k"的记账前提不成立，ESS-vs-k 曲线会被污染成两个变量的混合。
2. **partial 轨迹必须单独分桶**——用 `fully_async/partial/partial_ratio` 判断污染比例，用 `max_partial_span` 看最大跨版本数。若 `partial_ratio` 很低（比如 <5%），可以先忽略；若很高，E3 的结论要打折扣。
3. **这本身是个可发表的发现**：*partial rollout 破坏了 sequence-level IS 的同分布前提*。上游连 `routed_experts` 都做了版本感知处理，却漏了 logprobs——这是一个具体、可修的 issue（最小修法：把 `min/max_global_steps` 扩展成 per-segment 的版本边界数组，让 TIS 能按段计算）。

### 9.6 已有的 staleness metrics：**够用，不需要自己加埋点**

| 指标 | 出处 | 含义 |
|---|---|---|
| `fully_async/count/stale_trajectory_processed` | `trainer.py:761` | 累计使用过的陈旧轨迹数（k≥1） |
| `fully_async/count/current_param_version` | `trainer.py:762` | 当前参数版本，算 k 的基准 |
| `fully_async/count/total_generated_samples` | `rollouter.py:1110` | 累计生成样本数 |
| `fully_async/count/dropped_stale_samples` | `rollouter.py:1112` | **背压失效告警**（队列溢出丢样本） |
| `fully_async/count/staleness_samples` | `rollouter.py:1111` | 当前未消费的样本数 |
| `fully_async/partial/total_partial_num` | `detach_utils.py:156` | 本周期跨版本的样本数 |
| `fully_async/partial/partial_ratio` | `detach_utils.py:157` | **partial 污染比例** |
| `fully_async/partial/max_partial_span` | `detach_utils.py:158` | **最大跨越版本数**（= max(max−min)） |
| `fully_async/rollouter/idle_ratio` / `active_time` / `version_time` | `rollouter.py:582-584` | rollouter 空转率 |
| `static/staleness_threshold` / `max_required_samples` | `rollouter.py:1114-1116` | 配置回显 |

**★ 最有价值的是 `batch.meta_info` 里的两个（不在 W&B，但代码里可直接取）**：

```python
"param_version_diversity": len(set(trajectory_param_versions)),   # detach_utils.py:164
"trajectory_param_versions": trajectory_param_versions,           # :165  ← 每条轨迹的版本号列表！
```

**有了 `trajectory_param_versions`，k 的完整分布可以零埋点算出来**：`k_i = current_param_version − trajectory_param_versions[i]`，直方图、P50/P99 全都有了。这正是 §4 度量方案需要的 x 轴。

**两个缺口**：

1. **没有 k 的分布指标**，只有计数（`stale_trajectory_processed`）。但上面的 `trajectory_param_versions` 让我们能自己算——**加 3 行代码即可，是个很小的上游 PR**。
2. **`fully_async/count/stale_samples_processed` 是个幽灵指标**：README 的 Key Metrics 表里列了它，`detach_utils.py:207` 的聚合规则里也声明了它，**但全仓库没有任何地方计算它**（`grep -rn "stale_samples_processed"` 只有这两处）。**照 README 去 W&B 里找这个指标会找不到。**

### 9.7 对 §4 度量方案的修订

基于本节发现，§4.4 的实验矩阵需要两处调整：

- **新增前置格 A0'**：`partial_rollout=False` 下跑完整的 A1–A3，作为"干净"的 ESS-vs-k 曲线；`partial_rollout=True` 单独作为 E4 的对比，**不混进主曲线**。
- **x 轴改用实测 k**：不用 `trigger_parameter_sync_step` 当代理，直接用 `current_param_version − trajectory_param_versions[i]` 算出的 per-trajectory k，按 P50/P99 分桶。因为 §9.4 说明了 k 是涌现量，不是设定量。

同时新增一条待验证项：**`partial_ratio` 有多高**。若在我们的长尾 agentic 任务上很高（长轨迹更容易被参数同步打断），那 §9.5 的问题就不是边角案例，而是主要矛盾——**那样研究问题的重心应该从"sequence-level 的 T 依赖"扩展到"partial rollout 破坏 IS 同分布前提"**，后者更具体、更好修、上游价值更高。

### 9.8 ★ 兑现：M7 实测（2026-08-14，fully_async / 150 步）—— 上面那条待验证项有答案了

```
partial_ratio    0.000      ← 没有任何一条轨迹跨越参数版本边界
分布漂移 TV      0.000      ← 下发 7200 = 训练 7200，一条不差
ESS/N            0.74–0.88  ← 整跑稳定；离线合成 k=0 的预测是 0.846，落在区间内 ✅  ⛔(21)
陈旧轨迹          576 / 7200 = 8.0%  ⛔(21)
```

⇒ **§9.7 那条待验证项的答案是「很低，低到 0」**，所以研究重心**不转向** partial rollout。
根因是**尺度不匹配**：轨迹 ~700 token / 4.5 工具步，而同步周期 4 步 ≈ 300 s ——
**根本碰不到版本边界**。这同时说明 §2 结尾那句「fully-async → k 大」
**在当前配置下不成立**：k 是**涌现量**（§9.4 已论证），而这一跑它就没涌现出来。

⚠️ **对 §4 度量方案的后果**：x 轴的 k **不会自己长出来**。要拿到 ESS-vs-k 曲线，
必须**主动制造 k** —— 调大 `--latency-scale`（长尾拉长）或**缩短同步周期**，
二选一，否则 A1–A3 三格量到的会是同一个点。
⇒ 与之互补的是**离线合成**（`train/staleness.py`）：k 由我们给定、精确可控，
已产出第一个点 σ²(0) ≈ 2.0e-4/token。两条路的分工见 `06-rl-run-protocol.md` §4.1。
