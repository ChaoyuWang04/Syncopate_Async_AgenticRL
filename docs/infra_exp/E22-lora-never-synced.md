# E22 · 异步模式下 **LoRA 从没被推给 rollout** —— 生成数据的策略两个月没变过

> 状态：✅ **已确证 → 已定根因 → 已找到止血方案并验证**　建于 2026-08-18
> ⚠️⚠️⚠️ **这是正确性 bug，影响面比 [`E21`](E21-ddp-not-syncing.md) 更大。**
> 上游草稿：[`../upstream/verl-lora-adapter-never-synced-disaggregated.md`](../upstream/verl-lora-adapter-never-synced-disaggregated.md)
> 作废清单与重跑队列：[`00-INFRA-HANDOFF §5`](00-INFRA-HANDOFF.md)

---

## 0 · 结论卡片

| | |
|---|---|
| **问题** | `fully_async` / `one_step_off`（disaggregated）下，每次权重同步推给 vLLM 的都是**未经修改的冻结基座**；LoRA adapter **一个字节都没推过去** |
| **怎么发现的** | 0-B 排查「权重同步的**内容**」时，读发送侧代码发现一条可疑分支 ⇒ 离线复现分支行为 ⇒ 真实跑挂探针实测 |
| **判据** | 推出去的 `q_proj.base_layer.weight` `‖W‖=75.377708`，与**磁盘上起点模型逐位相同**；4 次独立短跑、每跑 2 次同步、全部一致 |
| **后果** | **rollout 永远用起点策略采样** ⇒ 整个 RL 回路是断开的：策略梯度算的是"当前策略"，数据来自"起点策略"，而这个偏离**随训练单调增大** |
| **根因** | `engine_workers.py:698` 在 disaggregated 分支上**只调一次** `get_per_tensor_param()` 且不传参 ⇒ `base_sync_done=False` ⇒ `collect_lora_params` **显式跳过所有 `lora_` 张量**。colocate 那条路**调两次**，是对的 |
| **止血** | `model.lora.merge=True`（`launch_rl --lora-merge`）⇒ 实测推出去的权重开始随训练变化。⚠️ **但它是 bf16 合并，有数值损失，见 §6** |
| **不受影响** | **colocate 全部正确**；所有吞吐 / 通信 / kernel 类测量不受影响 |

⇒ ⛔ **我们从来没有跑过一次正确的异步 RL。** 所有 fully_async 的**学习类**结论作废。

---

## 1 · 怎么发现的（方法上值得记：它是 E21 的直接产物）

E21 之后列的排查清单里有一条一直是红的：

```
🔴 权重同步的内容   推给 vLLM 的到底是不是**当前 trainer 的权重**？
                   （只验过"同步耗时"，没验过"同步内容"）
```

主线也在 `MAINLINE-HANDOFF §1.2` 把它提成 I3。**它是清单上唯一只有 infra 能做的一条。**

★ 关键在于**问对了问题**：此前 E12 花了一整轮研究「权重同步为什么慢」，
把「只推 132 MB LoRA」当成**已知前提**（而那是**算出来的**：66M × 2 B），
**从没问过"推的是什么"**。⇒ 又一次 `feedback-measure-dont-infer`。

---

## 2 · 代码路径

### 2.1 disaggregated：只调一次，用的是默认参数

```python
# verl/workers/engine_workers.py:695-700  ActorRolloutRefWorker.update_weights
effective_mode = mode if mode != "auto" else self.config.rollout.checkpoint_engine.backend
if effective_mode != "naive":
    per_tensor_param, _ = self.actor.engine.get_per_tensor_param()   # ← 不传参 ⇒ base_sync_done=False
    await self.checkpoint_engine.send_weights(per_tensor_param, global_steps=global_steps)
    return                                                            # ← 没有第二次
```
⚠️ 第二个返回值 `peft_config`（"这是个 LoRA 模型"的信号）在这里被**丢弃**。

### 2.2 colocate：调两次，先基座后 adapter —— **这条是对的**

```python
# engine_workers.py:711-731
per_tensor_param, peft_config = ...get_per_tensor_param(base_sync_done=True)   # adapter
if do_lora_base_sync:                                                          # 首次才做
    per_tensor_param_base, _ = ...get_per_tensor_param(base_sync_done=False)   # 基座
    await self.rollout.update_weights(per_tensor_param_base, peft_config=..., base_sync_done=False)
await self.rollout.update_weights(per_tensor_param, peft_config=..., base_sync_done=True)
```

### 2.3 `base_sync_done=False` 会显式扔掉 LoRA

```python
# verl/utils/fsdp_utils.py:700-713  collect_lora_params
for name, param in model.state_dict().items():
    if any(x in name for x in ["_flat_param", "lora_"]):      # ← **跳过所有 LoRA 张量**
        continue
    name = name.replace("_fsdp_wrapped_module.", "").replace(".base_layer", "")
```
随后 `replace_lora_wrapper`（`fsdp_utils.py:749`）把名字**改回** `...q_proj.base_layer.weight`
—— 让 vLLM 那个已启用 LoRA 的包装层收下基座权重。

⇒ `base_sync_done=False` 的语义本来是「vLLM 还没有基座，**先**推基座」，
设计上应当跟着一次 `base_sync_done=True`。**disaggregated 只有前半句。**

---

## 3 · 证据

### 3.1 离线：分支行为（小模型，不起训练，`scripts/probe_weight_sync_payload.py`）

| 分支 | 张量个数 | 总字节 | 含 `lora_` 的 |
|---|---|---|---|
| **`base_sync_done=False`（disaggregated 走的）** | 25 | 74.8 MiB | **0** |
| `base_sync_done=True`（colocate 会再调一次） | 28 | 0.2 MiB | **28** |

### 3.2 ★★★ 真实训练里的实测（`SYNCOPATE_SYNC_PAYLOAD=1`，4 次独立短跑）

```
[sync-payload] 本次同步推出去：399 个张量 / 8,414.1 MiB / 其中 lora_ 0 个
[sync-payload] 盯住的层 model.layers.0.self_attn.q_proj.base_layer.weight  ‖W‖=75.377708
```

**判据（"两个东西应当相同"型，不设阈值）**：

```
推出去的     model.layers.0.self_attn.q_proj.base_layer.weight   ‖W‖ = 75.377708
磁盘上起点模型 models/Qwen3-4B-sft-v13-e1 的同一层（直接读 safetensors）  ‖W‖ = 75.377708
                                                                          ↑ **逐位相同**
且**两次同步之间完全一致**（冻结基座本来就不会变）
```

补充旁证：

```
8,414.1 MiB ≈ Qwen3-4B 基座 bf16 的完整大小（LoRA r=32 只有 ~132 MiB）
vLLM 以 `--enable_lora` 启动、`PunicaWrapperGPU` 已初始化
⇒ **adapter 槽位一直是空的。**
```

### 3.3 ✅ 止血方案的对照（`--lora-merge` ⇒ `model.lora.merge=True`）

| | `merge=False`（默认，我们一直跑的） | **`merge=True`** |
|---|---|---|
| 张量名 | `...q_proj.**base_layer**.weight` | `...q_proj.weight`（合并后的正常名） |
| 盯住层 `‖W‖` | **75.377708**，四跑六次同步**全都一样** | **75.397400 → 75.397392**，**随训练在变** |
| 与磁盘起点比 | **逐位相同** ⇒ 冻结基座 | 相对差 2.6e-4 ⇒ 增量在里面 |
| 载荷 | 399 张量 / 8,414.1 MiB | 399 张量 / **8,414.1 MiB（一样大）** |

⇒ **修复不额外花钱** —— 我们本来就在推 8.4 GB，只是没拿到货。

---

## 4 · 后果：它把注意力引向了完全错误的方向

⚠️ **在任何指标上都看不出来**：loss 会降、reward 会动、grad_norm 正常、熵正常、
`rollout_is_eff_sample_size` 有值、没有任何 warning。

**而且那些"看起来像陈旧度"的现象一直在替它顶包**：

| 我们观察到的 | 当时的解释 | **真实解释** |
|---|---|---|
| 固定 `sync_every`，`rollout_corr/kl` 单调涨 36×（主线观测 A） | 陈旧度累积 | π_old 恒为 π₀ ⇒ 量的是**累计位移** |
| ESS 沿「lr × 步数」重合，与 `sync_every` 无关（观测 B） | 巧合 | ESS 是**位移**的函数，与陈旧度无关 |
| `staleness_threshold` 0.1→0.5，陈旧轨迹 6×，ESS 纹丝不动（观测 C / B10） | 阈值不敏感 | **这个旋钮没接到任何东西** |
| 权重同步"与数据量无关"（8 GB 与 132 MB 同耗时，E12 §253） | 固定开销主导 | **一直都是 8 GB**；"132 MB"是算出来的 |
| 跑完一整个 epoch 能力几乎没变（E20 的起点） | 序列级 IS + 更新次数少 | 两条都成立，**但还漏了这一条** |

★ **这条 bug 最坏的地方不是它错，是它制造了一整套自洽的错误解释**，
而我们据此追了两个月的"陈旧度"。

### 4.1 它与 E21、E20 叠在一起

```
名义   每次更新 6 题 × 8 采样 = 48 条序列，数据来自 k 步前的策略
E21    梯度不同步        ⇒ 每个 rank 只用 16 条
E20    序列级 IS 崩塌    ⇒ 剩下的 67% 被压平
E22    LoRA 没推过去     ⇒ **数据全部来自起点策略 π₀，且偏离随训练单调增大**
⇒ 实际发生的是：**用 π₀ 生成的数据、1/3 的样本量、崩掉的 IS 权重，做离线策略梯度。**
```

⇒ 「跑完一整遍数据集能力几乎没变」这个现象，**三条都有份**。

---

## 5 · 责任划分（照 E21 §5.5 的规矩，先说上游、再说我们）

### 5.1 上游的份（主要）

| 谁 | 问题 | 严重度 |
|---|---|---|
| **verl** | 同一个开关 `model.lora.merge` 在两条路径下语义不同，**默认值只对其中一条正确**；走错的那条**不报错**，而是推一份语义上空的基座 | 🔴 正确性级别的失败用静默方式处理 |
| **verl 文档** | `update_weights` 的 docstring 只有描述性的一句「when `model.lora.merge=True`, LoRA is merged into base weights before sync」——**没有说这是 disaggregated + LoRA 的必要条件** | 🟠 |

⚠️ **公道话**：`merge=False` 在 colocate 下是**正确且更优**的（推 132 MB adapter 而不是 8.4 GB 全量）。
所以这个默认值本身不能说"写错了"——**错在同一个默认值在另一条路径上是灾难，而框架不检查。**

### 5.2 我们的份

**① 我们量过这条路径的"耗时"，但从没量过它的"内容"。**
E12 是一份很扎实的耗时分解（编排 8 步分别计时、两点反解出传输分量），
**但它整份建立在"稳态只推 132 MB LoRA"这个从没量过的前提上**。
⇒ 而且 E12 §253 自己记下了反常：「与数据量无关（8 GB 与 132 MB 同耗时）」——
**那条反常就是这个 bug 在敲门，我们把它归给了"固定开销主导"。**

**② 排查清单上这一条红了整整一天没人动。**
E21 之后我们自己列的「🔴 权重同步的内容」，主线也提成 I3 ——
**清单是对的，只是没被排进执行。** ⇒ `observed-needs-an-owner` 的又一次。

**③ 一个可以更早发现的旁证被忽略了**：`--enable_lora` 的 vLLM 起着，
而我们从没问过"它的 adapter 槽位里有东西吗"。

---

## 6 · ⚠️ `--lora-merge` 只是止血，不是修好

**它在 bf16 里做合并**（`fsdp_merge_unmerge` → PEFT 的 in-place merge，基座是 bf16）。
而主线 `18 §3.3` 已经实测过这一级增量在 bf16 里会发生什么：

```
SFT 的增量占基座 0.42%   → 合并进 bf16 后 保真残差 0.36，幅度比 1.04    可用
RL  的增量占基座 0.056%  → 合并进 bf16 后 保真残差 **0.87**，幅度比 0.68  🔴 **方向被舍入噪声打乱**
⚠️ 损失来自**存储精度**，不是累加精度 —— 在 fp32 里相加再存 bf16，结果一样
```

### 6.1 ✅ R0-b 已实测：**止血不够，合并毁掉了 LoRA 一半的作用**

尺子：`scripts/probe_merge_logprob_fidelity.py`（**同引擎、同 dtype、同一批 prompt，只差合并这一步**
⇒ 差异只可能来自 bf16 合并；引擎差异与陈旧度**刻意不混进来**）。
输入是 R0-a 那份干净 ckpt（`r0a_clean/global_step_6`，24 步）。

```
adapter 本身对 logprob 的作用     中位 3.428e-02      ← 探针自检，同时证明它有能力失败
bf16 合并造成的偏移              中位 **1.717e-02**   ← **正好是 adapter 作用的 50%**
                                 p95  1.201e-01
                                 最大 2.370e-01
                                 每条序列 Σ|Δ| = **35.47**
引擎噪声地板（同版本 vLLM↔FSDP）   3.4e-04            ← 合并损失是它的 **50 倍**
```

⇒ 🔴 **判定：`--lora-merge` 不能当正式方案。**

**三个角度看这个数有多大**：
1. **相对 adapter 自己**：50% —— 推过去的策略，**只保留了 LoRA 一半的作用**。
2. **相对引擎噪声**：50× —— 远在"数值失配"能解释的范围之外。
3. **相对我们此前称为"陈旧度"的东西**：坏基线上 `log_ppl_diff` 最坏是 **0.0231**，
   而合并损失每 token 就有 **0.0172** ⇒ **这个止血会注入一个与"被研究对象"同量级的失配。**
   ⇒ 拿它去重跑 E20，等于**用一个新的混淆变量替换掉旧的**。

⇒ 与主线 `18 §3.3` 的权重侧测量互相印证（保真残差 0.87、幅度比 0.68）：
**幅度留住了、方向没了** —— 在 logprob 上表现为"作用只剩一半"。

### 6.2 ⇒ 正确性线的重跑改走 **colocate**

`colocate`（naive）那条路 **`get_per_tensor_param(base_sync_done=True)` ⇒ 直接推 LoRA 张量**
（[§3.1](#31-离线分支行为小模型不起训练scriptsprobe_weight_sync_payloadpy) 实测 28 个 `lora_` 张量），
再由 `rollout.update_weights(..., peft_config=..., base_sync_done=True)` 交给 vLLM 按 adapter 装载
⇒ **全程不做 bf16 合并。**

⇒ **它是目前唯一能把当前策略正确交付给 rollout 的模式。**
⚠️ **[推断，基于代码 + §3.1 的离线实测；运行时未单独验证]** ——
但验证是**免费**的：colocate 下陈旧度恒为 0 ⇒ `rollout_corr/kl` 应当**每步都贴着地板**。
第一次 colocate 重跑就能确认。

⇒ **异步线的正确性实验要等上游修法①**（把 adapter 单独推过去）。
⚠️ 自己实现的话不是小改动：`CheckpointEngineWorker.update_weights` 那一侧
**没有 adapter 装载入口**（`base.py:323`：`receive_weights` → `server_adapter.update_weights(weights)`，
签名里根本没有 `peft_config`）。⇒ **这也让上游 issue 的论点更硬。**

⇒ **真正的修法仍是上游那条 ①**：把 adapter **单独推过去**（vLLM 本来就支持，槽位都建好了），
根本不做 bf16 合并。**这也让上游 issue 的论点更硬：`merge=True` 不只是慢，它在数值上是有损的。**

---

## 7 · 已落地

| 落点 | 内容 |
|---|---|
| 🆕 `verl_patches._patch_sync_payload_probe`（`SYNCOPATE_SYNC_PAYLOAD=1`） | 每次同步打「张量数 / 字节 / `lora_` 个数 / 盯住层 ‖W‖」。给 `SYNCOPATE_SYNC_REF` 才下判定，**否则只报数** |
| 🆕 `scripts/probe_weight_sync_payload.py` | 离线复现发送侧两个分支的行为，不占卡 |
| 🆕 `launch_rl --lora-merge` | 打开 `model.lora.merge=True`（默认关，见 §8 的待决事项） |
| 🆕 `docs/upstream/verl-lora-adapter-never-synced-disaggregated.md` | 上游草稿，含三条修法 |

### 7.1 ★ 我在写这个探针时**两次**踩了"空判据被读成通过"

```
第一次  盯住的层名写错（少了 .base_layer）⇒ ‖W‖=None，那行却照样打「与磁盘起点相同」
第二次  修完之后，merge=True 那跑打出 75.3974（明明不同），**那行还是打「相同」**
        —— 因为我把结论**硬编码进了文案**，根本没做比较
```
⇒ 现在：绑不上就报红并打出真实名字样例；没有参考值就**只报数、不下判定**。
⇒ ★ 这是 `blank-thresholds-are-not-passes` 的第三条 ——
**而我是在写"专门防这个"的探针时踩的它。判据行的文案本身也是判据的一部分。**

---

## 8 · 待决 / 下一步

1. ⬜ **`--lora-merge` 要不要在 disaggregated 下默认开**（并在 `lora_rank>0 且非 colocate 且没开` 时直接报错）——
   见 §6，它是止血不是修好，但**不开一定是错的**。⇒ 等 Chaoyu 定。
2. ⬜ **R0-b：量 vLLM ↔ trainer 的 logprob 一致性**（§6 那条判据）—— 决定止血够不够。
3. ⬜ **上游 issue 是否提交**（三份草稿都等 Chaoyu 点头）。
4. ⬜ **重跑队列**见 `00-INFRA-HANDOFF §5`。
