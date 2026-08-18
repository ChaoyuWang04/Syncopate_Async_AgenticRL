# E22 · 异步模式下 **LoRA 从没被推给 rollout** —— 生成数据的策略两个月没变过

> 状态：✅ **已确证 → 已定根因 → 止血方案被自己否掉 → 已自己实现真正的修法并验证**　建于 2026-08-18
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
| **修法** | ✅ **已自己实现修法①并验证跑通（§6.4）**：`SYNCOPATE_LORA_ADAPTER_SYNC=1`，vLLM 引擎里 `list_loras()=[123]`、载荷 8,414→252 MiB、`kl` 回到地板、`param_sync` 6.25→0.97 s。<br>⛔ 而 `model.lora.merge=True` **已被 R0-b 否掉**（bf16 合并毁掉 adapter 一半的作用，§6.1）⇒ 正确性实验改走 **colocate**（§6.2）；异步线可自己补 adapter 推送（**§6.3：两端能力都在，缺中间传参，估 30–60 行**） |
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

### 5.1.1 🆕 上游情报（2026-08-18 检索）：**verl 知道这条路不支持 LoRA，而且当年是会报错的**

[verl#2048 「[Async VLLM] LoRA support?」](https://github.com/verl-project/verl/issues/2048)：

> LoRA 目前**只支持同步的 `vLLMRollout`**；async worker 的 `inference_engine` 没有
> `llm_engine` 属性，**因此会抛出错误**。

**该 issue 被关成 `not planned`。**

⇒ ★ **这把我们的论点整个改写了**，而且改硬了：

```
不是「你们有一个 bug」
而是「你们**知道**异步这条路不支持 LoRA，早期版本会**明确报错**；
      而在现在的 CheckpointEngine 架构里，它**不报错了** —— 改成静默推一份冻结基座」
⇒ **失败模式从"响的"退化成了"哑的"。**
```

**静默地把一个已知不支持的组合跑下去，比直接报错坏得多** —— 报错只损失一次启动，
静默损失的是两个月的实验和一整条结论线。

同族的脆弱迹象（都指向 async+LoRA 这条路一直没人走顺）：
[verl#3654](https://github.com/verl-project/verl/issues/3654)（async rollout + LoRA 加载时崩）·
[verl#3882](https://github.com/volcengine/verl/issues/3882)（LoRA 占位路径 `FileNotFoundError`）。

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

**④ 🆕 这个限制是**公开的**，我们从没搜过。**
`verl#2048` 明写「LoRA 只支持同步 rollout」，**两个月前就能搜到**。
⇒ **纪律补一条：撞到"框架行为不符合预期"时，先花五分钟搜上游 issue/论坛。**
成本几乎为零 —— 而这次它能省下的是两个月。（E21 那条同样：PyTorch 论坛上有人报过。）

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

## 6.3 ★★ 那 LoRA + 训推分离到底做不做得了？—— **做得到，缺的只是中间那段传参**

这是最要紧的一问，答案不是"不行"，也不是"现在就行"：

```
两端的能力**都已经在了**，断的是中间：

trainer 侧   get_per_tensor_param(base_sync_done=True)  ⇒ 直接吐 LoRA 张量 + peft_config   ✅ 有
transport    send_weights 是流式 (name, tensor)，LoRA 只有 ~132 MB                          ✅ 能过
🔴 断点      CheckpointEngineWorker.update_weights(global_steps)  ← **签名里没有 peft_config**
             （base.py:323）它只会 receive_weights() 然后原样交给 server_adapter
rollout 侧   vllm_rollout.update_weights(..., **kwargs) ⇒ update_weights_from_ipc(peft_config=…)
             ⇒ _update_weights ⇒ **TensorLoRARequest(lora_tensors=weights) + add_lora**     ✅ 有
             （`vllm_rollout/utils.py:262`，**能直接从张量装 LoRA，不需要文件路径**）
```

⇒ ★ **vLLM 那侧根本不缺能力** —— `--enable_lora` 起着、`PunicaWrapperGPU` 建好了槽位、
从张量装 adapter 的代码就在那儿，**colocate 每次同步都在用它**。
⇒ **缺的只是让 `peft_config` / `base_sync_done` 沿 CheckpointEngine 这条管子传下去，
并且让 trainer 侧首次推基座、之后推 adapter。**

### 6.3.1 自己补这条路的成本与风险（供决策）

| | |
|---|---|
| **改动面** | 三处：trainer 侧调用参数 · `CheckpointEngineManager/Worker` 的传参 · 首次/后续的 `base_sync_done` 状态 |
| **量级** | 估 **30–60 行**（在 `verl_patches` 里，不改 site-packages） |
| **主要风险** | ① `peft_config` 要跨 Ray actor 序列化（colocate 是同进程，没验过跨进程）② `@register` 装饰器包着的 worker 方法改签名要小心（E12 §262 踩过：包装丢了装饰器导致组级方法直接消失） |
| **验证** | 现成的：`SYNCOPATE_SYNC_PAYLOAD=1` 应当看到 **含 `lora_` 的张量数 > 0、载荷从 8.4 GB 掉到 ~132 MB**；再加 `kl` 每次同步回落到地板 |
| **附带收益** | 载荷 8,414 MiB → ~132 MiB（**64×**）⇒ 权重同步的时间构成整个重来（E12 要重写的那部分正好一起做了） |

### 6.3.2 ⇒ 三条路，按可用时间排

```
① 现在就能跑正确实验     **colocate**（推 adapter、不合并）—— 吞吐差约 1.9×，但**是对的**
② 一天左右的工作         **自己补 adapter 推送**（上面这条）⇒ 异步线的正确性实验解锁
③ 长期                   **上游修**（我们的 issue 建议的方案①）
```

⛔ **`--lora-merge` 不在这三条里** —— R0-b 已证明它注入的失配与被研究对象同量级（§6.1）。

★ **为什么建议做 ②**：异步 RL 是本项目 infra 线的**第二目标本身**，Track B 的整条叙事
（"agentic RL 训练系统的框架级改造"）都建立在"能正确地跑异步"上。
**只有 colocate 能跑正确实验 = 这条线的兑现物做不出来。**
⚠️ 但**先拿一份 colocate 的干净基线**（R0-c），否则补完了也没有正确的东西可以对照。

## 6.4 ✅✅ 修法① 已自己实现并验证 —— **异步 RL 第一次真正跑通**（2026-08-18 晚）

补丁：`verl_patches._patch_lora_adapter_sync`（`SYNCOPATE_LORA_ADAPTER_SYNC=1`）。
就是 §6.3 说的那段"没接上的管子"，两处各补一段：

```
trainer 侧  ActorRolloutRefWorker.update_weights
            自己记 `_syncopate_base_sync_done` ⇒ 首次 get_per_tensor_param(base_sync_done=False)（基座）
            之后 base_sync_done=True（**adapter**）
rollout 侧  CheckpointEngineWorker.update_weights
            首次原样装基座；之后带上 peft_config + base_sync_done=True
            ⇒ 一路到 `TensorLoRARequest` + `add_lora`
```

★ 两处**各自**记状态、**不跨进程传标志** —— 它们由 `CheckpointEngineManager.update_weights`
在同一步里成对调用（`base.py:497-500`），天然同步；判据行把两侧状态都打出来。
★ `peft_config` 也**不跨进程传**，在 rollout 侧用同源的 `model_config` 就地重建
（`PEFTHelper.from_dict` 只要 `r / lora_alpha / target_modules`）—— **少一条序列化路径就少一个静默失败点**。
⚠️ 两侧都用 `func.__dict__.update(orig.__dict__)` 保住 `@register` 的元数据
（E12 §262 踩过：包装丢了装饰器 ⇒ 组级方法直接消失）。

### 6.4.1 验证（60 步 / 15 个 param_version，`r0d_adapter`，**0 错误**）

| 判据 | 结果 |
|---|---|
| **★ vLLM 引擎里有没有 adapter**（新探针 `[lora-probe]`，挂在 `vLLMHttpServer.set_global_steps`） | `step=0 list_loras()=[]`（首次只推基座，符合设计）→ **`step≥1 list_loras()=[123]`** ✅<br>`123` 就是 `VLLM_LORA_INT_ID` —— `vllm_async_server.py:527` 正是拿它决定生成时挂不挂 LoRA |
| **载荷** | 第 1 次 `399 张量 / 8,414.1 MiB / lora_ 0`（基座）→ 之后 **`504 张量 / 252.0 MiB / lora_ 504`** ✅ |
| **`rollout_corr/kl`** | `0.00041 0.00257 0.00074 0.00184 0.00114 **0.00032 0.00034**` —— **震荡并回到地板**<br>对照坏基线同位置：`0.00034 0.00028 0.00168 0.00122 0.00160 0.00208 **0.00344**` —— **单调爬升**<br>⇒ 第 7 个版本上 **相差 10×**，判据这次**有能力分辨**了 |

⇒ ⭐ **异步 RL 第一次真正跑通**：rollout 用的是当前策略，不是 π₀，也不是 bf16 合并的残骸。

### 6.4.2 ⛔ 预测偏了一处（照纪律记下来）

```
预测  第 2 次起载荷 ~132 MiB
实测  **252.0 MiB**
原因  LoRA 参数是 **fp32**（66M × 4 B = 264 MB ≈ 252 MiB），我按 bf16 算了
⇒ 降幅是 **33×**，不是我说的 64×。张量数 504 的预测是对的。
```

### 6.4.3 ★ 白捡的一大块：`param_sync` 6.25 s → **0.974 s**（6.4×）

| | R0-a（merge 止血，每次推 8.4 GB） | **R0-d（推 adapter）** |
|---|---|---|
| 稳态 `param_sync` | **6.25 s** | **0.974 s**（中位；各次 1.04/0.97/0.92/1.10/1.02/0.84/0.79） |
| 占一步 | ~6.5% | **0.8%** |
| 首次（推基座） | — | 13.3 s，**一次性** |

⇒ **这直接改写 E12 的结论**：那份「权重同步 99.9% 不是传输」是在**误以为只推 132 MB**、
而实际推 8.4 GB 的前提下算的。**按真实载荷，同步就是 1 秒的事。**
⇒ 一步的构成随之变成：**三次前向占 88.9%**（`update_actor` 56.8% + `old_log_prob` 17.2% + `ref` 14.9%）
—— 吞吐线的靶子现在毫无争议地是 E17/B12。

## 6.5 ★★ 数值正确性验证（Chaoyu 追问："推了 adapter ≠ 推对了"）

**问题成立**：前面的判据只证明了「推的是 adapter、大小对、引擎里有」。
而补丁里有一个**重建**环节（`peft_config` 在 rollout 侧就地重建）——
**`scaling = lora_alpha / r` 一旦错了，一切表象都正常，而策略被整体缩放。**

### 6.5.1 V1 · 两侧 `peft_config` 逐字段比 ✅

```
trainer 真实   [dict] r=32 alpha=64 **scaling=2.0**
               target=['down_proj','gate_proj','k_proj','o_proj','q_proj','up_proj','v_proj']
rollout 重建         r=32 alpha=64 **scaling=2.0**
两侧共同源头   model_config: rank=32 alpha=64   ← 同一个 HFModelConfig
```
⇒ **scaling 一致。** 这是唯一"重建"出来的量，也是最可能错的地方。

⚠️ `target_modules` 两侧表示形式不同（trainer 是 PEFT **解析后**的 7 个模块名，
rollout 传的是**未解析的** `"all-linear"` 字符串）。
**这不影响装载**：vLLM 的 `hijack__load_adapter` 用的是**张量名字**决定权重落到哪个模块，
`expected_lora_modules` 取自模型自身而非 `peft_config`；`PEFTHelper` 也接受 `list[str] | str`。
⇒ 但**不靠这句话下结论**，靠下面 6.5.2 的端到端数值。

### 6.5.2 ★★★ V3 · 端到端数值：`log_ppl_diff` / `kl` 落在**同版本地板**上 ✅

`rollout_corr/log_ppl_diff` 比的就是 **vLLM 返回的 logprob** 与 **trainer 重算的 logprob**，
**同一批 token**。⇒ **只要 scaling 错、模块漏装、张量值损坏中的任何一项发生，它都会远离地板。**

| | 全程 `kl`（15 个 param_version） |
|---|---|
| **坏基线**（M7-b，推冻结基座） | `0.00034 0.00028 0.00168 0.00122 0.00160 0.00208 0.00344 0.00259 0.00407 0.00429 0.00488 0.00437 0.00657 0.00843 0.00732`　**单调爬升，均值 0.00385** |
| **修好后**（R0-d，推 adapter） | `0.00041 0.00257 0.00074 0.00184 0.00114 0.00032 0.00034 0.00037 0.00062 0.00163 0.00035 0.00042 0.00024 0.00036 0.00214`　**无趋势、反复回到地板，均值 0.00082** |

独立复现（`v_numeric`，12 步、**全部走新默认值**）：
`log_ppl_diff = 0.00053 / 0.00051 / 0.00014`，`kl = 0.00060 / 0.00051 / 0.00022` —— **全在地板上**。

⇒ ✅ **数值正确性端到端确认**：rollout 引擎算出的概率与 trainer 算出的概率一致到
**vLLM↔FSDP 的数值实现差异**这一层（~3.4e-4），没有可归因于 scaling / 漏装 / 值损坏的额外偏差。

### 6.5.3 ⛔ 我在这一步又踩了一次"判据自己失效"

```
第一版 V1 探针用 getattr 读 peft_config ⇒ 全打成 None（它是 dict 不是对象）
        ⇒ **判据无效，却打成了"无"** —— 差点被读成"没问题"
第二版 修成类型无关，又对字符串做了 sorted()
        ⇒ "all-linear" 被拆成 ['-','a','a','e','i','l','l','l','n','r']
        ⇒ **看起来像"重建错了"，其实是探针的显示 bug**
```
⇒ 两个方向都栽过了：**判据可以假通过，也可以假失败。**
⇒ 纪律再加一层：**判据打印本身要能区分"读不到"和"读到了是空"**，且**不要对未知类型做格式化操作**。

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
