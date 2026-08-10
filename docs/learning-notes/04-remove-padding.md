# 04 · remove padding（变长序列打包）在 verl 里的实现路径

> 调查日期：2026-07-28
> 代码基准：`reference/industrial_posttrain_training_release/verl/upstream/`（verl `0.8.0.dev` 快照，下简称 `UP/`）
> 老师 GRPO 默认开启：`actor_rollout_ref.model.use_remove_padding=True`

---

## 0. 一句话结论

verl 0.8 **不再用经典的 "cu_seqlens 手工穿线" 写法**，而是把变长打包封装成 **`torch.nested` jagged NestedTensor**：cu_seqlens 变成 NestedTensor 的 `offsets()`，随张量一起流动。unpad 只发生在 **driver 侧一处**（`left_right_2_no_padding`），pad 回来发生在 **loss 之前一处**（`no_padding_2_padding`），中间的 packed 坐标系**完全封闭在 forward 内部**。

对我们最重要的三个结论：

1. **actor / ref / critic 没有各写一份 forward** —— 只有一份，critic 靠继承复用；
2. **packed position_ids 确实每段从 0 重置**，但机制是"padded 坐标下 per-row cumsum + 按 mask gather"，**不是**按 cu_seqlens 做偏移减法；
3. **log_prob 在 packed 状态下就算完了**，`(bsz, seq, vocab)` 的巨型 logits 张量**从未 materialize** —— 这是 remove padding 最大的显存收益来源，比省 attention FLOPs 更值钱。

---

## 1. 谁消费 `use_remove_padding`

### 1.1 配置流转链

| 层 | 位置 | 作用 |
|---|---|---|
| 源头默认值 | `UP/verl/trainer/config/model/hf_model.yaml:40` | `use_remove_padding: True` |
| actor 继承 | `UP/verl/trainer/config/actor/dp_actor.yaml:43` | `${oc.select:actor_rollout_ref.model.use_remove_padding,false}` |
| ref / critic 继承 | `UP/verl/trainer/config/ref/*.yaml`、`workers/config/critic.py:272` | 同上 |
| 注入 engine | `UP/verl/workers/engine_workers.py:113, 335, 389, 535, 571` | 分别给 actor / ref / critic 的 engine_config 赋值 |
| 校验 | `UP/verl/workers/config/actor.py:329-332` | **开 sequence parallelism 必须同时开 remove_padding**（Ulysses SP 依赖 packed 布局切分） |

老师的脚本在 `REL/scripts/train_grpo_verl.py:90` 显式传 `actor_rollout_ref.model.use_remove_padding=True`，三个角色一起生效。

### 1.2 真正消费它的地方：**只有一处**

```
UP/verl/workers/engine/fsdp/transformer_impl.py
├── :85    class FSDPEngine(BaseEngine)
├── :922   @EngineRegistry.register(model_type="language_model", backend=["fsdp","fsdp2"])
│          class FSDPEngineWithLMHead(FSDPEngine)
│          ├── :923  prepare_model_inputs()    ← ★ unpad 逻辑唯一实现处
│          └── :1067 prepare_model_outputs()   ← ★ packed logits → log_prob
└── :1297  @EngineRegistry.register(model_type="value_model", ...)
           class FSDPEngineWithValueHead(FSDPEngineWithLMHead)   ← 继承！
           └── :1303 prepare_model_outputs()   ← 只覆写输出处理（取 value 而非 logits）
```

**回答"actor/ref/critic 是不是各有一份 forward"：不是。**

- **actor 和 ref 用同一个 `FSDPEngineWithLMHead`**，区别只在传给 `forward_step` 的 `loss_function` 和 metadata（`compute_loss=False` / `calculate_entropy=False` / `no_lora_adapter=True`）；
- **critic 的 `FSDPEngineWithValueHead` 直接继承 LMHead 版**，只覆写 `prepare_model_outputs`——`prepare_model_inputs` 里的整套 unpad 逻辑**一行没重写**。

> ⚠️ **踩坑预警**：`UP/verl/trainer/ppo/rollout_corr_helper.py:54` 的 docstring 写着 "Used in `dp_actor.py` for distributed worker computations"——**这个文件在本快照里已经不存在了**（`verl/workers/actor/` 目录整个没了）。0.8 把 legacy 的 `dp_actor.py`/`dp_critic.py` 重构成了统一的 `workers/engine/` 抽象，但注释没跟上。**照着旧博客/旧教程找 `dp_actor.py` 会扑空**，这也解释了为什么网上大量 verl remove-padding 讲解和这份代码对不上。

---

## 2. unpad → forward → pad 完整调用链

### 2.1 `unpad_input` 来自哪个包

**flash-attn 的，但套了一层 verl 自己的 dispatcher。**

`UP/verl/utils/attention_utils.py:90` 是统一入口，内部惰性加载（`:25-38`）：

- CUDA → `from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input`
- NPU → `verl.utils.npu_flash_attn_utils`，再退到 `transformers.modeling_flash_attention_utils._unpad_input`

套这层的目的是昇腾 NPU 移植，CUDA 上就是 flash-attn 原版。另外 Megatron 路径直接硬 import flash-attn（`UP/verl/utils/megatron/pipeline_parallel.py:23`，注释 "flash 2 is a must for Megatron"）。

### 2.2 九跳调用链

| # | 位置 | 动作 |
|---|---|---|
| 1 | `UP/verl/trainer/ppo/ray_trainer.py:1221 / 1236 / 1261 / 1301 / 1338` | **driver 侧**调用 `left_right_2_no_padding(batch_td)`，分别在 `_compute_values` / `_compute_ref_log_prob` / `_compute_old_log_prob` / `_update_actor` / `_update_critic` |
| 2 | `UP/verl/workers/utils/padding.py:53` | `input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)` |
| 3 | `padding.py:56` | `torch.nested.nested_tensor_from_jagged(input_ids_rmpad.squeeze(-1), offsets=cu_seqlens)` —— **cu_seqlens 就此变成 NestedTensor 的 offsets** |
| 4 | `padding.py:54` | `indices` 存进 non-tensor data 备用 |
| 5 | `padding.py:58-67` | position_ids 单独 gather（见 §3） |
| 6 | worker `transformer_impl.py:955-960` | `input_ids_rmpad = input_ids.values().unsqueeze(0)` —— 从 nested 取出 packed 平铺值 `(1, total_nnz)` |
| 7 | `transformer_impl.py:1001-1006` | 组 `model_inputs = {"input_ids": ..., "attention_mask": None, "position_ids": position_ids_rmpad}`，注释写明 *"only pass input_ids and position_ids to enable flash_attn_varlen"* |
| 8 | HF 模型内部 | `attention_mask=None` + 打包的 position_ids → 触发 varlen 分支，HF/flash-attn **自己从 position_ids 重推 cu_seqlens** |
| 9 | `padding.py:99 no_padding_2_padding` | 输出切回 `(bsz, max_response_len)` |

### 2.3 `cu_seqlens` / `indices` / `max_seqlen` 分别去了哪

这是本节最容易记混的地方，逐个说：

**`cu_seqlens`** → 立刻被塞进 `nested_tensor_from_jagged(offsets=cu_seqlens)`（`padding.py:56, 79, 90`）。之后所有需要它的地方都用 `xxx.offsets()` 取回：
- `transformer_impl.py:1103, 1140` 重建输出的 nested tensor
- `padding.py:118-119` `prompt_ids.offsets().diff()` 反推每条序列长度
- **不再作为独立参数在函数间传递** —— 这是与经典写法最大的差别。

**`indices`** → 存进 non-tensor data（`padding.py:54`），只用于 `index_first_axis` 把**辅助的 per-token 张量**也打包成同样布局：
- `padding.py:77` `routed_experts`（MoE 路由回放）
- `padding.py:87-88` `teacher_logprobs` / `teacher_ids`（蒸馏）

  注意 **它没有被用来 pad 回去**——回程走的是 §2.4 的偏移量切片，不是 `pad_input(indices=...)`。

**`max_seqlen`** → **被丢弃了**。第 2 跳的 `*_` 吃掉了 `unpad_input` 返回的第 4、5 个值。verl 不显式携带它，需要时现算（`padding.py:146 build_attention_mask_from_nested` 里 `seq_lens.max()`）。

### 2.4 回程：`no_padding_2_padding` 的两件事

`padding.py:99-143` 不用 `pad_input`，而是自己算偏移量切片：

```python
sequence_lens   = prompt_lens + response_lens
sequence_offsets = sequence_lens.cumsum(dim=0)
for resp_len, seq_offset in zip(response_lens, sequence_offsets):
    pad_size = max_response_len - resp_len
    # left-shift model output by one token for log_probs/values
    response_list.append(F.pad(values[seq_offset - resp_len - 1 : seq_offset - 1], (..., 0, pad_size)))
output = torch.stack(response_list, dim=0)     # (bsz, max_response_len)
```

它同时干了两件事，缺一不可：
1. **从整条序列里切出 response 段**（丢掉 prompt 段的输出）；
2. **`-1` 的左移**：第 t 个位置的 logits 预测的是第 t+1 个 token。切片区间 `[seq_offset - resp_len - 1, seq_offset - 1)` 意味着**从最后一个 prompt token 开始取**——因为正是它的 logits 预测了第一个 response token。这就是 next-token 对齐点。

---

## 3. ★ position_ids 怎么处理（本次调查重点）

### 3.1 结论

**每段从 0 重置，成立。但不是按 cu_seqlens 逐段重置的**——是两步接力：

```
① padded 坐标下按行 cumsum 构造（天然每行从 0 起）
                ↓
② 按 attention_mask 逐行 gather 出有效位（拼接后自然形成 [0..L₀-1][0..L₁-1]...）
```

### 3.2 第一步：构造（`UP/verl/utils/model.py:240-241`）

```python
def compute_position_id_with_mask(mask):
    return torch.clip(torch.cumsum(mask, dim=-1) - 1, min=0, max=None)
```

**`dim=-1` 是关键**：沿每一行（每条序列）自己的序列轴做 cumsum，所以每条序列的计数器天然独立、天然从 0 开始。举个左 padding 的 prompt：

```
attention_mask : [0, 0, 1, 1, 1]
cumsum         : [0, 0, 1, 2, 3]
-1             : [-1,-1, 0, 1, 2]
clip(min=0)    : [0, 0, 0, 1, 2]     ← pad 位被压成 0（无所谓，下一步会被丢掉）
                       ↑ 第一个真实 token 正好是 0
```

`clip` 的作用就是把左 padding 产生的 `-1` 压回 0，避免负索引；pad 位的值是垃圾但不会被使用。

调用点：
- **agentic 路径**：`UP/verl/experimental/agent_loop/agent_loop.py:896`，在 `_compute_position_ids()` 里（纯文本、`self.processor is None` 时走这条）
- 通用路径：`UP/verl/trainer/ppo/padding_utils.py:45`
- rollout schemas：`UP/verl/workers/rollout/schemas.py:319`

### 3.3 第二步：打包（`UP/verl/workers/utils/padding.py:58-67`）

```python
position_ids_list = []
for i in range(attention_mask.shape[0]):
    curr_mask = attention_mask[i].bool()
    curr_pos_ids = position_ids[i]
    if curr_pos_ids.dim() == 1:          # (seq_len,)  纯文本
        valid_ids = curr_pos_ids[curr_mask]
    else:                                 # (4, seq_len) 多模态 mrope
        valid_ids = curr_pos_ids[:, curr_mask]
    position_ids_list.append(valid_ids)
position_ids_nested = torch.nested.as_nested_tensor(position_ids_list, layout=torch.jagged)
```

拼起来就是 `[0,1,...,L₀-1, 0,1,...,L₁-1, ...]`——**每段从 0 重置得到保证**。

### 3.4 三个值得注意的细节

1. **position_ids 走的是和 input_ids 不同的 gather 路径**。input_ids 用 flash-attn 的 `unpad_input`（返回 `indices`），position_ids 用**纯 Python for 循环 + 布尔索引**。两者结果必须一致（都由同一个 `attention_mask` 决定），但代码上是两条独立路径。这是个潜在的一致性风险点，也是个小的性能损失（batch 维度未向量化）。

2. **打包后的 position_ids 是 flash-attn varlen 的唯一线索**。第 7 跳传的是 `attention_mask=None`，HF 内部靠 position_ids 从 0 跳变来切分序列边界。**所以如果某条序列的 position_ids 没有从 0 开始，attention 就会跨样本泄漏**——这是 remove padding 最危险的正确性 bug，而 verl 靠"cumsum 天然从 0"规避了。

3. **多模态是 4 维 mrope**：`position_ids.dim() == 3` 时（`(bsz, 4, seq_len)`），`transformer_impl.py:957-958` 会 `unsqueeze(1)` 成 `(4, 1, total_nnz)`。Qwen2.5-VL 的 mrope 需要 (t, h, w) 三个空间维 + 1 个文本维。我们 Phase 3-D 做 VLM agentic 时会碰这条。

---

## 4. logits 在 packed 还是 padded 状态下算 log_prob

### 4.1 **在 packed 状态下算完，padded 的巨型 logits 从未存在**

`UP/verl/workers/engine/fsdp/transformer_impl.py:1111-1123`：

```python
logits_rmpad = output.logits.squeeze(0)          # (total_nnz, vocab_size)   ← packed！
logits_rmpad.div_(temperature_rmpad.clamp(min=1e-8).unsqueeze(-1).to(logits_rmpad.dtype))

inplace_backward = True
if calculate_entropy:
    inplace_backward = False
log_probs = logprobs_from_logits(
    logits=logits_rmpad,
    labels=input_ids_rmpad_rolled,               # :963 torch.roll(input_ids_rmpad, shifts=-1)
    inplace_backward=inplace_backward,
)
```

得到 `log_probs` 形状 `(total_nnz,)`，之后才由 `no_padding_2_padding` 变回 `(bsz, max_response_len)`。

**顺序是：packed logits → packed log_probs → padded log_probs。** logits 永远不会是 `(bsz, seq_len, vocab)`。

### 4.2 为什么这才是 remove padding 的主要收益

直觉上 remove padding 是"省 attention 的 FLOPs"，但显存账上 **logits 才是大头**：

| 张量 | padded 形状 | packed 形状 |
|---|---|---|
| hidden states | `bsz × seq × 4096` | `total_nnz × 4096` |
| **logits** | `bsz × seq × 151936` | `total_nnz × 151936` |

vocab 比 hidden_size 大 37 倍（Qwen3 词表 151936 vs hidden 4096），所以 logits 张量比任何中间激活都大一到两个数量级。

**按我们 Phase 0 smoke 配置估**（Qwen3-0.6B，vocab 151936，bf16）：
- 单条序列 prompt ~7000 + response ~800 ≈ 7800 token
- packed logits：`7800 × 151936 × 2B ≈ 2.4 GB`（micro_batch=1）
- 若 padded 到 `max_prompt(8192) + max_response(2048) = 10240`：`10240 × 151936 × 2B ≈ 3.1 GB`
- **省 ~23%**——因为老师的数据 prompt 长度极其整齐（P50/P99 只差 1.3%），填充率本来就高

> **反直觉但重要**：在 prompt 长度整齐的任务上，remove padding 的收益主要来自 **response 长度的方差**，不是 prompt。老师 `max_response_length=4096` 但多轮 agentic 的实际 response 长度分布很散——**这正是 Syncopate 关心的长尾**。所以：**长尾越严重，remove padding 省得越多，而这恰好和"异步化收益"同向**。Phase 1 量长尾分布时，可以顺手算出 remove padding 的实际填充率，作为长尾严重度的另一个佐证指标。

### 4.3 另外两层显存优化（配套读）

1. **`inplace_backward=True`**（`:1116`）：flash-attn 的 cross-entropy 支持反向时**原地把 logits 覆写成梯度**，省掉一份同样大的 `(total_nnz, vocab)` 缓冲。**但开 entropy 计算时必须关掉**（`:1114-1115`）——因为 entropy 还要再读一遍 logits。老师配置 `entropy_coeff=0`，`calculate_entropy` 因此为 False，`inplace_backward=True` 生效。**这是老师配置里一个隐藏的显存优化，改 entropy_coeff 会连带让显存涨一大截**，调参时要意识到。
2. **`use_fused_kernels`**（`:1092-1093`）：走 fused linear+CE 内核，`output.log_probs` 直接出，`:1080` 注释明说 *"fused kernels do not materialize the full logits tensor"*——logits 完全不落地。代价是拿不到 Σπ²（optimal baseline 估计器用）。
3. **`entropy_checkpointing` / `entropy_from_logits_with_chunking`**（`:157-165, 1127-1131`）：entropy 必须对整个 vocab 做 softmax，用重计算或分块来削峰。

---

## 5. agentic 场景下 response_mask 怎么跟着走

### 5.1 老师的 adapter 在**哪个坐标系**下构造 mask

**都不是** packed，**也不是** padded——是**"单条序列的局部坐标"**，一个随生成过程增长的普通 Python list：

`REL/train/verl_agent_loop_adapter.py`：

```python
response_mask: list[int] = []                              # :106

while len(response_mask) < self.response_length:           # :120
    ...
    response_mask.extend([1] * len(generated_ids))         # :165  模型生成 → 1
    ...
    response_mask.extend([0] * len(token_ids))             # :363  工具/反馈 → 0（在 _append_non_model_messages 内）
```

这是 verl AgentLoop 抽象最舒服的一点：**adapter 作者只需要在"这条轨迹的第几个 token"这个最直观的坐标系里思考**，pad→pack→pad 的三次坐标变换全部由框架负责。

### 5.2 三次坐标变换

| 阶段 | 坐标系 | 位置 |
|---|---|---|
| ① 构造 | 单序列局部，变长 list | `REL/train/verl_agent_loop_adapter.py:106, 165, 363` |
| ② 右 pad | `(1, response_length)` padded | `UP/.../agent_loop.py:750-756`，再 `:764` 乘 attention_mask 清掉 pad 位：<br>`response_mask = response_mask_output["input_ids"] * response_output["attention_mask"]` |
| ③ 进 loss | **仍然是 padded** `(bsz, max_response_len)` | `UP/verl/workers/utils/padding.py:71` |

### 5.3 ★ 关键发现：response_mask **根本没有被 unpad**

`padding.py:71` 只有轻飘飘一行：

```python
data["loss_mask"] = data["response_mask"]
```

函数 docstring（`:37`）也明说：*"we will remove `attention_mask`, `response` in the return data, but **`response_mask` is kept**"*。

也就是说 —— **mask 全程待在 padded 坐标系里，从没进过 packed 世界**。因为回程 `no_padding_2_padding` 已经把模型输出变回 `(bsz, max_response_len)`，两者在 loss 里天然对齐：

```python
# UP/verl/workers/utils/losses.py:60
log_prob = no_padding_2_padding(model_output["log_probs"], data)   # packed → padded
...
# 之后 policy loss 用 data["response_mask"]（padded）做 masked aggregation
```

**所以 packed 坐标系是完全封闭在 forward 内部的一段"内部表示"**，进去之前和出来之后都是 padded。这个设计的好处是：**写 AgentLoop 的人、写 loss 的人，都不需要知道 remove padding 的存在**。

### 5.4 唯一的例外：SFT loss 在 packed 坐标下 roll

`UP/verl/workers/utils/losses.py:45-50` 的 SFT 路径是另一套：

```python
loss_mask_flatten = torch.roll(loss_mask_flatten, shifts=-1, dims=0)
loss = -masked_sum(log_prob_flatten, loss_mask_flatten) / batch_num_tokens * dp_size
```

它**不回 padded 坐标**，而是在 packed 平铺状态下把 mask 整体左移一位来做 next-token 对齐。等价于 PPO 路径里 `no_padding_2_padding` 的 `-1` 切片（§2.4），但省掉了一次 pad 回去的开销。

> ⚠️ 注意 `torch.roll(dims=0)` 在**平铺**张量上做——序列 i 的最后一个位置会被卷到序列 i+1 的第一个位置。这在 SFT 里靠 loss_mask 恰好把每条序列的末位标 0 来消解。**如果我们改造 SFT 数据构造，破坏了"末位 mask=0"这个隐含前提，就会引入跨样本的静默污染。** 这是我在这次调查里发现的最隐蔽的坑。

### 5.5 agentic 特有的正确性检查清单

老师的 adapter 里 mask 的三处来源，改造时都要保持：

| token 类型 | mask | 位置 |
|---|---|---|
| 模型生成 | 1 | `:165` |
| 工具 observation | 0 | `:269` 注释 + `:275` 传入 `_append_non_model_messages` |
| parse_error 反馈 | 0 | `:204` 注释 + `:210` |
| 右 pad 位 | 0 | `agent_loop.py:764` 乘 attention_mask |

`response_logprobs` 也必须**同步补 0.0**（`:363-364`）保持等长，否则 §4.1 的 `rollout_log_probs` 会和 `response_mask` 错位 —— 这会直接毒化 [[02-train-inference-mismatch]] 里的 TIS 权重计算（log_ratio 逐 token 相减会对错位置）。

---

## 6. 仍不确定 / 待验证

1. **`left_right_2_no_padding` 在 driver 还是 worker 执行**：调用点在 `ray_trainer.py`（driver / single-controller），但 NestedTensor 要跨 Ray 序列化传给 worker。**打包后再传输是否比传 padded 张量更省网络**，没测；理论上是（少传 pad token），但 NestedTensor 的 Ray 序列化开销未知。
2. `padding.py:58-67` 的 Python for 循环在 batch 很大时是否成为 CPU 瓶颈——Phase 1 用 nsys 抓 timeline 时可以顺带看一眼。
3. `no_padding_2_padding:132` 有个断言 `assert not prompt_lens.eq(0).any()`（"seq_offset - resp_len - 1 assumes prompt_len > 0"）。**agentic 场景下 prompt 永远非空所以安全**，但如果我们改造出某种 prompt 为空的任务会炸。
4. Megatron 后端走的是另一套（`UP/verl/utils/megatron/pipeline_parallel.py:23-30`，直接用 flash-attn 且是经典 cu_seqlens 写法）。Phase 3-B 切 Megatron 时这套 NestedTensor 结论**不适用**，要重读。

---

## 7. 与 Syncopate 的关联

- **§4.2 的填充率就是长尾的一个免费度量**：Phase 1 建同步基线时，`total_nnz / (bsz × max_seq_len)` 直接反映 response 长度分布的离散程度。长尾越重 → 填充率越低 → remove padding 省得越多，**和异步化收益同向**。可以作为"这个任务值不值得异步化"的一个先导指标，成本几乎为零。
- **§5.5 的 mask/logprob 对齐**是我们自建任务时最容易静默出错的地方，且错了不会报错、只会让 reward 曲线诡异地不涨。Phase 0 验收标准里应该加一条：**dump 一条轨迹的 `token_trace`，逐段核对 mask 与 logprob 的长度和边界**（老师的 `token_trace` 落盘设计正是为这个准备的）。
- **§1.2 的 `dp_actor.py` 已消失**要写进 Phase 0 的读码笔记——网上绝大多数 verl remove-padding / FSDP 讲解都基于旧结构，照着找会浪费时间。
