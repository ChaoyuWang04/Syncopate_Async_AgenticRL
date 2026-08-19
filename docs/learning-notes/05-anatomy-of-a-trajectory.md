# 05 · 解剖一条真实轨迹

> 调查日期：2026-07-28
> 数据：`REL/data/sft/stage5/train.jsonl`（121 条 gold 轨迹）、`REL/data/rl/stage5/train.jsonl`（2273 条 prompt-only）
> **token 数全部由真实 tokenizer 算出**（本地 `/home/samwang/code/projects/models/Qwen3-0.6B`，走 Qwen3 chat template），不是字符估算。

---

## 0. 本篇要回答的问题

第二章的核心命题是「训练对象从 `P(y|x)` 切到 `π(a_t | s_t, h_t)`，最小训练单元是 trajectory 不是 response」。这句话落到 token 层面，就是一个很具体的问题：

> **一条序列里，哪些 token 是模型的选择（要算梯度），哪些是环境给的条件（不算梯度）？**

答案先放这里：**在 response 区，模型自己产的 token 只占 39%，另外 61% 是工具返回，全部被 mask 掉。**

---

## 1. 任务 A：一条完整轨迹的逐段解剖

**样本**：`LBL_FR_standard_order_given_d10`（121 条里的 P50，11 条消息 / 5 个 assistant 轮 / 4 次工具调用，难度 L3）
**业务**：法国市场客户说"帮我这单开退货标签"，Agent 查订单 → 查客户档案 → 查政策 → 建退货面单 → 回复客户。

### 1.1 逐段表

| # | role | 谁产的 | 内容摘要 | token | 进 response_mask 吗 |
|---|---|---|---|---|---|
| 0 | `system` | **模板拼装** | 售后 Agent 行为守则（960）+ **31 个工具 schema（4369）** | **5329** | ❌ 在 prompt 区 |
| 1 | `user` | **环境注入** | `ticket_id / customer_id / market=FR / order_id=O_6987DA` + 客户原话「帮我这单开退货标签。」 | 60 | ❌ 在 prompt 区 |
| — | *(生成头)* | 模板 | `<\|im_start\|>assistant\n` | — | — **prompt 到此为止** |
| 2 | `assistant` | 🟢 **模型生成** | `<think></think>` + `<tool_call>{"name":"oms.get_order","arguments":{"order_id":"O_6987DA"}}</tool_call>` | 36 | ✅ **mask=1** |
| 3 | `tool` | **环境注入** | `ok:true, status=delivered, paid_amount=80 EUR, items=[ITM_1×1]` | 100 | ❌ mask=0 |
| 4 | `assistant` | 🟢 **模型生成** | `<tool_call>` → `crm.get_customer(customer_id=C_E6BCAA)` | 34 | ✅ **mask=1** |
| 5 | `tool` | **环境注入** | `tier=standard, risk_flag=false, market=FR` | 59 | ❌ mask=0 |
| 6 | `assistant` | 🟢 **模型生成** | `<tool_call>` → `policy.search(market=FR, topic=return_label)` | 35 | ✅ **mask=1** |
| 7 | `tool` | **环境注入** | `policy_id=P_LABEL_FR`，规则「退货运费由 customer 承担」，例外「tier==plus 则 seller 承担」 | 91 | ❌ mask=0 |
| 8 | `assistant` | 🟢 **模型生成** | `<tool_call>` → **`returns.create_label`（写操作）**，6 个参数含 `shipping_paid_by=customer` | 91 | ✅ **mask=1** |
| 9 | `tool` | **环境注入** | `label_id=LBL_O_6987DA_g4, return_id=RET_..., status=created` | 74 | ❌ mask=0 |
| 10 | `assistant` | 🟢 **模型生成** | 面向客户的中文终答：「已为你生成退货面单…本次退货运费需由你承担。」 | 33 | ✅ **mask=1** |

### 1.2 合计

```
prompt 区   = 5389 token   (system 5329 + user 60)
response 区 =  553 token
   ├─ 模型生成 (mask=1) = 229  (41.4%)   ← 只有这些算梯度
   └─ 工具返回 (mask=0) = 324  (58.6%)
整条序列    = 5942 token
```

**模型真正"做决策"的 token 只有 229 个，占整条序列的 3.9%。** 其余 96% 全是环境给定的条件——system 指令、工具 schema、工单上下文、工具返回。

这就是「trajectory 而非 response」在 token 层面的样子：不是"一段连续的模型输出"，而是**模型 token 和环境 token 交错的一条链**，梯度只流经其中的 5 段。

### 1.3 `<tool_call>` 块和工具返回的确切边界

用真 tokenizer 渲染出的实况（第 2 条消息）：

```
<|im_start|>assistant
<think>

</think>

<tool_call>
{"name": "oms.get_order", "arguments": {"order_id": "O_6987DA"}}
</tool_call><|im_end|>
```

工具返回（第 3 条）：

```
<|im_start|>user
<tool_response>
{"ok": true, "result": {...}}
</tool_response><|im_end|>
<|im_start|>assistant
```

**两个反直觉的细节**：

1. **Qwen3 把 `role: "tool"` 渲染成 `user` 消息 + `<tool_response>` 包裹**，不是独立的 tool role。所以 verl 配置里的 `max_user_turns=8` 计的就是工具返回轮数。
2. **下一轮的生成头 `<|im_start|>assistant\n` 被算进"工具返回"这一段**（模板 `add_generation_prompt=True` 产生），因此它 **mask=0**。这是正确的——它是模板拼的，不是模型选的。

---

## 2. 任务 B：121 条轨迹的形态统计

### 2.1 全表

| 指标 | P10 | P50 | P90 | max | mean |
|---|---|---|---|---|---|
| assistant 轮数 | 3 | **5** | 8 | 10 | 5.3 |
| 工具调用次数 | 2 | **4** | 7 | 9 | 4.3 |
| 🟢 模型 token（mask=1） | 125 | **222** | 381 | 534 | 229 |
| ⬜ 工具 token（mask=0） | 181 | **347** | 590 | 837 | 366 |
| response 区总 token | 301 | **538** | 973 | 1371 | 595 |
| prompt 区 token | 5485 | **5487** | 5491 | 5515 | 5488 |
| **模型 token 占 response 比** | 0.337 | **0.381** | 0.440 | 0.541 | **0.390** |

### 2.2 三个关键读数

**① mask 掉 61%。** 模型 token 占 response 区的比例极其稳定：P10=0.34、P90=0.44，全体均值 0.39。也就是说 **loss 只作用在 response 区约五分之二的 token 上**，另外五分之三是环境注入。

- 对 GRPO：`compute_grpo_outcome_advantage` 把标量 reward 挂在最后一个 token 上再对整条求和，advantage 再 broadcast 回 `response_mask`——所以**有效的信用分配目标只有这 39%**。
- 对 [[02-train-inference-mismatch]] 的 TIS：`masked_sum` 求 `S = Σδ_t` 时只累加这 39%，所以 **T ≈ 222（P50）而不是 538**。这个数比我上次用字符估的 280 低 21%，需要修正 [[../syncopate/23-research-question]] §3 的表格。

**② prompt 长度方差几乎为零，response 长度方差 4.5 倍。**

```
prompt   : P10=5485 → max=5515      变化 0.5%
response : P10=301  → max=1371      变化 355%
```

因为 prompt 里 81% 是完全相同的工具 schema。**这对 remove padding 的收益判断是决定性的**（见 [[04-remove-padding]] §4.2）：填充浪费**全部来自 response 侧**，prompt 侧几乎没有可省的。按 `max_response_length=2048` 算，gold 轨迹的填充率只有 538/2048 = 26%——**remove padding 在这个任务上能省的主要就是这块**。

**③ 长尾在 response 侧，而且是"工具返回"和"模型生成"同步放大。** 两者的 P10→max 倍数分别是 4.3× 和 4.3×，比例反而稳定。说明长尾的来源是**轮数**，不是单轮变长。

### 2.3 工具调用分布

519 次调用，用到 24 个不同工具（schema 里有 31 个，**7 个从未在 gold 里出现过**）。Top-8：

| 工具 | 次数 | 占比 |
|---|---|---|
| `oms.get_order` | 119 | 22.9% |
| `policy.search` | 68 | 13.1% |
| `attachment.list` | 61 | 11.8% |
| `attachment.inspect` | 61 | 11.8% |
| `oms.list_orders` | 58 | 11.2% |
| `reshipment.create` | 22 | 4.2% |
| `payment.get_charge` | 17 | 3.3% |
| `finance.simulate_refund` | 15 | 2.9% |

分布高度符合 system prompt 里写死的行为守则：
- `oms.get_order` 几乎每条都调（"调查先于任何回复"）；
- `attachment.list` 和 `attachment.inspect` **次数完全相等（61/61）**——因为守则明确要求"有附件的先 list 再 inspect"，两者严格配对；
- 读工具（get/search/list/inspect）压倒性多于写工具，符合"先查证后动手"。

**这说明 gold 轨迹是按 system prompt 的规则生成/筛选的，工具使用模式是被规则强约束的，不是自由探索出来的。** RL 训练要学的正是这套约束。

### 2.4 parse_error / 重试痕迹：**一条都没有**

- 519 条工具 observation，**`ok != true` 的数量 = 0**；
- 单条 assistant 消息含多个 `tool_call` 的次数 = **0**（符合 `max_parallel_calls=1` 和守则"每步只输出一个 tool call"）；
- assistant 既无 content 又无 tool_calls 的次数 = **0**。

gold 数据是**纯净的成功路径**，没有任何错误恢复的样本。

> ⚠️ **这是个值得警惕的缺口**：老师的 adapter 里有一整套 parse_error 反馈循环（`verl_agent_loop_adapter.py:196-215` + `agent/prompts/tool_error_feedback.txt`），但 **SFT 阶段模型从没见过"工具报错→修正"的样本**。这条路径只能靠 RL 阶段自己踩出来。
> 对我们的 Phase 0 有直接影响：小模型（0.6B）大概率高频 parse_error，而它**没有任何 SFT 先验知道该怎么恢复**——所以「parse_error 100% 是正常的」这个判断依然成立，但要意识到这不只是模型小，也是数据里根本没教过。

---

## 3. 任务 C：RL 侧对照

### 3.1 一条 RL prompt 的 token 拆解（n=200 实测）

RL parquet 是 **prompt-only**，`prompt` 字段固定 2 条消息（system + user），工具 schema 由 AgentLoop 在 `apply_chat_template` 时注入。

| 成分 | token | 占比 |
|---|---|---|
| system prompt 正文（行为守则） | 960 | 17.5% |
| **31 个工具 schema** | **4457** | **81.2%** |
| user 业务上下文（工单 + 客户消息） | 70（P90=74, max=98） | **1.3%** |
| **合计** | **5487**（P10=5485, max=5515） | 100% |

**最反直觉的一条：真正的"这个 case 是什么"只有 70 个 token，占 1.3%。** 其余 98.7% 是每条样本完全相同的指令和工具定义。

含义：
- **KV cache 的 prefix 复用潜力极大**——5417 个 token 的前缀在所有样本间完全一致。vLLM 的 automatic prefix caching 理论上能省掉几乎全部 prefill。但注意 [[03-resource-scheduling]] §3.2 说过，每次 wake_up 后会 `reset_prefix_cache`（权重变了旧 KV 必须作废），**所以 colocate 模式下这个红利每个参数版本只能吃一轮**。异步分离模式下 rollout 权重更新更稀疏，反而更能吃到 prefix cache——**这是异步化的一个此前没被提及的额外收益，Phase 2 值得单独测**。
- 我上次（Task 02 §D2）估的"6000–7000 token"**偏高 15–25%**，实测 5487。`max_prompt_length=8192` 仍然安全（余量 49%），结论不变。

### 3.2 一轮循环里 prompt 怎么被拼长

模板文件在 `REL/agent/prompts/`：

**`system.txt`（960 token）** —— 纯静态，全部 case 共用。内容是行为守则，关键几条：
- "每步只输出一个 tool call"（对应 `max_parallel_calls=1`）
- "**调查先于任何回复**"、"客户的文字陈述本身不构成证据"（强制先读后写）
- "不要输出隐藏推理过程" ← **和 §4.3 的发现直接冲突**
- "如果工具返回 error，必须根据 error observation 修正下一步 tool call"

**`step_user.txt`（~70 token）** —— Jinja2 模板，唯一的 per-case 变量：

```jinja
当前客服会话：
ticket_id={{ case.ticket_id }}
customer_id={{ case.customer_id }}
customer_market={{ case.market }}
{% if case.order_id %}order_id={{ case.order_id }}
{% else %}order_id=客户尚未明确提供或当前会话未绑定{% endif %}

客户消息：
{{ case.customer_message }}
```

**逐轮增长过程**（`verl_agent_loop_adapter.py` 的 while 循环）：

```
第 0 轮  prompt_ids = [system + 31 tools + user + "<|im_start|>assistant\n"]   = 5487 token
                                                                    ↑ prompt 区到此冻结

第 1 轮  ├─ 模型生成 <think>…</think><tool_call>{…}</tool_call>      ~35 token  mask=1
         └─ 追加 <|im_start|>user\n<tool_response>{…}</tool_response>
              <|im_end|>\n<|im_start|>assistant\n                     ~90 token  mask=0
第 2 轮  同上 …
   …     直到模型输出不含 <tool_call> 的纯文本（= final reply）
         或撞上 max_assistant_turns=8 / response_length 预算
```

**关键**：verl 里 `prompt_ids` 只有第 0 轮那一次，**之后所有内容（模型生成的 + 工具返回的）全部进 `response_ids`**，靠 `response_mask` 区分。所以：
- `max_prompt_length` 只需要覆盖 5487；
- `max_response_length` 要覆盖**整条多轮对话的其余部分**——gold 的 P90 是 973、max 1371，老师设 4096 留了很大余量（给 RL 阶段的探索、错误重试和 thinking）。

**`tool_error_feedback.txt`** 在 parse_error 时以 **user 角色**插入（`adapter:204-215`），同样 mask=0：

```
上一个 tool call 无法执行。
工具 error：{{ error_observation }}
请根据当前 tool schema、稳定实体和 error observation，输出一个修正后的 tool call，或直接给出安全回复。
除非参数已经修正，否则不要重复同一个无效 tool call。
```

---

## 4. 与之前理解不符的地方

### 4.1 我之前的 token 估算普遍偏高 15–25%

| 量 | 之前（字符估算） | 实测 | 偏差 |
|---|---|---|---|
| RL prompt 总长 | 6000–7000 | **5487** | 高估 15–25% |
| 工具 schema | 4000–5500 | **4457** | 区间正确 |
| T（模型 token，P50） | ~280 | **222** | 高估 26% |

**要改的地方**：[[../syncopate/23-research-question]] §3 的安全边界表用的是 T=300/800，实测应改成 **T=222(P50) / 381(P90) / 534(max)**。方向不变（同步 colocate 下安全），但边界更宽松一点。

### 4.2 「response 区 61% 被 mask」这个数字之前完全没量过

之前只知道"工具 token 要 mask"，没量过比例。0.39 这个数同时决定三件事：
1. loss 的有效 token 数；
2. TIS 里 `masked_sum` 的求和长度 T；
3. remove padding 的收益基数。

### 4.3 ★ `<think>` 块的 mask 归属：SFT 和 RL **模板不一致**

这是本次最重要的发现，牵涉正确性。

**Qwen3 模板对 `enable_thinking` 的行为**（实测）：

```python
add_generation_prompt=True                        → "…<|im_start|>assistant\n"
add_generation_prompt=True, enable_thinking=False → "…<|im_start|>assistant\n<think>\n\n</think>\n\n"
```

即：`enable_thinking=False` 时**空 think 块由模板注入**（进 prompt，mask=0）；不传时**模型自己生成 think 块**（进 response，**mask=1**）。

**两侧的实际配置**：

| | 配置来源 | 结果 |
|---|---|---|
| **SFT** | `scripts/train_sft.py:94-95` 显式传 `data.enable_thinking_key=enable_thinking` + `data.enable_thinking_default=False`，数据每行 `enable_thinking: False` | 渲染出**空** `<think>\n\n</think>`，作为训练目标 |
| **RL (GRPO)** | `scripts/train_grpo_verl.py` **完全没传** `data.apply_chat_template_kwargs`；verl 默认 `{}`（`agent_loop.py:272`），`add_generation_prompt=True` 硬编码（`:354, :379`） | 生成头只到 `<\|im_start\|>assistant\n`，**模型自由生成 thinking，token 拿 mask=1** |

**三个后果**：

1. **SFT 教的是"输出空 think 块"，RL 允许的是"输出真实 thinking"。** 而 README 明说 GRPO 默认从 `models/original_model/Qwen3-8B` 冷启、不依赖 SFT 输出——那模型会按 Qwen3 的默认习惯输出真实 thinking。
2. **thinking token 占 `max_response_length` 预算，而且拿梯度。** gold 轨迹 response 区 P50 才 538 token；真实 RL rollout 里 thinking 很可能是 response 长度的主要来源，**也是长尾的主要来源**。这直接影响 [[../syncopate/23-research-question]] 的核心变量 T ——**T 的实际分布可能远比 gold 数据宽**。
3. **和 system prompt 冲突**：守则明写"不要输出隐藏推理过程"，模板默认却鼓励它。`strip_reasoning_blocks`（`agent/runtime.py:462-470`）只把 think 块从 `final_text` 剥掉（供 verifier 打分），**token 仍然全额进 loss**。

**Phase 0 必做的一次验证**：dump 一条真实 rollout 的 `token_trace`，看 `<think>` 块到底出现在哪个 segment、占多少 token、mask 是 1 还是 0。这是零成本的（老师的 `token_trace` 就是为这个设计的），而且直接决定我们对 T 的估计对不对。

如果确认 thinking 大量出现且不想要，一行 override 即可关掉：
```
+data.apply_chat_template_kwargs.enable_thinking=False
```

### 4.4 SFT 和 RL 的工具 schema **序列化不一致**（键序差异，88 token）

`REL/data/sft/stage5/train.jsonl` 里存的 `tools` 字段是 **`sort_keys=True` 字母序**序列化的：

```json
{"function": {"description": "...", "name": "approval.create_case", "parameters": {...}}, "type": "function"}
```

而运行时 `ToolFactory().tool_schemas()` 是**自然序**：

```json
{"type": "function", "function": {"name": "approval.create_case", "description": "...", "parameters": {...}}}
```

**内容完全相同**（dict 相等、JSON 字符数都是 13619），但**键序不同 → BPE 切分不同 → token 数差 88**（4369 vs 4457，1.6% 的 prompt）。

语义上无害，但意味着 **SFT 阶段模型看到的工具 schema token 序列和 RL rollout 时不完全一样**。属于 SFT→GRPO 冷启一致性上的一个小裂缝，量级很小，记录备查即可。

### 4.5 gold 数据没有任何错误样本（见 §2.4）

parse_error 恢复路径在 SFT 里**完全没被教过**，只能靠 RL 探索出来。

### 4.6 顺带：本地已有 Qwen3-0.6B

`/home/samwang/code/projects/models/Qwen3-0.6B/`（含完整 `model.safetensors` + tokenizer）。
之前 Task 02 §D2 说"本地没有任何模型、需下载 1.5GB"——**过时了**。Phase 0 的 smoke test 可以直接用它，`MODEL=/home/samwang/code/projects/models/Qwen3-0.6B`，省掉下载环节。

---

## 5. 一句话总结

一条轨迹在 token 空间里长这样：**5487 个 token 的固定条件（其中 81% 是工具 schema），后面跟着 553 个 token 的交错链，链里只有 39%（222 个 token）是模型的选择**。RL 要做的，就是用一个标量 reward 去调整这 222 个 token 的分布——**而它们分散在 5 个不连续的片段里，中间被 324 个不可控的环境 token 隔开**。这就是 agentic RL 和普通 `P(y|x)` 微调的根本差别。
