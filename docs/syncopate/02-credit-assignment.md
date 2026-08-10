# Syncopate · 02 — 步级信用分配的可行性探底

> 调查日期：2026-07-29
> 代码：`REL/envs/sandbox_state.py`、`REL/agent/verifier.py`、`REL/agent/runtime.py`、`REL/train/verl_agent_loop_adapter.py`、`UP/verl/workers/utils/losses.py`
> 数据：121 条 SFT gold 轨迹

---

## 0. 结论先行

**技术上可以直接做（零埋点），但在这个任务上信息量近乎为零。**

| 问题 | 答案 |
|---|---|
| 谓词翻转时刻拿得到吗 | ✅ **免费可得**。每条 sandbox 记录都带 `tool_call_id`，而 RL 路径下它必然是 `tc_{step}` |
| 区分度够吗 | ❌ **远远不够**。平均 5.29 步只翻 **0.61** 个谓词，**46.3% 的轨迹一个谓词都不翻**；写动作 88% 落在倒数一两步，同 intent 内步号标准差仅 0.4–0.9 |
| 有没有更好的步级信号 | ✅ **caps 归因**。9 个 cap 里 **6 个能定位到步**，而 caps 才是真正拉开 reward 差距的东西 |
| verl 接口支持吗 | 🟡 `response_mask` **不行**（被 `.to(bool)`），但 **`rollout_is_weights` 是现成的连续逐 token 通道** |
| 分叉（档位 2）可行吗 | ❌ **结构上做不了**。N 条 rollout 在 dispatch 前就 `repeat` 展开，各自独立跑到底 |

**判定：不是"做不了"，是"做了没用"** —— 至少在老师这个任务上。要做步级信用分配，正确的切入点是 **caps 归因**，不是谓词翻转。

---

## 1. 任务 A：谓词翻转是否可追溯

### 1.1 前提更正：`verifier_input_fact` 和 `idem_key` 都不存在

```
grep -rn "idem_key\|idem"        →  零命中
grep -rn "verifier_input_fact"   →  零命中
```

课件说的 `sandbox_refund_ledger` 里有 `idem_key = "R1-step7"` —— **代码里没有这个字段**。这和 [[../learning-notes/07-sandbox-environment]] §2.3 的结论一致：**整套代码没有 idempotency 概念**。

同样地，[[../learning-notes/06-verifier-and-reward]] §3.1 已确认 `verifier_input_fact` / `policy_matched` / `risk_checked` / `evidence_attached` 四个字段里只有 `refund_issued` 真实存在。

**真实的"谓词"是 `WRITE_TOOL_FACTS` 的 17 个 fact key。**

### 1.2 每个谓词依赖 sandbox 的哪张表

`envs/sandbox_state.py:32-67`，17 个写工具 → (台账, 谓词) 的完整映射：

| 谓词（fact key） | 台账（ledger） | 台账形状 | 写工具 |
|---|---|---|---|
| `refund_issued` | `sandbox_refund_ledger` | 事件流 | `finance.issue_refund` |
| `return_label_created` | `sandbox_returns` | 事件流 | `returns.create_label` |
| `reshipment_created` | `sandbox_reshipments` | 事件流 | `reshipment.create` |
| `order_cancelled` | `sandbox_order_state` | **dict（覆盖）** | `oms.cancel_order` |
| `order_modified` | `sandbox_order_state` | **dict（共用！）** | `oms.modify_order` |
| `carrier_investigation_opened` | `sandbox_carrier_investigation` | 事件流 | `carrier.open_investigation` |
| `shipment_intercept_requested` | `sandbox_carrier_intercept` | 事件流 | `tms.intercept_shipment` |
| `shipment_reroute_requested` | `sandbox_carrier_reroute` | 事件流 | `tms.reroute_shipment` |
| `approval_created` | `sandbox_approval_cases` | 事件流 | `approval.create_case` |
| `payment_dispute_opened` | `sandbox_payment_disputes` | 事件流 | `payment.open_dispute_case` |
| `invoice_updated` | `sandbox_invoice_changes` | 事件流 | `invoice.update_vat` |
| `subscription_cancelled` | `sandbox_subscription_state` | **dict（覆盖）** | `subscription.cancel` |
| `security_case_opened` | `sandbox_security_cases` | 事件流 | `account.update_security_case` |
| `message_sent` | `sandbox_message_log` | 事件流 | `message.reply` |
| `ticket_closed` | `sandbox_ticket_state` | **dict（共用）** | `ticket.close` |
| `ticket_handoff` | `sandbox_ticket_state` | **dict（共用）** | `ticket.handoff` |

**"依赖哪些行"的答案**：`records_for_tool(tool, namespace_id)` 三层过滤（`sandbox_state.py:188`）——① 取台账 ② 按 `namespace_id` 过滤 ③ 共用台账再按 `fact` 字段 / `tool` 名细分。

> ⚠️ **三张 dict 台账是"覆盖式"的**（`DICT_LEDGERS`，`sandbox_state.py:69-73`）：同一对象 key 后写覆盖先写，**只留最新态**。所以在 `sandbox_order_state` 上先 cancel 再 modify，**cancel 的记录会被物理抹掉**——连带它的 `tool_call_id` 一起没了。**这是谓词翻转历史的一个真实丢失点。**

### 1.3 ★ A2：step 号可靠吗 —— **可靠，但机制不是 `idem_key`**

每条 sandbox 记录的 enriched 字段里都有 **`tool_call_id`**（`sandbox_state.py:150`），审计日志里也有（`:180`）。所以问题变成：**`tool_call_id` 能不能反推 step？**

**RL 路径（我们关心的）：能，而且必然能。**

```python
# train/verl_agent_loop_adapter.py:228
tool_call_id = tool_call.get("id") or _tool_call_id(assistant_turns, call_index, len(tool_calls))

# :379-384
def _tool_call_id(step, call_index, total_calls):
    return f"tc_{step}" if total_calls == 1 else f"tc_{step}_{call_index}"
```

看起来 `tool_call.get("id")` 优先、有被模型污染的风险。**但不会**——`parse_tool_calls`（`agent/runtime.py:52-92`）明确只保留两个字段：

```python
# 只保留 runtime 后续真正需要的字段；provider 原生 tool_call 的 id 在主循环中另行兼容。
tool_calls.append({"name": name, "arguments": arguments})
```

**`id` 被显式丢弃。** 所以 RL 路径下 `tool_call.get("id")` 恒为 `None`，`tool_call_id` **必然**是 `tc_{step}`。

而 `assistant_turns` 在 `:228` 处已经自增过（生成后立即 `+= 1`，`:154`），所以它就是该 tool_call 所属 assistant 轮的 1-indexed 步号。

**唯一的例外**：同一步多个 tool_call 时格式变成 `tc_{step}_{idx}` —— 但那种轨迹会被 `multi_tool_per_step_cap` **直接判 reward=0**，不进入训练信号，无所谓。

**gold 数据：也能，已实测验证。** gold 的 `tool_calls[].id` 是 `g1/g2/g3/...`，我逐条核对了 121 条轨迹的全部 519 次调用：

```
gold 里 tool_call id (g<N>) 与 step 不一致的次数: 0
```

**完全单调对应。**

**另有一条更直接的路径**：`trajectory.parsed_actions` 里**每个 action 都显式带 `"step"` 字段**（`adapter:230`）：

```python
parsed = {"step": assistant_turns, "tool_call_index": call_index,
          "tool_call_id": tool_call_id, "name": ..., "arguments": ...}
```

而 `parsed_actions` 和 `tool_observations` 通过 `tool_call_id` 关联。**所以「第 k 步执行了什么工具、成功没有、翻了哪个谓词」这条链是完整的，一行埋点都不用加。**

### 1.4 A3 结论：**免费可得**

需要的信息全在，两条独立路径互为冗余：

```
路径 1（推荐）：trajectory.parsed_actions[i]["step"]  ← 显式字段
                └─ tool_call_id ─→ tool_observations[j]（看 ok）
                                └─→ sandbox 记录（看 verified_fact_key）

路径 2（兜底）：sandbox 记录的 tool_call_id → 正则 ^tc_(\d+) → step
```

**改动量：0 行埋点。** 只需要一个约 20 行的解析函数把它们拼起来。

**唯一的真实损失**：§1.2 提到的三张 dict 台账覆盖式写——如果同一对象被写两次，前一次的记录（含其 `tool_call_id`）被抹掉。但 **`sandbox_audit_log` 是 append-only 的，保留了全部写历史**（`sandbox_state.py:170-181`），可以从那里恢复。

---

## 2. 任务 B：gold 轨迹实测

### 2.1 ★ B1：翻转极其稀疏，和预期差一个数量级

| 指标 | 实测 |
|---|---|
| 轨迹数 | 121 |
| 总步数 | mean **5.29**，P50 5，max 10 |
| **翻转谓词的步数** | mean **0.61** |
| 翻转步数分布 | `{0 步: 56 条, 1 步: 56 条, 2 步: 9 条}` |
| **完全不翻转的轨迹** | **56 / 121 = 46.3%** |
| **翻转步占全部步的比例** | **9.6%** |

**预期是"5-6 步里翻 4-5 个"，实测是"5.29 步里翻 0.61 个"——差了约 7 倍。**

原因很清楚：**近一半的 case 根本不需要写动作**。它们是 inform / deny / clarify 类——查清楚了直接回复客户。这和 verifier 的 `calculate_outcome` 三分支设计完全吻合（[[../learning-notes/06-verifier-and-reward]] §2.3）：

```python
if has_write and has_info:  return 0.75*write + 0.25*info
if has_write:               return write_score
return info_score           # ← 46.3% 的轨迹走这条
```

**对步级信用分配的直接含义**：**近一半的轨迹上，"谓词翻转"这个信号完全不存在，全部 reward 来自 `info_score`（LLM 判文字覆盖）——那是个轨迹级信号，天然无法拆到步。**

### 2.2 B2：哪些工具从不翻转谓词

519 次调用里，**读工具 445 次（85.7%）、写工具 74 次（14.3%）**。

**从不翻转的（读工具，按频次）**：

| 工具 | 次数 | 是"准备"还是"无用" |
|---|---|---|
| `oms.get_order` | 119 | **准备**——几乎每条轨迹第一步，守则要求"调查先于回复" |
| `policy.search` | 68 | **准备**，且**直接被 `policy` 子分（0.20）计分** |
| `attachment.list` | 61 | **准备**，与 inspect 严格 1:1 配对 |
| `attachment.inspect` | 61 | **准备** |
| `oms.list_orders` | 58 | **准备**——order_id 未知时定位订单 |
| `payment.get_charge` | 17 | 准备 |
| `finance.simulate_refund` | 15 | **准备，且是强制前置**（`missing_dry_run_cap` 0.25） |
| `crm.get_customer` | 14 | 准备 |
| 其余 9 个 | 各 ≤8 | 准备 |

**判断：gold 轨迹里没有"无用步骤"。** 三条独立证据：

1. **verifier 的 `evidence` 子分（权重 0.20）直接数 `required_read_tools` 的完成率**——读工具本身就是被计分的目标，不是纯开销；
2. **`efficiency` 子分会扣冗余步**（`1 - 0.05·max(0, 实际步 - expected)`，其中 `expected = |required_read_tools| + |required_side_effects|`），gold 是按最优路径构造的，不含冗余；
3. `attachment.list` 和 `inspect` 次数**完全相等（61/61）**，说明是被守则严格约束的配对，不是自由探索。

**所以"读工具不翻谓词"不等于"这一步没价值"——它的价值已经被 `evidence` 和 `policy` 两个子分（合计权重 0.40）显式定价了。** 这本身就削弱了"用谓词翻转做步级信号"的动机：**verifier 已经在用另一套（更细的）方式给读步骤定价了。**

### 2.3 ★ B3：区分度 —— **低到不足以支撑步级信号**

写动作发生在第几步（74 次写，绝对步号）：

```
step:  3   4   5   6   7   8   9
次数:  1  19  21  14  11   7   1
```

- **相对位置 mean = 0.83**（1.0 = 最后一步）
- **88%（65/74）的写发生在倒数第一或第二步**

同一 `primary_intent` 内的步号分布（这是"同类任务下写动作位置是否固定"的直接度量）：

| intent | n | 步号分布 | **std** |
|---|---|---|---|
| `wrong_item_received` | 14 | `{5:3, 6:6, 7:4, 8:1}` | 0.86 |
| `damaged_item_refund` | 12 | `{7:5, 8:6, 9:1}` | 0.62 |
| `payment_dispute_or_chargeback` | 8 | `{4:5, 5:3}` | 0.48 |
| `duplicate_charge` | 8 | `{4:2, 5:5, 6:1}` | 0.60 |
| `reshipment_or_replacement` | 7 | `{5:2, 6:5}` | 0.45 |
| `invoice_vat_change` | 6 | `{4:5, 5:1}` | 0.37 |
| `return_label_or_pickup_issue` | 4 | `{4:2, 5:2}` | 0.50 |
| `warranty_or_repair` | 3 | `{6:1, 7:2}` | 0.47 |

**标准差全部在 0.37–0.86 之间，即"同类任务的写动作位置几乎固定，只在相邻 1 步内浮动"。**

**结论（回答 B3）**：**结构高度固定，步级信号的区分度很低。** 轨迹几乎都是同一个模板：

```
[读] [读] [读] … [写]? [回复]
 ↑ 变长的调查阶段        ↑ 位置几乎确定
```

如果给"翻转谓词的那一步"加权，**等于给"倒数第二步"加权**——那不是信用分配，那是位置先验。

### 2.4 ⚠️ 一个必须标注的局限

**gold 轨迹是最优路径，它系统性低估了真实 RL rollout 的方差。** 真实 rollout 会有：

- parse_error 重试（gold 里 **0 次**，见 [[../learning-notes/05-anatomy-of-a-trajectory]] §2.4）
- 调错工具、走弯路
- 重复调用（触发 `duplicate_side_effect_cap`）
- 撞 `max_assistant_turns=8` 截断

**所以上面的 std ≈ 0.5 是下界。** 真实 rollout 的步号方差会更大——但**方差变大主要来自"垃圾步"而非"有意义的路径差异"**，那不会让步级信用分配更有价值，反而更嘈杂。

**这一条必须在 Phase 1 用真实 rollout 复测**（用 `token_trace` + `parsed_actions` 就能算，零成本）。

---

## 3. 任务 C：caps 的归因能力

**这是本次调查最有价值的发现：caps 比谓词翻转更适合做步级信号。**

| cap | 封顶 | 知道是哪一步吗 | 依据 |
|---|---|---|---|
| `multi_tool_per_step_cap` | 0.0 | ✅ **知道，但丢弃了** | `_has_multi_tool_step` 按 `action["step"]` 分组计数，最后 `any(c>1)` 把"哪个 step"扔了（`verifier.py:92-102`） |
| `missing_dry_run_cap` | 0.25 | ✅ **知道，但丢弃了** | `enumerate(actions)` 拿到了索引 `i`，`return True` 时丢掉（`verifier.py:117-121`） |
| `customer_harm_cap` | 0.25 | ✅ **可恢复** | 遍历 `records_for_tool("returns.create_label")`，每条 record 带 `tool_call_id`（`:908-931`） |
| `wrong_object_cap` | 0.25 | ✅ **可恢复** | `write_consistency_caps` 逐条遍历 records（`:933-961`） |
| `duplicate_side_effect_cap` | 0.30 | ✅ **可恢复** | `records_for_tool` 返回的多条记录各带 `tool_call_id` |
| `unauthorized_action_cap` | 0.30 | 🟡 **需回查** | 集合差只留 tool 名，但可用 tool 名回查 `records_for_tool` |
| `wrong_policy_cap` | 0.45 | 🟡 **部分** | policy 子分分支只留 `called/correct` 布尔；但 `write_consistency` 分支遍历 records 可恢复 |
| `missing_evidence_cap` | 0.55 | 🔴 **本质不可归因** | "该查的没查"是**缺失**，不是某一步的动作 |
| `false_promise_cap` | 0.35 | 🔴 **本质不可归因** | `claimed(来自 final_text) − executed(全轨迹)` 的集合差，是**轨迹级属性** |

**汇总：6/9 可定位到步，1/9 部分可，2/9 本质不可。**

### 为什么 caps 是更好的步级信号

1. **caps 才是拉开 reward 差距的东西。** 子分加权和最多把 raw 拉到 [0,1] 连续变化；而 cap 一命中直接把最终 reward 摁到 0.25–0.55（[[../learning-notes/06-verifier-and-reward]] §4）。**归因 cap = 归因了 reward 的主要方差来源。**
2. **cap 对应的是"具体的错误动作"**，天然是步级的：哪一步写错了对象、哪一步重复退款、哪一步一步发了多个工具。而谓词翻转对应的是"正确动作"，位置几乎固定（§2.3）。
3. **两个"知道但丢弃了"的 cap 是最容易摘的果子**：`_has_multi_tool_step` 和 `_issued_refund_without_dry_run` 内部**已经算出了 step**，只是 return 布尔值时扔掉了。改成返回 `(bool, step)` 各是 **2–3 行**。

**最小改造方案（约 30 行）**：给 `score_trajectory` 的返回值加一个 `cap_steps: dict[str, list[int]]`，让 `cap_reasons` 从"文本原因"升级成"文本原因 + 责任步号"。**不影响任何现有逻辑，纯增量。**

---

## 4. 任务 D：verl 接口够不够

### 4.1 ★ D1：`response_mask` 只能是 0/1，但**有一个现成的连续通道**

**坏消息**：PPO 路径显式把 mask 转成 bool（`UP/verl/workers/utils/losses.py:94`）：

```python
response_mask = data["response_mask"].to(bool)   # ← 连续权重在这里被销毁
```

同样的 `.to(bool)` 出现在 `:52`（sft_loss 的 padded 分支）和 `:166`（value loss）。所以**往 `response_mask` 里塞 `c_t ∈ [0,2]` 会被直接抹平成 1**。

底层其实支持——`masked_sum` / `masked_mean` 的 docstring 明写 "Boolean **or numeric** mask"，`agg_loss` 内部是 `masked_sum(loss_mat, loss_mask)` 和 `loss_mat * loss_mask`（`core_algos.py:1168-1196`），**纯数值运算**。**是 `losses.py:94` 那一行卡住了，不是数学卡住了。**

**好消息：`rollout_is_weights` 就是一个现成的、端到端打通的连续逐 token 乘子通道。**

```python
# UP/verl/trainer/ppo/core_algos.py:1357-1358
if rollout_is_weights is not None:
    pg_losses = pg_losses * rollout_is_weights     # ← 无任何 bool 转换
```

它的完整链路（见 [[../learning-notes/02-train-inference-mismatch]] §4）：
```
rollout_corr_helper 算出 (bs, seq_len) 张量
  → batch.union() 进 DataProto
  → losses.py:88-90 select 进 fields
  → :98 data.get("rollout_is_weights")
  → :112 传给 policy_loss_fn
  → core_algos:1357 逐 token 相乘
```

**所以实现步级权重 `c_t` 有两条路**：

| 方案 | 改动 | 风险 |
|---|---|---|
| **A. 复用 `rollout_is_weights`** | 在 `compute_rollout_correction_and_add_to_batch` 之后把 `c_t` 乘进去 | 🔴 **和 TIS 语义冲突**——它已经承载了重要性采样权重，两者相乘会污染 `rollout_corr/*` 全部诊断指标，也破坏 [[00-research-question]] 的 ESS 分析 |
| **B. 新增平行字段 `step_credit_weights`** | ① `losses.py:88` 加 2 行 select ② `:112` 传参 ③ `core_algos:1358` 后加 1 行乘法 ④ AgentLoop 侧产出该张量 | 🟢 **推荐**。约 10 行框架改动 + adapter 侧生成逻辑 |

**方案 B 的关键实现点**：`c_t` 必须在 AgentLoop 里生成并塞进 `AgentLoopOutput.extra_fields`，然后由 `_agent_loop_postprocess` pad 成定长（和 `response_logprobs` 完全同构，`agent_loop.py:758-761` 就是现成模板）。**因为只有 AgentLoop 知道每个 token 属于第几步**（`token_trace["segments"]` 里已经有 `step` 字段了）。

> 顺带一提：`token_trace` 的每个 segment 都带 `{"type", "step", "token_count", "mask"}`（`adapter:109-113, 175-185, 366-376`）。**这就是 token → step 的映射表，老师已经落盘了，不用自己建。**

### 4.2 D2：AgentLoop **不支持 prefix 分叉** —— 档位 2 结构上做不了

N 条 rollout 在**进 AgentLoop 之前**就展开了：

```python
# UP/verl/trainer/ppo/ray_trainer.py:1448
gen_batch_output = gen_batch.repeat(repeat_times=rollout_n, interleave=True)
```

之后每个副本走**完全独立**的 AgentLoop：独立的 `request_id = uuid4().hex`、独立的 `SandboxState`、独立的 `namespace_id`。`AgentLoopWorker` 对每条样本起一个协程，**1 输入 → 1 `AgentLoopOutput`**。

**技术上状态快照是可行的**——`SandboxState.export()` 是 deepcopy，`SandboxState(initial, namespace_id)` 能重建；`messages` / `all_token_ids` 都是普通 Python 对象。**真正的阻塞是 verl 的批次契约**：`_agent_loop_postprocess` 假设 1 输入 1 输出，分叉会产生 N 输出，破坏 batch 对齐、破坏 `sample_index` → GRPO 分组的映射（`agent_loop.py:1122-1142`）。

**要做分叉，得改的是 verl 的 AgentLoopWorker 批次契约，不是老师的 adapter。** 这远超"加埋点"的量级。

> 补充：`experimental/` 下没有任何 tree-search / branching recipe（只有 agent_loop / fully_async_policy / one_step_off_policy / separation / reward_loop）。**上游也没有这个能力。**

---

## 5. 最终判定与建议路线

### 判定

> **步级信用分配在这套代码上「可以直接做，但不该用谓词翻转做」。**

- ✅ **可以直接做**：step 号零埋点可得（§1.3），token→step 映射已落盘（`token_trace`），verl 侧加 ~10 行就能吃连续权重（§4.1）
- ❌ **不该用谓词翻转做**：46% 轨迹零翻转、翻转步只占 9.6%、位置方差 <1 步（§2）
- ✅ **应该用 caps 归因做**：6/9 可定位到步，且 caps 是 reward 方差的主要来源（§3）
- ❌ **档位 2（分叉）结构上做不了**：需要改 verl 的批次契约（§4.2）

### 建议路线（按性价比）

**第 0 步（Phase 1 必做，零成本）**：用真实 rollout 复测 §2 的三个数字。gold 是最优路径，系统性低估方差。用 `parsed_actions[i]["step"]` + `token_trace` 就能算，**不需要任何改动**。**如果真实 rollout 的翻转率仍然 <15%，就应该彻底放弃谓词路线。**

**第 1 步（最容易摘的果子，~30 行）**：给 verifier 加 `cap_steps` 输出。`_has_multi_tool_step` 和 `_issued_refund_without_dry_run` 内部**已经算出了 step**，只是 return 时丢了；另外 4 个 cap 遍历 records 时顺手记 `tool_call_id` 即可。**纯增量，不改任何现有逻辑。**

**第 2 步（~10 行框架 + adapter 逻辑）**：新增 `step_credit_weights` 平行字段（**不要复用 `rollout_is_weights`**，会污染 TIS 诊断）。

**第 3 步（实验）**：最朴素的形态——`c_t = 1.0` 默认，命中 cap 的那一步的 token `c_t = α`（α > 1，放大惩罚）。对照组是当前的均匀广播。

### 与 Syncopate 主线的关系

这条线**和异步化研究是正交的**，但有一处交汇值得注意：

**步级信用分配会加剧 sequence-level TIS 的问题。** 如果 `c_t` 让不同步的 token 权重差异变大，而 [[00-research-question]] §8 的 `S = Σδ_t` 是无权求和——**两者对"一条轨迹内 token 不等价"这件事的处理是不一致的**。真要同时上，得先想清楚 TIS 的求和是否也该加权。

**建议：Phase 2 先专注异步化主线，步级信用分配作为 Phase 3 的可选模块。** 但**第 0 步的测量应该在 Phase 1 顺手做掉**——它零成本，而且结论（翻转率）本身就是这个任务的一个有价值的刻画。
