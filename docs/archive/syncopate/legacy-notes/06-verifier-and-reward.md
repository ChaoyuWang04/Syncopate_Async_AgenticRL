# 06 · Verifier 与 Reward 设计的完整拆解

> 调查日期：2026-07-28
> 代码：`REL/agent/verifier.py`（1196 行）、`REL/schemas/reward_schema.py`、`REL/envs/sandbox_state.py`
> 版本标识：`verifier_simple_v1`
> **本篇一律以代码为准。课件与代码不符处单独列在 §5。**

---

## 0. 三句话结论

1. **是五维不是六维。** `write` 和 `info` 不是并列子分，它俩合成 `outcome`。"风控"根本不是子分——它是 caps 机制。
2. **打分同时看过程和结果，而且过程分的权重更高（0.50 vs 0.45）——但 caps 全部看结果，且能把总分压到 0.25。** 所以准确的说法是：**过程分决定你能拿多少，sandbox 结果决定你的上限。**
3. **`confidence` 在当前版本硬编码为 1.0。** 课件里那个 `× conf 0.72` / `× conf 0.88` 在代码里不存在。

---

## 1. 打分骨架

```python
# agent/verifier.py:229-236
raw_reward = Σ subscores[name] * WEIGHTS[name]           # 五维加权和
cap_value  = min(ACTIVE_CAP_VALUES[c] for c in active_caps, default=1.0)
confidence = 1.0                                          # ← 硬编码，注释："留作未来扩展位"
reward     = min(raw_reward, cap_value) * confidence
```

一条快捷通道：**同一 step 输出多个 tool_call → 整条直接 return reward=0**（`:161-176`），连 judge 都不调。零容忍。

---

## 2. 任务 A：五维 subscore

### 2.1 课件 ↔ 代码 对照表

| 课件维度 | 代码字段 | 课件权重 | **代码权重** | 备注 |
|---|---|---|---|---|
| 结果 | `outcome` | 0.4 | **0.45** | = 0.75·write + 0.25·info |
| 政策 | `policy` | 0.2 | **0.20** | ✅ 一致 |
| 证据 | `evidence` | 0.15 | **0.20** | 差 0.05 |
| 效率 | `efficiency` | 0.1 | **0.10** | ✅ 一致 |
| 沟通 | `communication` | 0.1 | **0.05** | 差一半 |
| **风控** | **不存在** | 0.05 | **—** | 风控由 **caps 机制**实现，不是子分 |

代码里的权威定义（`verifier.py:52-58`，与 `schemas/reward_schema.py` 一致）：

```python
WEIGHTS = {"outcome": 0.45, "policy": 0.20, "evidence": 0.20,
           "efficiency": 0.10, "communication": 0.05}   # 合计 1.0
```

> ⚠️ 另需修正一个提法：**`write` 和 `info` 不是六维中的两维。** `calculate_outcome`（`:683`）按三种情况合成：
> - 既有必做写又有信息点 → `outcome = 0.75·write + 0.25·info`
> - 只有写 → `outcome = write`
> - 只有信息点（inform / deny / escalate 类 case）→ `outcome = info`

### 2.2 ★ A1：每一维的输入是过程还是结果

**这是本次调查最关键的问题。逐维用代码证实：**

| 维度 | 输入来源 | **过程 / 结果** | 判定方式 | 代码位置 |
|---|---|---|---|---|
| **write**（outcome×0.75） | **sandbox 台账** `records_for_tool()` + env 真值 | 🔴 **纯结果** | 纯规则 | `:568-637` |
| **info**（outcome×0.25） | `final_text`（LLM 盲抽）+ env 真值（规则比） | 🔴 **结果**（文字产出） | LLM 抽 + 规则比 | `:638-682` |
| **policy** | `facts.tool_calls` 里 `policy.search` 的参数与返回 | 🔵 **纯过程** | 纯规则 | `:701-740` |
| **evidence** | `facts.tool_calls` 的工具名 + `ok` 标志 | 🔵 **纯过程** | 纯规则 | `:743-769` |
| **efficiency** | `facts.tool_calls` + `trajectory.tool_errors` | 🔵 **纯过程** | 纯规则 | `:771-810` |
| **communication** | LLM 对 `final_text` 的 `forbidden_hits` / `clear` | 🟡 结果（文字风格） | **纯 LLM** | `:812-822` |

**按权重汇总：**

```
看结果（sandbox / final_text）: outcome 0.45  +  communication 0.05  =  0.50
看过程（tool_calls 序列）     : policy 0.20 + evidence 0.20 + efficiency 0.10 = 0.50
其中真正读 sandbox 的只有 write: 0.45 × 0.75 = 0.3375
```

**所以"主要看 sandbox 状态"这个判断需要修正——单看 subscore 权重，过程分（0.50）不低于结果分。**

**但这不是全部，caps 才是决定性的。** 9 个 cap 里有 6 个直接读 sandbox 台账：

| cap | 读什么 | 封顶 |
|---|---|---|
| `false_promise_cap` | `claimed − executed_write_tools`（sandbox） | 0.35 |
| `unauthorized_action_cap` | `executed_write_tools`（sandbox） | 0.30 |
| `duplicate_side_effect_cap` | `records_for_tool()` 条数（sandbox） | 0.30 |
| `customer_harm_cap` | sandbox 记录字段 vs policy 真值 | 0.25 |
| `wrong_object_cap` | sandbox 记录的 id vs `case.entities` | 0.25 |
| `missing_dry_run_cap` | `parsed_actions` 顺序（过程） | 0.25 |

**cap 把 reward 压到 0.25–0.55，覆盖力远超任何单一权重。** 一条过程分满分（policy+evidence+efficiency = 0.50）的轨迹，只要 sandbox 结果出错（比如伤客），最终 reward 也只有 0.25。

> **✅ 最终结论（回答 A1）**：
> **过程分决定"你能拿多少"，sandbox 结果决定"你的上限是多少"。**
> 权重层面过程与结果各占一半；但 caps 层面几乎全部基于 sandbox 结果，且封顶值（0.25–0.55）比子分能拉开的差距更狠。
> 所以「主要看结果」在**决定性**意义上成立，在**权重**意义上不成立。

### 2.3 A2：各维的计算逻辑

**write_score（纯规则，查 sandbox）** `:568`

```
write_score = Σ_target [ sandbox 有该写工具的记录 AND required_correct 每条字段 == env 真值 ] / |required_side_effects|
```
- "做了" = sandbox 台账有记录（**dry-run / 失败不写台账**）
- "做对了" = `required_correct` 里每条 `{sandbox_field: value_source}` 都满足 `record[field] == resolve(value_source)`
- `required_correct` 为空 `{}` → 发生即算对（**这是个漏洞**，由 `write_consistency_caps` 补，见 §3.3）
- 无 `required_side_effects` → 记 1.0（outcome 改走 info）

**info_score（LLM 盲抽 + 规则比对）** `:638`

```
info_score = Σ_point [ covered AND (无 value_source OR stated_value == env 真值) ] / |required_response_points|
```
- coverage point：LLM 判 covered 即命中
- valued point：还要 LLM 抽出的 `stated_value` 等于规则侧解析的真值（说了但值错 → 不命中）

**policy_score（纯规则，四档）** `:701`

| 情况 | 分 |
|---|---|
| `policy_required=false` | 1.0 |
| 需要但没调 `policy.search` | 0.0 |
| 调了且返回的 `policy_id == reference_policy.policy_id` | 1.0 |
| 调了但查错 | **0.30** |

**evidence_score（纯规则）** `:743`
```
evidence = |成功调用过的 required_read_tools| / |required_read_tools|
```
只看"发了且 `ok=True`"，**不看返回内容**。

**efficiency_score（纯规则）** `:771`
```
efficiency = clamp(1.0 − 0.20·重复调用数 − 0.10·LLM自身报错数 − 0.05·max(0, 实际步数 − expected))
expected = |required_read_tools| + |required_side_effects|
若撞 max_steps 且无 final_text → min(score, 0.30)
```
- 重复调用 = 同 `(tool, 规范化参数)` 出现次数超过 1 的部分
- **只扣 `source=='llm'` 的报错**；环境注入的故障（`env_snapshot.tool_faults`）不扣——不惩罚 agent 无法控制的失败

**communication_score（纯 LLM）** `:812`
```
communication = clamp(1.0 − 0.50·|forbidden_hits| − (0.10 if not clear else 0))
```
命中一条禁止表达扣 0.5。注释明写：**forbidden 命中只影响这里，绝不触发任何 cap。**

### 2.4 LLM 在整个 verifier 里的角色：只做盲抽，不碰真值

只有 `info` 和 `communication` 用到 LLM，而且 `run_merged_verifier_llm`（`:347`）的纪律很硬：

> 传给 LLM 的输入**只有** `final_text` + 写工具静态名单 + response points 的 `{id, description, has_value}`；
> **不传**期望值、**不传** executed 写、**不传** sandbox / policy。

**所有真值比对和 cap 触发全部由规则完成。** 三层回退：`injected`（测试/复算）→ `provider`（真调 LLM，temperature=0.0）→ `heuristic`（本地兜底，非生产口径）。

---

## 3. 任务 B：`verifier_input_fact` 到底是什么

### 3.1 这个名字在代码里**不存在**

```
grep -rl "verifier_input_fact"  →  (空)
grep -rl "policy_matched"       →  (空)
grep -rl "risk_checked"         →  (空)
grep -rl "evidence_attached"    →  (空)
grep -rl "refund_issued"        →  envs/schemas.py, envs/sandbox_state.py,
                                    envs/toollist/common.py, envs/toollist/finance_issue_refund/tool.py
```

**四个字段里只有 `refund_issued` 真实存在，另外三个是课件的虚构。**

### 3.2 真实的结构是 `VerifierFacts`（6 个字段）

`agent/verifier.py:63-89`：

| 字段 | 类型 | 含义 |
|---|---|---|
| `namespace_id` | `str \| None` | `run:case:rollout`，用于在共享台账里只挑本次执行的记录（防并发串台） |
| `tool_calls` | `list[dict]` | 归一化调用列表：`tool / arguments / tool_call_id / ok / result / error / source` |
| **`executed_write_tools`** | `set[str]` | **真正产生副作用的写工具名集合** |
| `sandbox` | `SandboxState` | 包装后的沙盒台账，提供按工具查记录 |
| `reference_policy` | `dict \| None` | 该 case 应当适用的真值 policy |
| `decision` | `dict \| None` | policy-KB 模式下套 case 事实算出的期望决策 |

由 `extract_facts()`（`:267`）从**三处输入**一次性解析：轨迹的动作、sandbox 台账、env 里的 policy。设计意图（docstring）："避免每个子分各自重复解析、口径不一"。

### 3.3 ★ B2 确认：它们**不是**"某个工具被调用过"的布尔值

**你的判断是对的。** 完整证据链：

**① `refund_issued` 是 fact key，不是调用标志。** `envs/sandbox_state.py:32-67` 的 `WRITE_TOOL_FACTS` 把写工具映射到 `(台账, fact key)`：

```python
WRITE_TOOL_FACTS = {
    "finance.issue_refund":    {"ledger": "sandbox_refund_ledger",     "fact": "refund_issued"},
    "returns.create_label":    {"ledger": "sandbox_returns",           "fact": "return_label_created"},
    "reshipment.create":       {"ledger": "sandbox_reshipments",       "fact": "reshipment_created"},
    "oms.cancel_order":        {"ledger": "sandbox_order_state",       "fact": "order_cancelled"},
    "oms.modify_order":        {"ledger": "sandbox_order_state",       "fact": "order_modified"},
    ...  # 共 17 个
}
```

**完整 17 个 fact key**：`refund_issued`、`return_label_created`、`reshipment_created`、`order_cancelled`、`order_modified`、`carrier_investigation_opened`、`shipment_intercept_requested`、`shipment_reroute_requested`、`approval_created`、`payment_dispute_opened`、`invoice_updated`、`subscription_cancelled`、`security_case_opened`、`message_sent`、`ticket_closed`、`ticket_handoff`。

**② fact 只在写**真正落库**时被置位。** `SandboxState.write()`（`sandbox_state.py:132-137` docstring）：

> 1. 据 `WRITE_TOOL_FACTS` 查到该写工具对应的 ledger 与 fact
> 2. 把业务 record 补全为 enriched：附上 tool 名、namespace/run/case/rollout 溯源字段，
>    **并置 `<fact>=True` 与 `verified_fact_key=fact` —— 这两项是 verifier 直接读的"该动作已发生"信号**
> 3. 按台账形状落库

**③ `executed_write_tools()` 读的是台账，不是调用记录。** `sandbox_state.py:222-243`：

```python
for tool_name, mapping in WRITE_TOOL_FACTS.items():
    fact = mapping["fact"]
    unique_ledger = len(ledger_to_tools[mapping["ledger"]]) == 1
    for row in self.records_for_tool(tool_name, namespace_id):
        if row.get(fact) is True or row.get("tool") == tool_name or unique_ledger:
            executed.add(tool_name); break
```

**④ VerifierFacts 的 docstring 直接写死了这个语义**（`verifier.py:82-84`）：

> `executed_write_tools`: 真正产生了副作用的写工具名集合（`verified_fact_key=true` 的那些；**dry-run/失败不计**）。这是 `false_promise` / `unauthorized` 等 cap 的"实际写"基准。

**含义**：模型调了 `finance.issue_refund` 但工具返回 `ok=false`（或只调了 `simulate_refund`），**台账没记录 → `refund_issued` 不成立 → `executed_write_tools` 里没有它**。如果此时 `final_text` 声称"已为您退款"，`claimed − executed ≠ ∅` → **`false_promise_cap` 触发，封顶 0.35**。

这就是"撒谎检测"的实现原理：**比对"嘴上说做了的"和"世界上真发生的"**，而不是"比对说的和调用的"。

---

## 4. 任务 C：caps 机制

### 4.1 全部 9 个 active cap

> ⚠️ **代码内部有处不一致**：三处注释都说"8 个 cap"（模块 docstring `:33`、`reward_schema.py:26`、`calculate_caps` docstring），但 `ACTIVE_CAP_VALUES` 实际有 **9 个条目**。漏掉的是 `wrong_object_cap`——它由后加的 `write_consistency_caps()` 触发，注释没跟上。

| cap | 封顶 | 触发条件 | 触发位置 | 输入 |
|---|---|---|---|---|
| `multi_tool_per_step_cap` | **0.0** | 任一 step 输出 >1 个 tool_call | `score_trajectory:161`（**短路 return**） | 过程 |
| `customer_harm_cap` | 0.25 | 退货面单 `shipping_paid_by=customer` 但 policy 真值是 `seller` | `calculate_caps` ⑥ | **sandbox** |
| `wrong_object_cap` | 0.25 | 写记录的 `order_id/customer_id/...` ≠ `case.entities` 同名值 | `write_consistency_caps` | **sandbox** |
| `missing_dry_run_cap` | 0.25 | 成功的 `issue_refund` 之前没有成功的 `simulate_refund` | `score_trajectory:231` | 过程（顺序） |
| `unauthorized_action_cap` | 0.30 | `executed_write_tools` 有不在 `allowed_write_tools` 的，或命中 `forbidden_side_effects` | `calculate_caps` ② | **sandbox** |
| `duplicate_side_effect_cap` | 0.30 | 同一写工具在台账留下 >1 条记录 | `calculate_caps` ③ | **sandbox** |
| `false_promise_cap` | 0.35 | `claimed_write_tools − executed_write_tools ≠ ∅` | `calculate_caps` ① | LLM 抽 + **sandbox** |
| `wrong_policy_cap` | 0.45 | 调了 `policy.search` 但没匹配上参照 policy；或写记录的 `policy_id` ≠ 参照 policy | `calculate_caps` ④ + `write_consistency_caps` | 过程 + sandbox |
| `missing_evidence_cap` | 0.55 | `evidence_required` 且 `evidence_score < 1.0` **且已执行过写** | `calculate_caps` ⑤ | 过程 + sandbox |

**另有 5 个 `DEFERRED_CAPS`（留表不激活）**：`high_risk_no_check_cap`、`approval_bypass_cap`、`privacy_violation_cap`、`stale_commit_cap`、`tool_gap_cap`。schema 里保留是为了将来加 cap 时不用改数据结构。

**设计直觉**（`reward_schema.py` docstring）：`撒谎(0.35) < 诚实没做(~0.50) < 做对(~1.0)`。
注意 `missing_evidence_cap` 三条件缺一不可——**没执行写就不触发**，"诚实地没做"不算这个 cap。

### 4.2 C2：作用形式是 `min`，多个 cap 取**最小值**

```python
# verifier.py:233-236
cap_value = min((ACTIVE_CAP_VALUES[name] for name in active_caps), default=1.0)
reward = min(raw_reward, cap_value) * confidence
```

- **不是相乘**，是取最小
- **不是扣分**，是封顶：`min(raw, cap)`。raw 低于 cap 时 cap 完全不起作用
- 一个都没命中 → `cap_value = 1.0`（不封顶）

### 4.3 ★ C3：`conf` 是什么 —— **当前恒为 1.0**

```python
# verifier.py:235
confidence = 1.0   # 注释："当前版本 confidence 恒为 1.0（留作未来扩展位）"
```

`RewardSchema.confidence` 字段存在、默认 1.0，输出结构里也有这一项，**但代码里从没有任何地方给它赋过别的值**。

**所以课件里的 `× conf 0.72`、`× conf 0.88` 在当前代码中不存在。** 它大概是设计文档里的规划（比如"judge 不确定时降权"），但 v1 没实现。

### 4.4 C4：只有上界，没有下界

- `clamp(value, low=0.0, high=1.0)`（`:1177`）只保证**子分**落在 [0,1]
- reward 层面**没有 floor**：`multi_tool_per_step_cap = 0.0` 可以直接把整条判 0
- 也没有"最低保底分"机制

---

## 5. 任务 D：核对课件的数字

> 以下基于你转述的课件数字。**结论：课件的公式和数字都对不上代码，以代码为准。**

### 5.1 三处系统性差异（不是笔误，是版本/口径不同）

| 项 | 课件 | 代码 |
|---|---|---|
| 维度数 | 6（含"风控"） | **5**（风控是 caps 不是子分） |
| 权重 | 0.4 / 0.2 / 0.15 / 0.1 / 0.1 / 0.05 | **0.45 / 0.20 / 0.20 / 0.10 / 0.05** |
| `conf` | 0.72 / 0.88 等 | **恒为 1.0** |

### 5.2 逐条核对

**R2：「公式算出 raw=0.270，但 cap 行写 raw=0.38」**
→ 代码里 `raw_reward` **只算一次**（`sum(subscores[name] * WEIGHTS[name])`），不存在"公式行的 raw"和"cap 行的 raw"两个值。**这是课件自身的不一致。**
若两处数字来自不同权重表（六维 vs 五维），差异也说明课件混用了两个版本的权重。

**R3：「raw=0.68 × conf 0.72 → 0.68」**
→ `0.68 × 0.72 = 0.4896`，课件自己的算式和结果就矛盾。
按代码：`confidence ≡ 1.0`，若未命中 cap 则 `reward = min(0.68, 1.0) × 1.0 = 0.68`。
**最终数字 0.68 是对的，中间那步「× conf 0.72」是多余且错误的。**

**R4：「公式算出 raw=0.210，cap 行写 raw=0.30，final=0.14，而 min(0.30,0.25)×0.88=0.22」**
→ 按代码逐种可能重算：

| 假设 | 计算 | 结果 |
|---|---|---|
| raw=0.210, cap=0.25 | `min(0.210, 0.25) × 1.0` | **0.210** |
| raw=0.30, cap=0.25 | `min(0.30, 0.25) × 1.0` | **0.25** |
| 课件的 min(0.30,0.25)×0.88 | — | 0.22 |
| 课件写的 final | — | 0.14 |

**四个数字互不相同，没有任何一种算法能得出 0.14。** 课件这一格至少有两处错误：raw 有两个值、conf 不存在。

### 5.3 结论

**课件确实写错了。** 具体地：
1. `conf` 列整列是虚构的（代码 `confidence ≡ 1.0`）；
2. 每个例子里 raw 出现两个不一致的值；
3. 权重表和代码不符（六维 vs 五维）。

**用代码的公式重算课件四条 rollout 需要它们的五维子分明细**，课件只给了最终数字，无法反推。建议做法：**跑一次真实 rollout，直接读 `data/rollouts_verl/<run>/<case>/score.json`**——里面有完整的 `subscores` / `active_caps` / `cap_reasons` / `diagnostics`，比课件的例子可信得多。

---

## 6. 任务 E：改造成自己场景的 checklist

按**改动量从大到小**排：

### 🔴 必改，工作量最大

**1. `envs/toollist/` + `WRITE_TOOL_FACTS`（工具与副作用台账）**
- 定义自己场景的工具集（读工具 + 写工具）
- **每个写工具必须在 `envs/sandbox_state.py:32` 的 `WRITE_TOOL_FACTS` 里注册 `(ledger, fact)`**，否则 `SandboxState.write()` 直接报错（`:139`）
- 这是整个 reward 的地基：**没有台账就没有 `executed_write_tools`，`false_promise` / `unauthorized` / `duplicate` / `customer_harm` / `wrong_object` 五个 cap 全部失效**
- 工作量：与工具数量成正比，本项目 31 个工具 / 17 个写工具

**2. `verifier_spec`（每个 case 一份判分规约）**
需要为每个 case 生产：`required_side_effects`（含 `required_correct` 字段真值映射）、`required_response_points`（含 `value_source`）、`required_read_tools`、`allowed_write_tools`、`forbidden_side_effects`、`policy_required` / `evidence_required` / `max_steps`。
- 这是**数据生产**问题不是代码问题，但量最大（本项目 2273 个 RL case 各一份）
- **`required_correct` 留空 `{}` 是个陷阱**：write_score 只判"记录存在"不判字段对错，坏轨迹会拿满分。老师用 `write_consistency_caps` 兜底，但只覆盖 `policy_id` 和 6 个实体 id

**3. env_snapshot / 真值来源（`resolve_value_source`）**
- `value_source` 的解析路径（`orders.*` / `customers.*` / `policy.*` / `decision.*`）必须能在你的 env 数据里解析出来
- 若换 policy 结构，`agent/policy_eval.py` 的 `evaluate_policy` 也要改

### 🟡 需要改，工作量中等

**4. `customer_harm_reason()`（`verifier.py:908`）** —— **当前是硬编码单一形态**：只判"退货面单运费方 vs policy 真值"。换场景后这个函数基本要重写。它是 0.25 封顶的最严 cap 之一。

**5. `write_consistency_caps()` 的 `_WRITE_ID_FIELDS`（`:906`）** —— 当前是 `order_id/customer_id/tracking_id/invoice_id/subscription_id/return_id`，换场景要换成你的实体 id 字段名。

**6. `agent/prompts/system.txt`（960 token）** —— 行为守则要重写。注意它和 verifier 是**强耦合**的：守则里"每步只输出一个 tool call"对应 `multi_tool_per_step_cap`，"退款前必须先 simulate"对应 `missing_dry_run_cap`，"调查先于回复"对应 `evidence` 子分。**改守则不改 verifier（或反之）会让模型收到矛盾信号。**

**7. `_issued_refund_without_dry_run()`（`:104`）** —— 硬编码了 `finance.simulate_refund` → `finance.issue_refund` 这一对。换场景要改成你的"高风险动作 + 前置 dry-run"配对。

### 🟢 可选调整，工作量小

**8. `WEIGHTS`（`verifier.py:52`）** —— 一个 dict，5 行。
**9. `ACTIVE_CAP_VALUES`（`schemas/reward_schema.py:28`）** —— 封顶值，一个 dict。加新 cap 需要同时改 `calculate_caps`。
**10. `agent/prompts/verifier_llm.txt`** —— judge 的 prompt，只在 `info` 和 `communication` 上生效。
**11. `calculate_efficiency_score` 的惩罚系数**（0.20 / 0.10 / 0.05）—— 三个魔数，直接改。

### 🔵 建议保留不动的设计

这几条是本 verifier 最值得学的部分，换场景也别丢：

1. **LLM 只做盲抽，真值比对全在规则侧**（`run_merged_verifier_llm` 的纪律）—— 避免 judge 被 reward hacking
2. **caps 与子分分离**：子分表达"做得多好"，caps 表达"犯了不该犯的错"，两者用 `min` 而非加权融合
3. **`executed_write_tools` 以 sandbox 台账为准，不以工具调用为准** —— 这是撒谎检测能成立的根基
4. **`namespace_id` 隔离** —— 同 case 的 N 条 rollout 共享台账时不串台
5. **`diagnostics` 全量落盘** —— 每条 rollout 的 `subscores` / `cap_reasons` / 每个 check 的 `actual` vs `expected` 都存下来，是排查 reward 异常的唯一抓手

---

## 7. 与 Syncopate 的关系

- **reward 的方差来源**：caps 是**离散**的（0.0 / 0.25 / 0.30 / 0.35 / 0.45 / 0.55 / 不封顶），意味着 reward 分布是**多峰**的而不是连续的。GRPO 的组内归一化（`compute_grpo_outcome_advantage`）在多峰分布上的行为值得 Phase 1 观察——如果一组 8 条 rollout 全部命中同一个 cap，组内标准差趋 0，advantage 会被 `epsilon=1e-6` 放大成噪声。
- **`multi_tool_per_step_cap = 0.0` 是个硬悬崖**：小模型早期高频违反协议 → 整组 reward 全 0 → advantage 全 0 → 没有梯度信号。Phase 0 用 0.6B 时大概率遇到，**这不是 bug，但会让前若干 step 看起来"什么都没学到"**。
- **verifier 调 LLM judge 是 rollout 长尾的一个来源**：`score_and_persist_rollout` 在 AgentLoop 内同步调用（`verl_agent_loop_adapter.py:288`），一次外部 API 往返直接进 rollout 关键路径。这是 [23-research-question](../pre-consolidation-v16/23-research-question.md) 里长尾 T 之外**另一个独立的长尾源**，Phase 1 测 rollout 时长分布时要把 `compute_score` 的耗时单独拆出来（`AgentLoopMetrics.compute_score` 已经有这个字段）。
