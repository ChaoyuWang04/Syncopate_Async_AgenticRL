# 07 · 沙盒环境：课件的 8 组件 vs 老师的真实实现

> 调查日期：2026-07-29
> 代码：`REL/envs/`（1 224 行）+ `REL/agent/observations.py` / `runtime.py` / `rollout_store.py`
> 数据：`REL/data/batches/stage5_full/env_snapshots/*.env.json`（2 737 份）

---

## 0. 三句话结论

1. **8 个组件里 7 个都存在**，但形态和课件不同——最大的差异是 **Tool Gateway 不做权限校验**，它是一条"薄执行管线"而非"守门人"。
2. **故障注入机制完整实现了，但 2 737 份 env_snapshot 里零配置**——有机制、无数据。
3. **`envs/` 层完全没有时间和随机数**，环境本身是确定性的；不确定性全部来自 LLM（采样 + judge），不来自沙盒。

---

## 1. 任务 A：8 组件对照表

| # | 课件组件 | 代码位置 | 有没有 | 与课件的差异 |
|---|---|---|---|---|
| 1 | **Case Store** | `data/batches/stage5_full/cases/*.json`，经 parquet 的 `extra_info.case_path` 指路 | 🟡 **数据有，无抽象层** | 没有 CaseStore 类。消费方各自 `json.load()`（`adapter:67`、`verl_reward_adapter:37`） |
| 2 | **State Builder** | `SandboxState.from_env_snapshot()`（`sandbox_state.py:98`）+ `default_sandbox()` | ✅ **有** | 用 `sandbox_initial` 打底 + 空台账骨架。`schemas/env_schema.py` 的 `model_validator` 补齐缺失键 |
| 3 | **Sandbox State Store** | `envs/sandbox_state.py`（243 行）`SandboxState` | ✅ **有，且最完整** | 13 个台账 + 审计日志，namespace 隔离，dict/list 两种台账语义 |
| 4 | **Tool Gateway** | `envs/toolfactory.py:154 ToolFactory.execute()` | 🟡 **有，但不校验权限** | 见 §1.1——**这是最大的设计差异** |
| 5 | **Mock Tool** | `envs/toollist/`（35 个薄包装）+ `common.py`（1 005 行真实现） | ✅ **有** | 三层结构：`tool.py` 一行 `make_tool(name)` → `common.TOOL_SPECS` → `_read_handler` / `_write_handler` |
| 6 | **Observation Gen** | `agent/observations.py:34 project_observation_for_model()` | ✅ **有，是真加工层** | 见 §1.3 |
| 7 | **Terminal Checker** | ❌ **散在两处循环里**，无独立组件 | 🔴 **无** | 见 §1.2 |
| 8 | **Trajectory Logger** | `agent/trajectory.py` + `agent/rollout_store.py` | ✅ **有** | `write_rollout_artifacts()` 落 trajectory/score/token_trace/sandbox_final_state |
| 9 | **Verifier Input Adapter** | `train/verl_reward_adapter.py:26 score_and_persist_rollout()` | ✅ **有** | 重新从磁盘读 case/env/verifier_spec，保证 reward 用的是权威文件 |

> 说明：课件列的是 9 个名字（题目说 8 个），上表按 9 个逐一对照。

### 1.1 ★ Tool Gateway：存在，但**故意不做权限校验**

`ToolFactory.execute()`（`toolfactory.py:154-210`）的五步管线：

```python
1. tool = self.get(tool_name)          # 不存在 → unknown_tool (source=llm)
2. tool.validate_args(arguments)       # 缺必填参数 → <name>_required (source=llm)
3. injected = self._maybe_fault(...)   # 故障注入，命中则直接返回 (source=environment)
4. if tool.handler is None: raise tool_not_implemented (source=runtime)
5. result = tool.handler(args, env_snapshot, sandbox, context)
```

**做的校验只有两项：工具存在性 + 必填参数。没有权限校验、没有前置依赖校验。**

`allowed_tools` 参数还在签名里，但是死的（`:189-191`）：

```python
# allowed_tools 是旧接口残留。runtime 不在这里做 case 级授权，避免"执行层截断轨迹"；
# 是否允许该写动作由 verifier 看完整轨迹后评分。
_ = allowed_tools  # compatibility only; verifier handles authorization, runtime no longer gates tools.
```

**这是一个明确的设计决策，不是遗漏。** 理由写在注释里——**执行层截断轨迹会让 RL 拿不到"做错了"的样本**。让模型把错误动作执行完，再由 verifier 用 `unauthorized_action_cap`（0.30）教它，比在工具层直接拒绝更有训练价值。

工具层的原则叫 **permissive**，`common.py` 里反复出现：

> 工具 permissive：只拦**物理现实**（订单存在 / 不超付）。是否先 simulate、政策是否允许、policy_id 对不对——这些"该不该退"的正确性**不在工具拦**，由 verifier 评分教模型。**上线时可由规则层再加硬拦。**

**含义**：这套沙盒是**为训练设计的，不是为生产设计的**。要上线得自己加一层硬拦。

### 1.2 ★ Terminal Checker：**没有独立组件，散在两处**

**没有任何函数叫 `is_terminal` / `check_done`。** 终止判定分散在两个互不共享代码的循环里：

**① standalone runtime**（`agent/runtime.py`）：
- `limit = max_steps or case.get("max_steps") or 20`（`:256`，三级回退）
- 终止方式一：模型输出不含 `<tool_call>` → `final_text = strip_reasoning_blocks(...)` → `break`（`:341-346`）
- 终止方式二：`for...else` 跑满 limit 步没 break → `final_text = ""`（`:417-421`）

**② verl AgentLoop adapter**（`train/verl_agent_loop_adapter.py`）：
- `while len(response_mask) < self.response_length`（token 预算）
- `if assistant_turns >= self.max_assistant_turns: break`（`:122`）
- `if user_turns >= self.max_user_turns: break`（`:124`）
- 无 tool_call → `final_text` → `break`（`:221-222`）

**两条路径的终止条件不一致**：runtime 用 `max_steps`（默认 20，或 case 自带），adapter 用 verl 的 `max_assistant_turns=8` / `max_user_turns=8` + token 预算。**同一个 case 在两条路径下能跑的步数不同。**

> ⚠️ 这对我们有直接影响：verifier 的 `efficiency` 子分里有 `hit_max_steps = actual >= spec.max_steps and not final_text`，用的是 **`spec.max_steps`**（verifier_spec 里的字段），**和上面两个都不是同一个数**。三处 max_steps 口径各自独立——改造时要统一。

### 1.3 Observation Gen：**不是工具返回值本身，有白名单投影层**

`agent/observations.py:34`：

```python
MODEL_VISIBLE_OBSERVATION_KEYS = {"ok", "result", "error", "message", "source"}

def project_observation_for_model(observation):
    projected = {k: observation[k] for k in MODEL_VISIBLE_OBSERVATION_KEYS if k in observation}
    projected["tool_name"] = observation.get("tool_name")
    projected["tool_call_id"] = observation.get("tool_call_id")
    return projected
```

**隐藏了 `namespace_id` 和 `arguments`**（完整 observation 仍存进 `trajectory.tool_observations` 供 verifier/审计）。注释说明理由："避免 namespace_id、内部审计字段或未来新增的敏感字段泄漏进 prompt"。

这一层还顺带定义了错误归因：`is_environment_error()` 要求 `ok is False AND source == "environment"` 同时成立——这是 verifier `efficiency` 只惩罚 `source=='llm'` 报错的依据。

---

## 2. 任务 B：四个关键机制核对

### 2.1 namespace 隔离：三元组拼字符串，**碰撞风险来自 uuid4**

`envs/namespace.py` 全文只有一个函数：

```python
def build_namespace_id(run_id: str, case_id: str, rollout_id: str) -> str:
    return f"{run_id}:{case_id}:{rollout_id}"     # 形如 "run123:CASE_B:rollout01"
```

调用点（`verl_agent_loop_adapter.py:71-72`）：

```python
rollout_id = f"rollout_{extra_info.get('index', 0):04d}_{uuid4().hex[:8]}"
namespace_id = build_namespace_id(self.run_id, case_id, rollout_id)
```

- `run_id` = `VERL_RUN_ID` 环境变量（一键脚本设的时间戳串），兜底 `make_run_id("verl")`
- `case_id` = case 标识
- `rollout_id` = 数据集 index + **uuid4 前 8 位**

**碰撞风险评估**：
- uuid4 前 8 位 = 32 bit。同一个 (run_id, case_id, index) 下要撞才有问题，而同一 case 的 rollout_n（默认 8）条各带独立 uuid → 生日碰撞概率 ≈ `8²/2/2³²` ≈ **7×10⁻⁹**，可忽略。
- 真正的风险不是碰撞，而是**不可复现**：uuid4 让 namespace_id 每次运行都不同（见 §4.3）。

**隔离的实际作用**：所有写记录带 `namespace_id`（`sandbox_state.py:158`），`records_for_tool(tool, namespace_id)` 按它过滤。注意 `export()` 导出的是**完整 sandbox 不做过滤**（`:110`），verifier 读取时才按 namespace 过滤——这个设计在共享 factory 的并发场景下是对的。

另外 `ToolFactory._fault_counts` 的 key 也是 `(namespace_id, tool_name)`（`:82`），保证 `transient_error` 的"前 N 次失败"配额各 rollout 独立。

### 2.2 simulate vs issue：**只有 1 对，且真的分开实现**

35 个工具文件里，**只有 refund 这一对**是 simulate/issue 拆分的：

| | `finance.simulate_refund` | `finance.issue_refund` |
|---|---|---|
| 分类 | **读工具**（`_read_handler`，`common.py:256`） | **写工具**（`_write_handler`，`common.py:494`） |
| 副作用 | 无，不写台账 | 写 `sandbox_refund_ledger`，置 `refund_issued=True` |
| 校验 | 订单存在 + `amount ≤ paid_amount` | **完全相同的两条** |
| 返回 | `simulation_id / allowed / policy_found / reasons` | `refund_id / status=issued` |

**两者的物理校验完全一样，唯一区别就是写不写台账。** 注释直说："与 simulate_refund 的区别就在这里会落台账"。

其余"读-写配对"是隐式的（比如 `oms.get_order` → `oms.modify_order`），但没有 dry-run 语义。

### 2.3 idempotency_key：**没有这个概念**

全仓 `grep -ri "idempot"` → **零命中**。

代码里最接近的是 `_id()`（`common.py:151`）：

```python
def _id(prefix, context, *parts):
    """生成确定性的对象 id（如 RF_<order>_<tool_call_id>）。
    带上 tool_call_id 后缀保证同一次调用幂等、可 replay，且不同调用不会撞 id。"""
    suffix = context.get("tool_call_id", "tc")
    return f"{prefix}_{clean}_{suffix}"
```

**注意这只是 ID 的确定性，不是写入的幂等性。** `tool_call_id` 在 adapter 里是 `tc_{step}`（`adapter:294`），所以：

- **同一步重放** → 同样的 `tool_call_id` → 同样的 `refund_id`，但 `_write` 仍会**再 append 一条**到事件流台账（`sandbox_state.py:167`），不会去重
- **不同步重复调用同一工具** → 不同 `tool_call_id` → 不同 id → 台账两条记录

**"同一条 rollout 重试会怎样"的答案**：**会产生重复记录，然后被 verifier 的 `duplicate_side_effect_cap`（封顶 0.30）惩罚**（`verifier.py:900` 的 `duplicate_write_tools` 检查 `len(records_for_tool(tool)) > 1`）。

> **设计取向很明确：不在环境层做幂等，而是让重复发生、再用 reward 教模型别重复。** 和 §1.1 的 permissive 原则一致。
> **上线必须自己加幂等**——这是训练沙盒和生产系统最危险的一处差距。

### 2.4 evidence 强制：**issue_refund 不会因缺 evidence 而 reject**

**在任何一层都不 reject。** 逐层确认：

| 层 | 有没有拦 |
|---|---|
| Tool Gateway (`ToolFactory.execute`) | ❌ 只校验工具存在 + 必填参数 |
| 工具内部 (`_write_handler` 的 `issue_refund` 分支) | ❌ 只拦订单存在 + 不超付 |
| **Verifier** | ✅ **只在这里"扣分"** |

`common.py:494-501` 的注释是决定性证据：

> 工具 permissive：只拦物理现实（订单存在 / 不超付）。是否先 simulate、政策是否允许、policy_id 对不对——这些"该不该退"的正确性**不在工具拦**，由 verifier 评分教模型（`required_read_tools` 含 simulate → 跳过扣 evidence；deny case 把 issue_refund 列 forbidden → `unauthorized_action_cap`；`wrong_policy_cap` 等）。**上线时可由规则层再加硬拦。**

而且 `evidence_ids` 这个参数本身：
- 在 `carrier.open_investigation`（`common.py:850`）和 `payment.open_dispute_case`（`:861`）的 schema 里是 `arg("array")`，**非必填**
- 值只是原样存进台账记录（`:474, :488`），**没有任何校验**
- `finance.issue_refund` 的 schema 里**根本没有 `evidence_ids` 参数**，只有一个 `simulation_id`，而且标注是 `arg("string", False, "可带 finance.simulate_refund 返回的 simulation_id（**informational**）")`——**声明就是"仅供参考"，不校验**

verifier 侧的相关约束是 `missing_dry_run_cap`（0.25，`verifier.py:104` 按 `parsed_actions` 顺序检查 simulate 是否在 issue 之前）和 `missing_evidence_cap`（0.55）。**都是事后扣分，不是事前拦截。**

---

## 3. 任务 C：故障注入

### 3.1 **机制完整存在**：`ToolFactory._maybe_fault()`

`toolfactory.py:212-261`，由 `env_snapshot.tool_faults[tool_name]` 驱动：

| mode | 行为 | 备注 |
|---|---|---|
| `none` | 放行 | 默认 |
| `latency` | **放行**（同 none） | ⚠️ 注释明说"当前版本只是'慢但成功'，**本函数不模拟延迟**" |
| `transient_error` | 前 `fail_times` 次失败，之后成功 | 用 `(namespace_id, tool_name)` per-rollout 计数 |
| `hard_error` | 每次都失败 | 确定性 |

注入的失败一律标 `source="environment"`，与模型自己用错的 `source="llm"` 严格区分——**verifier 的 `efficiency` 只惩罚后者**（`verifier.py:790`），不让模型为环境故障背锅。

配置形状（来自 `schemas/env_schema.py`）：

```json
"tool_faults": {
  "tms.get_tracking": {"mode": "transient_error", "fail_times": 2, "error": "carrier_timeout"}
}
```

### 3.2 ★ **但 2 737 份 env_snapshot 里零配置**

```
env_snapshot 总数 = 2737
配了 tool_faults 的 = 0        ← 全部是 {}
故障 mode 分布 = {}
```

**这是"有机制、无数据"。** 整套故障注入代码从未在这批数据上被触发过。

同样地：
- `external_services`（外部服务罐头响应）：2 737 份里只有 **278 份非空**（10.2%）
- `reference_now`（参考时钟，值是 `2026-06-01T00:00:00+00:00`）：**数据里每份都有，但代码里没有任何地方读它**（`grep -rl reference_now --include=*.py` 只命中 `schemas/env_schema.py` 自己的 schema 声明）。又一个"声明未实现"的字段。

### 3.3 我们改造时要补什么

**好消息：故障注入这一块不用自己写，机制已经有了，只需要造数据。** 最小方案就是在 `env_snapshot.tool_faults` 里填配置，零代码改动。

但如果要做 Syncopate 关心的**长尾**，现有机制**不够**——`latency` 是空实现。最小补丁：

```python
# envs/toolfactory.py::_maybe_fault，在 mode 分支里加：
if mode == "latency":
    # 确定性延迟：用 (namespace, tool, 调用序号) 派生，避免真随机破坏复现性
    import time, hashlib
    key = f"{context.get('namespace_id','')}:{tool_name}:{self._fault_counts[k]}"
    jitter = int(hashlib.sha256(key.encode()).hexdigest()[:4], 16) / 0xFFFF
    base_ms = float(fault.get("latency_ms", 0))
    time.sleep((base_ms * (0.5 + jitter)) / 1000.0)   # 0.5x~1.5x 抖动
    return None    # 慢但成功
```

**关键设计点（和 seed 挂钩）**：不用 `random`，用 **`hashlib` 对 `(namespace_id, tool_name, 调用序号)` 做哈希**派生抖动。这样：
- 同一条 rollout 重放 → 完全相同的延迟序列（可复现）
- 不同 rollout → 不同延迟（有方差，能造出长尾）
- 不引入全局随机状态，不受 `random.seed()` 干扰

⚠️ 但注意：**AgentLoop 是 asyncio 协程，`time.sleep` 会阻塞整个 event loop**，必须用 `await asyncio.sleep()`。而 `ToolFactory.execute` 是同步函数——所以这个补丁要么把 execute 改成 async，要么在 adapter 层（`_append_non_model_messages` 之前）注入延迟。**后者改动更小。**

---

## 4. 任务 D：复现性

### 4.1 版本字段：**有 3 个，但都不覆盖 env 内容**

| 字段 | 位置 | 值 | 说明 |
|---|---|---|---|
| `env_snapshot.version` | 数据文件 | `"env_v1"`（2 737 份全部相同） | **schema 版本，不是内容版本**（`env_schema.py:97`） |
| `prompt_template_version` | `Trajectory`（`trajectory.py:55`） | 常量 | prompt 模板版本 |
| `prompt_hash` / `tool_schema_hash` | `Trajectory` | SHA-256 | `templates.py:51 stable_hash()` |
| `verifier_version` | verifier 输出 | `"verifier_simple_v1"` | 硬编码字符串 |

**没有 `env_version`（内容版本）也没有 `tool_schema_version`**——工具 schema 靠 `tool_schema_hash` 追踪，比版本号更严格（内容变了哈希就变）。

`templates.py:11-14` 说明了这些哈希的用途，很关键：

> GRPO group 要求同一组内 prompt 版本、`tool_schema_hash`、env/verifier/model 版本全部一致，consistency audit 靠这些哈希对齐 rollout 侧与 training 侧。

### 4.2 seed：**代码里根本没有 seed 概念**

`grep -rn "seed" envs/ agent/` → 零命中（只有 verl 侧的 `data_loader_seed`）。

**因为环境不需要 seed——它完全确定性**（见 §4.3）。

### 4.3 ★ 同一个 case 跑两次，observation 会完全一样吗

**给定相同的模型输出序列，会完全一样。** 证据：

```
grep -rn "datetime|time\.|now()|uuid|random" envs/*.py envs/toollist/*.py  →  零命中
```

**整个 `envs/` 层没有任何时间函数、没有任何随机数。** 所有工具都是纯函数：读 `env_snapshot.readonly_tables`（静态 JSON）→ 返回 `deepcopy`。

**三处不确定性来源，逐一分析它们会不会污染 observation：**

| 来源 | 位置 | 会污染 observation 吗 |
|---|---|---|
| `uuid4()` in `rollout_id` | `adapter:71` | ❌ **不会**。它只进 `namespace_id`，而 `project_observation_for_model` 的白名单**不含 `namespace_id`**（`observations.py:20-31`） |
| `uuid4()` in `request_id` | `adapter:115` | ❌ 不会。只用于 vLLM sticky session |
| `datetime.now()` in `make_run_id` | `rollout_store.py:41` | ❌ 不会。只影响 artifact 目录名；verl 路径下 `run_id` 来自 `VERL_RUN_ID` |

**业务 id 是确定性的**：`_id()` 用 `tool_call_id` 做后缀，而 adapter 里 `tool_call_id = f"tc_{step}"`（`adapter:294`）——纯步数派生。所以 `refund_id = RF_{order_id}_tc_3` 每次都一样。

**结论：沙盒是确定性的。真正的不确定性全在 LLM 侧：**

1. **模型采样**（temperature > 0）——rollout 每次不同，这是 RL 要的
2. **LLM judge**（`VERIFIER_PROVIDER=qwen`）——即使 `temperature=0.0`（`verifier.py:359`），远端 API 仍可能因批处理/硬件差异给出不同结果，**影响 reward 不影响 observation**
3. **dict 遍历顺序**——Python 3.7+ 保证插入序稳定；`_first_by()` 做线性查找（`common.py:48`）依赖表的插入顺序，但表来自同一份 JSON，顺序固定 → **不引入不确定性**

> **一句话**：环境可复现，reward 不完全可复现（judge），rollout 本来就不该复现（采样）。
> **对我们的价值**：可以用"注入 judgement"（`run_merged_verifier_llm` 的第 1 层回退，`verifier.py:365`）做完全确定性的 reward 复算——这是调试 reward 逻辑的正确姿势。

---

## 5. 重点结论：改造自己场景时怎么办

### 🟢 可以直接复用（几乎零改动）

| 组件 | 为什么能复用 |
|---|---|
| **`SandboxState`**（`sandbox_state.py`） | 台账机制与业务无关。只需换 `SANDBOX_KEYS` 和 `WRITE_TOOL_FACTS` 两张表 |
| **`build_namespace_id`** | 24 行，纯字符串拼接 |
| **`ToolFactory` 执行管线** | 五步管线与业务无关；动态加载让"加工具=加一行路径" |
| **故障注入 `_maybe_fault`** | 完整可用，只需在 env_snapshot 里造数据（`latency` 需补，见 §3.3） |
| **`project_observation_for_model`** | 白名单投影，换白名单即可 |
| **`Trajectory` + `rollout_store`** | 落盘结构与业务无关 |
| **`verl_reward_adapter`** | verl 接线层，与业务无关 |

### 🔴 必须自己写（工作量从大到小）

| # | 要写什么 | 量 | 说明 |
|---|---|---|---|
| 1 | **`TOOL_SPECS` + `_read_handler` / `_write_handler`**（`common.py`） | **最大** | 1 005 行里绝大部分是这个。每个工具一个分支 |
| 2 | **`WRITE_TOOL_FACTS` + `SANDBOX_KEYS`** | 中 | 每个写工具必须登记 `(ledger, fact)`，否则 `write_record` 直接 `KeyError`。**这是 verifier 五个 cap 的地基**（见 [06-verifier-and-reward](06-verifier-and-reward.md) §3.3） |
| 3 | **`env_snapshot` 数据**（readonly_tables + policies） | 中（数据活） | 2 737 份 JSON。是"真值的唯一来源" |
| 4 | **`case` + `verifier_spec` 数据** | 大（数据活） | 见 [06-verifier-and-reward](06-verifier-and-reward.md) §6 |
| 5 | **Terminal Checker 统一** | 小但重要 | 见 §1.2——现在三处 max_steps 口径不一致，改造时该收敛成一处 |

### 🟡 老师也没有，我们要补的

| 缺什么 | 严重程度 | 说明 |
|---|---|---|
| **`latency` 故障的真实实现** | 🔴 **对 Syncopate 是关键** | 现在是空实现。**长尾 rollout 正是我们要研究的东西**，没有它就造不出可控的长尾 |
| **idempotency_key** | 🟡 训练可不要，上线必须 | 现在靠 `duplicate_side_effect_cap` 事后罚 |
| **Tool Gateway 的权限硬拦** | 🟡 训练故意不要 | 注释明说"上线时可由规则层再加硬拦" |
| **独立的 Terminal Checker** | 🟡 | 现在散在两个循环，且和 verifier 的 `spec.max_steps` 又是第三个口径 |
| **Case Store 抽象层** | 🟢 可不补 | 现在各处 `json.load()`，能跑 |
| **`reference_now` 的实际使用** | 🟢 | 数据里有、代码没读。若业务涉及时效（退货窗口/ETA）必须接上，否则"时间"概念不存在 |

---

## 6. 对 Syncopate 的直接影响

**① 长尾必须自己造。** 现有 2 737 份 env_snapshot 零故障配置、`latency` 是空实现，**当前这个任务的 rollout 时长方差只来自轮数（P10=3 轮 / max=10 轮，见 [05-anatomy-of-a-trajectory](05-anatomy-of-a-trajectory.md) §2.1）**，没有工具延迟贡献。而 Phase 1 要"建立一个值得被异步化的同步基线"——**如果不补 latency，长尾可能不够严重，异步收益测不出来**。

按 §3.3 的哈希派生方案补 `latency`，我们就能**精确控制长尾形态**（改 `latency_ms` 和命中的工具比例），这比"碰运气看任务本身有多长尾"强得多，**而且让 Phase 2 的"长尾严重度 × 异步收益"曲线变成可控变量而非观测变量**。这可能是整个 Phase 1 最值得先做的一件事。

**② 环境确定性是个好消息。** 沙盒无时间无随机，意味着 rollout 时长的方差**纯粹来自模型生成长度 + LLM judge 的 API 往返**，不来自环境噪声。这让长尾归因变干净：

```
rollout 总时长 = Σ(生成时长) + Σ(工具执行时长≈0) + judge API 往返
                    ↑ 模型侧          ↑ 确定性且极快      ↑ 外部依赖，方差大
```

**③ judge 是被低估的长尾源。** 工具执行是纯内存操作（读 JSON + deepcopy），耗时可忽略；而 verifier 的 LLM judge 是**同步进 rollout 关键路径**的一次外部 API 往返。Phase 1 测长尾分布时，`AgentLoopMetrics` 已经把 `generate_sequences` / `tool_calls` / `compute_score` 三段分开计时了——**直接读这三个数就能验证"judge 占多大比重"**，零埋点。
