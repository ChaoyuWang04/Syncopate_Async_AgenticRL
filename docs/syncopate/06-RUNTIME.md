# Syncopate · Runtime 与 RAG

> 本文是模型交互、AgentLoop、工具、安全闸、会话和 RAG 的唯一现行说明。
> API、队列、数据库运维和发布看 [07-SERVING.md](07-SERVING.md)。

## 1. Runtime 负责什么

Runtime 把模型输出变成可控的业务动作：

```text
system prompt + 会话历史 + 当前用户请求 + 工具菜单
  → Decider / 模型
  → think、tool call 或自然语言终答
  → ActionGate
  → 工具、RAG 或 session 信令
  → observation 回到模型
  → 终答或等待状态
```

模型只提出动作，不能绕过 ActionGate 直接调用外部系统。

## 2. 当前实现

| 能力 | 正式实现 | 作用 |
|---|---|---|
| 消息与行为协议 | [contract.py](../../syncopate/core/contract.py)、[parsing_v15.py](../../syncopate/core/parsing_v15.py) | 解析 think、XML/JSON 工具调用、session 信令和终答 |
| 工具注册 | [tool_registry.py](../../syncopate/core/tool_registry.py) | 工具 schema 的单一来源 |
| 沙盒 | [sandbox.py](../../syncopate/core/sandbox.py) | 训练与评测中的确定性世界 |
| 生产工具治理 | [tool_governance.py](../../syncopate/runtime/tool_governance.py) | 权限、超时、幂等、重试与输出约束 |
| AgentLoop | [agent_loop.py](../../syncopate/runtime/agent_loop.py) | 多轮决策、工具观测、快照与收场 |
| ActionGate | [action_gate.py](../../syncopate/runtime/action_gate.py) | 外部副作用的唯一出口 |
| Decider | [decider.py](../../syncopate/runtime/decider.py) | 构造真实线上 prompt 并调用模型端点 |
| 会话历史 | [db.py](../../syncopate/runtime/db.py)、[prior_turns.py](../../syncopate/core/prior_turns.py) | 保存并恢复真实 user/assistant 轮次 |
| RAG | [retrieval.py](../../syncopate/runtime/retrieval.py) | 多租户政策与经验检索 |
| 回流 | [flywheel.py](../../syncopate/runtime/flywheel.py) | 从线上 run 提取、脱敏、标注并导出训练候选 |

Runtime 主体能力已经实现并有测试保护；它在目标部署环境中的正式验收属于 Serving 的 T3。

## 3. v15 交互协议

### 三个通道

- `think`：模型推理；v15 默认 think-on。
- `tool`：业务工具或 session 信令，所有机器结构只走这里。
- `final`：给用户看的自然语言，不允许残留 JSON 壳、字段壳、think 或 tool 标签。

### Session 信令

| 信令 | 含义 | Runtime 结果 |
|---|---|---|
| `session.defer` | 当前数据不足，需要以后复查 | 结束当前轮并留下可复查语义 |
| `session.clarify` | 缺少用户信息 | run 进入 `waiting_for_user` |
| `session.reject` | 越权、离题或政策拒绝 | 结束并审计 |
| `session.report` | 报告机器可核字段 | 非终止性，记录后继续生成自然语言 |

行为由实际动作推导，不再要求模型填写 `behavior` 标签。运行态身份参数由系统按租户注入，模型看不到也不能覆盖。

## 4. AgentLoop

AgentLoop 每轮执行：

1. 读取真实会话历史和当前请求。
2. 用共享模板、预算和工具 schema 调用 Decider。
3. 解析模型提出的一个动作。
4. 将动作交给 ActionGate。
5. 把工具观测或 session 结果追加到历史。
6. 保存可恢复快照。
7. 在自然语言终答、终止性信令、取消、预算或步数边界到达时结束。

恢复时从最新完整快照继续，不能重放已经完成的只读工具，也不能在回放事件时重新执行副作用。
不认识的 checkpoint 版本进入人工处理，不能猜格式。

## 5. ActionGate

ActionGate 是所有动作的强制收口。它负责：

- 工具是否注册；
- 参数 schema 与运行态注入；
- 租户、资源归属和权限；
- 自动化档位与发布开关；
- 审批、取消、预算和步数安全点；
- 外部幂等键和重复调用；
- 写动作执行前的意图记录；
- 工具结果、错误、耗时、事件和审计；
- 对“结果未知”与“明确失败”做不同处理。

原则：

- 写操作的幂等键绑定业务对象和动作，不绑定 run ID。
- 明确失败可以按登记策略重试；副作用结果未知不能自动重发。
- C 档写动作需要审批，D 档不自动执行。
- 读不到发布状态、权限、检索或副作用结果时，按保守方向处理。
- 工具实现不能自行绕过 Gate。

## 6. 沙盒与线上同形

训练沙盒是 Runtime 的确定性子集。二者必须共享：

- 工具名、参数 schema 和返回字段；
- session 信令与终止语义；
- 运行态参数注入规则；
- 错误类别和可重试含义；
- 多轮 history、分页和异步任务形状；
- 写工具的副作用与幂等语义。

允许的差异必须明确登记。例如线上异步审核不能在一次工具调用中阻塞等待，沙盒可以用确定性时钟模拟；两边仍要给模型等价的可观察状态。

工具新增或修改时，必须同时通过 registry 完整性、字段对照、行为对照和负向测试。

## 7. RAG

### 数据

Runtime 使用 PostgreSQL 保存：

- 有生效期和取代关系的政策条款；
- 按租户隔离的历史经验或 insight。

结构化条件先由 SQL 精确过滤，再对剩余文本做确定性词法排序。

### 三态契约

| 状态 | 含义 | 模型应该做什么 |
|---|---|---|
| `ok` | 找到达到阈值的结果 | 使用证据继续 |
| `no_match` | 服务正常，但没有足够相关的结果 | 如实说明没有，不返回“最像的一条” |
| `unavailable` | 检索服务不可用 | 进入保守降级，不把它伪装成“查不到” |

沙盒与 Runtime 使用同一种词法打分，但候选集不同，因此阈值不同：

- 沙盒阈值：0.35，定义在 [corpus.py](../../syncopate/domains/adcampaign/corpus.py)。
- Runtime 阈值：0.53，定义在 [retrieval.py](../../syncopate/runtime/retrieval.py)。

阈值只能依据正例和应留空负例重新标定，不能因为召回少就向下移动。

向量检索不是当前依赖。只有积累足够真实查询，并证明词法检索的非结构化漏召回超过预注册门槛时，才重新评估；这类研究没有进入当前 TASKS。

## 8. Runtime 不变量

- 训练、评测和线上使用同一协议、预算和渲染规则。
- 模型不能直接触碰平台；所有副作用经过 ActionGate。
- 每个写动作都有权限、幂等、审批、意图记录和审计。
- “失败”“没有结果”“服务不可用”“结果未知”是四种不同状态。
- 会话恢复不重复已完成副作用。
- 事件回放只读，不执行工具。
- 跨租户对象表现为不存在，不能泄露可枚举信息。
- Runtime 变化必须同时检查数据同形和训练回放。

## 9. 验证入口

- 纯协议与循环：`tests/runtime/test_agent_loop.py`
- 设计符合性：`tests/runtime/test_design_conformance.py`
- 工具与幂等：`tests/runtime/test_tool_runtime_k6.py`、`test_idempotency.py`
- 会话与恢复：`tests/runtime/test_conversations.py`、`test_recovery_k8.py`
- RAG：`tests/runtime/test_retrieval.py`
- 沙盒/线上对照：`tests/runtime/test_tool_field_parity.py`、`test_tool_behaviour_parity.py`

需要 PostgreSQL 或 Redis 的测试如果被跳过，只能说明环境不完整，不能算验收通过。

旧设计、初审失败和施工过程保存在
[Runtime 历史索引](../archive/syncopate/pre-consolidation-v16/README.md#runtime-与-rag)。
