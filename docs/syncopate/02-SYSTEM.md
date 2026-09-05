# Syncopate · 系统总览

> 本文只说明整套系统如何连接、每个模块负责什么、哪些文件是唯一事实来源。
> 当前工作排期看 [01-TASKS.md](01-TASKS.md)，专题细节看 03～07。

## 1. 系统要交付什么

Syncopate 的第一目标是一个能调用工具、连续多步完成业务任务的 agent。

系统必须同时满足三件事：

1. 数据里的请求、历史、工具、观测和回答与线上真实请求同形。
2. SFT、RL、OPD 和评测使用同一套消息契约、模型路径和长度预算。
3. 线上每个动作经过权限、审批、幂等、审计和恢复机制，而不是让模型直接碰外部世界。

异步 RL 和训练基础设施服务于这个产品闭环，不是独立终点。

## 2. 两条主流程

### 训练流程

```text
cases → menus → split → gates/supply
      ├→ rl-data
      └→ teacher → sft-data
                    → sft-train → sft-eval → sft-select → merge → exam
                    → rl-train → rl-adapter → rl-eval
                    → opd-train → opd-eval
```

固定入口是 [scripts/v16_pipeline.sh](../../scripts/v16_pipeline.sh)。数据细节在
[03-DATA.md](03-DATA.md)，训练和评测细节在 [04-TRAINING.md](04-TRAINING.md)。

### 线上流程

```text
用户/API
  → PostgreSQL 中的 run + outbox
  → dispatcher
  → Redis/Celery
  → worker
  → AgentLoop
  → ActionGate
  → 工具 / RAG / session 信令
  → 事件与审计
  → SSE / 前端 / 用户反馈
```

Runtime 负责“模型如何思考和行动”，Serving 负责“请求如何可靠地进入、执行、恢复和发布”。
两者边界分别在 [06-RUNTIME.md](06-RUNTIME.md) 和 [07-SERVING.md](07-SERVING.md)。

## 3. 模块边界

| 模块 | 负责 | 不负责 |
|---|---|---|
| 数据 | 造题、菜单、切分、SFT/RL 数据、门禁 | 训练参数、线上排队 |
| 训练 | SFT、Exam、RL、OPD、模型选择与评测 | API、数据库运维 |
| Compute | Modal、GPU、镜像、Volume、依赖和机器探针 | 模型质量结论 |
| Runtime | 消息契约、AgentLoop、工具、安全闸、会话和 RAG | 服务发布、队列运维 |
| Serving | API、数据库、队列、worker、事件、恢复、发布和 SLO | 训练数据构造 |
| Infra | 分布式训练、通信、kernel 和性能实验 | 主线任务排期 |

## 4. 当前契约

### 版本

- **v15 是交互协议版本**：行为通过 `session.defer`、`session.clarify`、
  `session.reject` 和 `session.report` 表达，终答是自然语言。
- **v16 是数据版本**：题库、切分和 SFT/RL 数据都从 v16 常量派生。
- 协议没有变化时，不因为数据升级而机械地把 v15 重命名成 v16。

### 三个输出通道

| 通道 | 内容 |
|---|---|
| think | 模型内部推理；v15 默认启用，可为空但必须符合统一模板 |
| tool | 业务工具和 session 信令；机器可读结构只走这里 |
| final | 给用户看的自然语言，不包含 JSON 壳、字段壳或工具标签 |

行为不是模型另填的标签，而是由实际动作推导：

- 调业务工具并以自然语言结束：`tool_call`
- 不调工具直接回答：`answer`
- 调终止性 session 信令：分别得到 `defer`、`clarify`、`reject`

## 5. 唯一事实来源

| 事实 | 唯一来源 |
|---|---|
| 数据版本和默认数据目录 | [syncopate/pipeline/split.py](../../syncopate/pipeline/split.py) |
| 学生、教师和测试分词器 | [syncopate/core/model_paths.py](../../syncopate/core/model_paths.py) |
| 协议、session 信令和字段规则 | [syncopate/core/contract.py](../../syncopate/core/contract.py) |
| 长度、thinking 和采样预算 | [syncopate/train/rollout_budget.py](../../syncopate/train/rollout_budget.py) |
| 工具 schema | [syncopate/core/tool_registry.py](../../syncopate/core/tool_registry.py) |
| Runtime 工具治理 | [syncopate/runtime/tool_governance.py](../../syncopate/runtime/tool_governance.py) |
| 数据到 OPD 的执行顺序 | [scripts/v16_pipeline.sh](../../scripts/v16_pipeline.sh) |
| Modal 镜像和机器探针 | [modal_app/stack_probe.py](../../modal_app/stack_probe.py) |
| 数据库结构 | [syncopate/runtime/migrations](../../syncopate/runtime/migrations) 与 [schema.snapshot.txt](../../syncopate/runtime/schema.snapshot.txt) |
| 当前任务 | [01-TASKS.md](01-TASKS.md) |

文档解释这些来源，不复制一套可漂移的默认值。代码与文档冲突时，先核对真实入口和审计证据，再修正文档或代码。

## 6. 当前系统状态

- 数据已经到达可用于真实训练冒烟的状态。
- B02 已用真实 v16 数据完成 SFT 30 次、RL 2 次和 OPD 1 次真实更新，并证明合并模型、checkpoint 与 adapter 连续传递。
- 固定 smoke 的机械链路已通过；Exam、RL 长回复和 OPD 短评测仍有质量 WARN，所以不是 candidate 或稳定性能 baseline。
- Runtime 与 Serving 主体能力已经实现；Serving 的当前环境正式验收仍未结束。
- B200 是现行训练环境；B300 没有经过本项目复核。

## 7. 不变量

- 训练样本与线上请求逐项同形。
- EVAL、SFT、RL 三桶从源头隔离，写盘后再独立复核。
- 注册、配置或文件存在不等于机制生效；真实调用路径必须留下判据。
- 下一阶段必须证明自己读取了上一阶段的确切产物。
- smoke 只证明线路和机制；candidate 才讨论质量和晋级。
- 模型只能提出动作，所有外部副作用必须经过 ActionGate。
- 不确定、结果未知或依赖不可用时，系统保守停止并留下审计，不猜成功。
- 旧实验数字只用于历史追溯；引用前看
  [作废数字档案](../archive/syncopate/pre-consolidation-v16/21-invalidated-numbers.md)。
