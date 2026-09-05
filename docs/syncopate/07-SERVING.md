# Syncopate · Serving

> 本文是 API、数据库、队列、worker、事件、恢复、发布和 SLO 的唯一现行说明。
> 模型的多轮行为、工具和 RAG 契约看 [06-RUNTIME.md](06-RUNTIME.md)。

## 1. 当前状态

Serving 的 K0–K11 主体能力已经施工完成，代码和测试覆盖了 API、数据库、Outbox、Celery、
状态机、AgentLoop、工具、SSE、恢复、发布闸和回流。

当前结论仍是：**已施工，未完成正式验收**。

原因不是缺少主体模块，而是当前部署环境上的 Celery 负载、连接预算、真实发布/回滚、
模型权重恢复、生产备份、前端和干净 SLO 基线尚未全部演练。所有工作只登记在
[TASKS 的 T3](01-TASKS.md#t3--serving-正式验收)。

## 2. 生产结构

```text
客户端
  → FastAPI
  → PostgreSQL：run + event + outbox 同事务
  → dispatcher：先发布、后标记
  → Redis broker
  → Celery worker
  → AgentLoop + ActionGate + ToolRuntime + RAG
  → PostgreSQL：状态、事件、审计、用量、checkpoint
  → SSE / 前端

sweeper / reconciler
  → 回收过期 lease
  → 重投丢失任务
  → 对账结果未知的写动作
  → 清理到期数据
```

PostgreSQL 是业务事实来源，Redis/Celery 是投递层，不能把任务返回值或业务状态另存成第二份事实。

## 3. 组件

| 组件 | 正式入口 | 职责 |
|---|---|---|
| API | [api.py](../../syncopate/runtime/api.py) | 鉴权、建 run、查询、取消、恢复、审批、反馈和 SSE |
| 数据库 | [db.py](../../syncopate/runtime/db.py) | 状态、事件、Outbox、幂等、用量和审计事务 |
| Migration | [migrations](../../syncopate/runtime/migrations) | 数据库结构唯一来源 |
| Dispatcher | [dispatcher.py](../../syncopate/runtime/dispatcher.py) | 从 Outbox 发布到 Celery |
| Broker | [celery_app.py](../../syncopate/runtime/celery_app.py) | Redis 队列与 Celery 设置 |
| Worker | [worker.py](../../syncopate/runtime/worker.py) | claim、lease、执行、事件和收尾 |
| Sweeper | [sweeper.py](../../syncopate/runtime/sweeper.py) | 恢复、对账、超龄检查和清理 |
| 事件分层 | [event_layer.py](../../syncopate/runtime/event_layer.py) | 区分 public、trace 和 internal 字段 |
| 发布闸 | [release.py](../../syncopate/runtime/release.py) | 全局停机和 automation tier 上限 |
| 指标 | [metrics.py](../../syncopate/runtime/metrics.py) | 队列、运行、工具、成本与告警 |
| 回流 | [flywheel.py](../../syncopate/runtime/flywheel.py) | 反馈、标注和训练候选导出 |

## 4. 状态与投递

run 只有六种状态：

```text
queued → running → waiting_for_user
                 → succeeded
                 → failed
                 → cancelled
```

终态不能重新打开；rerun 必须创建新 run 并关联 parent。所有状态变化统一经过
`transition_run`，不能直接写 status。

建 run 时，业务记录、`run.created` 事件和 Outbox job 在同一事务提交。
Dispatcher 必须先把消息发布到 broker，再标记 dispatched：

- 发布成功但标记失败：下轮可能重复发布，由 worker 的幂等与终态检查兜住。
- 先标记再发布：进程在中间崩溃会永久丢任务，因此禁止。

Celery 采用至少一次投递；正确性来自数据库状态机、lease、checkpoint、幂等键和唯一约束，不依赖“消息只来一次”。

## 5. 可靠性与副作用

### Lease 与恢复

- worker claim 后持有 lease，并按固定间隔续租。
- 续租失败时停止继续执行。
- 过期 lease 只由 sweeper 回收，claim 不能悄悄接管。
- 恢复从最新完整 checkpoint 继续。
- 已完成的只读工具不重调，写工具依靠外部幂等键和意图日志防重。

### 写工具

- 执行前先记录意图。
- 同一业务幂等键只允许一条真实执行记录。
- `response_lost` 表示副作用可能成功但响应丢失，禁止自动重发。
- Reconciler 按幂等键查询平台账本；能确认就回填，不能确认就进入人工处理。
- 手工 repair 必须记录操作者、原因、前后值、事件和审计。

### 事件

- 事件序号由数据库原子分配，不能用 `MAX(seq)+1`。
- SSE 断线后按游标补发；public 流隐藏内部字段。
- 回放只读取事件，不调用工具。
- public 事件只要求严格递增，不要求连续，因为 internal 事件可能被过滤。

## 6. 启动顺序

正式环境的逻辑顺序是：

1. PostgreSQL，并运行 Alembic 到 head。
2. Redis，启用认证、AOF 和 no-eviction。
3. 模型服务与 Decider 端点。
4. Dispatcher。
5. Sweeper/Reconciler。
6. Celery workers，按 interactive、batch、maintenance 分队列。
7. FastAPI。
8. Runtime smoke、SLO 读数和告警检查。

对应入口：

- [scripts/serving/pg_bootstrap.sh](../../scripts/serving/pg_bootstrap.sh)
- [scripts/serving/redis_bootstrap.sh](../../scripts/serving/redis_bootstrap.sh)
- `python -m syncopate.runtime.dispatcher`
- `python -m syncopate.runtime.sweeper`
- `celery -A syncopate.runtime.celery_app worker ...`
- `uvicorn syncopate.runtime.api:app ...`
- [runtime_smoke.py](../../scripts/serving/runtime_smoke.py)
- [slo_readout.py](../../scripts/serving/slo_readout.py)

旧 4×5090 的模型服务启动脚本不是 B200 正式验收结论。当前环境的进程数、池大小、
队列和模型端点必须在 T3 中按真实机器重新登记。

## 7. 发布与回滚

发布顺序：

1. 迁移和 schema 快照一致。
2. Runtime、数据库和 broker 测试全绿，不能把依赖缺失的 skip 算通过。
3. 在干净环境运行 smoke、故障演练和 SLO。
4. 从 D 档只读开始，再到 C 档逐次审批，最后才考虑更高自动化。
5. 每一级只按预注册门槛放量。

当前已有的控制面：

- `SYNCOPATE_RELEASE_HALTED=1`：全局停止新副作用。
- `SYNCOPATE_RELEASE_MAX_TIER`：限制最高自动化档位。
- `SYNCOPATE_DISABLED_TOOLS`：按工具停用。
- Celery cancel consumer：停止继续取某个队列。
- SIGTERM warm shutdown：让 worker 完成当前任务后退出。

回滚不是直接恢复数据库旧结构。数据库遵守 expand/contract；代码回滚、队列 drain、
旧 worker 退场和新 worker 启动必须留下事件与审计。

## 8. SLO 与告警

现行面板至少覆盖：

- API 创建成功率与延迟；
- 意图端到端延迟；
- 队列最老等待时间；
- 卡死 run 和过期 lease；
- 写工具错误率、重复拦截和待对账数量；
- SSE 补发完整性；
- 单 run 与 org token/成本；
- 预算触顶与人工等待；
- 发布档位和被禁工具。

旧开发库读数含测试污染，不能作为生产基线。正式验收必须在干净数据库和当前部署形态重新生成。

## 9. 六张故障卡

| 故障 | 第一原则 |
|---|---|
| queue lag 上升 | 先分清 Outbox、broker、worker 还是模型端点；先停写和 batch，不重跑全部 running |
| run 卡死 | 先查 lease、sweeper、waiting 和 attempts；不手工改 status |
| 写工具报错 | 先假设副作用已经发生；结果未知只对账，不重发 |
| SSE 断线 | 执行事实仍在数据库；按游标补发，不重跑任务 |
| migration 失败 | 先查版本与漂移；不为迁就脏库反向修改 schema 快照 |
| token 异常 | 先找不收敛 run 和 org 流量；降低预算或暂停，不把 waiting 记成失败 |

完整历史卡片保存在
[30 号归档](../archive/syncopate/pre-consolidation-v16/30-serving-release-checklist-and-runbook.md)；
现行查询脚本是 [runbook_queries.py](../../scripts/serving/runbook_queries.py)。

## 10. 正式验收边界

Serving 只有在以下证据都齐全后才能写“验收通过”：

- 当前机器上的真实 Celery 负载与连接预算；
- 发布停机、分队列停止、warm drain 和代码回滚实演；
- 数据层与模型权重层灾备恢复；
- 真人灰测前的数据库备份；
- 对外场景的多层并发限制；
- 前端构建、反馈入口和 SSE；
- 干净环境的新 SLO 基线；
- 每个失败场景都能指向可执行 runbook。

这些项目的状态只在 TASKS 的 T3 维护，本文不保存第二份待办表。

## 11. 历史资料

Serving 的施工计划、风险调查、K0 盘点和旧上线清单已归档：

- [27-serving-harness-plan.md](../archive/syncopate/pre-consolidation-v16/27-serving-harness-plan.md)
- [28-serving-middleware-hazards.md](../archive/syncopate/pre-consolidation-v16/28-serving-middleware-hazards.md)
- [29-serving-k0-inventory.md](../archive/syncopate/pre-consolidation-v16/29-serving-k0-inventory.md)
- [30-serving-release-checklist-and-runbook.md](../archive/syncopate/pre-consolidation-v16/30-serving-release-checklist-and-runbook.md)

它们保留施工证据和决策背景，但不代表当前部署已经通过验收。
