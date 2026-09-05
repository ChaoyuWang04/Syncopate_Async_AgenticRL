# Syncopate v16 文档整合前快照

> 本目录是历史档案，不是当前操作说明。
>
> 当前状态从 [START](../../../syncopate/00-START.md) 进入，当前任务只看
> [TASKS](../../../syncopate/01-TASKS.md)。这里保留整合前的原始文档、施工过程、
> 失败记录、旧机器信息、旧数字和决策背景。

## 为什么归档

整合前的 `docs/syncopate/` 同时混放了当前规范、施工日志、验收证据、旧计划和历史决策。
同一主题跨多个时期存在多份文档，容易把“当时正确”误读成“现在仍然正确”。

本次整合把现行说明收敛为 8 份：

- [00-START](../../../syncopate/00-START.md)
- [01-TASKS](../../../syncopate/01-TASKS.md)
- [02-SYSTEM](../../../syncopate/02-SYSTEM.md)
- [03-DATA](../../../syncopate/03-DATA.md)
- [04-TRAINING](../../../syncopate/04-TRAINING.md)
- [05-COMPUTE](../../../syncopate/05-COMPUTE.md)
- [06-RUNTIME](../../../syncopate/06-RUNTIME.md)
- [07-SERVING](../../../syncopate/07-SERVING.md)

## 数据

| 档案 | 当时的用途 | 现行去向 |
|---|---|---|
| [13-diversity-gates.md](13-diversity-gates.md) | D1–D11 与 L1/L2 形成过程 | [03-DATA](../../../syncopate/03-DATA.md) |
| [18-pipeline-assumption-probes.md](18-pipeline-assumption-probes.md) | 管线前提的一次性审计 | 数据不变量进 03，训练不变量进 04 |
| [24-unified-conversation-training.md](24-unified-conversation-training.md) | v14 统一会话训练计划 | 被 v15/v16 方案取代 |
| [25-v15-contract.md](25-v15-contract.md) | v15 契约施工与决策 | 现行协议进 02、03、04、06 |
| [26-repair-rulers-and-data.md](26-repair-rulers-and-data.md) | v15/v16 数据维修与 run 记录 | 当前数据进 03，训练状态进 04 |

## 训练

| 档案 | 当时的用途 | 现行去向 |
|---|---|---|
| [06-rl-run-protocol.md](06-rl-run-protocol.md) | 旧训练协议、指标和停机规则 | [04-TRAINING](../../../syncopate/04-TRAINING.md) |
| [18-pipeline-assumption-probes.md](18-pipeline-assumption-probes.md) | merge、rank、配对比较探针 | [04-TRAINING](../../../syncopate/04-TRAINING.md) |
| [21-invalidated-numbers.md](21-invalidated-numbers.md) | 旧 RL 作废数字的机器可读登记 | 继续作为历史隔离源；现行规则进 04 |
| [23-research-question.md](23-research-question.md) | 早期异步 RL 研究问题 | 暂不属于当前主线任务 |
| [24-unified-conversation-training.md](24-unified-conversation-training.md) | v14 SFT/RL/OPD 计划 | 被 v15/v16 取代 |
| [25-v15-contract.md](25-v15-contract.md) | v15 R0–R8 施工书 | 稳定训练契约进 04 |
| [26-repair-rulers-and-data.md](26-repair-rulers-and-data.md) | W0–W5 与 B200 施工证据 | 最新状态进 03～05 |

## Compute

| 档案 | 当时的用途 | 现行去向 |
|---|---|---|
| [08-machine-and-environment.md](08-machine-and-environment.md) | 5090、Serving 环境和早期 Modal 读数 | [05-COMPUTE](../../../syncopate/05-COMPUTE.md) |
| [31-modal-and-new-stack.md](31-modal-and-new-stack.md) | Modal/B200 迁移过程和最新施工现场 | 当前环境进 05，训练状态进 04 |
| [modal-app-README.md](modal-app-README.md) | 新旧 Modal 探针混合说明 | [现行 Modal README](../../../../modal_app/README.md) |

## Runtime 与 RAG

| 档案 | 当时的用途 | 现行去向 |
|---|---|---|
| [07-toolbox-and-runtime-design.md](07-toolbox-and-runtime-design.md) | Toolbox、失败注入和早期 Runtime 设计 | [06-RUNTIME](../../../syncopate/06-RUNTIME.md) |
| [09-runtime-handoff.md](09-runtime-handoff.md) | Runtime 启动与多年施工流水 | Runtime 进 06，Serving 进 07 |
| [10-rag-retrieval.md](10-rag-retrieval.md) | 沙盒与 Runtime RAG 选型、标定和旧任务 | RAG 现行契约进 06 |
| [11-runtime-acceptance.md](11-runtime-acceptance.md) | 初审失败、修复和旧验收表 | 不变量进 06，部署验收进 07 |
| [25-v15-contract.md](25-v15-contract.md) | session 信令与自然语言终答形成过程 | [06-RUNTIME](../../../syncopate/06-RUNTIME.md) |

## Serving

| 档案 | 当时的用途 | 现行去向 |
|---|---|---|
| [27-serving-harness-plan.md](27-serving-harness-plan.md) | K0–K11 施工计划与完成记录 | [07-SERVING](../../../syncopate/07-SERVING.md) |
| [28-serving-middleware-hazards.md](28-serving-middleware-hazards.md) | Celery、Redis、PG 风险和实验候选 | 稳定运维规则进 07 |
| [29-serving-k0-inventory.md](29-serving-k0-inventory.md) | K0 代码与能力盘点 | 架构结论进 07 |
| [30-serving-release-checklist-and-runbook.md](30-serving-release-checklist-and-runbook.md) | 旧发布清单、六张卡和基线 | 现行 runbook 与验收边界进 07 |

## 治理、入口与对外材料

| 档案 | 当时的用途 | 现行去向 |
|---|---|---|
| [00-START.md](00-START.md) | 整合前导航快照 | [现行 START](../../../syncopate/00-START.md) |
| [01-TASKS.md](01-TASKS.md) | 整合前任务队列快照 | [现行 TASKS](../../../syncopate/01-TASKS.md) |
| [22-decision-log.md](22-decision-log.md) | 主线决策与调查日志 | 永久历史资料 |
| [NARRATIVE-AND-RESUME.md](NARRATIVE-AND-RESUME.md) | 整合前简历快照 | [现行简历](../../../narrative/syncopate-resume.md) |
| [claude-memory-syncopate-docs-map.md](claude-memory-syncopate-docs-map.md) | 整合前的 Claude 文档地图 | [现行记忆地图](../../../../.claude/memory/syncopate-docs-map.md) |
| [claude-memory-modal-migration-state.md](claude-memory-modal-migration-state.md) | 整合前的 Modal 施工记忆 | [现行 Modal 记忆](../../../../.claude/memory/modal-migration-state.md) |

## 使用规则

- 可以从这里追查“为什么这样做”和“当时发生了什么”。
- 不从这里决定当前任务、启动命令、模型路径、依赖版本或验收状态。
- 档案中的未完成方框不自动成为当前任务；只有现行 TASKS 中登记的工作才开着。
- 档案中的旧数字仍受 [21-invalidated-numbers.md](21-invalidated-numbers.md) 约束。
- 文件顶部的归档横幅是状态标识；正文保持整合前快照，便于追溯。
- 快照正文里的相对路径也按当时位置原样保留；当前导航和有效去向以本索引为准。
