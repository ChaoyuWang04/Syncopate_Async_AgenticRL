---
name: syncopate-docs-map
description: Syncopate 主线 8 份、infra 7 份现行文档的职责、阅读顺序与历史归档入口
metadata:
  node_type: memory
  type: project
  originSessionId: 254d8707-7512-4e9b-bd89-6e1eeec39011
  modified: 2026-09-04T00:00:00.000Z
---

# Syncopate 文档地图

主线现行目录只保留 8 份文档：

```text
docs/syncopate/00-START.md       唯一导航：文档在哪、现在在哪、下一步
docs/syncopate/01-TASKS.md       唯一当前任务队列
docs/syncopate/02-SYSTEM.md      系统全景、模块边界、协议和唯一事实来源
docs/syncopate/03-DATA.md        v16 数据、切分、门禁和产物
docs/syncopate/04-TRAINING.md    SFT / Exam / RL / OPD / Eval
docs/syncopate/05-COMPUTE.md     Modal / B200 / 依赖 / Volume / 探针
docs/syncopate/06-RUNTIME.md     AgentLoop / 工具 / ActionGate / 会话 / RAG
docs/syncopate/07-SERVING.md     API / PG / Redis / Celery / 发布 / SLO / DR
```

Infra 现行目录只保留 7 份文档：

```text
docs/infra_exp/00-START.md                 唯一导航
docs/infra_exp/01-TASKS.md                 唯一当前 B 系列队列
docs/infra_exp/02-SYSTEM.md                边界、研究层次和证据流
docs/infra_exp/03-TRAINING.md              分布式训练、异步 RL、训推一致性
docs/infra_exp/04-SERVING.md               引擎拓扑、调度、缓存、解码与 SLO
docs/infra_exp/05-COMPUTE-AND-KERNELS.md   B200/B300、通信、精度与 kernel
docs/infra_exp/06-EXPERIMENTS.md           B 编号、预注册、证据与报告生命周期
```

阅读顺序：

1. `AGENTS.md`
2. `docs/syncopate/00-START.md`
3. `docs/infra_exp/00-START.md`
4. `docs/syncopate/01-TASKS.md`
5. `docs/syncopate/02-SYSTEM.md`
6. 只读与当前任务有关的一份专题文档

历史施工、旧机器、旧数字与决策都在：

```text
docs/archive/syncopate/pre-consolidation-v16/README.md
docs/archive/infra_exp/legacy-4x5090/README.md
```

其中：

- 旧 RL 数字引用前查归档里的 `21-invalidated-numbers.md`。
- 旧决策原因查归档里的 `22-decision-log.md`。
- v15/v16 与 Modal/B200 的详细施工证据查归档里的 `26`、`31`。
- Serving K0–K11 的施工证据查归档里的 `27`～`30`。
- 归档中的未完成方框不是当前任务；只有 `01-TASKS.md` 中的任务有效。

对外叙事和简历保存在：

```text
docs/narrative/syncopate-resume.md
docs/narrative/infra-resume.md
```

维护规则：

- 状态变化先更新对应专题，再同步 TASKS 和 START。
- START 不放历史；TASKS 不放完成记录；专题不放散落待办。
- 同一事实只在一处写全，其他文档只链接。
- 跨线事项只放唯一负责方 TASKS，沟通用 Codex 独立任务消息工具，不建交互文档。
- 代码常量和审计证据高于记忆摘要；记忆只负责导航。
