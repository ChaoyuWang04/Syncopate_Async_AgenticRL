# Infra 旧栈归档 · 4×5090 / E 系列

> 本目录保存 infra 整理前的原始文档。它们用于追查历史、实验方法和旧证据，**不代表当前 B200/B300 状态或默认配置**。
>
> 当前入口：[Infra START](../../../infra_exp/00-START.md)；当前任务：[Infra TASKS](../../../infra_exp/01-TASKS.md)。

## 1. 适用边界

这些材料主要形成于以下环境中的一个或多个：

- 4×RTX 5090、sm_120、无 P2P/NVLink；
- verl 0.8、vLLM 0.12、FSDP1；
- Qwen3-4B 或旧 MoE 模型；
- v12/v13/v14 数据和旧训练契约。

当前环境是 Modal 2×B200、verl 0.9、vLLM 0.28 和新的 MoE 学生。旧绝对数字、噪声地板、默认参数与硬件结论不能直接迁移。

## 2. 原入口与过程材料

| 文件 | 历史职责 |
|---|---|
| [00-START.md](00-START.md) | 整理前的入口、5090 运行规则和历史状态 |
| [01-TASKS.md](01-TASKS.md) | 旧完成记录、停放项和最初 B200 候选队列 |
| [02-DECISIONS.md](02-DECISIONS.md) | 旧栈默认值和决定过程 |
| [README.legacy.md](README.legacy.md) | 原 E 报告索引和模板 |
| [TRACKS.md](TRACKS.md) | 旧 Track A/B 视图 |
| [PRIMER-precision-sm120.md](PRIMER-precision-sm120.md) | sm_120 低精度入门 |
| [STORY-async-lora-weight-sync.md](STORY-async-lora-weight-sync.md) | FSDP1/LoRA 权重同步历史故事 |
| [NARRATIVE-AND-RESUME.md](NARRATIVE-AND-RESUME.md) | 旧 5090 简历与长叙事 |
| [MAINLINE-INFRA.retired.md](MAINLINE-INFRA.retired.md) | 已退役的两线人工交互文档 |
| [focus-migration-2026-08.md](focus-migration-2026-08.md) | 5090 机器迁移与焦点变化 |
| [distributed-training-design-v0.1.md](distributed-training-design-v0.1.md) | 4×5090 分布式训练设计 |
| [ostinato-project-design-v0.2.md](ostinato-project-design-v0.2.md) | 旧单卡/4卡 infra 与 kernel 设计 |
| [claude-memory-infra-line-state.md](claude-memory-infra-line-state.md) | 整理前 617 行 infra 历史记忆快照 |

`MAINLINE-INFRA` 只保留作历史。以后跨线沟通使用 Codex 任务消息工具；未完成工作只进入唯一负责方的 TASKS。

### 退役时的未闭合事项去向

| 旧交互文档中的事项 | 现在的去向 |
|---|---|
| 新软件栈和 B200 搬家 | 已成为主线 Compute 与 infra SYSTEM 的当前基线，不再是待办 |
| B200 通信、FA4、低精度、异步和 PD 探针 | infra B01～B10 |
| `logprob_coverage` 约 0.1% 占位值 | infra B02；先在新栈复现，再决定是否修 |
| Modal 抢占恢复、缓存和观测 | infra B11 |
| B300 | infra B12 负责硬件/系统，主线 T4 负责固定业务全链 |
| v16、CoT、OPD 和真实上游接线 | 主线 T1/T2 |
| 当前部署的真实 Serving 验收 | 主线 T3；引擎拓扑实验归 infra B08/B09 |
| W&B 全链读数 | 合并进主线 T1-7 的证据条件 |
| worker 日成本 CLI 透传 | 代码已存在，不再列任务 |
| 旧 sm_120 上游草稿 | 保留历史包；只有 B06/B10 在新栈复现后才决定是否继续 |

## 3. E 系列报告地图

| 主题 | 报告 |
|---|---|
| 步结构、异步与等待 | E01、E08、E13、E14、E17 |
| 数据并行、通信与同步正确性 | E02、E03、E18、E21、E22 |
| MoE、稀疏与训练喂数 | E07、E11、E25、E26 |
| 训练学习、采样和评测 | E20、E23、E24、E27 |
| 低精度、sm_120 与 kernel | E16、E19、E30、E31 |
| checkpoint | E29 |
| 4×5090 Serving 与编排 | E32、E33 |

目录中的报告文件名就是完整索引。缺失的编号表示历史上未建档、已停放或存放在更早的归档；编号不连续不是文件丢失。

## 4. 仍可复用的方法

- 比较两个本应相同的东西：跨 rank 梯度、推送前后权重、逐 token 输出。
- 机制存在不等于真实路径已经接上。
- 跑前登记预测、阈值、正负对照和证据位置。
- 一次只改变一个关键变量，并先量 before 的噪声。
- 先证明正确，再看组件速度，最后看端到端与任务质量。
- 负面结果、不适用范围和被判据推翻的解释都要保留。

这些是方法，不是对当前硬件的性能承诺。

## 5. 原始证据

旧 E 系列的平铺证据仍保存在 [`_audit/infra/`](../../../../_audit/infra/)。新的 B 系列改用 `_audit/infra/Bxx/<arm>/`，不会覆盖旧文件。
