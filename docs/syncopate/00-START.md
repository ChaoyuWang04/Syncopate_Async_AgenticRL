# Syncopate · START

> 2026-09-07 起，训练与推理 infra 是本项目主线。请先进入 [Infra START](../infra_exp/00-START.md) 与 [Infra TASKS](../infra_exp/01-TASKS.md)。

## 当前方向

以真实训练入口的速度画像为起点，观察瓶颈后安排定位、改动和原入口A/B；full/LoRA、SFT/RL/async/OPD等是场景与参考维度，不是预设探针队列。范围上限与数据选择只认Infra SYSTEM。实验负责人负责调查、复现、测试、结论和 DRAFT；正式 PR 考据与提交由 upstream 负责人处理。

原场景数据构建和完整学习排期停止，模型仅作为计算负载，不要求业务能力提升。两个 Lab 默认独立；Harness/tool runtime 不在本项目当前施工范围。

## 本目录用途

| 文档 | 当前作用 |
|---|---|
| [TASKS](01-TASKS.md) | 原业务任务的停止/暂缓边界 |
| [SYSTEM](02-SYSTEM.md) | 保留业务系统实现说明，不驱动新实验 |
| [DATA](03-DATA.md) | 已停止构建的 v16 数据结构和历史证据 |
| [TRAINING](04-TRAINING.md) | 现有业务 runbook 的真实能力和历史训练结果 |
| [COMPUTE](05-COMPUTE.md) | 现行主要资源平台、硬件调查范围、隔离与运行边界 |
| [RUNTIME](06-RUNTIME.md)、[SERVING](07-SERVING.md) | 保留系统实现与未验收边界；不接管独立 Lab |

B01 认证、B02 机械全链、B03 局部正确性和 B13 微测试均为既有证据，不代表新的模型/卡型/性能探针已经验收。不存在当前完整学习预算，也没有 candidate 通过声明。
