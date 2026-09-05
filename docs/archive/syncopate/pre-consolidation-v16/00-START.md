# Syncopate · 00 · START — 文档在哪 · 现在在哪 · 下一步做什么

> 📦 **历史快照，不代表当前状态。** 现行入口是
> [docs/syncopate/00-START.md](../../../syncopate/00-START.md)。

> 这是新人进入主线时读的第一份文档，也是主线唯一的导航。
>
> 它只回答三件事：每份文档放什么、项目现在到哪一步、下一步做什么。
>
> 历史和决策原因去 [22-decision-log.md](22-decision-log.md)，详细施工记录去对应专题文档，当前任务只看 [01-TASKS.md](01-TASKS.md)。

## 0 · 三十秒读懂

- **项目目标**：做一个能调用工具、连续多步办事的业务 agent；训练、评测和 serving 必须形成完整闭环。
- **当前方案**：从 [25-v15-contract.md](25-v15-contract.md) 开始，主线由旧的单轮 function calling 转为**多轮 + CoT + OPD**；当前数据版本是 v16。
- **当前机器**：训练之家已经从本地 4×5090 搬到 **Modal 2×B200**，全链跑通后再处理 B300。
- **已经完成**：v16 数据已经用最新脚本生成并通过全部现行门禁；SFT、考场、RL、OPD 的机器与机制冒烟都已跑通。
- **尚未完成**：固定管线的 smoke 档还没有把真实 SFT 产物一路接进 RL 和 OPD，因此真实 v16 数据尚未跑完一次 **SFT → RL → OPD** 全链冒烟；serving 的 27–30 号施工已经完成，但尚未做最终验收。

主线与 infra 共用一个仓库：

- **主线**：数据、SFT、RL、OPD、评测、serving。本文件负责导航。
- **infra 线**：训练基础设施、并行、通信和 kernel。入口是 [infra 00-START.md](../infra_exp/00-START.md)。
- 两线之间仍在处理的事项只写 [MAINLINE-INFRA.md](../../MAINLINE-INFRA.md)。

## 1 · 新人阅读顺序

1. [AGENTS.md](../../AGENTS.md)：全项目必须遵守的工作方式和安全边界。
2. [01-TASKS.md](01-TASKS.md)：主线唯一的当前任务队列。
3. [31-modal-and-new-stack.md](31-modal-and-new-stack.md)：Modal、B200、新技术栈和当前现场的最新详细说明。
4. 做训练或数据施工时，再读 [25-v15-contract.md](25-v15-contract.md) 与 [26-repair-rulers-and-data.md](26-repair-rulers-and-data.md)。
5. 做 serving 验收时，再读 [30-serving-release-checklist-and-runbook.md](30-serving-release-checklist-and-runbook.md)。
6. 只有需要引用旧 RL 数字或追查决策原因时，才读 [21-invalidated-numbers.md](21-invalidated-numbers.md) 与 [22-decision-log.md](22-decision-log.md)。

## 2 · 文档地图

文档按编号从早到晚排列。**编号大只表示形成得更晚，不表示可以跳过它依赖的协议。**

| 文档 | 放什么 | 当前角色 |
|---|---|---|
| **[00](00-START.md)** | 导航、当前状态、下一步 | 每位新人先读 |
| **[01](01-TASKS.md)** | 仍然开着的主线任务、顺序和完成条件 | 当前进度唯一入口 |
| **[06](06-rl-run-protocol.md)** | RL 运行协议、健康度和停机规则 | 训练参考；后续整理 |
| **[07](07-toolbox-and-runtime-design.md)** | Toolbox 与训练/runtime 设计 | 专题参考；后续整理 |
| **[08](08-machine-and-environment.md)** | 机器、环境、数据再生链和机器实测 | 环境参考；最新现场以 31 为准 |
| **[09](09-runtime-handoff.md)** | Runtime 现状、启动方式与生产级 agent loop | Serving 参考 |
| **[10](10-rag-retrieval.md)** | RAG 选型、验收和 runtime 契约 | RAG 专题 |
| **[11](11-runtime-acceptance.md)** | Runtime 验收口径 | Serving 验收参考 |
| **[13](13-diversity-gates.md)** | 数据多样性与三桶隔离门禁 | v16 数据门禁参考 |
| **[18](18-pipeline-assumption-probes.md)** | 训练管线关键假设的探针与证据 | 调查参考；后续整理 |
| **[21](21-invalidated-numbers.md)** | 已失效数字与仍可引用数字的登记 | 引用旧 RL 数字前查 |
| **[22](22-decision-log.md)** | 决策、调查过程和历史 | 不用于判断当前进度 |
| **[23](23-research-question.md)** | 早期研究问题 | 后续判断是否归档 |
| **[24](24-unified-conversation-training.md)** | v14 统一会话训练方案 | 上一阶段方案；后续归档 |
| **[25](25-v15-contract.md)** | 多轮 + CoT + OPD 的 v15 契约 | 当前协议分界线 |
| **[26](26-repair-rulers-and-data.md)** | v15/v16 尺子、数据维修和新栈施工细节 | 当前施工与证据参考 |
| **[27](27-serving-harness-plan.md)** | Serving K0–K11 施工记录 | 已施工，未验收 |
| **[28](28-serving-middleware-hazards.md)** | Serving 中间件风险与防护登记 | 验收和排障参考 |
| **[29](29-serving-k0-inventory.md)** | Serving K0 现状盘点与证据 | 已施工，未验收 |
| **[30](30-serving-release-checklist-and-runbook.md)** | Serving 上线清单、runbook 和复盘模板 | 当前 serving 验收入口 |
| **[31](31-modal-and-new-stack.md)** | Modal、B200、新栈、最新进度与下一步 | 当前现场最新权威 |
| **[NARRATIVE](NARRATIVE-AND-RESUME.md)** | 项目叙事与简历材料 | 对外说明，不是施工入口 |

docs/archive/ 中的文件只表示历史；旧文档的归档会在 00/01 恢复后再逐份处理。

## 3 · 现在到哪一步

### 数据与训练

- v16 题库、切分和 SFT 训练数据已经生成，现行出厂检查全部通过。
- SFT、考场、verl 0.9 RL、OPD 都已分别证明“机制能跑”。
- 这些结果还不能证明真实数据上的完整训练链健康，也不能当作候选模型验收。

### Serving

- [27](27-serving-harness-plan.md) 到 [30](30-serving-release-checklist-and-runbook.md) 的功能施工已经完成。
- 这部分仍是“施工完成、尚未验收”，不能写成已经可以发布。

如果本节与其他旧文档冲突，当前现场以 [31](31-modal-and-new-stack.md) 为准，并同步修正 [01](01-TASKS.md)。

## 4 · 下一步做什么

当前唯一队首是：先补齐固定管线 smoke 档的上下游接线，再在 **Modal 2×B200** 上用真实 v16 数据跑通：

**sft-train → sft-eval → sft-select → merge → exam → rl-train → rl-adapter → rl-eval → opd-train → opd-eval**

完成条件：

1. 先用本地检查证明 smoke 档的 SFT 输出会被选点、合并和考场读取，RL 从本轮合并模型开始，OPD 从本轮 RL/SFT adapter 开始。
2. 每个阶段都从唯一入口 [scripts/v16_pipeline.sh](../../scripts/v16_pipeline.sh) 启动，阶段间产物能直接衔接。
3. 每个阶段退出正常、内置判据全绿，模型与 adapter 产物可被下一阶段真实加载。
4. OPD 必须产生有效蒸馏步，不能用裸底座上的“几乎全跳步”冒充通过。
5. 关键读数和产物路径写入审计文件，并据此更新 [31](31-modal-and-new-stack.md) 和 [01](01-TASKS.md)。

这只是**真实数据全链冒烟**，不是候选训练完成，也不是 serving 上线验收。实际操作顺序见 [31](31-modal-and-new-stack.md) 和 [modal_app/README.md](../../modal_app/README.md)，本文不复制命令。

## 5 · 必须记住的项目底线

- 产生 Modal 费用或改变云端状态前，必须得到用户对该次运行的明确授权。
- 训练和评测只走固定管线；共享 Volume 同一产物同时只能有一个写者。
- “代码写完”“机制冒烟通过”“最终验收通过”是三件不同的事。
- 训练样例必须与 runtime 真实请求同形；数据隔离必须由代码和落盘复核共同保证。
- 不从旧聊天或旧状态表猜进度，不引用 [21](21-invalidated-numbers.md) 中已失效的数字。

## 6 · 维护这两份入口

- **00 只保留**文档地图、当前状态和下一步；不放日期流水账、命令大全、决策过程或经验教训。
- **01 只保留**仍然开着的任务、依赖、负责人和完成条件；做完的任务不留成历史长表。
- Modal/B200 的详细现场就地更新 31；决策原因写 22；专题结论回到对应专题文档；历史由 git 保留。
- 项目状态变化时，先把证据写进对应权威文档，再在同一次修改中更新 01 和本文件的简短摘要。
