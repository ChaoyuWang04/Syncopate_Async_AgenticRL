# Syncopate · START

> 这是新人进入主线后读的第一份文档，也是唯一导航。
>
> 它只回答三件事：文档在哪里、项目现在到哪一步、下一步做什么。
> 当前任务只看 [01-TASKS.md](01-TASKS.md)；历史过程只看
> [归档索引](../archive/syncopate/pre-consolidation-v16/README.md)。

## 1. 三十秒读懂

- **目标**：在模拟业务场景中把训练、评测、runtime 和 serving 接成可靠闭环，理解 SFT/RL/OPD 的实际作用；不以业务上线或求职材料为驱动。
- **当前协议**：模型交互契约是 v15；当前数据版本是 v16。两者名称不同是正常的。
- **当前机器**：只使用 Modal CPU / B200，按任务分配 CPU、单卡或双卡；独立实验可同时运行。
- **已经完成**：v16 数据已生成；B01 上云前认证通过；B02 已在 2×B200 上把本轮 SFT、合并模型、RL adapter 和 OPD adapter 连续传到底。
- **当前准确结论**：B02 全链程序和产物接线通过，但不是“全绿”。35B 的 B03 c/d 批验过真实更新、adapter、同步载荷与 token；小模型验过独立梯度、AdamW 和正常恢复。仍有质量 WARN，不能称为模型问题根治。candidate 从未启动。
- **正在做**：冻结数据，补齐 35B 的 trainer/vLLM 概率、GDN/MoE 路由和三次重复；B04/B06 的正确性前置可并行。SFT/OPD 归一化只过 CPU 数值前置。
- **尚未完成**：没有 candidate 模型或正式性能基线；Serving 主体已施工，正式验收未结束。

主线与 infra 共用仓库：

- 主线：本文和 [01-TASKS.md](01-TASKS.md)。
- infra： [infra START](../infra_exp/00-START.md)。
- 跨线未完成事项只放在唯一负责方的 TASKS；负责人之间直接使用 Codex 任务消息工具，不再维护交互文档。

## 2. 阅读顺序

1. [AGENTS.md](../../AGENTS.md)：工作方式、安全边界和开工规则。
2. [01-TASKS.md](01-TASKS.md)：唯一的当前任务队列。
3. [02-SYSTEM.md](02-SYSTEM.md)：整套系统和模块边界。
4. 按任务只读一个专题：
   - 数据：[03-DATA.md](03-DATA.md)
   - 训练与评测：[04-TRAINING.md](04-TRAINING.md)
   - Modal 与机器：[05-COMPUTE.md](05-COMPUTE.md)
   - Agent Runtime 与 RAG：[06-RUNTIME.md](06-RUNTIME.md)
   - Serving 与运维：[07-SERVING.md](07-SERVING.md)
5. 追查旧决定、失败过程或旧数字时，再进入
   [历史归档](../archive/syncopate/pre-consolidation-v16/README.md)。
6. 对外表达和简历材料在 [syncopate-resume.md](../narrative/syncopate-resume.md)，不属于施工入口。

## 3. 文档地图

| 文档 | 只放什么 |
|---|---|
| [00-START.md](00-START.md) | 导航、当前状态、下一步 |
| [01-TASKS.md](01-TASKS.md) | 仍然开着的任务、顺序和完成条件 |
| [02-SYSTEM.md](02-SYSTEM.md) | 系统全景、边界和唯一事实来源 |
| [03-DATA.md](03-DATA.md) | v16 数据生命周期、门禁、产物和当前读数 |
| [04-TRAINING.md](04-TRAINING.md) | SFT、Exam、RL、OPD、评测和晋级规则 |
| [05-COMPUTE.md](05-COMPUTE.md) | Modal、B200、软件栈、Volume 和探针 |
| [06-RUNTIME.md](06-RUNTIME.md) | AgentLoop、工具、安全闸、会话、RAG |
| [07-SERVING.md](07-SERVING.md) | API、队列、数据库、发布、恢复和 SLO |

## 4. 当前进度

| 部分 | 状态 | 能说明什么 |
|---|---|---|
| v16 题库与切分 | 已通过 | 2030 个 case；EVAL/SFT/RL 三桶为 401/597/1032，跨机器重建一致 |
| v16 SFT 数据 | 可供 smoke | 1222 行、18 桶；现行结构闸和三桶隔离通过，candidate 严格带宽尚未冻结 |
| SFT | B02 smoke 健康通过 | 新归一化修复只过 CPU，B200 验收仍归 B04；不套用旧结果 |
| Exam | B02 链路通过、质量 WARN | 40/40 记录齐，6 个判卷失败，1 个终答仍有机器语法 |
| RL | 35B 身份通过、小模型数值与恢复通过、质量 WARN | 真更新、adapter、同步/token 已验；35B 跨引擎概率和路由仍待解释，不是性能基线 |
| OPD | B02 smoke 健康通过、质量待解 | 新 token/均值/真实更新审计已接线；目标 CPU、B200 两次更新和摆放比较归 B06 |
| 全链 smoke | B02 机械链路通过 | 不是同一固定源码 clean smoke；新一轮全链仍等 B07，不能把 B03 局部复验混入旧账本 |
| Serving | 已施工、未验收 | K0–K11 主体能力存在；当前机器上的正式演练和生产门槛未全部完成 |

## 5. 下一步

队首先处理 [T1 / 工程接线与学习运行](01-TASKS.md)，由 infra
[B03](../infra_exp/01-TASKS.md) 补齐 35B 训推身份与重复性，建立固定源码、可重复的 B200 尺子；B04/B06 正确性前置同时准备。质量告警继续记录，不等待语义清洗。最终采用经对照验证的配置完成全链学习运行，具体口径见 [训练文档](04-TRAINING.md)。

最终必须从统一入口 [scripts/v16_pipeline.sh](../../scripts/v16_pipeline.sh) 证明：

```text
sft-train → sft-eval → sft-select → merge → exam
          → rl-train → rl-adapter → rl-eval
          → opd-train → opd-eval
```

每一段必须真实读取上一段产物。B02 已证明这条产物链能接通；详细证据见
[B02 报告](../../_audit/infra/B02/REPORT.md)。它仍不代表候选模型通过、性能已经最优或 Serving 可以发布。

## 6. 维护规则

- START 不写历史、决策过程、失败流水、命令大全或长篇经验。
- 所有实际待办只写进 TASKS；专题文档只写现行系统和已验证事实。
- 同一事实只在一个专题写全，START 只放一句摘要。
- 状态变化时，先更新专题证据，再同次更新 TASKS 和本页。
- 资源、授权和并行规则只看 [Compute](05-COMPUTE.md)；TASKS 不保存决定和流水账。
