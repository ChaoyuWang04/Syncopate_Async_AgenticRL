# Syncopate · START

> 这是新人进入主线后读的第一份文档，也是唯一导航。
>
> 它只回答三件事：文档在哪里、项目现在到哪一步、下一步做什么。
> 当前任务只看 [01-TASKS.md](01-TASKS.md)；历史过程只看
> [归档索引](../archive/syncopate/pre-consolidation-v16/README.md)。

## 1. 三十秒读懂

- **目标**：做一个能调用工具、连续多步完成业务任务的 agent，并把数据、训练、评测、runtime 和 serving 接成闭环。
- **当前协议**：模型交互契约是 v15；当前数据版本是 v16。两者名称不同是正常的。
- **当前机器**：训练主场是 Modal 的 2×B200；B300 复核排在 B200 全链稳定之后。
- **已经完成**：v16 数据已生成；B01 上云前认证通过；B02 已在 2×B200 上把本轮 SFT、合并模型、RL adapter 和 OPD adapter 连续传到底。
- **当前准确结论**：全链程序和产物接线通过，但不是“全绿”。Exam 有质量 WARN，RL 有 25% 长回复截断，OPD 短评测多样性只有 5/8；candidate 从未启动。
- **正在做**：先关闭这些质量欠账，并用固定源码建立可重复的 B200 baseline；随后才做 SFT `DP=2`/`TP=2` 和 RL/OPD 单因素实验。
- **尚未完成**：没有 candidate 模型、正式性能基线或 B300 结论；Serving 主体已施工，正式验收未结束。

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
| SFT | 本轮 smoke 健康通过 | 30 次真实更新；adapter、选点和 310/310 层合并均已验；不是候选模型 |
| Exam | 链路通过、质量 WARN | 40/40 记录齐，6 个判卷失败，1 个终答仍有机器语法 |
| RL | 本轮双卡 smoke 健康通过、质量 WARN | 2 次真实更新、权重同步、step-2 checkpoint 和 350 组 LoRA 导出通过；每步 25% rollout 撞响应上限 |
| OPD | 本轮 smoke 健康通过、质量待解 | 2 次尝试中 1 次有效，完成 1 次真实更新、84 个蒸馏 token 和有限 KL；短评测多样性 5/8 |
| 固定全链 smoke | 机械链路通过 | `pipeline_ok=true`、`all_passed=false`；说明线路通但仍有质量 WARN，不是 candidate 通过 |
| Serving | 已施工、未验收 | K0–K11 主体能力存在；当前机器上的正式演练和生产门槛未全部完成 |

## 5. 下一步

队首先处理 [T1 / smoke 质量欠账](01-TASKS.md#t1--smoke-质量收口与固定源码复验)，并由 infra
[B03](../infra_exp/01-TASKS.md#b03--固定源码的重复性与训推身份尺子) 建立固定源码、可重复的 B200 尺子；随后再做 SFT 双卡 DP/TP 和 RL/OPD 单因素实验。

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
- 产生 Modal 费用或改变云端状态前，仍需用户对那次运行明确授权。
