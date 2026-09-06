# Infra · 系统总览

> 本文只说明 infra 研究什么、怎样接入主线、事实放在哪里。
> 当前排期只看 [01-TASKS.md](01-TASKS.md)。

## 1. 这条线要交付什么

Infra 不另造一套模型产品。它使用主线固定数据、模型和业务负载，交付三类东西：

1. **正确性证据**：梯度、权重、token、logprob、MoE 路由和产物身份真实一致。
2. **系统收益**：端到端训练速度、rollout goodput、Serving SLO、显存、稳定性或成本得到可复验改善。
3. **可迁移结论**：说明结果依赖哪种硬件、软件、模型和流量，环境变化时知道什么必须重测。

工程问题先在 B 报告记清，再按 [upstream 规则](../upstream/README.md) 留适用的 DRAFT。没有必须提交 PR、写论文或凑简历数字的任务。对外材料在 [infra-resume.md](../narrative/infra-resume.md)，只能在有证据后更新。

## 2. 与主线的关系

```text
主线：冻结模拟数据 + 固定管线 + 学习观察 + 服务工程
                    │
                    ▼
infra：正确性尺子 → 训练/Serving/Kernel 单变量实验
                    │
                    ▼
      _audit 原始证据 → B 报告 → 当前专题 / 适用的 upstream DRAFT
```

| 范围 | 主线负责 | infra 负责 |
|---|---|---|
| 数据和契约 | v16 数据、消息形状、工具和预算 | 检查实验两臂使用同一输入，不另建数据真源 |
| SFT/RL/OPD | 固定入口、质量门槛、候选晋级 | 分布式、异步、低精度、权重同步和性能实验 |
| Runtime | AgentLoop、工具治理、RAG、业务语义 | 只测它造成的实际流量和系统成本 |
| Serving | 生产可靠性、发布、回滚与业务 SLO | 引擎拓扑、调度、缓存、解码、量化和容量边界 |
| Compute | Modal 环境、Volume、当前机器入口 | B200 画像、通信、kernel 和成本实验 |

一件跨线工作只放在唯一负责方的 TASKS。另一条线只保留依赖链接；负责人之间使用 Codex 任务消息工具直接核对，不再创建交互文档。

## 3. 研究层次

每个优化按下面顺序向下走；上一层没有证明，下一层数字没有意义。

1. **身份层**：代码、镜像、模型、数据、配置、拓扑是否是预期对象。
2. **正确性层**：梯度、权重、token、mask、logprob、路由和产物是否一致。
3. **机制层**：开关是否真的改变目标路径，正负对照是否显形。
4. **组件层**：通信、kernel、同步、缓存或调度本身快了多少。
5. **端到端层**：训练步、有效 token、成功请求和 goodput 是否改善。
6. **学习观察层**：冻结任务上哪些变好、变差或没动。行为差不阻塞测量；若工作量或训练目标改变，不能把读数当成纯系统加速。

## 4. 当前系统

- 云端与机器：Modal CPU / B200，资源按任务申请，见 [主线 Compute](../syncopate/05-COMPUTE.md)。
- 模型与常量：只认 [model_paths.py](../../syncopate/core/model_paths.py)、[rollout_budget.py](../../syncopate/train/rollout_budget.py) 和 [contract.py](../../syncopate/core/contract.py)。
- 业务管线：只认 [scripts/v16_pipeline.sh](../../scripts/v16_pipeline.sh)。
- Modal 编排和已有探针：只认 [stack_probe.py](../../modal_app/stack_probe.py) 与 [Modal README](../../modal_app/README.md)。
- 原始实验证据：`_audit/infra/<B-id>/`。
- 当前任务：只认 [01-TASKS.md](01-TASKS.md)。

## 5. 当前事实边界

已经能说：

- B200 主栈、双卡通信和模型服务可工作；B02 已把本轮 SFT、合并模型、双卡 RL、RL adapter 和 OPD 连续传递到底。
- B01/B02 可以证明上云前检查与机械链路；B02 跨多次源码修复且有质量 WARN，不能当性能 baseline 或 candidate 结果。
- 历史事故的判据与负对照可以复用，必须在当前路径重新验证。

还不能说：

- B200 上已经有正式的训练吞吐或 Serving SLO 基线。
- 旧 FSDP1、PrefixGrouper、LoRA 同步、NCCL、FP8、四引擎路由或 PD 决策在新栈仍是默认最优。

## 6. 文档与证据流

```text
01-TASKS 登记问题
  → 06-EXPERIMENTS 规定判据
  → _audit/infra/Bxx/REPORT.md 写施工报告和证据索引
  → _audit/infra/Bxx/<arm>/ 落原始证据
  → 验收后把当前结论写进 03/04/05
  → 完整报告移入 docs/archive/infra_exp/b-series/
  → 有足够证据后更新 docs/narrative/infra-resume.md
```

专题文档不保存施工日志，TASKS 不保存完成历史，START 不复制专题细节。
