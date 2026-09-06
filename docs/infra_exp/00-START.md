# Infra · START

> 这是新人进入 infra 线后读的第一份文档，也是唯一导航。
>
> 它只回答三件事：文档在哪里、现在到哪一步、下一步做什么。
> 当前任务只看 [01-TASKS.md](01-TASKS.md)；历史归档不作为运行入口。

## 1. 三十秒读懂

- **目标**：用项目实际负载把训练、rollout、serving、通信和 kernel 做对、测清楚；遇到问题就解决并记录，不为求职、PR 或论文制造实验。
- **当前机器**：Modal CPU / B200；CPU、单卡或双卡按任务申请，可并行起多组资源。
- **当前栈**：PyTorch 2.13、vLLM 0.28、verl 0.9、Transformers 5.10；精确值只认主线的 [Compute 文档](../syncopate/05-COMPUTE.md) 和锁文件。
- **已经完成**：B01 上云前认证通过；B02 真实 v16 全链机械接通；35B 的 B03 c/d 已验双卡 RL 更新、adapter、同步载荷与 token。小模型又验过独立梯度、同状态 AdamW、连续两步和正常恢复。SFT 和 OPD 的 token 均值目标修复已过 CPU 数值检查，未做 GPU 比较。
- **当前准确结论**：B02 的程序与产物链通过，但 `all_passed=false`；它是边跑边修的 smoke，不是固定源码性能 baseline，也不是 candidate。
- **尚未完成**：还没有可重复的 B200 性能 reference、胜出优化组合；旧软件栈的实验只作历史。
- **当前队首**：B03 用 35B 真实路径解释 trainer/vLLM 概率及 GDN/MoE 路由差，再关闭身份探针做三次固定源码重复；B04/B06 的正确性前置可同时准备。

## 2. 阅读顺序

1. [AGENTS.md](../../AGENTS.md)：开工、交接、安全和云端费用规则。
2. [01-TASKS.md](01-TASKS.md)：infra 唯一当前队列。
3. [02-SYSTEM.md](02-SYSTEM.md)：infra 的范围、边界和证据流。
4. 按任务只读一个专题：
   - 训练与异步 RL：[03-TRAINING.md](03-TRAINING.md)
   - rollout 与 Serving：[04-SERVING.md](04-SERVING.md)
   - B200、通信和 kernel：[05-COMPUTE-AND-KERNELS.md](05-COMPUTE-AND-KERNELS.md)
   - 实验编号、判据和归档：[06-EXPERIMENTS.md](06-EXPERIMENTS.md)
5. 历史结论在 `docs/archive/`，不用于选择当前运行环境。
6. 对外表达和简历只看 [infra-resume.md](../narrative/infra-resume.md)。

## 3. 文档地图

| 文档 | 只放什么 |
|---|---|
| [00-START.md](00-START.md) | 导航、当前状态、下一步 |
| [01-TASKS.md](01-TASKS.md) | 仍然开着的 B200 任务、依赖和完成条件 |
| [02-SYSTEM.md](02-SYSTEM.md) | 系统边界、研究层次、事实来源和交接方式 |
| [03-TRAINING.md](03-TRAINING.md) | 分布式训练、异步 RL、权重同步和训推一致性 |
| [04-SERVING.md](04-SERVING.md) | rollout/serving 拓扑、调度、缓存、解码与 SLO |
| [05-COMPUTE-AND-KERNELS.md](05-COMPUTE-AND-KERNELS.md) | B200、精度、通信、attention 和 GEMM |
| [06-EXPERIMENTS.md](06-EXPERIMENTS.md) | B 编号、实验协议、证据和报告生命周期 |

## 4. 当前状态

| 部分 | 状态 | 结论边界 |
|---|---|---|
| B200 环境 | 可工作 | 依赖、CUDA、双卡 NCCL、vLLM 单卡/EP 和真实训练机械全链已通过 |
| 主线全链 | 机械通过、质量 WARN | 本轮 SFT、RL 和 OPD 产物已连续传递；Exam、RL 截断和 OPD 质量仍未关闭 |
| 上云前认证 | B01 通过 | 本机可测项无失败；本机缺环境的 5 项已在 Modal CPU 目标镜像 5/5 通过 |
| B200 smoke | B02 已收口 | 可证明链路接通；跨多次源码修复，不能当性能 baseline |
| B200 性能基线 | 未建立 | B03 才用固定源码、重复计时、拓扑、利用率和费用建立；不能拿 B02 最好单点或旧环境数字当 before |
| 训推一致性 | 同步/token 和小模型梯度/恢复通过，35B 成套认证未完 | 小模型两臂两步及恢复已验；35B 的跨引擎概率、GDN/MoE 路由和重复性仍待完成 |
| B200 训练/Serving 优化 | 已做部分 CPU 正确性前置 | SFT 切 micro-batch 的梯度权重及 OPD 总和/均值不一致已修并过 CPU 对拍；尚无 B200 优化加速比 |

## 5. 下一步

1. 完成 B03 的 35B 训推概率、GDN/MoE 路由解释和三次固定源码重复；不要重跑已经通过的小模型 GPU 子问题。
2. B04 先做 tiny DP=2/TP=2 的前后向与 LoRA 保存正确性，再比较同工作量与各自最佳 micro-batch；不再测 1 卡与 2 卡 DP 的速度。
3. B06 的规格与代码复审已通过；下一步先跑目标 CPU 零失败、零跳过，再用指定 RL adapter 做两次真实 OPD 更新。
4. 正确性前置通过后，B04/B05/B06、推理及 kernel 的独立性能实验尽量并行；B07 验证采用组合，再进入完整学习运行。B02 质量告警只观察，不重造数据。

## 6. 维护规则

- START 不写历史、实验过程、长命令、决策争论或完整待办。
- 所有未完成工作只写进 TASKS；专题文档只保存现行系统和已验事实。
- 两条线不再使用 `MAINLINE-INFRA`、HANDOFF、REPLY 或“信件”文档。跨线事项归入唯一负责方的 TASKS，沟通直接使用 Codex 任务消息工具。
- 状态变化时，同次更新证据、专题、TASKS 和本文摘要；不在文末不断追加时间线。
