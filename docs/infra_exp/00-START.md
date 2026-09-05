# Infra · START

> 这是新人进入 infra 线后读的第一份文档，也是唯一导航。
>
> 它只回答三件事：文档在哪里、现在到哪一步、下一步做什么。
> 当前任务只看 [01-TASKS.md](01-TASKS.md)；旧 4×5090 实验只看
> [历史归档](../archive/infra_exp/legacy-4x5090/README.md)。

## 1. 三十秒读懂

- **目标**：在主线真实负载上研究训练、rollout、serving、通信和 kernel，产出可复验的正确性与性能结论，并把最新 B200/B300 工作沉淀成简历材料。
- **当前机器**：Modal 的 2×B200；B300 只在 B200 基线稳定后复核。
- **当前栈**：PyTorch 2.13、vLLM 0.28、verl 0.9、Transformers 5.10；精确值只认主线的 [Compute 文档](../syncopate/05-COMPUTE.md) 和锁文件。
- **已经完成**：B01 上云前认证通过；B02 在 2×B200 上把真实 v16 全链机械接通。
- **当前准确结论**：B02 的程序与产物链通过，但 `all_passed=false`；它是边跑边修的 smoke，不是固定源码性能 baseline，也不是 candidate。
- **尚未完成**：还没有可重复的 B200 性能 reference、胜出优化组合或 B300 结论；旧 E01～E33 全部属于 4×5090 或旧软件栈。
- **当前队首**：B03 固定源码、补齐重复性/噪声和训推身份尺子；随后 B04 在 2×B200 上比较 SFT 的 `DP=2` 与 `TP=2`。

## 2. 阅读顺序

1. [AGENTS.md](../../AGENTS.md)：开工、交接、安全和云端费用规则。
2. [01-TASKS.md](01-TASKS.md)：infra 唯一当前队列。
3. [02-SYSTEM.md](02-SYSTEM.md)：infra 的范围、边界和证据流。
4. 按任务只读一个专题：
   - 训练与异步 RL：[03-TRAINING.md](03-TRAINING.md)
   - rollout 与 Serving：[04-SERVING.md](04-SERVING.md)
   - B200/B300、通信和 kernel：[05-COMPUTE-AND-KERNELS.md](05-COMPUTE-AND-KERNELS.md)
   - 实验编号、判据和归档：[06-EXPERIMENTS.md](06-EXPERIMENTS.md)
5. 需要追查旧结论时，再进入 [4×5090 归档](../archive/infra_exp/legacy-4x5090/README.md)。
6. 对外表达和简历只看 [infra-resume.md](../narrative/infra-resume.md)。

## 3. 文档地图

| 文档 | 只放什么 |
|---|---|
| [00-START.md](00-START.md) | 导航、当前状态、下一步 |
| [01-TASKS.md](01-TASKS.md) | 仍然开着的 B200/B300 任务、依赖和完成条件 |
| [02-SYSTEM.md](02-SYSTEM.md) | 系统边界、研究层次、事实来源和交接方式 |
| [03-TRAINING.md](03-TRAINING.md) | 分布式训练、异步 RL、权重同步和训推一致性 |
| [04-SERVING.md](04-SERVING.md) | rollout/serving 拓扑、调度、缓存、解码与 SLO |
| [05-COMPUTE-AND-KERNELS.md](05-COMPUTE-AND-KERNELS.md) | B200/B300、精度、通信、attention 和 GEMM |
| [06-EXPERIMENTS.md](06-EXPERIMENTS.md) | B 编号、实验协议、证据和报告生命周期 |

## 4. 当前状态

| 部分 | 状态 | 结论边界 |
|---|---|---|
| B200 环境 | 可工作 | 依赖、CUDA、双卡 NCCL、vLLM 单卡/EP 和真实训练机械全链已通过 |
| 主线全链 | 机械通过、质量 WARN | 本轮 SFT、RL 和 OPD 产物已连续传递；Exam、RL 截断和 OPD 质量仍未关闭 |
| 上云前认证 | B01 通过 | 本机可测项无失败；本机缺环境的 5 项已在 Modal CPU 目标镜像 5/5 通过 |
| B200 smoke | B02 已收口 | 可证明链路接通；跨多次源码修复，不能当性能 baseline |
| B200 性能基线 | 未建立 | B03 才用固定源码、重复计时、拓扑、利用率和费用建立；不能拿 B02 最好单点或 5090 数字当 before |
| 训推一致性 | 部分有读数、待成套认证 | B02 有 logprob、同步和 adapter 身份读数；完整正负对照与噪声地板仍归 B03 |
| B200 训练/Serving 优化 | 未开始验收 | 队列已经登记，尚无可写成新成果的 after |
| B300 | 未验证 | 卡型、软件、kernel 与性能都必须重新探测 |

## 5. 下一步

1. 完成 B03：固定源码重跑重复性与噪声，并证明梯度、权重、token、logprob 和 MoE 路由在训推两侧一致。
2. B04 以 `DP=2` 为 reference，对比 `TP=2` 的同工作量与各自最佳 micro-batch；不再测 1 卡与 2 卡 DP 的速度。
3. 再按 B05 → B06 做 RL、OPD 单因素实验，B07 组合复验和 clean smoke。
4. B02 暴露的模型质量问题由主线 T1 负责；infra 只提供相同输入、性能和机制证据。

## 6. 维护规则

- START 不写历史、实验过程、长命令、决策争论或完整待办。
- 所有未完成工作只写进 TASKS；专题文档只保存现行系统和已验事实。
- 两条线不再使用 `MAINLINE-INFRA`、HANDOFF、REPLY 或“信件”文档。跨线事项归入唯一负责方的 TASKS，沟通直接使用 Codex 任务消息工具。
- 4×5090、sm_120、verl 0.8 和 vLLM 0.12 的数字只能标成历史，不能写成 B200 当前结论。
- 状态变化时，同次更新证据、专题、TASKS 和本文摘要；不在文末不断追加时间线。
