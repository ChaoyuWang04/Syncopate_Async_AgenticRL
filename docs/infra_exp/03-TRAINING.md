# Infra · 训练与异步 RL

> 本文是分布式训练、异步 RL、训推一致性和训练侧低精度的现行说明。
> 未完成工作只看 [01-TASKS.md](01-TASKS.md)。

## 1. 当前基线

| 项目 | 当前值或状态 |
|---|---|
| 机器 | Modal 2×B200 |
| 学生 | `Qwen3.6-35B-A3B`；精确路径只认 `model_paths.py` |
| 框架 | PyTorch 2.13、verl 0.9、vLLM 0.28 |
| 训练入口 | `scripts/v16_pipeline.sh` 的 SFT、RL、OPD 阶段 |
| 已证明 | B02 本轮 SFT → merge → RL → RL adapter → OPD 连续传递；各训练段有真实更新和可加载产物 |
| 当前告警 | RL 每步 25% rollout 撞响应上限；OPD 短评测工具错误多、多样性 5/8；这些是质量问题，不抹掉机械链路，也不能进入 candidate |
| 未证明 | 固定源码可重复的 B200 before；完整训推身份负对照；各优化方案的质量不退化 |

主线的当前训练状态和产物关系只看 [04-TRAINING.md](../syncopate/04-TRAINING.md)。Infra 不复制其训练命令或模型晋级规则。
本轮完整机械证据见 [B02 报告](../../_audit/infra/B02/REPORT.md)。

## 2. 正确性契约

性能实验开始前必须先证明：

- 每个 rank 读取相同版本的数据、模型、LoRA 和配置。
- 梯度归约作用在正确进程组；健康路径的跨 rank 结果满足预期关系。
- rollout 生成使用本轮 trainer 推出的真实权重，而不是起点或旧 adapter。
- trainer 与 rollout 对同一 token 的 logprob 差有实测噪声地板。
- token 序列、attention/loss mask、截断和采样设置逐项同形。
- MoE 模型要记录两侧专家选择；路由不同不能被误报成普通浮点噪声。
- 正对照能通过，故意改错权重、token、mask 或路由的负对照必须失败。

“补丁存在”“参数已经注册”“日志没有报错”都不算机制生效。

B02 已有两步 logprob、权重同步和 adapter shard 身份读数，但它跨过多次源码修复，不能据此冻结噪声地板；B03 必须在同一源码下补齐成套正负对照和重复性。

## 3. 并行与训练形态

当前 B200 需要重新比较的层次是：

```text
模型状态：FSDP2 ↔ Megatron-Bridge / expert parallel
进程摆放：trainer 与 rollout colocate ↔ 1+1 分离
时间关系：sync ↔ colocate async ↔ separate async
模型结构：attention 层 ↔ GDN/线性注意力层 ↔ MoE 专家层
```

旧 5090 上“DDP/FSDP-size-1 必选”“TP 净负”“mb=8 最优”等结论依赖旧框架、旧模型和无 P2P 拓扑，不能直接成为当前默认值。

每个对照必须保持相同有效 batch、更新数、数据顺序、预算和评测集合。若框架让两臂无法同形，报告必须写明差异，不能硬算加速比。

当前 SFT 是自有 PyTorch + PEFT 训练循环，支持显式 1/2 卡数据并行，没有 TP。用户已接受既有
DDP 扩展证据，因此不再重复测 1 卡与 2 卡 DP 的速度；目标默认形态是 `DP=2`，但固定 runbook
目前仍是 1×B200，改默认前至少要用当前源码通过一次双卡梯度/权重一致性 smoke。

B04 真正比较的是同样占用 2×B200 的 `DP=2` 与 `TP=2`。先固定相同 global micro-batch、
effective batch、每次更新 token 和总工作量，只隔离并行方式；再允许两臂各自把 micro-batch
调到最大稳定值，同时减少梯度累积，做最佳实用配置比较。不同 global effective batch 的结果
不能作为纯性能结论。当前稀疏词表投影会绕过模型根 `forward`，所以 TP 必须先证明官方训练路径、
LoRA、optimizer、adapter 保存/合并和监督 mask 全部兼容；否则 B04 记“不适用”，不为实验手搓 TP。

## 4. 异步 RL 的账

异步方案至少同时记录：

- 生成、打分、logprob、更新、权重同步、保存和等待的墙钟分解；
- trainer 与 rollout 每张卡的忙闲、显存和功耗；
- policy version、陈旧度分布、轨迹完成/中止/重复；
- rollout correction、IS/ESS、coverage 和截断；
- 有效更新数、有效 token 和任务级配对结果。

只报“同步搬得更快”或“GPU 利用率更高”不够。最终要回答每单位墙钟时间产生了多少健康训练信号，以及质量有没有退化。

B02 的 RL step 1 为冷启动 251.11s / 188.68 tok/s，step 2 为较热的 98.45s / 421.27 tok/s；只有一个较热样本，没有中位数、离散度或稳定机器指纹，不能当性能 reference。两步权重同步分别约 6.33s 和 6.27s，只能说明同步真实发生。

### 动态分池的边界

GRPO 在同一道题的多次 rollout 上算相对优势；如果这一组 reward 全相同，本批相对优势为零，
这组 rollout 通常不给策略梯度。动态分池据此降低这类题下一次被抽中的概率，并用权重地板和
长时间未抽后的回捞保护回归。它只看当前观测到的方差，不按“简单题”标签直接删除题。

但它会改变训练分布，所以可能提高有效梯度/卡时，也可能让已经学会的简单题回退。旧栈质量证据
受权重同步问题污染。现行 baseline 因此使用官方均匀采样；动态分池只有通过 B05 的 OFF/ON
任务级配对、覆盖率和回归率后才能进入默认。

## 5. 权重同步与 checkpoint

- 下一轮 rollout 必须能证明自己读取了指定 policy version。
- adapter、全量权重和量化权重分别登记身份；不同格式不能只比较文件名。
- 同步后同时检查载荷内容、目标引擎状态和同输入输出，不以 API 成功返回代替内容正确。
- checkpoint 必须能独立加载和继续训练；只存增量时要验证与正确底座合成后的逐项身份。
- Modal 抢占恢复不能重复消费样本、跳过更新或覆盖另一个实验臂。

旧 FSDP1/LoRA 同步事故与修复保存在 [5090 归档](../archive/infra_exp/legacy-4x5090/README.md)，只作为设计负对照。

## 6. 低精度训练

BF16 是当前比较基线。FP8、MXFP8 或其他格式进入 trainer 前必须登记：

- 权重、激活、前向、dgrad、wgrad、累加和 optimizer 各用什么格式；
- scale 的粒度、布局、更新规则和两侧是否相同；
- 哪些模型层实际命中低精度 kernel；
- 数值误差是否随层数、序列长度和 MoE 路由放大；
- 端到端速度、显存和任务质量是否同时受益。

只在一个 GEMM 上变快，或只有一侧量化，都不能称为“训推统一低精度”。

## 7. SFT 稀疏词表投影

当前 SFT 只有约 4% token 参与监督，所以训练器先挑出这些 hidden states，再只对它们做
`lm_head` 和交叉熵；本地单测已与官方完整 logits 路径对拍 loss 与监督位置梯度。

通用 Transformers 默认需要返回所有位置的 logits，而普通预训练通常几乎每个 token 都监督；
稀疏选择还可能绕过模型根 `forward` 的 DDP/FSDP hook、TP 词表切分、fused linear+CE、packed
sequence 或 compile 的静态形状。因此“实现简单”不等于适合作为通用官方默认。本项目已经为
绕过 DDP hook 使用手动梯度归约，这正说明集成成本存在。B10 必须重新对拍数值、显存、端到端
速度和这些融合/并行兼容性，再决定保留还是回到官方路径。

## 8. 验收口径

训练优化只有同时满足以下条件才可以进入主线候选配置：

1. 正确性契约及负对照全部有效。
2. 同尺子端到端收益超过实测噪声带。
3. loss、梯度、同步、checkpoint 和数值健康。
4. 冻结任务评测没有越过预注册退化线。
5. 结果写明适用的硬件、软件、模型、并行和流量边界。

本机缺少目标 CUDA、B200、完整 verl/vLLM 或服务权限时，不重配本机来模拟目标栈；先缩成最小定向测试，再到 Modal CPU/B200 镜像运行并保存证据。
