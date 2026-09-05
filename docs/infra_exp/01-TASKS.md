# Infra · TASKS

> 这是 infra 唯一的当前任务队列。这里只放尚未完成的 B200/B300 工作、顺序和完成条件。
> 旧 4×5090 证据在 [历史归档](../archive/infra_exp/legacy-4x5090/README.md)。

## 当前边界

- 当前环境是 Modal 2×B200。B01 上云前认证已通过；B02 已完成机械全链 smoke，报告分别在 [B01](../../_audit/infra/B01/REPORT.md) 和 [B02](../../_audit/infra/B02/REPORT.md)。
- B02 是多次源码修复拼成的证据链，只能证明程序和产物关系接通，不能当可重复性能 before；`pipeline_ok=true`、`all_passed=false`。
- 默认 baseline 是“当前官方稳定、适合生产的优化已经打开”的配置，不拿“全部关闭”做简历稻草人。
- 官方开关也不自动可信。每项都要证明真实命中、开关两态数据/数值正确、端到端确有收益；全关闭臂只用于排错。
- 主线拥有训练语义、模型质量和发布；infra 拿同一条真实管线做正确性与性能实验，不复制主线任务。
- 主线 T1 正在处理 RL 25% 长回复截断、Exam 6 个失败/1 个机器壳和 OPD 短评测质量；infra 不复制修复责任，但下一份 baseline 必须等待影响性能尺子的缺口关闭。
- 付费 GPU 运行前仍要登记费用和本次授权；本机没有目标硬件/栈时，不重配本机，改在 Modal 做最小定向验证。

## P0 · 固定源码，立正确性和性能尺子

### B03 · 固定源码的重复性与训推身份尺子

**前置**：[B02](../../_audit/infra/B02/REPORT.md) 已证明机械链路；主线 T1-1 先关闭会污染工作量的 25% 长回复截断；再次付费运行前登记预算并取得授权。

**目标**：用同一源码同时建立两样东西：可重复的官方生产 baseline，以及“trainer、rollout、adapter 是同一策略”的正确性尺子。

**baseline**：官方稳定生产优化开启；动态分池、PrefixGrouper、自研 kernel、实验低精度关闭。固定模型、数据、seed、有效 batch、更新数、响应预算、镜像、代码 SHA、region/机器指纹和预热窗口。

**工作与完成条件**：

1. 至少重复三个同工作量 smoke 计时窗口，报告中位数、离散度、冷启动与稳态；补齐 GPU busy、显存、功耗、通信、完整拓扑和每健康更新成本。
2. 同 prompt/token 建 trainer ↔ rollout logprob 噪声地板；对 token、mask、权重和 MoE 路由做逐项正对照。
3. 验 LoRA/全量权重同步前后身份、跨 rank 梯度、FSDP2 shard → adapter、checkpoint 加载与继续训练。
4. 故意换错权重、token、mask、adapter 或路由的负对照必须失败；没有判据行视为机制未生效。
5. 当前官方开关逐项登记真实命中；无法在相同机器上复现时结论写“无结论”，不从 B02 单点补数。

**产物**：`_audit/infra/B03/`。B03 完成前，仓库没有可用于简历或优化加速比的 B200 性能 baseline。

## P1 · 用 smoke 做单因素训练实验

### B04 · SFT 双卡 DP 与双卡 TP

**前置**：B03；使用同一镜像、数据和 SFT 起点。当前训练器只实现了数据并行，固定 runbook 仍是 1×B200；文档调整不表示 TP 已经接通，也不表示默认值已经改完。

**问题**：同样占用 2×B200 时，是两张卡各跑一份模型、各吃不同样本的 `DP=2` 更快，还是两张卡共同切一份模型的 `TP=2` 能靠更大的 micro-batch 更快？用户已接受既有 DDP 扩展证据，不再做 1 卡与 2 卡 DP 的性能 A/B；B04 以 `DP=2` 为 reference。

**阶段 0 · 先证明 TP 值得接线**：先只读核对当前 Transformers、模型、PEFT 和 PyTorch 是否提供训练可用的官方 TP 路径。现有 SFT 会绕过模型根 `forward` 做稀疏词表投影，并手动归约 LoRA 梯度；TP 必须证明分片后的前向、反向、optimizer、adapter 保存/合并和现有 loss mask 都接在真实路径上。若官方路径不支持，或接线成本明显超过可预期收益，结论记为“不适用”，不为做实验临时手搓一套 TP。

**阶段 1 · 只比较并行方式**：`DP=2` 对 `TP=2` 使用相同模型、LoRA、数据顺序、seed、精度、有效 batch、每次更新的输入/监督 token、更新数和评测。两臂先固定相同的 global micro-batch；DP 把样本分给两个副本，TP 让两个 rank 共同处理同一批样本，梯度累积按同一个有效 batch 派生。这样测到的差异才主要来自并行方式。

**阶段 2 · 比较各自最佳实用配置**：两臂分别寻找不 OOM、数值稳定的最大 global micro-batch，但仍固定同一个有效 batch、每次更新 token 和总工作量；micro-batch 变大时同步减少梯度累积。若 TP 只在更大 micro-batch 下获胜，报告必须写成“TP 释放显存后带来的 batch 收益”，不能写成纯 TP 加速。不得用不同 global effective batch 做速度结论。

**正确性门槛**：DP 先通过当前栈的跨 rank 梯度/权重一致性；TP 再用 B03 噪声带对拍 token、loss、监督位置梯度、可训练参数与合并后模型输出，并证明没有整层意外复制、漏分片或错误重复计算。两臂的 adapter 都必须能保存、加载、合并并进入同一评测；故意改错 shard、mask 或 adapter 的负对照必须失败。

**性能与完成条件**：同一 2×B200 拓扑上交替运行并至少取得三个预热后窗口，比较中位 step time、监督 tokens/s、峰值显存、GPU busy、通信占比、端到端墙钟和每固定工作量 GPU-hours，同时检查 loss、梯度和冻结任务质量。运行前用 B03 离散度预注册胜出线；差异落在噪声带内就算没有胜者。TP 只有在正确性全过、端到端显著更快且成本/质量不退化时才晋级，否则保留 `DP=2`。当前 runbook 从 1×B200 改为 `DP=2` 是单独的落地动作，至少先过一次当前源码的双卡正确性 smoke，不再为它另做 1×/2×速度实验。

### B05 · RL 官方形态、官方开关与动态分池

**前置**：B02、B03。

**顺序**：

1. 官方均匀采样 + `sync` 是 reference。
2. 单独比较 `colocate_async`；只有资源和模型能同形时再比较 trainer/rollout 1+1 的 `separate_async`。
3. PrefixGrouper、CUDA Graph、checkpoint、batch/并发等官方开关逐个做 OFF/ON 机制与数值探针。
4. 最后单独比较 `--dynamic-pool` OFF/ON，不与并行模式同时改变。

动态分池只会降低“当前组内 reward 没方差”的题的抽样概率，并保留地板和超时回捞；它不按静态难度标签删简单题。它可能省 rollout，也可能改变训练分布、伤害简单题保持，所以必须看冻结任务配对质量、回归率、覆盖率、有效梯度/卡时和端到端 goodput，不能只看步速。

**完成条件**：每个开关证明真实命中；开关前后 token、样本、权重和数值差异符合预期；获胜项逐个给出采用/拒绝/不适用/无结论。

### B06 · OPD 角色摆放与有效更新率

**前置**：B02、B03。

**目标**：在两张 B200 上选择学生、教师、锚的摆放与 batch，并减少“生成了但没有可蒸 token”的空转。

**完成条件**：所有臂读取同一 RL adapter；`max_steps` 只数真实 optimizer update，另记 attempts/skips；比较有效更新/小时、KL/mask 数值、显存、通信和任务质量。SFT fallback 只能作为显式诊断臂，不能冒充全链 OPD。

### B07 · 胜出组合的全链复验与 candidate 冻结

**前置**：B04～B06 中要进入默认的项目都已经单因素验收。

**目标**：把胜出项合起来重跑一次完整 smoke，排除组合交互，再冻结 candidate 配方。

**完成条件**：与 B03 的固定 reference 同尺子比较，正确性和质量不退化、端到端收益超过噪声带；主线 runbook 默认值、门禁、文档和审计身份一致。任何组合回归都回到单因素定位。

## P2 · 深层训练与 kernel 研究

### B08 · FSDP2 与 Megatron-Bridge / EP=2

**前置**：B03；使用隔离镜像。只有预估收益能覆盖接线成本时才放在 candidate 前。

**完成条件**：同数据、effective batch 和更新数，比较步速、tokens/s、显存、通信和 loss；先证明梯度、路由和 checkpoint 正确，再决定采用、拒绝或不适用。

### B09 · Transformer Engine FP8 / MXFP8

**前置**：B03，且明确学生每类层的实际计算路径。

**完成条件**：BF16 为 reference；格式、scale、作用层、前向/dgrad/wgrad/累加逐项登记；训练与 rollout 两侧误差、速度、显存、稳定性和任务质量同时过闸。旧 sm_120 结论不能代替 B200。

### B10 · Attention、MoE 与执行层

**前置**：B03；每项单独预注册。

候选包括 FA4、grouped GEMM、`torch.compile`、选择性重计算、序列打包、LoRA 挂载范围、PrefixGrouper，以及 SFT 稀疏词表投影。

B02 还登记了两个直接可测的 serving/rollout 执行问题：FlashInfer MoE shape 超出 tuning bucket，以及 `_fused_moe_lora_one_shot_kernel` 等 kernel 在正式推理时才 JIT。要比较扩 bucket/AOT/预热前后首次请求、稳态延迟、缓存身份和端到端 goodput；不能把 JIT 时间静默删掉。

稀疏投影要把当前“只对约 4% 监督位置做 lm_head”与官方完整 logits 路径对拍：loss/梯度、峰值显存、端到端步速、DDP/FSDP/TP hook、fused linear-cross-entropy 和 compile 兼容性都要测。它实现简单，但会改变标准模型 forward 的返回形状和并行/fusion 路径，所以不能因本地单测等价就默认比官方路径更好。

**完成条件**：每项同时报告 kernel 与端到端收益、数值和任务回归；没有端到端收益的组件加速不晋级。

## P3 · Rollout、Serving 与硬件

### B11 · vLLM 0.28 双卡拓扑

比较单卡、DP=2、TP=2、EP=2 及当前版本真实支持的 DeepEP/EPLB 组合；同一 trace 报告吞吐、goodput@SLO、TTFT、TPOT、显存、cache、负载均衡和故障行为。

先把 vLLM 已弃用的 raw prompt InputProcessor 调用迁到 Renderer API，并用逐 token 对拍证明输入完全相同；API 迁移不能和拓扑 A/B 混成一个变量。

### B12 · 解码、缓存、精度与 PD 边界

在 B11 胜出拓扑上单变量评估 MTP、CUDA Graph、FlashInfer/TRT-LLM 后端、prefix cache、FP8 KV、低精度权重和 prefill-decode 分离；同时看单流、并发、质量和资源账。

### B13 · NVLink/NCCL 与 Tensor Core 画像

建立 B200 集合通信矩阵、消息大小/对齐/协议和 tcgen05/TMEM 块缩放 GEMM 的物理基线；所有读数绑定拓扑、驱动、编译器和 kernel 身份。

### B14 · Modal 抢占恢复、可观测性与成本

主动故障注入需另获授权。完成条件是 checkpoint 恢复无重复/漏样本、缓存绑定输入身份、监视器能识别进程退出，并能报告每个健康更新/有效 token/成功请求的成本。

### B15 · B300 迁移与复核

**前置**：B200 候选组合稳定，用户批准 B300 费用。

重新采集环境、通信、kernel 和模型加载；复跑 B200 胜出项，不继承 B200 数字。业务全链由主线 T4 验收。

## 旧 5090 能留下什么

这是计划盘点，不是 B200 验收结果：

| 处理 | 数量 | 当前用途 |
|---|---:|---|
| 直接保留的方法/尺子 | 5 | 梯度跨 rank、训推 logprob、LoRA 身份、稀疏投影等价、flash-attn 反向门禁；在 B03/B10 重跑 |
| 旧自定义实现由官方能力接班 | 6 | FSDP2、V1 权重同步/异步形态、LoRA-only checkpoint、PrefixGrouper、vLLM DP/EP 等；旧补丁不默认开，但官方实现仍逐项验 |
| 机制可能仍有价值、性能必须重测 | 5 | 动态分池、稀疏投影、前缀共享、CUDA Graph/compile、低精度与 kernel；进入 B05/B09/B10/B12 |
| 直接退役 | 2 | 无 NVLink 的 5090 专属拓扑调参、sm_120 专属 PTX/峰值数字；只留历史故事 |

B200 可直接继承的旧性能数字是 **0 个**。可继承的是问题拆法、探针和负对照。

## 执行顺序

1. 当前先完成 B03；B02 的质量修复依赖主线 T1，不在 infra 复制一份。
2. 随后按 B04 → B05 → B06 做单因素实验。
3. B07 只组合已单因素验收的胜出项，并承担固定源码 clean smoke。
4. B08～B14 按 B03 的瓶颈读数排序；B15 永远最后。

## 队列规则

- 每个实验必须写清问题、唯一变量、正负对照、证据目录、费用和停止线。
- 做完即从本页移除；现行结论进入专题，完整报告进入 B 系列归档。
- 跨线任务只保留一个负责人；另一条线只链接依赖。
- 路线变化、GPU 运行和云端费用都要重新获得用户确认。
