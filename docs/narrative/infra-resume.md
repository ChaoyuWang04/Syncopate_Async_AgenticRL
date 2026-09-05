# Infra 项目 · 当前简历材料

> 本文只保存可以对外使用的当前表述。技术事实以 `docs/infra_exp/` 和 B 系列证据为准。
> 旧 4×5090 简历保存在 [历史归档](../archive/infra_exp/legacy-4x5090/NARRATIVE-AND-RESUME.md)，不再作为当前投递稿。

## 1. 当前可以诚实写什么

项目已经迁到 Modal 2×B200，并建立了 PyTorch 2.13、verl 0.9、vLLM 0.28、Transformers 5.10 的可复现环境。当前学生是 MoE 模型；SFT、RL、OPD 和 Serving 的分段机制已经能运行，但连续产物交接仍等待主线全链验收。

已经验证的当前基础：

- B200 主栈、双卡 NCCL、模型权重和 vLLM 单卡/EP 启动可工作；
- 真实 v16 SFT 30 步、RL 双卡机制和 OPD 机制分别完成冒烟；
- 实验体系要求环境指纹、正负对照、原始证据和任务级质量门槛。

这些能说明新的研究环境已经建立，但还不能写成“B200 上性能提升了多少”。新的 before/after 要等 B 系列实验验收。

## 2. 新一代项目结构

### 项目一 · B200 MoE Agentic RL 训练系统

目标成果来自 B01～B07：

- 训推逐 token 概率、权重同步和 MoE 路由的一致性验证；
- FSDP2 与 Megatron-Bridge/EP=2 的同尺子对照；
- sync、colocate async、separate async 的端到端效率与陈旧度边界；
- B200 上 Transformer Engine 训推统一 FP8/MXFP8、FA4 和 MoE 执行层优化。

只有形成“问题 → 修复/方案 → 端到端数字 → 质量不退化”的闭环后，才改写成简历 bullet。

### 项目二 · B200 vLLM 推理与 Rollout 系统

目标成果来自 B08～B09：

- DP/TP/EP、DeepEP/EPLB 的真实业务负载拓扑选择；
- goodput@SLO、TTFT、TPOT、KV/cache、成本和故障行为的完整容量曲线；
- MTP、CUDA Graph、FlashInfer/TRT-LLM kernels、FP8 KV、NVFP4/FP8 权重和 PD 分离的适用边界。

### 项目三 · Blackwell 通信、低精度与 Kernel

目标成果来自 B10～B12：

- NVLink/NCCL 集合通信和对齐画像；
- tcgen05/TMEM 上 MXFP8/NVFP4 Tensor Core/GEMM 距物理峰值的测量与优化；
- Modal 抢占恢复、成本效率，以及 B200 胜出方案在 B300 上的兼容性复核。

## 3. 当前投递稿

在 B01/B02 之前，建议只使用下面这段，不填写未经验证的加速数字：

> 在 Modal 2×B200 上搭建面向多轮工具调用 Agent 的训练—推理一体化实验栈，覆盖 MoE 模型的 SFT、异步 RL、OPD 与 vLLM Serving；已经验证环境、双卡通信和分段训练机制，正在建立跨 trainer/rollout 的权重、逐 token 概率和路由一致性尺子，并围绕 FSDP2/Megatron-Bridge、expert parallel、训推统一低精度和 Blackwell kernel 开展同尺子端到端优化。

这段表达的是已经建立的系统和正在进行的方向，不把计划冒充成果。每完成一个 B 实验，就用实测数字替换其中对应的“正在”。

## 4. 更新门槛

- 数字必须能指向 `_audit/infra/Bxx/` 和验收报告。
- 不混用 5090 与 B200/B300 的吞吐、显存、通信和 SLO。
- 不把 microbenchmark 加速写成端到端加速。
- 不把短 smoke 写成模型质量或生产验收。
- 上游 issue/PR 只有真实提交并可访问后才写编号。
