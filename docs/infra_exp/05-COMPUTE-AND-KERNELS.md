# Infra · Compute 与 Kernels

> 本文是 B200、通信、精度和 kernel 实验的现行说明。
> 机器和依赖的唯一当前清单仍在主线 [05-COMPUTE.md](../syncopate/05-COMPUTE.md)。

## 1. 当前硬件边界

| 机器 | 地位 | 可以引用什么 |
|---|---|---|
| Modal CPU / B200 sm_100（按任务分配） | 当前 | 已落盘的 B200 探针与新的 B 系列结果 |

当前 B200 已验证的能力包括主栈导入、CUDA/attention 基础探针、双卡 NCCL、vLLM 单卡和 EP 启动，以及真实 v16 的 SFT、双卡 RL、adapter 导出和 OPD 机械全链。B03 还用小型 Text MoE 验过双卡独立梯度、同状态 AdamW 和正常恢复。详细证据见主线 Compute、[B02 报告](../../_audit/infra/B02/REPORT.md)与 [B03 报告](../../_audit/infra/B03/REPORT.md)；这些证明环境、线路和小模型数值机制可工作，不是正式性能画像。

## 2. 环境指纹

每份结果必须同时保存：

- GPU 名称、数量、compute capability、驱动、CUDA runtime/toolkit；
- CPU、NUMA、PCIe/NVLink/NVSwitch 可见拓扑；
- 容器镜像、Python、PyTorch、vLLM、verl、Transformers 和 kernel 包版本；
- 模型、dtype、shape、batch/sequence、并行和编译参数；
- git commit/工作树状态、实验入口、随机种子和计时窗口；
- 温度、功耗、时钟、预热与重复次数。

没有完整指纹的数字只可作为线索，不能进入当前默认或简历。

B02 的多次 Modal 调度落在不同 region/主机，且 `nvidia-smi topo -m` 没有成功输出完整矩阵；它的单点吞吐只能作为 B03 设计输入，不能跨次比较。B03 必须固定可比机器指纹并补齐重复窗口、利用率、功耗、通信和费用。

## 3. 通信

B200 通信画像至少分开：

- all-reduce、all-gather、reduce-scatter 和点对点；
- 两卡拓扑、消息大小、dtype、对齐与协议；
- 裸 microbenchmark 与真实训练张量；
- 算法带宽、bus bandwidth、延迟和端到端占比。

旧 E18 的 16 字节对齐悬崖是一个应复用的探测问题，不是 B200 上已经存在的事实。只有当前栈复现后，才讨论修复或上游提交。

B13 首轮合成微测试已在一次 2×B200 分配中完成 30 个通信 case，逐元素内容检查通过。每 rank 分块 64 MiB 时，三个窗口中位数的 bus 带宽为：all-reduce 377.847 GB/s、all-gather 336.247 GB/s、reduce-scatter 325.812 GB/s、单向点对点 392.838～399.572 GB/s。4096³ BF16 GEMM 为 1345.706～1355.915 TFLOP/s。完整口径、环境和原始证据见 [B13 REPORT](../../_audit/infra/B13/REPORT.md)。这些是一次机器分配内的初始读数；拓扑、可靠利用率、能耗和跨机器重复尚未完成，不能外推为训练加速或硬件峰值。

## 4. Attention 与模型结构

当前学生不是纯标准 Transformer。分析前先按实际模型配置区分 attention、线性注意力/GDN 和 MoE 层：

- FA4 只和真正命中的 attention 路径比较；
- kernel 前向、反向和数值都要验证，不能用 import 成功代替；
- 变长序列、长上下文和实际 head 形状必须进入 benchmark；
- 组件 TFLOPS 最终要对账到训练步速或 Serving goodput。

FA4 位于独立环境，不能为了探针破坏主 vLLM/训练依赖。

B02 已观察到 FlashInfer 的多组 MoE token shape 超出 tuning bucket，并在真实推理阶段触发 `_fused_moe_lora_one_shot_kernel` 等 Triton JIT。B10 要分别比较扩 bucket、AOT 资产和预热覆盖；首次请求与稳态都要报告，不能从计时里静默删掉编译。

## 5. 低精度与 GEMM

当前探索对象包括 BF16、FP8、MXFP8 和 NVFP4。每种格式都要回答：

1. 当前 B200 上实际执行了哪条指令或 kernel，而不是只看配置名。
2. 权重、激活、scale、累加和梯度的真实格式是什么。
3. 数值误差与 shape、层数、序列长度和路由的关系。
4. 距硬件物理峰值多远，瓶颈是计算、内存、布局、调度还是编译。
5. 组件收益能否转化为端到端收益和质量不退化。


## 6. 资源与成本

- 资源、持续授权和并行规则只看主线 Compute；主动故障注入另需批准。
- 本机不具备目标硬件或依赖时，不在本机重配模拟；把最小测试直接放进 Modal 的 CPU 或目标 GPU 镜像，并记录源码和镜像身份。
- 每个实验臂使用独立审计目录，同一共享路径只有一个写者；不同实验目录可以并行写入。
- 编译缓存可复用，但必须绑定 GPU 架构、依赖和源代码身份。
- 停止服务和删除产物前解析精确对象；原始证据与必要 checkpoint 的保留规则在实验前登记。
