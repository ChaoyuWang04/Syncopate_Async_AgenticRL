# Infra · 算子、量化、MoE 与通信

> G4/G6/G7 的定位参考，无真实画像信号不主动扫描。硬件候选和资源约束只看 [Compute](../syncopate/05-COMPUTE.md)，任务见 [TASKS](01-TASKS.md)。

## 1. 先定位真实瓶颈

从训练/推理 trace 提取 shape、dtype、stride、token/专家数、调用频率与耗时/显存，再缩成微测试。优化预期说明减少的 FLOPs、内存字节、临时分配、launch 或传输；结合算力/带宽上限和热点占比估计收益，不以理论峰值代替实测。

## 2. 算子与编译

attention、GDN/FLA、grouped GEMM、MoE dispatch/combine、稀疏词表投影、CE/KL 分块融合分别确认实际后端。已有 fused kernel、AOT 或官方调优桶先验证，确认仍有缺口再写实验性补丁。

前向、反向、监督 mask、动态 shape、root/hook 与 TP/EP 兼容分别有正负对照。JIT/AOT、预热、Graph/compile 重捕获与稳态时间分开记录，诊断 profiler 不混进测速。

## 3. 量化

逐部件登记权重、激活、KV、dgrad/wgrad、累加和 optimizer 的格式、scale 粒度与更新规则。BF16 基线和目标格式要走真实受支持算子；QAT/fake quantization 不代表原生低精度反向。

容量节省、计算开销、转换开销、逐层误差、路由变化和稳定性各自报告；top-k/稀疏近似改变目标时独立实验。不是每张 Blackwell 卡都可直接复用相同软件/指令配置，换架构必须重验。

## 4. 通信与传递

仅在真实通信瓶颈需要时分别测 all-reduce、all-gather、reduce-scatter、点对点、EP all-to-all 和权重传递各阶段。通信内容逐元素核对，传输完整权重或 adapter 同时记录 source/receiver 内容和策略版本。

消息矩阵来自实际梯度、专家 token、adapter 或参数桶形状；小消息/对齐边界可以作定位对照，但不冒充真实负载占比。字节口径、算法带宽/bus 带宽、host/CUDA event 时间与最慢 rank 一致。

计算通信重叠必须从依赖与时间线证明；更高利用率不自动等于更低端到端时间。当前平台没有跨节点/RDMA 权限时，记录不适用，不擅自换平台。

## 5. 既有硬件证据

[B13](../../_audit/infra/B13/REPORT.md) 的首轮 2×B200 完成通信内容与 BF16 GEMM 微测试。它缺完整拓扑、可靠利用率/能耗和独立跨机器重复；旧 4×5090 对齐悬崖也只是待调查线索，不能直接宣布当前 NCCL 有同一问题。

当前限定Modal最多8卡B200。其他硬件不在本轮执行范围；未来明确调整范围后，跨卡对照仍只针对已观察的问题或实际需求，固定目标 workload 并记录各自最佳合理配置与差异。硬件比较不是纯软件 A/B；没有同卡前后对照不能声称补丁提速。
