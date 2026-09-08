# Infra · 推理引擎与 Rollout

> G3 研究模型引擎计算与调度；业务 API/队列、Harness/tool runtime 由独立 Lab 负责。本项目不承担原 Serving 完整验收排期。

## 1. 基线

范围只含vLLM、SGLang、TensorRT-LLM。采用当前官方稳定、适合目标模型和真实训练接线的配置；TensorRT-LLM用于有需求的低精度/NVIDIA特性探索，不自动视为所有RL框架已支持。先查 GitHub 最新源码、issue/PR 和后端支持，固定具体 revision、容器与 kernel；已有 B200 vLLM 启动成功不代表新模型/新功能兼容，当前没有新的 SGLang 对照结论。

直接使用公开 base 与冻结输入，不要求原业务模型。可控 replay 测固定工作量；真实采样测请求调度和输出差异，不能将回答缩短算成引擎提速。

## 2. 有现实意义的负载

| 负载 | 观察什么 |
|---|---|
| 长 prompt / 短输出 | prefill、chunking、KV 分配、prefix cache |
| 短 prompt / 持续 decode | 访存、每 token 延迟、batch 与 launch 开销 |
| 长短混合和动态到达 | 队头等待、批内位置、scheduler、公平性和尾延迟 |
| MoE + 少 token/专家 | 路由不均、grouped GEMM、dispatch/combine |
| RL 周期性权重更新 | 引擎暂停、加载、版本、缓存失效和恢复吞吐 |

每个选中负载先登记长度/到达率/并发的来源，不构造无人会采用的极端组合来凑 bug。压力边界须说明对应的实际容量情境。

## 3. 画像指标与优化后的验收

初始探索不全面对拍概率；形成优化后按受影响路径跟踪“请求下发 → 实际 batch/位置 → 算子 → 输出概率”，同引擎重复与跨引擎差分开。

延迟至少包括 TTFT（首 token 等待）、TPOT（后续 token 时间）和端到端尾部；吞吐包括请求、有效输出 token 与给定延迟约束内的成功吞吐。记录失败/取消、排队、prefill/decode、KV/显存、编译与缓存，不能只统计最快成功请求。

TP/DP/EP/PD 先证容量和通信可行。拓扑不同只能作整体方案比较，不能将全部差异归因于某个 kernel。多卡并非必然优于单卡；收益需在同任务负载下证明。

## 4. 单因素开关

Graph/compile 覆盖与重捕获、prefix cache 命中、chunked prefill、量化 KV/权重、投机解码、LoRA sleep/wake 分开验证。投机解码核对采样语义和接受率；低精度量化另设数值/输出回归，不能只验程序退出码。

## 5. 上游与既有线索

旧 B02 的 JIT/tuning bucket/Renderer 警告只能导航调查。先查当前实现是否已修或有官方配置，找到具体调用路径后再复现；引擎排行榜不是贡献本身。

从真实训练rollout路径出发，出现瓶颈后才按需复用 [vLLM benchmark](https://github.com/vllm-project/vllm/tree/main/benchmarks) 与 [SGLang](https://github.com/sgl-project/sglang) 的官方测试，实验 REPORT 钉住实际使用 commit，不能让动态 main 链接代替身份。
