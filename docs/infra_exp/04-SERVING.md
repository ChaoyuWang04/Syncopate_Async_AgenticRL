# Infra · Rollout 与 Serving

> 本文是模型引擎拓扑、调度、缓存、解码、量化和性能评测的现行说明。
> 生产 API、数据库、队列和发布规则看主线 [07-SERVING.md](../syncopate/07-SERVING.md)。

## 1. 当前状态

- 当前引擎栈是 vLLM 0.28，运行在 Modal B200。
- 单卡启动和双卡 EP 启动已经通过环境探针。
- B02 的真实 SFT/RL/OPD 评测已在 vLLM 上运行，暴露了 FlashInfer tuning bucket 未覆盖、推理期 Triton JIT 和 raw prompt API 弃用警告；这些是待测线索，不是性能结论。
- 主线 Serving 主体代码已经施工完成，但当前部署环境的正式验收尚未结束。
- 还没有 B 系列的吞吐、goodput、TTFT、TPOT、cache 或容量曲线。
- 旧 E32/E33 的 4×5090、vLLM 0.12 和自研亲和路由结果只属于历史。

因此当前不能把“四引擎默认”“PD no-go”“FP8 KV 默认”或旧并发数字直接搬到 B200。

## 2. 两种负载

Infra 必须分开报告：

| 负载 | 主要目标 | 不能省略 |
|---|---|---|
| RL rollout | 单位墙钟内产生健康轨迹 | 权重版本、采样契约、陈旧度、轨迹完整性 |
| 业务 Serving | 在 SLO 内完成真实请求 | 成功终态、TTFT、TPOT、排队、恢复和任务质量 |

两者可以共用 vLLM，但不能共用一个模糊的“吞吐更高”结论。

## 3. 拓扑比较

候选形态包括单卡、DP=2、TP=2、EP=2，以及当前版本实际支持的 DeepEP/EPLB。比较前先证明：

- 模型、LoRA、MoE 路由和采样契约一致；
- 两张卡都收到并完成了预期份额，不能只看进程数量；
- 请求没有被日预算、错误快速终止或重试污染；
- 每个拓扑使用相同 trace、预热、并发阶梯和统计窗口；
- 失败和降级行为也被计入，而不是只统计成功快请求。

## 4. 指标

每次 Serving 实验至少报告：

- request/s、output tokens/s 和 **goodput@SLO**；
- TTFT、TPOT、端到端延迟的中位数与尾部；
- 排队时间、prefill/decode 时间、batch 大小和调度等待；
- KV 占用、prefix cache 命中、显存、GPU 忙闲与功耗；
- 每个成功请求或有效输出 token 的成本；
- 任务级质量，以及超时、失败、取消和错误快速终止数量。

一个组件更快但 goodput、质量或成本没有改善，只能写组件结果。

## 5. 解码、缓存与量化

- MTP、ngram 或其他投机解码必须报告接受率、额外模型成本、单流和并发两种结果，并做输出质量对照。
- Prefix cache 要使用真实重复结构与随机化对照，命中率必须和端到端收益一起读。
- FP8 KV、FP8/NVFP4 权重必须分别隔离；容量收益、kernel 收益和质量代价不能打包归因。
- CUDA Graph 要报告覆盖率、未捕获形状和重编译/重捕获成本。
- PD 分离必须先证明有足够 prefill 计算可卸载，再核算 KV 搬运和排队收益。

## 6. 与主线 Serving 的边界

Infra 可以产出引擎选型、容量曲线和候选默认值；是否进入生产由主线 Serving 验收决定。

当 infra 结果建议改变主线默认值时：

1. 完成 B 报告和原始证据；
2. 用 Codex 任务消息把结论、适用边界和复验方法发给主线负责人；
3. 主线在自己的 TASKS 和专题文档中决定是否采用；
4. 不创建新的交互文档，也不在两边复制状态。

## 7. 证据位置

- 当前环境和已有探针：[主线 Compute](../syncopate/05-COMPUTE.md)、[Modal README](../../modal_app/README.md)
- B 系列原始证据：`_audit/infra/Bxx/`
- B 系列报告生命周期：[06-EXPERIMENTS.md](06-EXPERIMENTS.md)
- 旧 5090 Serving 报告：[历史归档](../archive/infra_exp/legacy-4x5090/README.md)
