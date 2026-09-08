---
name: syncopate-project-framing
description: 真实训练画像驱动的性能优化与上游贡献
metadata:
  node_type: memory
  type: project
  modified: 2026-09-08
---

# 当前项目定位

先真实训练，观察瓶颈，再安排优化并回原入口A/B；速度第一，不从算子猜测开局。模型是计算负载，业务造数、教师扩写和完整学习排期保持停止。

- 主要平台Modal，上限8卡B200，按实际需要取卡，所有角色计入；其他Lab和5090负载不动。
- RL框架只含verl/AReaL/ROLL/NeMo-RL/Miles/slime，verl为主；训练FSDP2/Megatron，推理vLLM/SGLang/TensorRT-LLM。名单不是组合已支持声明。
- 固定一MoE一dense，优先复用Volume；公开数据只做必要转换，先看长度/多轮/前缀/并发等负载形状。
- 初始画像不要求先有bug或准确率达标；保留真实工作量/更新/交接和运行健康。优化方案形成后补受影响正确性与原入口A/B，未回归不称可采用。
- 支持/容量/源码调查先行；有瓶颈或实际功能需求才启动定位和移植。缺功能可成为贡献机会，不因框架差异自动制造任务。
- G地图为参考，不再每批强制三个预设微探针；唯一活动队列见[Infra TASKS](../../docs/infra_exp/01-TASKS.md)。
- 实验组提供实验与必要DRAFT，正式PR考据/提交/跟进归upstream；两个Lab的Harness/tool runtime仍独立。

范围与数据维度只认[Infra SYSTEM](../../docs/infra_exp/02-SYSTEM.md)，资源实测只认[Compute](../../docs/syncopate/05-COMPUTE.md)。
