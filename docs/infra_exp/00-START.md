# Infra · START

> 当前主线：真实训练负载驱动的训练与推理性能改进。唯一队列是[TASKS](01-TASKS.md)。

## 做什么

先跑真实入口，观察速度瓶颈，再选优化并在原入口A/B。模型仅为负载，不恢复业务造数、全链学习或准确率排期。不预设上游错误，不以底层微测试代替真实训练。

用户2026-09-08锁定：Modal最多8卡B200（按需分配）；RL框架verl/AReaL/ROLL/NeMo-RL/Miles/slime，以verl为主；训练FSDP2/Megatron；推理vLLM/SGLang/TensorRT-LLM；固定一MoE一dense。范围上限不等于每个组合已经支持。最低执行健康保留，详细正确性/质量评测在发现瓶颈并提出改动后按需验证。

## 阅读顺序

1. [AGENTS](../../AGENTS.md)与项目记忆索引。
2. [TASKS](01-TASKS.md)：当前准备与真实运行画像队列。
3. [SYSTEM](02-SYSTEM.md)：范围、负载数据维度、合理并行、仅供定位的旧方向地图。
4. [EXPERIMENTS](06-EXPERIMENTS.md)：探索/优化分阶段规则和唯一REPORT。
5. 按瓶颈读[训练](03-TRAINING.md)、[推理](04-SERVING.md)、[算子/通信](05-COMPUTE-AND-KERNELS.md)。
6. [Compute](../syncopate/05-COMPUTE.md)、[Modal README](../../modal_app/README.md)：资源盘点与现有实现。

## 当前入口与历史

先做Q21场景冻结，再Q22真实训练画像；尚未启动新GPU运行。B11停止主动下钻，B12/B13暂存为参考，重开须真实画像触发。原始已测/未解结论不改写成全部正常，具体记录见[实验文件夹地图](experiments/README.md)。

infra-probes编号继续递增，REPORT仍放`experiments/Bxx-主题/REPORT.md`，不另起一套编号。旧`_audit/infra/`与归档只作历史。两个Lab独立；本组提供实验和必要[upstream DRAFT](../upstream/README.md)，正式PR由其负责人处理。
