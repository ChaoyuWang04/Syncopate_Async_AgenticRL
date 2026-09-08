# Syncopate · Training & Inference Infra

面向开源贡献的训练与推理基础设施实验。先在合理的模型、硬件和并行组合跑真实训练，观察速度瓶颈，再引入优化并回原入口A/B，交付可复现证据和实验性修复；疑似上游问题以 DRAFT 移交，由 upstream 负责人完成正式考据、PR 与提交。

## 当前方向（2026-09-08）

- 工作重心：训练框架、SFT/RL/async/OPD、推理引擎、MoE、算子/量化、通信与权重传递。
- 现有业务场景的数据构建、教师扩写、清洗和全链学习排期停止。已有代码、数据和历史产物保留；模型只作为负载，固定一个MoE和一个dense；复用公开数据，仅做必要格式转换。
- 每个实验先做最新官方资料/GitHub 背景调查，先确认真实组合可运行并形成画像，再以瓶颈证据安排优化；不先猜算子错误。业务准确率后置，探索保留执行健康，优化后补定向回归。没有证据不登记“上游 bug”；已有可用修法先验证，不重复造轮子。
- 每个具体实验只有一份 REPORT，记录背景、完整设置、每次测试和结论。新实验不等待旧 B03 或 SFT→RL→OPD 产物链。
- 本项目与 `harness-lab/`、`sandbox-rl-MOPD-lab/` 默认完全独立。Harness/tool runtime 由其他 Lab 探索，本项目不复制其排期；未来交互另行约定。

## 入口

- [工作规则](AGENTS.md)
- [Infra START](docs/infra_exp/00-START.md) · [实际执行队列](docs/infra_exp/01-TASKS.md)
- [实验地图与组合筛选](docs/infra_exp/02-SYSTEM.md) · [背景调查与实验记录规范](docs/infra_exp/06-EXPERIMENTS.md)
- [资源、隔离和费用](docs/syncopate/05-COMPUTE.md) · [现有 Modal 入口](modal_app/README.md)
- [原业务线状态](docs/syncopate/01-TASKS.md) · [上游移交](docs/upstream/README.md)

## 实现与证据边界

`syncopate/` 放可复用 Python，`scripts/` 放固定入口和薄调度，`modal_app/` 管资源。当前实现仍包含 v16 业务 runbook；本次方向调整没有实现通用 base 探针入口或新卡型调度，不能把新规划当成已可运行能力。

Modal 是主要计算平台，本轮上限8卡B200，按真实场景需要分配，所有角色和并发计入；RL框架仅verl/AReaL/ROLL/NeMo-RL/Miles/slime，以verl为主；训练FSDP2/Megatron，推理vLLM/SGLang/TensorRT-LLM。每run的卡数、拓扑、镜像、时限和费用登记；共享主机可能有其他 Lab 的 5090 任务，停进程先核对归属。

Git 保存代码、配置、测试、报告和小型证据索引；模型、原始轨迹、checkpoint、缓存与遥测留在实验专属持久存储。历史资料只用于追溯，不驱动当前排期。

MIT License，见 [LICENSE](LICENSE)。
