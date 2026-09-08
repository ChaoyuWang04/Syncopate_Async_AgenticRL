---
name: syncopate-docs-map
description: 当前 Infra 队列及停止业务专题的文档导航
metadata:
  node_type: memory
  type: project
  modified: 2026-09-07
---

# 项目文档地图

进入项目先读 AGENTS、本目录 MEMORY 与相关条目，再读两份 START；当前施工以 [Infra TASKS](../../docs/infra_exp/01-TASKS.md) 为准。[原主线 TASKS](../../docs/syncopate/01-TASKS.md) 只登记停止项。

| 文档 | 唯一职责 |
|---|---|
| `docs/infra_exp/00-START.md` | 当前入口 |
| `docs/infra_exp/01-TASKS.md` | P/Q真实场景准备、画像与证据触发的优化队列；G参考已移到SYSTEM |
| `docs/infra_exp/02-SYSTEM.md` | 锁定范围、真实负载/数据维度、合理组合与定位参考地图 |
| `docs/infra_exp/03-TRAINING.md` | 训练、RL/async、OPD 的调查方法 |
| `docs/infra_exp/04-SERVING.md` | vLLM/SGLang/TensorRT-LLM推理与rollout |
| `docs/infra_exp/05-COMPUTE-AND-KERNELS.md` | 硬件、MoE、算子/量化和通信 |
| `docs/infra_exp/06-EXPERIMENTS.md` | 背景调查准入与唯一 REPORT 模板 |
| `docs/syncopate/05-COMPUTE.md`、`modal_app/README.md` | 资源规则与已实现入口 |
| `docs/syncopate/02～04`、`06～07` | 已停止业务系统的保留实现契约 |
| `docs/infra_exp/experiments/Bxx-主题/REPORT.md` | 一个具体实验的背景、设置、逐次测试、结论 |
| `docs/upstream/README.md` | 实验移交与独立 PR 负责人的分工 |

TASKS 不复制日志；REPORT 关闭后原地保留；历史资料的旧待办、默认值与定位不生效。项目记忆负责导航，不能替代当前证据。Harness 与 Sandbox Lab 各自维护文档，不在本项目排期。
