# Scripts 目录规则

`scripts/` 只放执行入口、临时调度 shell 和环境/行为探针。会被多处复用的 Python 组件必须放在
`syncopate/` 中，由 `python -m syncopate...` 调用；正规模块不得反向 import `scripts/`。

当前结构：

- `v16_pipeline.sh`：主线唯一固定入口；默认 `smoke/observe`，candidate/strict 必须显式选择。
- `v16/`：v16 数据、考场与行为探针；考场 shell 只负责编排。
- `serving/`：服务启停、压测、故障演练与运行读数。
- `infra/`：硬件、通信、kernel 和训练基础设施探针。
- `tools/`：磁盘、checkpoint 和长跑守卫等运维工具。
- `archive/`：退出 v16 的历史脚本；现行代码、测试和 runbook 不得调用。

从原脚本区迁出的正式组件按职责放在：

- `syncopate/pipeline/`：数据构建、菜单、门禁、切分隔离和训练材料。
- `syncopate/evaluation/`：考卷、考场执行、判卷、认证和分诊。
- `syncopate/train/`：选点、adapter/checkpoint 处理、候选门和训练分析。
- `syncopate/runtime/` 与 `syncopate/domains/`：serving 可复用逻辑和领域数据导入。
- `syncopate/authoring/`：可重复的数据标定工具。

需要新增 Python 文件时，先问一句：它是“可被 import/重用的功能”，还是“仅负责观测或现场执行”？
前者进 `syncopate/`，后者才进这里。

注意：这些脚本**不是每次都会跑**。日常主线只从 `v16_pipeline.sh` 进入，它调用
`syncopate/` 中的正式组件，考场阶段再调 `v16/exam_chain.sh`。`serving/`、`infra/`、
`tools/` 与其他 `v16/` 探针都只在对应验收、调试或运维时按需使用。
