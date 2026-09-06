# Syncopate

在模拟业务场景中，把多轮工具 agent 的 SFT → RL → OPD、评测与服务工程做好。用真实更新和可重复实验理解训练、推理与基础设施，不以业务上线、刷分、PR 或论文为目标。

## 从这里开始

- [项目规则](AGENTS.md)
- [主线入口](docs/syncopate/00-START.md) · [主线任务](docs/syncopate/01-TASKS.md)
- [Infra 入口](docs/infra_exp/00-START.md) · [Infra 任务](docs/infra_exp/01-TASKS.md)
- [数据](docs/syncopate/03-DATA.md) · [训练与评测](docs/syncopate/04-TRAINING.md)
- [计算资源](docs/syncopate/05-COMPUTE.md) · [Modal 操作](modal_app/README.md)

## 当前边界

Mac 用于阅读、修改、Git 和 CPU 检查。云端只使用 Modal：需要 CPU 就申请 CPU，需要单卡就申请一张 B200，双卡实验申请两张 B200。独立实验可以同时申请多组资源，各自保存产物。

数据、模型、checkpoint 和完整轨迹保存在 Modal 的 `syncopate-home` Volume。Git 只保存代码、配置、测试、文档和小型审计结论。历史归档用于追查证据，不是运行指南。

当前 v16 管线已经完成 B02 机械全链 smoke，各训练段有真实更新和可加载产物；质量仍有告警，尚未建立固定源码性能基线，也没有运行 candidate。最新进度只认两份 TASKS。

现有数据冻结，不再做语义清洗。质量告警保留观察，但不阻塞 infra；token/mask、数值、身份、数据隔离和真实更新仍必须正确。发现工程问题就修复并记录，通用问题按 `docs/upstream/` 留材料，不要求强行扩展成投稿。

## 固定入口

```bash
# Mac：按锁文件安装 CPU 开发和 Runtime 测试依赖
uv sync --frozen --extra dev --extra runtime

# 本地先检查命令和阶段连接
bash scripts/v16_pipeline.sh --dry-run --profile smoke --gate-mode observe all
```

云端由 `modal_app/stack_probe.py` 编排，业务步骤只调用 `scripts/v16_pipeline.sh`。默认 `smoke/observe`：质量告警留下证据并继续；错误身份、数据越桶、NaN、坏产物和零真实更新仍停止。Candidate 的门槛和运行需要单独批准。

## 目录

```text
syncopate/   可复用 Python 组件：数据、训练、评测和 Runtime
scripts/     固定入口、Shell 调度和探针
modal_app/   Modal 编排与冻结的云端依赖
configs/     数据和训练配置
tests/       自动检查
docs/        当前专题和历史归档
_audit/      小型报告；大体积原始证据保存在 Volume
```

模型路径由 `syncopate/core/model_paths.py` 定义，训练预算由 `syncopate/train/rollout_budget.py` 定义。不要在入口中复制默认值。

MIT License，见 [LICENSE](LICENSE)。
