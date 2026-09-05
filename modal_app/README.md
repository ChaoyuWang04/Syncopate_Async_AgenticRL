# Modal / B200 运行入口

> 当前机器、依赖、Volume 和安全规则见
> [docs/syncopate/05-COMPUTE.md](../docs/syncopate/05-COMPUTE.md)。
> 训练顺序见 [docs/syncopate/04-TRAINING.md](../docs/syncopate/04-TRAINING.md)，
> 当前任务见 [docs/syncopate/01-TASKS.md](../docs/syncopate/01-TASKS.md)。

## 两个入口

- [stack_probe.py](stack_probe.py)：创建 Modal 环境、检查机器、执行探针，并把结果写入审计目录。
- [scripts/v16_pipeline.sh](../scripts/v16_pipeline.sh)：数据、SFT、Exam、RL、OPD 的唯一业务管线。

Modal 层只负责选机器、准备容器、挂载 Volume 和调用固定管线，不复制模型路径、预算或训练参数。

## 当前环境

- Volume：`syncopate-home`
- GPU：B200 或 B200:2
- 镜像与依赖：[stack/pyproject.toml](stack/pyproject.toml) 和 [stack/uv.lock](stack/uv.lock)
- 容器 checkout：`/tmp/repo`
- 持久数据、模型和审计：`/vol`

精确布局和依赖版本只在 [05-COMPUTE.md](../docs/syncopate/05-COMPUTE.md) 维护。

当前验证状态：B01 上云前认证已通过；B02 已把 2×B200 真实 v16 全链机械接通。
详细读数和质量 WARN 见 [B02 报告](../_audit/infra/B02/REPORT.md)。这不是 candidate 或稳定性能 baseline。

## 常用探针

```bash
# 依赖和机器
modal run modal_app/stack_probe.py --steps image,verl,versions
modal run modal_app/stack_probe.py --steps gpu,fa4
modal run modal_app/stack_probe.py --steps nccl
modal run modal_app/stack_probe.py --steps vllm,vllm_ep

# 仓库与数据
modal run --detach modal_app/stack_probe.py --steps pytest
modal run --detach modal_app/stack_probe.py --steps rebuild_v16
modal run --detach modal_app/stack_probe.py --steps build_v16 --build-gates strict

# 已有分段冒烟
modal run --detach modal_app/stack_probe.py --steps sft_smoke --max-steps 30
modal run --detach modal_app/stack_probe.py --steps exam_v4 --exam-passes 1
modal run --detach modal_app/stack_probe.py --steps rl_cfg
modal run --detach modal_app/stack_probe.py --steps rl_smoke
modal run --detach modal_app/stack_probe.py --steps opd_smoke
modal run --detach modal_app/stack_probe.py --steps pipeline --pipeline-stage all \
  --pipeline-profile smoke --pipeline-gate-mode observe --pipeline-run-id <ID>
```

`pipeline` 是全链唯一云上入口，只转调 runbook；不会在 Modal 层再复制一套训练参数。
默认仍是 smoke/observe。判断时分开看：`manifest.pipeline_ok=true` 表示程序和产物链能继续；
`manifest.all_passed=false` 表示仍有 WARN。Modal 汇总会把后一种显示为红色提醒，但 observe 不会在 WARN 出现时提前杀掉训练。

## 固定业务管线

在容器内只使用：

```bash
bash scripts/v16_pipeline.sh [--dry-run] [--profile smoke|candidate] \
  [--gate-mode observe|strict] [--run-id ID] <stage|all>
```

阶段以脚本里的 `ALL` 数组为准。当前共 19 段：

```text
cases menus split gates supply rl-data teacher sft-data teacher-stop
sft-train sft-eval sft-select merge exam
rl-train rl-adapter rl-eval opd-train opd-eval
```

`--dry-run` 只检查命令形状和静态前置，不能代替数据、教师文本、GPU 或候选门禁。

本机没有目标 CUDA/B200、完整 verl/vLLM 或 PG/Redis 权限时，不在本机重配；把失败缩成最小测试，
用 `--steps pytest --pytest-args ...` 放到 Modal CPU 镜像，或用对应 pipeline stage 放到 B200。只跑需要的段，并保存 summary。

## 证据

- 本机汇总：`_audit/stack_probe/summary_*.json`
- Volume 汇总：`/vol/_audit/stack_probe/`
- v16 数据与训练：`/vol/_audit/v16/`
- 模型与训练产物：`/vol/models`、`/vol/checkpoints`

每个实验臂使用独立审计目录；不能让两个任务同时写同一 parquet、checkpoint 或缓存。

## 安全与收尾

- 产生费用或改变云端状态前，必须得到用户针对该次运行的明确授权。
- 密钥只放 Modal Secret 或本机受控配置。
- 长步骤使用 `--detach`，并确保阶段幂等、过程持续落盘。
- 新服务启动前先精确确认旧模型服务和子进程已经退出、显存回到底线。
- 不使用会匹配当前命令自身的宽泛进程杀法。
- 停 App：

```bash
modal app list
modal app stop <app-id> --yes
modal app list
```

- 删除 Volume 或大量产物前先核对精确目标；删除后再次列出确认。

## 历史

旧 `probe.py`、PRO 6000、4×5090 和旧依赖说明已经退出当前入口。整合前说明保存在
[modal-app-README.md](../docs/archive/syncopate/pre-consolidation-v16/modal-app-README.md)。
