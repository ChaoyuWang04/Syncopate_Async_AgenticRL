# Syncopate · Compute

> 本文是主线当前机器、Modal 环境、依赖栈、Volume 和机器探针的唯一现行说明。
> 训练步骤看 [04-TRAINING.md](04-TRAINING.md)，任务顺序看 [01-TASKS.md](01-TASKS.md)。

## 1. 当前环境

| 项目 | 当前值 |
|---|---|
| 云平台 | Modal |
| 持久化 Volume | `syncopate-home` |
| 当前训练卡 | B200 × 2 |
| 架构 | sm_100 |
| 基础镜像 | CUDA 13.0 devel / Ubuntu 24.04 / Python 3.12 |
| 主训练栈 | PyTorch 2.13.0、vLLM 0.28.0、verl 0.9.0、Transformers 5.10.4 |
| 注意力与线性注意力 | flash-attn 2.8.3、flash-linear-attention 0.5.2 |
| 当前学生 | `Qwen3.6-35B-A3B` |
| 当前教师 | `Qwen3.8-27B` |
| 测试分词器 | `Qwen3.5-0.8B` |

精确模型路径只认 [model_paths.py](../../syncopate/core/model_paths.py)，精确依赖只认
[modal_app/stack/pyproject.toml](../../modal_app/stack/pyproject.toml) 和锁文件。

FlashAttention 4 因依赖冲突放在独立环境中，只用于对应探针和内核实验，不与主 vLLM 环境混装。

## 2. Volume 布局

```text
/vol/repo.git                    bare 仓库镜像
/tmp/repo                        每个容器自己的 checkout
/vol/data/{batches,sft,rl}       数据产物
/vol/models                      模型权重
/vol/checkpoints                 SFT / RL / OPD checkpoints
/vol/flashinfer_cache            FlashInfer 编译与 AOT 缓存
/vol/vllm_cache                  vLLM 缓存
/vol/_audit/stack_probe          机器与依赖探针证据
/vol/_audit/v16                  v16 数据和训练证据
```

规则：

- 共享仓库、数据和审计路径同一时刻只有一个写者。
- 每个容器从 bare 仓库建立自己的 checkout，不在共享工作目录并发修改。
- 每个实验臂写自己的审计子目录，并记录镜像、GPU、拓扑和代码身份。
- 长任务分阶段落盘，允许被抢占后幂等重跑。
- 缓存必须绑定数据切分和构造器版本，不能靠人工记得清理。

## 3. 唯一入口

### 环境和机器

[modal_app/stack_probe.py](../../modal_app/stack_probe.py) 是 Modal 当前唯一探针与编排入口。
使用方式集中在 [modal_app/README.md](../../modal_app/README.md)。

常用只读或诊断步骤：

```bash
modal run modal_app/stack_probe.py --steps versions,gpu,nccl
modal run --detach modal_app/stack_probe.py --steps pytest
```

### 数据和训练

容器中的每个业务阶段都必须调用：

```bash
bash scripts/v16_pipeline.sh [--dry-run] [--profile smoke|candidate] \
  [--gate-mode observe|strict] [--run-id ID] <stage|all>
```

Modal 层不得复制训练命令、模型路径或预算。

## 4. 已验证的机器能力

当前 B200 环境已经验证：

- 主栈能够安装和导入；
- 模型权重完整性检查通过；
- CUDA、flash-attn 反向、FLA 训练核通过；
- 双卡 NCCL 通信通过；
- vLLM 单卡和双卡 EP 能启动；
- PostgreSQL、Redis 与 Runtime 测试可在镜像中运行；本机环境跑不了的 5 个定向测试已在该镜像 5/5 通过；
- v16 题库可在本机和 Modal 确定性重建；
- 真实 v16 SFT、Exam、双卡 RL、RL adapter 导出、OPD 和两段评测已连续运行；完整结果见 [B02 报告](../../_audit/infra/B02/REPORT.md)。

这些结果证明 B200 环境可工作，不证明候选训练或 Serving 正式验收完成。

## 5. 运行纪律

- 产生费用或改变云端状态前，必须获得用户对该次运行的明确授权。
- 网络下载、模型权重和重型编译默认放 Modal 容器和 Volume，本机主要做代码、检查和读取证据。
- 本机缺少目标 CUDA、B200、vLLM/verl 完整依赖或服务权限时，不在本机重配环境拖延；先把验证缩成最小定向项，再直接放进 Modal 的 CPU 或 B200 目标镜像。结果必须落盘，不能用“云上应该可以”代替实测。
- Modal 对象不能按环境变量条件定义；可选 Secret 也要用稳定对象形状。
- 密钥只放 Modal Secret 或本机受控配置，不进入仓库、日志和文档。
- GPU 任务结束后精确核对模型服务和子进程已经退出，不能用会误杀自身的宽泛进程匹配。
- 停止 App 后再次查看 App 列表；删除 Volume 或大目录前先解析并核对精确目标。
- 换镜像、卡型、CUDA、attention kernel 或训练框架后，先重跑对应探针，再运行训练。
- B200 的能力和性能读数不能直接套到 B300；B300 的复核条件见 TASKS 的 T4。

## 6. 环境判据

环境健康至少要能回答：

| 判据 | 证明什么 |
|---|---|
| versions | 实装版本与锁文件一致；偏离项有明确钉住原因 |
| models | 本地权重字节与来源声明一致 |
| gpu | 架构正确，关键 kernel 前向与反向均通过 |
| nccl | 目标双卡拓扑真实可通信 |
| vllm / vllm_ep | 当前模型能以登记形态启动和生成 |
| pytest | 带 PostgreSQL/Redis 的仓库回归正常结束 |
| rebuild_v16 | 本机和 Modal 的三份切分 SHA 完全一致 |
| stage 审计 | 每个固定管线阶段的输入、输出和退出状态落盘 |

空结果、跳过或没有出现判据行都不算通过。

## 7. 当前已知机器侧欠账

- B02 是跨多次修复拼成的机械证据链，不能作为稳定性能 reference；下一次要固定源码、预热窗口、重复次数和费用口径。
- 多次 Modal 分配的 region/主机不同，`nvidia-smi topo -m` 没有成功落出完整矩阵；跨次数字不能直接比较。
- vLLM/FlashInfer 对本轮多组 MoE shape 报 tuning bucket 未覆盖，推理时仍触发 Triton JIT；这会污染冷启动和延迟。
- vLLM 已警告 raw prompt 的 InputProcessor 接口将移除，需迁移到 Renderer API 并对拍 token 序列。
- 还没有全程 GPU busy、功耗、通信占比和可靠总费用，B02 的单点速度不能进入默认或简历。

这些工作只登记在 infra [01-TASKS.md](../infra_exp/01-TASKS.md)，本页不展开施工计划。

## 8. 历史资料

旧 4×5090、PRO 6000、旧依赖栈、迁移过程和历次失败均已移入：

- [08-machine-and-environment.md](../archive/syncopate/pre-consolidation-v16/08-machine-and-environment.md)
- [31-modal-and-new-stack.md](../archive/syncopate/pre-consolidation-v16/31-modal-and-new-stack.md)
- [modal-app-README.md](../archive/syncopate/pre-consolidation-v16/modal-app-README.md)

它们用于追查背景，不代表当前启动方法。
