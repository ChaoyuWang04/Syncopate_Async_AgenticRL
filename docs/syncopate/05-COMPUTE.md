# Syncopate · Compute

> 本文是主线当前机器、Modal 环境、依赖栈、Volume 和机器探针的唯一现行说明。
> 训练步骤看 [04-TRAINING.md](04-TRAINING.md)，任务顺序看 [01-TASKS.md](01-TASKS.md)。

## 1. 当前环境

| 项目 | 当前值 |
|---|---|
| 云平台 | Modal |
| 持久化 Volume | `syncopate-home` |
| 计算资源 | Modal CPU、1×B200 或 2×B200，按任务申请 |
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
/vol/flashinfer_cache            只读 FlashInfer 缓存种子
/vol/vllm_cache                  只读 vLLM 缓存种子
/vol/_cache_runs/<调用ID>/       每容器自己的可写缓存
/vol/_audit/stack_probe          机器与依赖探针证据
/vol/_audit/v16                  v16 数据和训练证据
```

规则：

- 共享仓库、数据和审计路径同一时刻只有一个写者。
- 每个容器从 bare 仓库建立自己的 checkout，不在共享工作目录并发修改。
- 每个实验臂写自己的审计子目录，并记录镜像、GPU、拓扑和代码身份。
- 长任务分阶段落盘，允许被抢占后幂等重跑。
- 缓存必须绑定数据切分和构造器版本，不能靠人工记得清理。

固定管线使用 Modal Dict 的原子占位登记写者；同一 run id 或正式数据版本被占用时，第二个写者拒绝启动。容器异常退出留下的占位不能直接清掉：先确认旧容器已退出。Volume 本身不提供可用的分布式文件锁，不能用 `flock` 代替这一检查。

缓存复制先在 CPU 完成，写出与源码绑定的 `cache_ready.json` 后才分配 GPU。GPU 只读取准备结果，不占着显卡复制大量小文件；被打断的复制没有完成标记，不会被复用。

诊断重跑可显式顺序复用已完成的缓存，但必须先核对旧运行已停止。新记录保留旧缓存来源，不冒充同一源码；GPU 对实际缓存目录另加原子写者占位，不允许两个臂同时写同一缓存。框架仍按自身缓存键判断编译产物是否可用。

源码上传前先冻结快照，容器只读取明确的 Git 提交，再叠加可选的不可变源码快照。同一 run 的 `source.json` 必须始终相等；不同源码不能使用 `--resume` 拼接结果。已有 B03/B13 运行都登记了 Git 底座、overlay、编排源码和镜像，不能冒充纯 Git 提交。当前改动推送后，后续正式重复应优先直接钉住新的明确 Git SHA；如果仍有 overlay，也必须单独登记并在各重复中完全相同。

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

云端 `train-all` 按阶段顺序换容器：SFT/评测/Exam 用单卡，merge/adapter 导出/选择用 CPU，RL/OPD 用双卡。具体资源由 [stages.py](../../syncopate/pipeline/stages.py) 维护。教师与 SFT 建库使用同一个单卡容器中的 `sft-data-group`，避免教师端点随容器消失。

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
- B03 的两层 full-attention Text MoE 小模型已在 2×B200 上通过独立梯度、同状态 AdamW、连续两步和正常 checkpoint 恢复；这不覆盖 35B、GDN、LoRA、rollout 或性能。
- B13 的首轮 2×B200 合成微测试完成 30 个通信 case 和两卡 BF16 GEMM。64 MiB 分块的 bus 带宽为 all-reduce 377.847、all-gather 336.247、reduce-scatter 325.812 GB/s；4096³ GEMM 为 1345.706～1355.915 TFLOP/s。数字只属于本次分配，不是训练 baseline。

这些结果证明 B200 环境可工作，不证明候选训练或 Serving 正式验收完成。

初始 Mac 接手检查（2026-09-05）：Git SSH 只读认证可用；当时远端 main 仍为交接提交。Modal CPU 实际依赖导入通过，W&B 探针写入并读回 3 个点。现有模型、SFT/RL 数据、教师缓存和 B02 产物仍在 Volume。证据为本机 `_audit/stack_probe/summary_2026-09-05_141830_image-verl-versions.json` 与 `summary_2026-09-05_142401_wandb.json`。Mac 只安装 CPU 开发依赖和小型 tokenizer 文件，不安装训练栈或下载权重。

## 5. 运行纪律

- 用户已批准以工程为主的 B03 基线、训练/推理/kernel 对比、采用项组合和全链学习计划。首批总费用上限 $300、最多同时 6 张 B200；每批仍登记卡数、时限和证据，接近总限额前停止新分配。Candidate、公开发布、主动故障注入及扩大资源范围另行确认。费用明细和批次记录归 [B03 REPORT](../../_audit/infra/B03/REPORT.md)，不能把 GPU 费用当全账单。
- 只需 CPU 的检查单起 CPU；模型能用一张卡的步骤单起 B200；双卡训练或对照申请 B200:2。允许同时运行多个独立资源组。
- 本地先做语法、契约、数据结构、假引擎和 runbook 检查。只有目标依赖或硬件缺失的检查放进 Modal CPU/B200；已知本地错误修好后再上 GPU。
- 正确性前置过后，可独立的性能臂尽量并行；每臂独立 checkout、缓存、run id、审计和产物路径。已冻结共享数据和模型只读。两个双卡臂可以同时使用共四张 B200。
- 并行机器分别记录拓扑、预热与噪声，必要时每组交叉运行两种配置。比较时同时报告墙钟和总卡时。
- observe 默认收齐质量红项；错误身份、越桶、NaN、程序异常、坏产物和零真实更新仍停止。
- 网络下载、模型权重和重型编译默认放 Modal 容器和 Volume，本机主要做代码、检查和读取证据。
- 本机缺少目标 CUDA、B200、vLLM/verl 完整依赖或服务权限时，不在本机重配环境拖延；先把验证缩成最小定向项，再直接放进 Modal 的 CPU 或 B200 目标镜像。结果必须落盘，不能用“云上应该可以”代替实测。
- Modal 对象不能按环境变量条件定义；可选 Secret 也要用稳定对象形状。
- 密钥只放 Modal Secret 或本机受控配置，不进入仓库、日志和文档。
- GPU 任务结束后精确核对模型服务和子进程已经退出，不能用会误杀自身的宽泛进程匹配。
- 停止 App 后再次查看 App 列表；删除 Volume 或大目录前先解析并核对精确目标。
- 换镜像、卡型、CUDA、attention kernel 或训练框架后，先重跑对应探针，再运行训练。

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
- B03 已补齐 35B 的真实更新、adapter、同步载荷与原始 token，也用小模型验过双卡梯度和正常恢复；35B trainer/vLLM 概率、GDN/MoE 路由及至少三次固定源码重复仍未完成。
- 多次 Modal 分配的 region/主机不同，`nvidia-smi topo -m` 没有成功落出完整矩阵；跨次数字不能直接比较。
- vLLM/FlashInfer 对本轮多组 MoE shape 报 tuning bucket 未覆盖，推理时仍触发 Triton JIT；这会污染冷启动和延迟。
- vLLM 已警告 raw prompt 的 InputProcessor 接口将移除，需迁移到 Renderer API 并对拍 token 序列。
- 还没有全程 GPU busy、功耗、通信占比和可靠总费用，B02 的单点速度不能进入默认或简历。
- B13 的短微测试已有通信与 GEMM 数字，但遥测采样太稀，不能据此声称利用率、稳态功耗、能耗或 Tensor Core 指令利用率；完整拓扑也仍缺失。

这些工作只登记在 infra [01-TASKS.md](../infra_exp/01-TASKS.md)，本页不展开施工计划。

## 8. 历史资料

旧环境和迁移过程只在 `docs/archive/` 中追查。当前执行只认本文和 Modal README，不从历史运行说明继承机器、模型或默认值。
