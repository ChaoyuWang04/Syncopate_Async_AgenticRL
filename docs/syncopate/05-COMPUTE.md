# Infra · Compute

> 本文是主线当前机器、Modal 环境、依赖栈、Volume 和机器探针的唯一现行说明。
> 训练实验看 [Infra TRAINING](../infra_exp/03-TRAINING.md)，当前任务看 [Infra TASKS](../infra_exp/01-TASKS.md)。
> [原业务 TRAINING](04-TRAINING.md) 只解释已停止管线的保留实现。

## 1. 当前资源上限与范围

2026-09-08用户明确：主要平台Modal，单次真实场景上限8张B200，不是每次都用8卡。按模型容量、角色摆放及合理TP/EP等并行关系取卡；需要4/8卡的真实场景不为节省而改成失去代表性的双卡微测试。上限包括trainer、rollout、ref/teacher同时占用；默认本组并发总占卡也≤8，A/B必要时顺序交叉。

每run仍登记卡数/角色、CPU/内存、硬时限、预计卡时和费用预留；8卡不是长期训练或无限金额授权。本次仅文档和只读盘点，没有GPU投递。其他卡型和旧预算不自动继承，当前不排B300/H100/RTX PRO 6000；今后明确调整范围再调查。

2026-09-08重查[Modal GPU文档](https://modal.com/docs/guide/gpu)：单容器B200支持最多8卡且同物理机，超过2卡通常排队更久。不同容器不保证同机；具体可分配性、拓扑/显存和镜像仍需实际核验。使用精确B200，不用可混配B300的B200+。历史CUDA13.0镜像不等于B300就绪。

本地5090可能承载其他Lab，不能假定空闲或按名称停止进程；不改变它们的环境/Volume/预算。

### 既有 B200 镜像（历史实测，不是全硬件默认）

| 项目 | 当前值 |
|---|---|
| 云平台 | Modal |
| 持久化 Volume | `syncopate-home` |
| 已实现旧入口资源 | Modal CPU、1×B200或2×B200；不是新8卡入口已实现声明 |
| 架构 | sm_100 |
| 基础镜像 | CUDA 13.0 devel / Ubuntu 24.04 / Python 3.12 |
| 主训练栈 | PyTorch 2.13.0、vLLM 0.28.0、verl 0.9.0、Transformers 5.10.4 |
| 注意力与线性注意力 | flash-attn 2.8.3、flash-linear-attention 0.5.2 |
| 历史学生/本轮MoE优先复用 | `Qwen3.6-35B-A3B` |
| 历史教师/本轮dense优先复用 | `Qwen3.8-27B` |
| 测试分词器 | `Qwen3.5-0.8B` |

新场景资源盘点见[2026-09-08资源记录](../infra_exp/storage/20260908-resources.md)：主MoE与dense索引引用分片均存在，未重哈希整模/加载。其他库存不增加主模型变量；公开数据尚未准备。新场景使用独立冻结配置，不继承学生/教师角色。

保留业务的精确模型路径只认 [model_paths.py](../../syncopate/core/model_paths.py)，精确依赖只认
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

现有业务固定管线使用 Modal Dict 的原子占位登记写者；同一 run id 或正式数据版本被占用时，第二个写者拒绝启动。容器异常退出留下的占位不能直接清掉：先确认旧容器已退出。Volume 本身不提供可用的分布式文件锁，不能用 `flock` 代替这一检查。

缓存复制先在 CPU 完成，写出与源码绑定的 `cache_ready.json` 后才分配 GPU。GPU 只读取准备结果，不占着显卡复制大量小文件；被打断的复制没有完成标记，不会被复用。

诊断重跑可显式顺序复用已完成的缓存，但必须先核对旧运行已停止。新记录保留旧缓存来源，不冒充同一源码；GPU 对实际缓存目录另加原子写者占位，不允许两个臂同时写同一缓存。框架仍按自身缓存键判断编译产物是否可用。

源码上传前先冻结快照，容器只读取明确的 Git 提交，再叠加可选的不可变源码快照。同一 run 的 `source.json` 必须始终相等；不同源码不能使用 `--resume` 拼接结果。已有 B03/B13 运行都登记了 Git 底座、overlay、编排源码和镜像，不能冒充纯 Git 提交。当前改动推送后，后续正式重复应优先直接钉住新的明确 Git SHA；如果仍有 overlay，也必须单独登记并在各重复中完全相同。

### 产物保留与清理

最近实测：[2026-09-07 清理记录](../infra_exp/storage/20260907-cleanup.md)。旧合并模型与部分训练恢复文件已删除，不能从历史 REPORT 推断它们仍可加载。

用户已授权：本项目实验结束后，确认不会再用于恢复、复验、后续实验或 upstream 复现的 checkpoint 与临时大产物直接清理，不必逐次重复询问。共享基模、活动任务使用的产物、归属或复用需求不明的文件保留待核对；不清理两个独立 Lab 的资源。

- 删除前列出精确 Volume 身份、路径、字节数、所属 run、删除理由，检查活动 App/容器和引用；不能从旧任务停止推断整个目录都无用。
- 先保留必要的配置、源码/模型身份、指标、验证结果、最小复现输入和产物指纹。若待移交问题必须依赖原 checkpoint，保留并写明用途；小型 adapter 可按实际复用价值保留，不因为文件名就自动删除。
- 删除后重新列目录确认目标不存在，并记录可核验的前后大小；文件逻辑字节数与平台计费占用分开报告，不保证账单立刻变化。失败或部分删除如实登记。
- 每份新实验 REPORT 的收尾写保留项/删除项、原因和索引；清理机读记录放对应 run 的小型索引。旧系列批量盘点记录独立存储维护摘要，不占用新实验 B00 编号。
- 这是每次实验的收尾规则，当前没有定时清理服务；禁止按年龄、文件名或“超过 1 TiB”无差别删除。

## 3. 登记入口

### 环境和机器

[modal_app/stack_probe.py](../../modal_app/stack_probe.py) 保留历史机器/业务编排；独立探针使用各实验已登记入口，包括 `infra_probes.py`、`lora_probe.py` 与 `probability_batch.py`。
使用方式集中在 [modal_app/README.md](../../modal_app/README.md)。

既有探针示例（会分配资源，不因列在文档中自动获准运行）：

```bash
modal run modal_app/stack_probe.py --steps versions,gpu,nccl
modal run --detach modal_app/stack_probe.py --steps pytest
```

### 保留的业务数据和训练入口（原排期已停止）

现有代码中的业务阶段仍必须调用下列入口；新的真实场景入口按Q21准备与按需实现，不能靠文档绕过现有身份闸：

```bash
bash scripts/v16_pipeline.sh [--dry-run] [--profile smoke|candidate] \
  [--gate-mode observe|strict] [--run-id ID] <stage|all>
```

Modal 层不得复制训练命令、模型路径或预算。

云端 `train-all` 按阶段顺序换容器：SFT/评测/Exam 用单卡，merge/adapter 导出/选择用 CPU，RL/OPD 用双卡。具体资源由 [stages.py](../../syncopate/pipeline/stages.py) 维护。教师与 SFT 建库使用同一个单卡容器中的 `sft-data-group`，避免教师端点随容器消失。

## 4. 已验证的 B200 历史能力

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

- 当前按TASKS的场景准备→真实画像推进，不再沿旧概率/通信批次排期。运行在用户明确范围内，每run登记时限/费用估计和证据路径；历史B03预算不作当前余额。公开发布和主动故障注入另行批准。
- 只需 CPU 的检查不申请 GPU；卡型和卡数按已登记的合理组合选择，实验按 TASKS 逐项推进，同实验独立 A/B 臂按下条并行，不为穷举硬件而扩资源。当前代码的阶段资源仍固定 B200，新卡型调度尚待实现和验证。
- 本地先做实验相关语法、契约、输入结构与假引擎检查，选用旧业务入口时才做其 runbook 检查。只有目标依赖或硬件缺失的检查放进 Modal 的目标 CPU/GPU 环境；已知本地错误修好后再上 GPU。
- 优化阶段定向回归通过后，同一实验可独立的性能臂在总占卡≤8内并行，否则顺序交叉；每臂独立 checkout、缓存、run id、审计和产物路径。已冻结共享数据和模型只读。两个双卡臂可以同时使用共四张 B200。
- 并行机器分别记录拓扑、预热与噪声，必要时每组交叉运行两种配置。比较时同时报告墙钟和总卡时。
- observe 默认收齐质量红项；错误身份、越桶、NaN、程序异常、坏产物和零真实更新仍停止。
- 网络下载、模型权重和重型编译默认放 Modal 容器和 Volume，本机主要做代码、检查和读取证据。
- 本机缺少目标 CUDA、B200、vLLM/verl 完整依赖或服务权限时，不在本机重配环境拖延；先把验证缩成最小定向项，再直接放进 Modal 的 CPU 或 B200 目标镜像。结果必须落盘，不能用“云上应该可以”代替实测。
- Modal 对象不能按环境变量条件定义；可选 Secret 也要用稳定对象形状。
- 密钥只放 Modal Secret 或本机受控配置，不进入仓库、日志和文档。
- 操作前核对 App/run-id、PID/启动时间、父子进程、工作目录、GPU 与所属任务；不明归属先询问。只停止本实验资源，随后核对退出；不使用宽泛 pkill 或“清空整台机器显存”。其他 Lab 的 Modal 与 5090 负载都必须保护。
- 停止 App 后再次查看 App 列表；删除 Volume 或大目录前先解析并核对精确目标。
- 换镜像、CUDA、attention kernel或训练框架后，补所需接口/硬件运行健康检查；不要求先做全套底层扫描，具体优化的数值回归按实验协议后续验收。

## 6. 环境判据

以下是既有探针的能力索引，不是每次实验的必跑清单。独立实验只登记与其问题相关的检查；原业务重建已停止，PostgreSQL/Redis 与业务 stage 不作为通用前置。

| 判据 | 证明什么 |
|---|---|
| versions | 实装版本与锁文件一致；偏离项有明确钉住原因 |
| models | 本地权重字节与来源声明一致 |
| gpu | 架构正确，关键 kernel 前向与反向均通过 |
| nccl | 目标双卡拓扑真实可通信 |
| vllm / vllm_ep | 当前模型能以登记形态启动和生成 |
| pytest | 实验相关回归；旧业务 PG/Redis 回归仅为保留能力 |
| rebuild_v16（已停止） | 历史业务重建的切分 SHA 对拍，当前不运行 |
| stage 审计 | 被选中的固定入口输入、输出和退出状态落盘 |

已登记为本实验必需的检查，空结果、跳过或没有判据行都不算通过；不相关项明确标为不适用，不推动业务建设。

## 7. 历史线索（按新任务调查，不自动续跑）

- B02 是跨多次修复拼成的机械证据链，不能作为稳定性能 reference；新实验各自固定源码、预热窗口、重复次数和费用口径。
- B03 已补齐 35B 的真实更新、adapter、同步载荷与原始 token，也用小模型验过双卡梯度和正常恢复；35B trainer/vLLM 概率、GDN/MoE 路由及至少三次固定源码重复仍未完成。
- 多次 Modal 分配的 region/主机不同，`nvidia-smi topo -m` 没有成功落出完整矩阵；跨次数字不能直接比较。
- vLLM/FlashInfer 对本轮多组 MoE shape 报 tuning bucket 未覆盖，推理时仍触发 Triton JIT；这会污染冷启动和延迟。
- vLLM 已警告 raw prompt 的 InputProcessor 接口将移除，需迁移到 Renderer API 并对拍 token 序列。
- 还没有全程 GPU busy、功耗、通信占比和可靠总费用，B02 的单点速度不能进入默认或简历。
- B13 的短微测试已有通信与 GEMM 数字，但遥测采样太稀，不能据此声称利用率、稳态功耗、能耗或 Tensor Core 指令利用率；完整拓扑也仍缺失。

这些仅为历史线索，不在活动队列；需真实画像触发才能重开，排期只认infra [TASKS](../infra_exp/01-TASKS.md)。

## 8. 历史资料

旧环境和迁移过程只在 `docs/archive/` 中追查。当前执行只认本文和 Modal README，不从历史运行说明继承机器、模型或默认值。
