# B03 · 固定源码、真实更新与性能基线

状态：35B 的 c/d 批已验真实更新、adapter、同步载荷、逐轮版本和原始 token。小模型 `training_20260906c` 又完成了独立梯度、同状态 AdamW、双卡连续两步和正常 checkpoint 恢复；原汇总只因 PyTorch 返回裸 UUID 而误报红，修正格式判断后对保存的两 rank 证据重新聚合通过。35B 的 trainer/vLLM 概率差、两侧 GDN/MoE 路由和三次可比重复仍未完成，因此 B03 仍不是性能 baseline。

本地前置 145 passed、11 skipped，证据为本目录 `local_preflight_20260905a.xml`；使用 Mac 已有的学生 tokenizer，不安装模型或训练栈。10 项缺目标依赖由 Modal CPU 补验，1 项不适用于 v15。正式 v15 配置下 `supply` 实跑通过：L2 为 172/150。最初直接调用漏设 v15，读到旧协议的 172/200，那次读数作废；没有改数据或阈值。

## 1. 现在要解决什么

用户已批准以工程为主继续项目：现有数据冻结，质量告警不再阻塞 infra；不以求职、PR 或论文倒推任务。先确认最近的 token/mask 修复进入真实 RL 更新，再建立可信的比较尺子。遇到通用工程问题按 `docs/upstream/README.md` 留 DRAFT，后续上游扩展由用户决定。

T1-1 的真实更新复验已在本实验完成，不再重复开同类任务。历史固定题目仍有模型行为告警，不宣称模型退化已根治。

## 2. 总体顺序和验收

1. 本地与 CPU 前置、2-step RL 接线复验。
2. B03 固定源码重复 reference，与 B13 一起记录硬件和时间分解。
3. 正确性通过后，B04/B05/B06、B11 及有瓶颈证据的 B10 独立臂并行。
4. B08/B09/B12/B14 分批筛选；B300 先做镜像与资源计划，再定向复验采用项。
5. B07 验证组合，随后从 SFT 到 OPD 完成学习运行。每个方向给出采用、拒绝、不适用或无结论，不穷举组合。

性能 reference 使用当前稳定的官方生产配置，未验自研和低精度开关关闭。至少三个独立重复；固定模型、数据、seed、有效 batch、输入/token 工作量、预算、源码与镜像。真实 on-policy 的生成长度另报，不能把回放冒充探索或把短回答当作系统提速。冷启动、预热、稳态分别记录；采集 step time、token/s、有效更新、GPU busy/显存/功耗、通信、成本及波动。主要采用目标见 [实验协议](../../../docs/infra_exp/06-EXPERIMENTS.md)。

正确性还须成套验证：同 token 的 trainer/rollout logprob 噪声、策略版本和同步载荷、跨 rank 梯度、MoE 路由、checkpoint 恢复、shard→adapter，以及故意错 token/mask/权重/adapter 的负对照。小模型已经覆盖独立梯度和正常恢复，但不含 35B、GDN、LoRA 或 rollout 引擎；不能用它替代剩余的真实模型检查。

## 3. 首批授权与费用账

- B03 `training_20260906a` 小模型正确性批次：CPU 前置后，同容器 2×B200 / 8 CPU / 32 GiB、最多 20 分钟，预留 $7；累计保守预留 **$101**。只用固定合成 token 和小模型，范围与阈值见下，不是大模型训练或测速。

- B13 首批 `hardware_20260906a`：CPU 前置、同一容器 2×B200 / 8 CPU / 32 GiB、GPU 容器最多 20 分钟，预留 $7，累计保守预留 $93；命令、负载、判据与证据见 [B13 REPORT](../B13/REPORT.md)。上云前另用目标 CPU 核对 torchrun 的进程关闭行为（$1），累计保守预留 **$94**。这些仍不是实际账单。

- `engine_sources_20260906a`：4 CPU / 16 GiB、脚本 5 分钟，无 GPU，补齐已确认的安装版 engine/config/checkpoint 方法源码，预留 $1；累计保守预留 $86。源码只用于设计最小梯度/正常恢复实验，不宣称已执行恢复。

- 新前置 `controls_cpu_20260906a`：4 CPU / 16 GiB、脚本 10 分钟，无 GPU，另预留 $1；连同路由源码只读批次累计保守预留 $85。检查 OPD 真正执行的零 mask 前后向与计数，不是质量或性能验收。

- 当前批组上限：$300；最多同时 6 张 B200。用户已批准本方案，不重复要求同范围授权。
- 本报告登记并保守预留每批费用，接近总限额时不发起新任务。没有拿到账单时写“估计”，不能写成实付。
- 官方基础价核验于 2026-09-05：B200 $0.001736/秒；CPU $0.0000131/物理核秒；内存 $0.00000222/GiB秒。Volume、超配和地区费用另算，见 [Modal 价格](https://modal.com/pricing)。
- a 批及缓存路径诊断只用了 CPU，保守预留 $1，未取得账单。
- b 批约 4 分钟双卡核函数检查及 CPU 前置，保守预留 $2，未取得账单；没有训练更新。
- c 批 `b03_health_20260905c`：CPU 前置 8 CPU/32 GiB、30 分钟；缓存准备 4 CPU/16 GiB、30 分钟。通过后只起一组 2×B200/32 CPU/256 GiB；训练子进程 60 分钟，容器含门禁/准备总上限 100 分钟。预留 $35，不使用指定 region 或非抢占溢价。随后 CPU 导出/匿名汇总另预留 $2。独立 B04 CPU 微测试预留 $1，以上合计预留 $41，均计入 $300 批组上限。
- 这是首批健康复验，不是三臂性能比较。后续资源独立预注册；不超范围启动 candidate、发布或主动故障注入。

## 4. 第一批输入、接线与判据

- 当前入口：`scripts/v16_pipeline.sh --profile smoke --gate-mode observe --run-id b03_health_20260905c --rl-input-run b02_20260905a rl-train`。
- 起点：CPU 校验 B02 的 merge PASS 和确切 run 身份，逐文件 SHA256 绑定已有合并 SFT 模型；只读复用，不把 B02 账本复制成新一轮 SFT PASS。
- 新 RL/checkpoint/adapter 全部写独立 run-id。新 `rl_input.json` 明确标注 `external_sft_for_rl_diagnostic`；外部输入参数不能用于 all/train-all/candidate。
- 数据：原有 RL parquet 及切分，记录 SHA 并复核三桶隔离；不重建、不做语义清洗。
- 源码：固定 Git 底座 + 不可变 overlay SHA + orchestrator SHA + 镜像；本轮含未提交修改，不能宣称结果来自纯 Git 提交。跨重复不换源码。
- 参数：沿用 smoke 的 sync、均匀采样、2 次更新、每题 4 条 rollout、每步 2 题、12,288 响应预算和现行 1.0/1.0/-1 采样。没有缩短回答预算、改奖励或切换分池。
- CPU：相关测试通过、冻结输入可读、隔离通过、真实 Hydra 配置合成通过；保存当前安装的 optimizer 方法源码，不能靠旧钩子名称猜更新。
- GPU：B200 架构、FlashAttention 反向、FLA 门禁；保存 UUID/PCI/拓扑信息和逐秒 GPU 采样。
- 真实 RL：模型 token 与原始生成相等，模板/工具 mask=0，模型 token 有完整有限 logprob；不满足就保存 artifact 并阻止进入训练。观察器缺字段不能被默认为正常。
- 验收：2 次真实 optimizer 调用、有限且非零梯度、两步权重同步、step-2 checkpoint；16 个完整 artifact 的 tensor 接线通过。CPU 导出 adapter 并核验 shard 身份。
- 模型重复、未闭合、工具错误、截断仍记 WARN；不杀掉健康运行，不把 WARN 改成 PASS。
- 这批不能证明跨引擎概率完全一致、全套跨 rank 梯度正确、能力改善或加速比；仍留 B03 后续验收。

## 5. 执行与证据

Mac 已核对 Modal 列表：所有可见旧 App 都为 stopped、tasks=0。显式复用已停止的 T1-1 d 批 full 缓存，保存原来源，并对其物理目录使用单写者占位。不复制大缓存占用 GPU，不把缓存复用写成性能收益。

```bash
modal run --detach modal_app/stack_probe.py --steps pipeline \
  --pipeline-stage rl-train --pipeline-profile smoke --pipeline-gate-mode observe \
  --pipeline-run-id b03_health_20260905c --pipeline-rl-input-run b02_20260905a \
  --pipeline-cache-from _audit/mainline/T1-1/t11_gen_20260905d/full/cpu \
  --pipeline-timeout 3600
```

CPU 不通过则不申请 GPU。训练只在本容器的进程组中运行，超时只回收本组；监视器随正常或异常退出回收，结束后核对 App 状态。

- 运行/前置：`/vol/_audit/v16/runs/b03_health_20260905c/`
- checkpoint：`/vol/checkpoints/grpo/v16_smoke_b03_health_20260905c/`
- 匿名报告：本目录；完整 prompt/token 只留 Volume，不回传原文。
- 本地汇总：`_audit/stack_probe/summary_*.json`

### a 批 CPU 结果与路径修复

- App `ap-CEabO93MJ5L0AzHUDA9gTb`；CPU 测试 142 passed、1 skipped（旧 v15 不适用项），测试 395.45 秒，完整前置 533.1 秒。
- RL train 825、val 207，三桶隔离通过。合并模型两片权重合计约 69.3 GB，CPU 已完整计算 SHA256；模型、tokenizer、RL 数据与切分身份见云端 `rl_input.json`、`preflight.json`。
- Git 底座 `61e125a6aa05d6d75641aeaa7adf4af5d8b3ac90`；overlay `69fa5ef105f1bfae71c3bf03187aa0ff4816f92dda79695b351c8a3faab27bcd`；镜像 `im-e4SA6QgSr31ESoRRdbRJ32`。这不是 b 批源码身份。
- 真实 Hydra 配置：temperature=1、top_p=1、top_k=-1、calculate_log_probs=true；rollout IS/RS 均未开启、bypass=false。这里只登记实值，不表示已验证跨引擎概率一致。
- GPU 分配前，缓存准备报“只能显式复用 Volume 审计目录中的完成缓存”。CPU 诊断证实 `/vol` 是 Modal 的软链接：代码只解析了待检查路径，没有解析允许目录，导致合法路径被误拒绝。
- 修复把两侧都解析成物理路径，仍拒绝绝对路径、`..` 和内部软链接越界；缓存单写者键也按物理路径计算，防止软链接别名绕开锁。本机相关测试 28 passed，含允许挂载软链接和禁止越界的正反例。
- a 批只生成配置与前置记录，没有 optimizer update 或 checkpoint。全部证据保留。改代码后换 b 批，不把两份源码拼成一次训练。
- 本地证据：`_audit/stack_probe/summary_2026-09-05_202406_pipeline_ffcb1c25.json`、`summary_2026-09-05_202659_exec_28e082cb.json`。

### b 批结果与入口修复

本地 155 passed、11 skipped（`local_preflight_20260905b.xml`）；Modal CPU 补验 152 passed、1 skipped，131.78 秒。冻结输入检查和缓存准备均通过。App `ap-H9MykJ5l2ENHpE6uUA5tH2`；overlay `f1547fbf0941a541bda4557df7cc7fa0a4c55de0e513f3c9f96938290fff3adb`；镜像 `im-8eWvohlXSFmspG2WhG6Lsw`。

GPU 看见两张 B200，FlashAttention 反向通过；FLA chunk 前向相对误差 0.00606，五组梯度最大相对误差 0.0067，解码核前向误差 0.00455。核函数在 GPU0 执行，GPU1 只确认架构，不能写成逐卡算子对拍。该机器 PCI 字段不可读、拓扑矩阵查询失败：保留 UUID/NVLink 信息，但不能作为完整性能拓扑指纹。

训练命令在 1.6 秒内因前置检查退出：`rl_input --model-only` 导入模块时输出两条提示，Shell 把提示和模型路径一起捕获成文件名。权重并未缺失；没有 rollout、optimizer update、checkpoint 或 adapter。容器已退出，证据保留在 `_audit/stack_probe/summary_2026-09-05_204450_pipeline_0da52208.json`。

修复：CLI 提示移到 stderr，stdout 只给路径或 JSON；新增 runbook `--check-inputs`，与真实训练共用 `rl_inputs` 检查模型及 train/val。CPU 前置必须实跑此模式，不能只有 Hydra 配置合成。两个新 CLI 测试修前确实因提示污染失败；新 Shell 正例及缺 train/val 负例补齐后，相关 43 项通过。测试夹具只有四字节权重和文件存在标记，不是真训练数据，也不冒充格式验收。

### c 批真实更新与产物验收

训练 App `ap-JaOrt9faomFpj7pTxUbs9E`；导出 App `ap-oPHwSxAPeSaaz9A9YiF3xY`。两者使用同一代码：Git 底座仍为 `61e125a6aa05d6d75641aeaa7adf4af5d8b3ac90`，overlay 为 `18e830a9111170ba62ca3a006aca10b1f3f2de9e9dadd9e2fe6cf6567ab3a648`，orchestrator 为 `19a0a90e3b28734f466eb8e1b70d89102166674e62b50e88e8eeacb24bcd7c09`，训练镜像为 `im-2r4joW8RvBbdi2mDHomUQz`。未提交工作树不等于纯 Git SHA 运行。

| 验收对象 | 实测结果 |
|---|---|
| 真正的 optimizer 状态 | 两个 rank 各 700 个参数状态，全部 `step=2`，数值有限；不是只数外层方法调用 |
| 梯度 | 两步 norm 为 0.4055548310、0.4165892601，有限且非零 |
| 权重同步 | 两步都有实际更新调用耗时 5.6941s、5.9274s；完整同步载荷/跨引擎概率对拍仍待后续 |
| 训练信号 | 16/16 artifact 通过；12,180 个模型 token 有有限 logprob，82 个补入模板 token 的 mask=0；原始采样与训练 token 相等 |
| 生成结束 | 82 次生成均为 `stop`；0 条 token 长度截断，3 条轮数耗尽。最长 response 为 2,981，原预算仍为 12,288 |
| 导出 | 从本轮两个 rank 的 actor shard 重建 350 组 LoRA A/B，共 700 个 tensor；全部有限，350 个 B 非零 |
| 身份 | actor 路径、两个 shard 哈希、adapter 权重及配置哈希均与 export manifest 相等 |
| GPU 采样 | 1,120 行、两张卡的 UUID；PCI/拓扑矩阵不可读，不能冒充完整拓扑指纹 |

adapter SHA256：`f0c284328b093757c7230a2dd414efa926b4f04d54d9bedd0c15985502b67b24`。GPU UUID 为 `GPU-28b12f01-c66f-5c66-676c-785f95625230` 与 `GPU-fbd9a961-16f3-1628-d532-1a6bb24b7215`；驱动 580.95.05，区域 Seattle。核函数在 GPU0 验过，GPU1 只确认架构，不能写成逐卡算子全验。

训练子进程 562.1 秒；两个 step 的计时为 158.3696s、35.9184s。只有两步，包含冷/热差异和不同 on-policy 生成工作量，不计算加速比，也不与 B02 最快点比较。没有跑新一轮 SFT、Exam、RL eval 或 OPD，不能把已有 B02 上游输入包装成新的全链通过。

原始账本保留 `pipeline_ok=true / all_passed=false`；RL WARN，adapter PASS。导出实际返回 0，用时 28.9 秒；外层 CLI 因整轮旧 WARN 返回 1，不等于导出失败。两个 rank 的 optimizer 状态和 adapter 内容由另一个 CPU 只读检查完成，不用外层退出码猜结果。

证据：`_audit/stack_probe/summary_2026-09-05_210946_pipeline_14e2422c.json`、本目录 `c_adapter_summary.json`、`c_clipping_summary.json`、`_audit/stack_probe/summary_2026-09-05_212237_exec_3c7a5d3e.json`。CPU 完整终态为 `/vol/_audit/infra/B03/c_health_summary.json`；此派生审计有自己的代码身份，不会混入原训练源码。

### 截断指标修正：登记与结果

c 批原始轨迹与 verl 的 `response_length/clip_ratio` 不一致。CPU 最小复现已确认：实际长度同为 `[2,4]`，只把 padding 宽度从 4 改成 12，指标就从 0.5 变成 0。下一步只修本仓库 `rl_run_gate` 的判读，不改 verl 源码、不改 c 批日志或旧账本。

判据：从本轮完整 token artifact 读取实际结束原因、裁掉的模型 token 和轨迹终止原因；分开记录 token、轮数、观测截断。固定 sync 运行的覆盖数从本轮已保存的启动配置派生，不能凭缺文件报零截断。padding 指标只保留为原始读数，不能决定质量灯。真 token 截断、轮数或观测用尽仍是 WARN；坏 token/mask/logprob 仍是硬失败。缺证据不能报 PASS；异步覆盖口径尚未接通时明确记未测，不套用 sync 的样本数。

先在 Mac 用正常、真截断、轮数、缺轨迹、坏 mask 的正负例检查；再用 Modal CPU 只读 c 批的 16 个 artifact，结果另存 `/vol/_audit/infra/B03/c_gate_reaudit.json`。不启动新的 GPU，不覆盖原 `rl_run_gate.json`。

结果：新增门禁回归在旧实现上 13 failed / 3 passed，失败在真实判读行为；修改后相关 27 项在 Mac 与 Modal CPU 都通过。新口径读到 0 条 token 截断、3 条轮数耗尽，整体仍为 WARN。缺完整轨迹或未接通的异步覆盖不会报零截断 PASS；坏 mask、缺原始结束原因或预算不符保持 FATAL。原门禁文件 SHA256 前后不变。完整记录为 `gate_regression_before.xml`、`gate_regression_after.xml` 和 `_audit/stack_probe/summary_2026-09-05_214355_exec_948c7901.json`。上游函数的独立 CPU 复现在 [DRAFT](../../../docs/upstream/DRAFT-verl-dynamic-padding-clip-ratio/1-finding.md)；未提交 issue/PR。

本次指标误读不能推翻 B02 的真实 4/16 长度截断：B02 有达到 12,288 的原始轨迹，两件事必须分开。

### c 批源码保全

代码归档为 `/vol/_audit/infra/B03/source_c_18e830a9.tar.gz`，605 个文件、2,618,342 字节，只含代码/配置/测试/文档，不含模型、数据或 checkpoint。解读 tar 后的 overlay SHA 与训练完全相等；归档文件 SHA256 为 `0b268f732831c3606fa6324e041c42ea2c6053f8447769aa6329569c743be845`，旁边的 `.json` 保存清单。

首次归档回读错误地用字符串顺序代替 `source_digest` 的路径段顺序，摘要误报不同，未写成功清单。修正后先在 Mac 对完整 c 副本做 tar 回读通过，再在 CPU 校验原云端归档通过；没有覆盖或删除归档内容。证据为 `c_archive_before_summary.json`、`c_archive_summary.json`。后续复核应读归档或冻结镜像，不依赖 Mac 临时目录。

### c 批后的工作

首批健康已过，下一批重点是同 token 的 trainer/rollout logprob 与 MoE 路由、真实同步载荷、跨 rank 梯度、checkpoint 恢复和故意错身份的负对照；先做局部接口与数值检查，再登记 B200 批次。测量使用新冻结源码，修正后的门禁不得与 c 批原代码混称同一次运行。完成这些后，至少三个独立可比重复才能形成 B03 性能 reference。

本轮收尾的 Mac 相关回归为 164 passed、12 skipped，证据 `local_final_20260905.xml`。跳过中 10 项缺目标依赖、1 项是整个 Torch 数值模块（8 项已在 B04 CPU 运行）、1 项是 v15 不适用的旧契约测试；不把跳过算通过。新增门禁另在 Modal CPU 27/27 通过，SFT CPU 为独立的 40/40，不能相加冒充同一套全库测试。固定 runbook 的 all dry-run、Shell 语法、`git diff --check` 和本次 101 个文档本地链接检查通过；dry-run 没有执行建库、训练或修改数据。

## 6. 成套身份检查：CPU 前置

用户已批准继续。先做 `identity_cpu_20260905a`：4 CPU / 16 GiB，最多 10 分钟，无 GPU，额外预留 $1，批组合计预留 $42。只读取安装的 verl/vLLM/Transformers 源码、本轮日志与匿名元数据，不生成轨迹、不改数据或模型。把涉及 V1 old-logprob、FSDP2 更新/同步、checkpoint 和路由的源码及 SHA 存到独立审计目录，供 Mac 做精确接口测试；不下载权重。

本地已发现旧探针只数外层 `optimizer_step`，不能分辨内层跳步；旧 DDP 探针只比范数，不能直接用来判断 FSDP2 不同 shard 是否正确。先核对当前实际实现，再做只观察、不改返回值的探针。CPU 前置的完成条件是：源码、版本、原始 c 批的训推偏差读数与可用策略版本字段有明确记录，缺项写缺失；新接口先通过本地与 CPU 正负例，再登记下一批 B200。

CPU 读取已通过：verl 0.9.0、vLLM 0.28.0、Transformers 5.10.4；29 个安装源码文件共 895,667 字节。归档 SHA256 为 `b926d9a550963f1cffc9660eca38fa4b53b5e681354e4e182aabec4145684550`，Mac 回读相等。c 批两个窗口的 k3 KL 为 0.00272765 / 0.00226589，策略陈旧度均值为 0；这些平均值不能替代逐 token 记录或噪声对照。82 次生成均未保存逐轮策略版本。证据 `identity_cpu_result.json` 和 `_audit/stack_probe/summary_2026-09-05_224619_exec_fe368a24.json`。安装版 `engine_workers.py` 只给 Megatron/VeOmni 启用路由重放，FSDP2 不继承此开关；不能宣称本项目已经做了路由重放。

### 身份观察器的 CPU 正负例

登记 `identity_tests_20260905a`：4 CPU / 16 GiB，最多 20 分钟，无 GPU，额外预留 $1，批组累计预留 $43。先在 Mac 做轻量接线与语法检查，再在完整栈跑真实 optimizer 方法、观察字段转发与 token logprob 对照。正例要求内层 optimizer 返回计数为 1、权重改变；非有限梯度跳步和异常的计数必须为 0。重复 logprob 后必须逐项恢复第一次训练使用的概率和 entropy，不能让诊断改变训练目标。不同字节但同范数、错 mask、非有限概率必须能被识别。保留本地跳过的事实。

观察开关默认关闭，当前只允许 sync/FSDP2。新观察器读本地 FSDP 分片、同步产生的完整 LoRA 载荷、同 token 训推概率和 trainer 重复噪声；不同分片不做伪“跨 rank 相等”断言。它有额外 CPU 拷贝和一次概率重算，后续正式测速必须关闭。同步接收端实际装载、跨 rank 梯度独立参考与恢复仍需继续验证；本批不提前宣称 B03 通过。

a 次 CPU 为 28 passed / 2 failed / 0 skipped。两个失败都在测试末尾把 TensorDict 当字典迭代，正确写法是遍历 `.keys()`；不是训练或概率恢复失败，也不算全通过。修正后登记 b 次复跑，仍为 4 CPU / 16 GiB、20 分钟、无 GPU，另预留 $1，批组累计 $44。增加安装版 V1 的真实 jagged tensor（变长张量）路径测试、接收端的真实 `_update_weights` 与 `add_lora` 返回检查，以及版本通过 IPC 接口的测试。失败原件保留在 `identity_tests_20260905a` 和 `_audit/stack_probe/summary_2026-09-05_225406_exec_e8e16bad.json`。

接收端观察扩展继承官方 vLLM 扩展，只记录实际收到和送入 `add_lora` 的字节、返回值及 slot。没有重写加载器，没有将“slot 存在”单独当作正确证据。训练批次在此 CPU 复跑通过前不会启动。

b 次 CPU 已通过 34/34、0 skipped，耗时 91.66 秒；包括安装版真实 V1 方法的变长张量、真实 FSDP engine 的内层更新/NaN 跳步、真实 LoRA 接收方法、延迟导入和转发。可选 Megatron/VeOmni 等未安装的警告保留；当前 FSDP2 测试没有因此跳过。证据 `_audit/stack_probe/summary_2026-09-05_225952_exec_113153c6.json`。Mac 调度签名测试仍写旧实参列表的问题已同步改正，该组 31/31 通过。另有 13 项纯离线汇总正负例通过，不提前算成 B200 证据。

## 7. d 批：真实双卡身份诊断

登记 `b03_identity_20260905d`。继续用 B02 的同一 merge 和冻结数据，2 步 sync smoke/observe，原预算、reward、uniform sampling 均不变。不是完整训练，也不是性能比较。

- CPU 前置和缓存准备分别最多 30 分钟，前置使用 8 CPU / 32 GiB。完整测试、输入 SHA、隔离和实际 Hydra 合成通过才分配 GPU。
- GPU 为 2×B200 / 32 CPU / 256 GiB，训练子进程最多 60 分钟，容器含门禁和准备最多 100 分钟。包括后续 CPU 匿名汇总，本批另预留 $37，累计保守预留 $81，仍在 $300 内。没有增加同时 6 卡的限额。
- 缓存只复用已退出的 c 批 `_audit/v16/runs/b03_health_20260905c/preparation/rl-train`；继续使用物理路径单写者锁。每次启动前只读检查 App。
- 固定源码由本次 Modal 上传冻结，保存 Git 底座、overlay、调度器和镜像身份；不把未提交 overlay 说成纯 Git SHA。
- 预期 4 份 optimizer 记录（2 rank × 2 次）、6 份同步源和 6 份接收记录（版本 0/1/2 × 2），两份逐 token logprob 对照。完整 LoRA 载荷在同版本的两端逐字节相同；版本随真实更新变化；内层 optimizer 完成、状态 step 前进、有限非零梯度与权重变化必须同时成立。
- 每步 trainer 重算一次同策略 logprob，保存逐 token 差，恢复第一次的训练概率。先报告噪声与跨引擎差，不跑完再发明可接受阈值。缺观察、错版本、错载荷、NaN、零真实更新为健康失败；回答质量仍只记 WARN。
- 本批不覆盖 MoE 两侧路由、独立梯度参考或恢复，仍不能关闭整个 B03。不能拿带重复前向和拷贝的 step time 报加速。

执行命令（只在上述本地检查通过后调用）：

```bash
PYTHONPATH=. modal run --detach modal_app/stack_probe.py --steps pipeline \
  --pipeline-stage rl-train --pipeline-profile smoke --pipeline-gate-mode observe \
  --pipeline-run-id b03_identity_20260905d --pipeline-rl-input-run b02_20260905a \
  --pipeline-identity-probe --pipeline-timeout 3600 \
  --pipeline-cache-from _audit/v16/runs/b03_health_20260905c/preparation/rl-train
```

CPU 前置已通过 204 passed、1 skipped（v15 不适用的旧契约测试），`health_ok=true`，输入/隔离/Hydra 全部通过；见本目录 `d_preflight.json`。双卡运行的 overlay 为 `ccee2f6c7499566f4fac546152b94fec28fc5caf370130f42413c3df04574e80`，镜像为 `im-AXS0RVbwQ96zQ6QHp38tWb`，App 为 `ap-vqDXa4DJLuHn2YupZugOWl`。本地已保全 615 个源码文件到 `source_d_ccee2f6c.tar.gz`，回读 overlay 相等，归档 SHA256 为 `7e453c950ec4c034d4d67fa40a9389398d3b0c96b754d92cfb37a04a74b18822`。后续本地编辑不属于这批训练。

收尾 CPU 预注册：d 批退出后用 4 CPU / 16 GiB、最多 20 分钟执行 `identity_collect.py`，费用包含在本批 $37 中，不新增 GPU。先过回读器的正负例，再逐条匹配原始 rollout 与 trainer 保存的 prompt、response、mask 和采样概率（按保存 dtype 转换后相等），复算逐 token 概率摘要，检查每次生成的策略版本，以及两个 rank 的 optimizer checkpoint 状态。结果只写 `/vol/_audit/infra/B03/identity_readback_20260905d/`，不改原训练账本。原始 token 不下载到 Mac；缺项就是未验/失败，不能用计数猜正常。

### d 批结果

实际 RL 子进程退出 0，耗时 702.8 秒，完成两次更新。`identity_gate.health_ok=true`；原 runbook 保留 `pipeline_ok=true / all_passed=false`，长度/轮数 WARN 没有提前中断。外层 CLI 的红灯不是训练异常。

| 对象 | 回读结果 |
|---|---|
| optimizer | 4 条内层成功调用记录；两卡各 700 个状态，checkpoint 全为 step=2 且有限 |
| 同步 | 版本 0/1/2 × 两卡的 6 份发送、6 份接收记录齐全；完整 LoRA 载荷逐字节相等，实际 `add_lora` 接收通过 |
| 训练输入 | 16/16 条原始 rollout 与 trainer 保存的 prompt、response、mask 和采样概率相等；覆盖 21,325 个模型 token |
| 逐轮版本 | 63 次生成完整登记：版本 0 为 25 次，版本 1 为 38 次；与对应更新窗口相符 |
| 模板边界 | 64 个补入模板 token 的 mask=0，未冒充模型采样 token |
| 生成质量告警 | 1/16 条达到 12,288 上限，另有 2 条轮数耗尽；62 次 stop、1 次 length |
| GPU | 1,400 行采样，双 UUID 齐；本次为 FIN-BM，PCI/拓扑矩阵仍不可读 |

一次小样本从 c 批 0 条截断变为 d 批 1 条，不能说退化已经根治，也不能把 6.25% 与 B02 的 25% 当成改善结论。现有数据和响应预算未改；质量继续观察。带身份拷贝和重复前向的计时不进入性能表。

两卡每次变化的本地张量数分别为 700、660，并非 40 个参数没有同步：只读回查确认 rank 1 的 40 个不变项全是 `mlp.shared_expert_gate.lora_B` 的空分片，完整形状 `[1,32]`、本地形状 `[0,32]`、0 字节。两版本的原始记录为 `optimizer-5498-b032d30ddd4f49bd89a0444c6adc7383.json` 和 `optimizer-5498-216758b5687b424cb8e2c294893b34ba.json`。这说明不能要求不同 FSDP 分片的变化计数相等；仍不代替梯度对独立参考的数值检查。

逐 token 的 logprob 差（自然对数单位，不是概率百分点）如下。两侧实际读取同一载荷、token 与 mask，trainer 的同策略重复重算为零差；跨引擎差还不能被叫作“已接受的浮点噪声”。

| 窗口 | 模型 token | trainer/vLLM 平均绝对差 | P99 绝对差 | 最大绝对差 | trainer 重算最大差 |
|---|---:|---:|---:|---:|---:|
| step 1 | 16,533 | 0.01818480 | 0.19427141 | 0.74348736 | 0 |
| step 2 | 4,792 | 0.03169360 | 0.25378725 | 1.46589327 | 0 |

GPU 结果为 `_audit/stack_probe/summary_2026-09-05_232418_pipeline_e49a80c0.json`。独立 CPU 回读前的 27/27 正负例通过、0 skipped；随后 10 项内容检查全部通过，JSON 摘要与原始 `.pt` 复算相等。结果为本目录 `d_identity_summary.json`、`_audit/stack_probe/summary_2026-09-05_232742_exec_2b7d9ae0.json` 和 `/vol/_audit/infra/B03/identity_readback_20260905d/`。派生回读代码 overlay 为 `57ae1bf34e5c0eb706a3ef5d9082db7c67329b5c8e9a2469285c44aba96e6c6c`，不是训练 overlay；未改原训练结果。CPU 只读 DTensor checkpoint 时有未初始化进程组的 DeviceMesh 警告；没有在此进程做分布式计算或恢复，不能把读取成功说成恢复通过。

本地归档及清单已上传到 `/vol/_audit/infra/B03/source_d_ccee2f6c.tar.gz`；从 Volume 流式回读的 SHA256 仍为 `7e453c950ec4c034d4d67fa40a9389398d3b0c96b754d92cfb37a04a74b18822`。本机当前相关回归 191 passed、16 skipped，见 `identity_local_final_20260905.xml`；缺 Torch/verl/vLLM 的模块由对应 CPU 批次补验，唯一旧 v15 不适用项仍保留跳过，不能把不同测试集合相加成“全库通过”。收尾的 Shell 语法、`git diff --check`、12 份本次文档的 144 个本地链接检查通过。本段源码和报告现已进入 `main`；实验实际运行身份仍以上文 Git 底座、overlay、调度器和镜像为准，不能倒写成由收尾提交直接运行。

GPU App、OPD CPU App、d 批回读 App 均已确认 stopped、tasks=0。含并行 B06 的 CPU 任务，本批组累计保守预留 $83；这是资源预算账，不是实际账单。

### 后续实验线索和队首

持续执行范围已确认到 B13；沿用 $300 / 最多同时 6 张 B200，不含 candidate、发布或主动故障注入。同一问题记录每轮假设、改动和结果；超过 5 轮仍未解或遇到重大决策时暂停询问。B03 增补逐层 MoE 路由一致性；B11 增补 continuous batching 的下发、实际位置及算子对拍，不预设差异原因。

`routing_sources_20260906a`：4 CPU / 16 GiB、脚本最多 10 分钟，无 GPU，预留 $1，累计保守预留 $84。只读取安装版 verl 的 Qwen3.5 前向替换、vLLM 路由/调度和 GDN 实现，保存源码 SHA 与归档到本实验独立目录。关键实际调用文件缺失则失败；可选路径缺失明确列出，不模糊匹配。Mac 先做 AST 编译检查；不读取模型或训练文本。

已取回 41 个安装版文件，归档 SHA256 `e4c4a02eb0f47d59ffb109db059c3d64507f99057cd103dafc66bbea7ce9a3ad`，本机 `routing_installed_sources.tar.gz` 与 Volume 相等；汇总 `_audit/stack_probe/summary_2026-09-06_094344_exec_acc3a482.json`。engine/config/checkpoint 的另 16 个文件也已取回，`engine_installed_sources.tar.gz` SHA256 `19f45d7b9e0f4450e31918e70934fae181bb7c07ed7cabe2881b564144daf26f`；汇总 `summary_2026-09-06_095141_exec_66b32fb8.json`。两次都没有缺失登记文件，也没有执行训练。

实际 B02 merge 的 config 已从 Volume 取回，SHA256 为 `2daa48e1589e8a1d13938a0668388b99107ab323d13d44480edc1ba1ff81c608`，与 d 批绑定的输入完全相等。模型是 `qwen3_5_moe_text / Qwen3_5MoeForCausalLM`，不会命中只接受多模态类型名的 verl Qwen3.5 特化分支。此前阅读中的相反推断已撤回。后续必须采真实 root/GDN 调用参数，再判断 packed 边界是否正确；d 每个 microbatch 只有一条，不能拿多条拼接的问题直接解释 d 概率差。

- 先在同一输入上分开检查两引擎的 mask/位置、计算精度、GDN 路径及 MoE 路由，再给跨引擎概率差解释；不根据这两次结果临时放宽阈值。
- 补独立梯度参考和正常 checkpoint 恢复；不主动注入故障。完成后冻结关闭身份诊断的测速源码，跑至少三个可比重复。
- B04 的 B200 DP/TP 正确性、B06 的真实 mask/多次更新可独立准备；性能采用决定等 B03 尺子。
- 启动日志显示 `OMP_NUM_THREADS=32` 的线程争用警告、部分 MoE shape 没有调优桶、推理中仍有 JIT。它们是 B10/启动成本的具体线索，不是已确认瓶颈或加速结果。本次没有据此临时改参数。

### 官方 engine 小模型梯度与正常恢复计划

本机实现与独立规格/代码审查已通过，定向组合 **136 passed / 15 skipped**；跳过均因 Mac 无 PyTorch，必须由同源码目标 CPU 补齐。独立额外正负对照 9/9 符合预期。修复了状态“数值相等冒充字节相等”、失败先抛异常导致原始观测丢失、汇总只信成功标志三类测试问题；全部先有失败反例，再修实现。汇总现在回查原始数值、两臂/步骤/token 身份、实际 root 与路由、跨 rank checkpoint 哈希和进程终态，不以 `health_ok` 字样放行。

启动命令：`modal run --detach modal_app/stack_probe.py --steps infra_probe --infra-probe b03_training --infra-run-id training_20260906a --infra-mode gpu`。父入口先运行目标 CPU 全部测试和公式/接口检查，零跳过通过才申请已经登记的 2×B200；最终源码/镜像以本次 summary 为准。首个实际 root 的输入和位置必须等于登记 row，路由 top-2 必须是合法不同专家、有限权重且和为 1（误差最多 1e-5）；这不等于已验证 vLLM 路由一致性。

`training_20260906a` 先隔离一个可解释问题：官方 FSDP2 根前向是否把两卡不同 token 数的梯度合对，保存后是否能原样接下一步。模型为两层 full-attention 的小型 `qwen3_5_moe_text / Qwen3_5MoeForCausalLM`：vocab 128、hidden 64、heads 2、kv heads 1、head dim 32、experts 4/top-k 2、expert/shared intermediate 64，保留真实 MoE。它不包含 GDN，也不覆盖 LoRA、同一 microbatch 多序列 packing、35B 训推差或性能收益。

两臂只改变 `use_remove_padding=False/True`。每 rank 两条长度/监督数不同的合成序列，每 microbatch 一条；全局四条的末 token mask 都是 0。正式 `EngineRegistry` 创建 `FSDPEngineWithLMHead`，走根 `forward_backward_batch`；参考是未分片的原生模型，CPU 逐条前向，用全局 token 均值独立构造交叉熵和 backward，不复用 engine 的 logprob/loss 帮助函数。回调必须读到正确的全局 token 分母。

全部 FP32、TF32 关闭；dropout、aux loss、编译、融合、梯度 checkpoint 和 offload 关闭。AdamW 为 lr=0.001、betas=(0.9,0.999)、eps=1e-8、weight_decay=0、foreach/fused=False；clip_grad=1e6 并断言实际未裁剪，constant scheduler、两次更新。先比裁剪前逐参数完整梯度，不能用 Adam 首步结果掩盖梯度倍数错误。

- loss/logprob：atol=1e-5、rtol=1e-5。
- 梯度：全局 relative L2≤1e-4；每 tensor 最大绝对误差≤1e-6+1e-4×参考最大绝对值。全部有限且梯度/参数真实变化非零。
- 参考 AdamW 更新后参数：atol=2e-6、rtol=1e-4。
- 两臂各在 step 1 正常保存 model/optimizer/extra，原 engine 接 step 2；新 engine 从相同初始文件启动并正常 load，比较模型、optimizer moments/step、scheduler、Python/NumPy/CPU/本 rank CUDA RNG 逐字节相等，再跑相同下一批，终态与原 engine 逐字节相等。失败保存差异，不临时降低标准。

记录真实 root callable/源码哈希、输入 kwargs、`pass_packed_cu_seqlens`、各层 router IDs/weights；GDN 明记零次调用、未测。CPU 前置使用实际安装接口、配置、TensorDict、mask 和原生小模型数学；GPU 只在同源码 CPU 前置通过后分配。每 rank 原始 state 和逐参数误差保存在本批独立目录；子进程 480 秒、进程组 90 秒、torchrun 收尾等待 105 秒，正常回收并保留失败证据。

a 批实际 CPU 为 **149 passed / 2 failed / 0 skipped**。错误是探针漏传 `get_non_tensor_data` 的必填 `default`，不是梯度或恢复失败；GPU 未申请。全扫描还发现 GPU loss 回调的两处同类调用，一并补测。App `ap-8hciJQIavahhpsxxKeY8Hf` 已停止，镜像 `im-LZyNIeg202zyTFKoHiBFuO`，overlay `50d1de163e7ae129df24dab7db5559d899a117bfb17476edad4d3de3a371ad7e`；证据 `_audit/stack_probe/summary_2026-09-06_103545_infra_probe_4a783523.json`。失败目录保留，不覆盖重跑。

为核对实际签名而非猜测，登记 `api_sources_20260906a`：4 CPU / 16 GiB、最多 5 分钟、无 GPU，补取 TensorDict API 和 vLLM monolithic 路由接口，不读取训练输入。预留 $1，累计保守预留 **$102**（不是实际账单）。接口修复通过本机回归后，用新 ID `training_20260906b` 复跑同一 CPU→2×B200 计划；另保守预留 $7，累计 **$109**，资源/阈值不变。

接口原件已取回，`api_installed_sources.tar.gz` SHA256 为 `7e417a01b49b84b36366696061c98dc7658273f33e019800a32cb26b001582cd`，6 个文件；2 个可选旧路径不存在，不能猜成已验证。真实 `get_non_tensor_data(data,key,default)` 没有默认值，4 处漏参已补；签名负测 4 failed→5 passed，组合本机 141 passed/15 skipped。CPU 读取记录 `summary_2026-09-06_104214_exec_c6d3c4a7.json`。

b 批 **CPU 156/156 通过、0 skipped**；双 B200 在首个 remove-padding=off 更新后停止。梯度、逐 token 概率、token 分母和非零梯度通过；两 rank 的首步 AdamW 状态均为 1，lr=0.001，scheduler 前进一步，裁剪未触发，参数确实改变。更新后对独立 CPU 参考的参数检查有 3 个 tensor 未通过，因此没有继续恢复或第二臂，不能关闭本项。

rank 0 的梯度 relative L2 为 `3.2700636185360354e-7`，最大绝对误差 `1.1920928955078125e-7`。更新后失败的是第 0 层 experts.down_proj、shared_expert.down_proj、q_proj，最大参数差分别约 `1.27116e-5`、`3.54671e-6`、`2.17408e-5`。下一步只读保存的逐元素梯度和 optimizer 状态，核对公式及近零梯度的数值敏感性；目前不认定根因，也不放宽参数阈值。

App `ap-yAYb3zlDtUNkIw5Wmcrlkz`，镜像 `im-TRr5dKmDbOTxMLMqv7NhBS`，overlay `7498309ab8911818546e6b2122fa3fead5ba50dcee0b3cefe3f72f80ed89d7ef`。GPU 子进程 79.7 秒、exit=1、无超时；汇总 `summary_2026-09-06_104719_infra_probe_61dcb99b.json`，本地 `training_b_rank0_post_step.json` / `training_b_rank1_post_step.json`。原始 `.pt`、root 观测及失败证据保留在 `/vol/_audit/infra/B03/training_20260906b/gpu/`。

登记只读 `optimizer_readback_20260906a`：4 CPU / 16 GiB、脚本最多 10 分钟，无 GPU，预留 $1，累计保守预留 **$110**。读取两 rank 的 initial/pre-clip/post-step 六份小模型 `.pt`，保存 SHA；逐个失败坐标读初始权重、梯度、moments、step、更新后值及实际参数组。独立首步公式 `p1=p0-lr*g/(abs(g)+eps)` 分别代入两侧实际梯度，再与实际更新对拍；全量比较跨 rank 字节。它只解释已发生的首步，不改变验收阈值、不重新训练。

CPU 回读通过，CUDA 未初始化：实际配置与登记一致；两个 rank 的梯度和更新前后参数全部逐字节相同。33 个 tensor 共 173,120 个参数中只有 **3 个坐标**不满足原参数对拍线。它们的差值被各自保存梯度代入 AdamW 公式准确预测，两侧公式与实际参数的全量最大差约 `3.9e-9`。用 GPU 保存梯度在独立 CPU AdamW 再算一步，33 个 tensor 全通过，最大差 `7.450580596923828e-9`。例如 q_proj 的两个梯度为 `4.65661e-10` 与 `2.32831e-10`，都很小，但 lr/eps=100000 会放大近零梯度差；实际参数差 `-2.1740794e-5`，公式预测 `-2.1740908e-5`。这证明了本例根因，不是一般性证明所有训推差都可接受。

报告 `optimizer_readback_20260906a.json` 包含六份原件 SHA、全部 33 tensor 读数和 3 个坐标。App `ap-qWQNmSWhTKDYTWbIpyMSdD`、镜像 `im-becjwN8cXDthEzyxXqgO0h`；汇总 `summary_2026-09-06_105650_exec_2ac38244.json`。这是派生解释，不覆盖 b 批的 FAIL。

### c 批：把两种数值检查分开

下一轮 `training_20260906c` 保留模型、两臂、工作量、所有数值阈值和正常恢复要求；修正的是比较对象：

1. 每步先把当前 engine 的完整参数复制给独立 CPU 原生模型，并逐字节核对。相同 token/mask 下独立算梯度，仍用原梯度/概率阈值；不把前一步 CPU/GPU 梯度的累计漂移混入下一步。
2. 在 optimizer 前保存实际参数组、moments/step 和 GPU 算出的完整梯度。另建独立 CPU AdamW，从同一状态、同一梯度计算一步；逐 tensor 比更新后的参数与 moments/step。参数线仍为 atol=2e-6、rtol=1e-4。不能把参考梯度差放大后的参数差误当 optimizer 实现差。
3. 同权重但各算梯度的 CPU 更新仍保存为旁证，允许显示差异，不用它替代梯度检查。保存/加载状态与恢复下一步仍要求同栈逐字节一致。

补充预先固定的细节：两条 CPU 更新都独立复制同一份实际更新前状态，不能沿用 CPU 上一步自己的 moments。step 1 状态为空；step 2 必须准确接上实际 step 1 的参数、moments、step、参数组与 scheduler，非零 moments 不能被重置。moment 的逐坐标误差线为 `8×FP32机器精度×max(|旧moment|, |本次梯度项|, |参考新moment|)`，一阶梯度项为 g、二阶为 g²；零尺度必须零差。step 和参数组逐字节相等。两个 rank 的完整参数、梯度、optimizer 和 scheduler 指纹相等，DP 组确为 {0,1}；不要求两 rank 的随机数流相等。参考计算前后还核对实际状态未被修改。新增重置 moments、错步数、同形参数错绑的负例；汇总必须读取逐坐标误差比和失败坐标数，不能只读成功标记。

先补本机可跑的规格/失败例与目标 CPU 的同状态 AdamW（含 step 2 非零 moments）检查，再启动同源码 CPU→2×B200。每 rank 保存参考起点、actual-gradient optimizer 对照及原始数据；删除不了解含义的 `max_abs_bound` 标签，参数摘要打印真正逐坐标阈值比。CPU/GPU资源与时限同前，另预留 $7，累计保守预留 **$117**。本批不修改正式训练 optimizer、原 b 结果或 35B 训推概率标准。

本机两探针测试为 129 passed / 29 skipped，后者需要目标 CPU 的 Torch/verl。规格和独立代码质量复审已通过；修复前用 8+3 个反例确认摘要会放过缺失/矛盾证据，修复后全部拒绝，独立追加 9 项规格与 7 项质量正负对照符合预期。参数超线的独立梯度旁证仍可保留，不阻止同状态 optimizer 主检查；它不能被删掉或写成已通过。固定 runbook dry-run、Shell 语法、本机供给检查、diff 空白检查通过；供给只数现成底题，没有造数或清洗。

### `training_20260906c` 结果与收尾

目标 CPU 实测 **182 passed / 0 failed / 0 skipped**，随后才申请 2×B200。两张卡、两个 rank、`remove_padding=False/True` 两臂都完成两次真实更新；每一步的独立梯度、同状态同实际梯度 AdamW、参数、moments、step、scheduler、跨 rank 完整状态均通过。两臂的正常保存、加载、RNG 恢复和恢复后的下一步也逐字节相等。GPU 子进程 29.1 秒退出，没有超时；两个 worker 都是 `health_ok=true`。

外层最初仍显示红色，唯一原因是汇总器要求 UUID 字符串以 `GPU-` 开头，而 `torch.cuda.get_device_properties().uuid` 实际返回 `023ce55b-...` 这种裸 UUID。卡名、sm_100、两张不同 UUID 和所有数值证据都存在。新增正反例后，校验器只接受标准 UUID，并同时接受 PyTorch 的裸格式和 `nvidia-smi` 的 `GPU-` 格式；没有改任何训练、阈值或原始证据。用修正后的汇总逻辑重新读取原两 rank 记录，结果为 `2 ranks / 2 arms / 2 baseline updates，ok=true`。派生记录为 `training_20260906c_readback.json`；它不覆盖原汇总中的红灯。

- 原汇总：`_audit/stack_probe/summary_2026-09-06_111617_infra_probe_a9c7ff75.json`
- App：`ap-g1Urk3BgazPEQYhUiE0Vbi`，已确认 stopped、tasks=0
- 镜像：`im-UzS6KPWORHkhUKVb2YrL5E`
- Git 底座：`61e125a6aa05d6d75641aeaa7adf4af5d8b3ac90`
- overlay：`4d084e2938af121462adcb778fa34cc7a7902fbdef05b83ba6c50922a46e2cad`
- 冻结探针源码：`a9cda552f82d0d063ceb6eddcab59f8fa5ca9a303992b56002a323ccef280e3d`

截至本次暂停，B03～B13 已启动批次的累计**保守预留上限为 $118 / $300**；这是按登记上限相加，不是实际账单。收尾时可见 Modal App 均为 stopped、tasks=0，没有遗留训练或模型服务。

这关闭的是“两层 full-attention Text MoE 小模型上的 FSDP2 梯度、AdamW 和正常恢复”子问题。下一步不要重复这批 GPU：先用 35B 同一真实 token 补齐 trainer/vLLM 的位置、精度、GDN 和逐层 MoE 路由解释，再关闭身份诊断，冻结性能源码，跑至少三个可比重复。
