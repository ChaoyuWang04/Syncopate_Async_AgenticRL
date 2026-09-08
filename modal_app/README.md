# Modal 运行入口与保留 B200 实现

> 当前机器、依赖、Volume 和安全规则见
> [docs/syncopate/05-COMPUTE.md](../docs/syncopate/05-COMPUTE.md)。
> 当前任务见 [Infra TASKS](../docs/infra_exp/01-TASKS.md)。原业务数据构建与完整学习链已停止。
> 以下是保留入口，不是执行清单。当前先真实训练画像再定位；Modal上限8卡B200，但现有双卡微探针不能冒充新的8卡真实训练入口。新入口按Q21支持/容量核验后按需实现。

B11追加测量入口为 `communication_baseline_batch.py --phase cpu|gpu|graph|direct --run-id ID [--preflight 精确CPUattempt]`：CPU在独立目录编译固定SHA的官方nccl-tests；同一派生MPI镜像的双B200对比native/PyTorch eager/Graph。实际执行状态只看B11 REPORT，不能把入口存在当基线通过。

## 保留定位探针入口（仅在真实画像触发后使用）

- [probability_batch.py](probability_batch.py)：概率与GDN定位探针，`--phase cpu|fsdp|vllm|layers|fp8|gdn|gemm_cpu|gemm|splitk_cpu|splitk|lora_cpu|lora`（新系列B04/B05已完成；`gemm_cpu/gemm`及固定M因果对照`splitk_cpu/splitk`用于B07，`lora_cpu/lora`用于B08；CPU gate绑定源码/镜像）；CPU 返回精确 attempt 路径作为 GPU 的 `--preflight`。每次投递独立目录，程序退出和必需结果内容均验证。
- 本批新增 `residual_cpu/residual`（B09，CPU→单B200）、`opd_cpu`（B10，只用CPU）、`communication_cpu/communication`（B11，CPU→双B200）；通信只核模型元数据，不读取整份模型权重。同问题后续入口为 `residual_follow_cpu/residual_follow`（B09严格重放与联合干预）及 `communication_focus_cpu/communication_focus`（B11交错尺寸与独立trace）。新入口按各REPORT的当前验证状态使用。每次函数使用独立容器，通信三次分配分别保留拓扑与全部样本。
- [gdn_fastpath_batch.py](gdn_fastpath_batch.py)：B06派生镜像，只增加固定causal-conv1d依赖；`--phase cpu|gpu`，GPU必须引用同源码/同派生镜像的CPU attempt。

- [infra_probes.py](infra_probes.py)：infra-probes/B00 长度指标CPU对照、B01公开MoE同token的CPU/单B200对照。
- [lora_probe.py](lora_probe.py)：infra-probes/B02 tiny MoE + LoRA的CPU/双B200更新对照。
- 参数/资源/判据和验证状态只认各 [实验REPORT](../docs/infra_exp/experiments/)。示例命令里的新ID必须替换，不得重写旧run。

```bash
PYTHONPATH=. modal run --detach modal_app/infra_probes.py --probe b01-cpu --run-id <new-cpu-id>
PYTHONPATH=. modal run --detach modal_app/infra_probes.py --probe b01-gpu --run-id <new-gpu-id> --cpu-run <passed-cpu-id>
PYTHONPATH=. modal run --detach modal_app/lora_probe.py --mode cpu --run-id <new-cpu-id>
PYTHONPATH=. modal run --detach modal_app/lora_probe.py --mode gpu --run-id <new-gpu-id> --cpu-run <passed-cpu-id>
```

这些入口复用冻结stack镜像，在`/opt/syncopate-current`执行实际源码快照，不同步共享bare repo或连接业务数据目录。GPU前置核对对应CPU的输入、源码和依赖身份；只在自己的`/vol/_audit/infra-probes/Bxx/<run-id>/`写日志、证据及私有缓存。诊断hook不用于测速；B01仅在自有引擎IPC内传已知观察函数。

## 保留入口

- [stack_probe.py](stack_probe.py)：保留机器/历史探针与业务管线编排。
- [scripts/v16_pipeline.sh](../scripts/v16_pipeline.sh)：保留数据、SFT、Exam、RL、OPD业务管线，当前停止排期。

## 保留入口的环境

- Volume：`syncopate-home`
- 资源：CPU、B200 或 B200:2，按实际步骤选择；独立对照可以同时起多组资源
- 镜像与依赖：[stack/pyproject.toml](stack/pyproject.toml) 和 [stack/uv.lock](stack/uv.lock)
- 容器 checkout：`/tmp/repo`
- 持久数据、模型和审计：`/vol`

精确布局和依赖版本只在 [05-COMPUTE.md](../docs/syncopate/05-COMPUTE.md) 维护。

历史验证边界（本次未重跑）：B01 上云前认证已通过；B02 已把 2×B200 真实 v16 全链机械接通。B03 的 35B 身份证据和小模型双卡梯度/正常恢复已有局部通过；B13 已取得首轮双卡通信与 BF16 GEMM 读数。B04/B06 仍在正确性前置阶段。
详细边界只看 [infra TASKS](../docs/infra_exp/01-TASKS.md) 和各 B 系列 REPORT。没有candidate或真实训练性能baseline；B11原生NCCL局部基线只覆盖其已测配置。

## 保留探针命令（按新实验准入后选用）

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

# 固定管线：按阶段自动选择 CPU / 单卡 / 双卡
modal run --detach modal_app/stack_probe.py --steps rl_cfg
modal run --detach modal_app/stack_probe.py --steps pipeline --pipeline-stage train-all \
  --pipeline-profile smoke --pipeline-gate-mode observe --pipeline-run-id <ID>
```

`pipeline` 是全链唯一云上入口，只转调 runbook；不会在 Modal 层再复制一套训练参数。
默认仍是 smoke/observe。判断时分开看：`manifest.pipeline_ok=true` 表示程序和产物链能继续；
`manifest.all_passed=false` 表示仍有 WARN。Modal 汇总会把后一种显示为红色提醒，但 observe 不会在 WARN 出现时提前杀掉训练。

B03 独立 RL 对照可显式传 `--pipeline-rl-input-run <旧 smoke run-id>`。CPU 先检查旧 merge 身份、完整权重 SHA、数据隔离、测试与 Hydra 配置，还实际执行 runbook 的 `--check-inputs ... rl-train`，通过后才申请 GPU；新产物写新 run-id，不伪造新 SFT。此参数不允许 all/train-all/candidate。具体批次、时限、费用和命令只在 [B03 REPORT](../_audit/infra/B03/REPORT.md) 登记。

`--pipeline-identity-probe` 是 B03 的额外身份诊断，只接受上述独立 smoke `rl-train`。它记录真实 optimizer 调用、LoRA 同步两端载荷、接收端加载返回、同 token 训推 logprob 与重复计算差，文件放在本轮 checkpoint 目录的 `policy_evidence/`。接收端使用继承官方加载器的观察扩展。观察器有 CPU 拷贝和重复计算开销，默认关闭，不能用其耗时当性能 baseline。具体未验证项与 CPU/B200 证据仍只看 B03 REPORT。

原始 token 与概率留在 Volume；收尾回读用 `policy_evidence.validate_batch_trace` 对照生成记录和 trainer 张量，不用 JSON 计数代替内容相等。实际上传的源码快照可用 `source_snapshot.archive_source` 归档并回读哈希；只包含源码白名单，已有归档不覆盖。归档必须匹配该次运行的 overlay，不能拿后来编辑的工作树冒名。

`--pipeline-timeout` 限制本阶段子进程秒数，`--pipeline-cache-from` 只能显式指向 Volume `_audit/` 下已经完成的缓存准备目录；复用前必须确认旧写者已退出。GPU 阶段先验 kernel，再逐秒记录可见 GPU 的利用率、显存与功耗。

独立硬件微测试使用受限入口 `--steps infra_probe --infra-probe b13_hardware --infra-run-id <ID> --infra-mode gpu`。
它先在同一镜像、同一源码的 CPU 容器实跑测试与数学对照；只有测试非空、零跳过/失败且 NCCL 接口可用，才申请一组 2×B200。只需 CPU 时传 `--infra-mode cpu`。
每个阶段在写日志前占用独立证据标记，失败重跑也必须换 run-id；GPU 运行保留逐秒采样和进程终态。工作量、时限和价格预留见 [B13 REPORT](../_audit/infra/B13/REPORT.md)。B13 不加载模型或数据。

同一受限入口的 `--infra-probe b03_training` 检验小型随机 Text MoE 在官方 FSDP2 上的梯度与正常 checkpoint 恢复。CPU 先创建本次独占的合成模型并验证真实接口，GPU 只消费其原样文件。每个探针执行自己登记的测试，不能借另一探针的通过记录。这不是业务训练的第二入口，也不替代 35B/LoRA/GDN 或性能基线；范围见 [B03 REPORT](../_audit/infra/B03/REPORT.md)。

`b03_training` 的首个完整双卡批次已经通过保存证据的重新聚合。原汇总只因 PyTorch 返回裸 UUID、旧格式检查只接受 `GPU-` 前缀而误报；修复只扩大合法 UUID 的格式识别，没有改梯度、更新、恢复或速度门槛，也没有重跑 GPU。原红色汇总保留不覆盖，修正读回见 `_audit/infra/B03/training_20260906c_readback.json`。

T1-1 的固定输入生成诊断使用 `--steps generation_probe`。具体题目、两臂参数、时限和命令只在
[T1-1 实验记录](../_audit/mainline/T1-1/REPORT.md) 登记，不是训练或性能 baseline。
两臂先各自通过 CPU 接口、输入和缓存准备，再同时申请单卡 B200。尚未写出 `cache_ready.json` 就不会分配 GPU。
生成结果写完后，最多再等 60 秒让程序正常退出；超时只回收该调用自己的进程组，保留结果并记 `timeout_reason=exit_after_result`。这属于程序收尾失败，不是质量 WARN，也不能算正常完成。
默认同时跑两臂；只复验现行采样时显式传 `--probe-arms full`，不会申请另一臂的资源。

## 固定业务管线

在容器内只使用：

```bash
bash scripts/v16_pipeline.sh [--dry-run] [--profile smoke|candidate] \
  [--gate-mode observe|strict] [--run-id ID] <stage|all>
```

阶段顺序由 `syncopate/pipeline/stages.py` 定义，Shell 和 Modal 共用。当前共 19 段：

```text
cases menus split gates supply rl-data teacher sft-data teacher-stop
sft-train sft-eval sft-select merge exam
rl-train rl-adapter rl-eval opd-train opd-eval
```

`--dry-run` 只检查命令形状和静态前置，不能代替数据、教师文本、GPU 或候选门禁。

`--check-inputs` 支持 `rl-train` 和 `opd-train`：实际解析上游模型路径并检查对应数据/adapter 文件存在，不启动训练或写阶段 PASS，不能与 `--dry-run` 合用。独立 OPD 的 runbook 参数 `--opd-input-run <RL run-id>` 只支持 smoke 的 opd-train/opd-eval，且与 `--rl-input-run` 互斥；须先在 CPU 用 `syncopate.pipeline.opd_input` 完成底座、源 actor 与 adapter 的内容绑定。Python 给 Shell 的输出只含机器路径/安全引用变量，提示走 stderr；数据内容和隔离仍由 CPU 前置单独验证。

云端对应 `--pipeline-opd-input-run <RL run-id>`。`opd-train` 自动先运行 CPU 测试、实际输入/adapter/词表检查，再准备缓存并申请两张 B200；任何失败或跳过不申请 GPU。`--pipeline-opd-real-steps 2` 只覆盖这次独立 smoke 的真实更新目标，仍由唯一 runbook 派生 attempts 和 batch。`opd-eval` 先复核同轮 CPU 证据，再申请一张 B200。单独 `--steps opd_preflight --pipeline-run-id <ID> --pipeline-opd-input-run <RL run-id>` 不申请 GPU；随后必须使用同源码 `--pipeline-resume`，不能覆盖已有前置。原始 token 门禁使用明确的本轮底座 tokenizer，不猜默认模型。

本机没有目标 CUDA/B200、完整 verl/vLLM 或 PG/Redis 权限时，不在本机重配；把失败缩成最小测试，
用 `--steps pytest --pytest-args ...` 放到 Modal CPU 镜像，或用对应 pipeline stage 放到 B200。只跑需要的段，并保存 summary。

## 证据

- 本机汇总：`_audit/stack_probe/summary_*.json`
- Volume 汇总：`/vol/_audit/stack_probe/`
- v16 数据与训练：`/vol/_audit/v16/`
- 模型与训练产物：`/vol/models`、`/vol/checkpoints`

每个实验臂使用独立审计目录；不能让两个任务同时写同一 parquet、checkpoint 或缓存。
同一 run 的每段可换 CPU/单卡/双卡容器，但源码身份必须相等。恢复使用 `--pipeline-resume`；不能复用没有源码身份的旧 B02 账本。

## 安全与收尾

- 在用户已批准的范围内运行，具体资源批次和时限写进实验记录；授权与并行规则只看 Compute。
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

`docs/archive/` 只保存历史证据，不能作为当前运行入口。
