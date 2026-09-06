# T1-1 · RL 重复思考与生成长度

状态：按最新工程范围关闭。接线修复已通过定向验证，并在 B03 c 批完成真实 RL 更新、同步、checkpoint 与 adapter 身份复验。历史固定题目的模型行为告警没有被宣布根治；用户已停止语义清洗，质量 WARN 不阻塞 infra。

## 要解决什么

B02 的 16 条 RL 轨迹有 4 条重复思考并撞到 12,288 token，另有一条接近上限但没有闭合思考。最初以修复行为为目标；用户随后把范围调整为保留诊断、修正训练接线并验证真实更新，不再追求消除所有模型质量告警。下面保留各批原始计划和实测，不把旧质量停止线重新当作当前前置。

本记录保存测量、决定和验收证据。任务顺序只放在主线 TASKS。

## 已确认的起点

- Git 起点：`61e125a6aa05d6d75641aeaa7adf4af5d8b3ac90`。
- 数据、合并 SFT 模型和 B02 原始轨迹全部在 Modal Volume。
- 四条截断样本中，三条首轮开始重复，一条在工具返回后重复；分数仍为 0.1～0.3。
- 当前循环没有独立单轮预算；引擎可以把整条轨迹预算给一次生成。
- 原始逐步 dump 没有保存结束原因。需要把实际结束原因和每轮读数留到完整 artifact。

## 本轮授权与资源

用户已批准按 T1 → B03 → B04/B05/B06 → B07 的顺序施工及所需 Modal 验证。只使用 CPU 和 B200；按需要申请 CPU、单卡或双卡。可独立的实验尽量并行；相互依赖的阶段先等上游产物通过检查。

本轮先运行 CPU 数据审计：读取现有 SFT parquet、Gold 和 B02 原始轨迹；不重建正式数据。预期一次 CPU 容器，4 CPU、16 GiB，30 分钟硬超时。GPU 运行前补充具体输入、工作量、时限和读数。

## 跑前登记的检查

1. 正常数据：统计每个 assistant 轮的 token 数、思考 token 数、是否闭合、连续重复长度和桶分布，报告 p50/p90/p99/max。
2. 实际输入：记录模板、system、消息历史、完整工具菜单及 token 身份；原文只留 Volume。
3. 仪器：每轮保存 token 数、剩余轨迹预算、生成结束原因、思考闭合、解析结果和工具错误。数值摘要进入 RL 指标，完整结构进入 artifact。
4. 本地正负对照：首轮重复、工具后重复、未闭合、近上限、合法长工具链、预算与 mask 对齐。先用固定假引擎验证这些读数，再上云。
5. 修复参数在正常分布量完后登记；整条轨迹预算保持 12,288，不以降低总预算消除告警。
6. 修复验收：固定输入的退化样本不再出现长段重复或未闭合近上限；正常长工具链不误杀。随后 2-step RL 要有两次真实更新、权重同步和可导出的 checkpoint/adapter。

质量红项使用 observe 收齐证据；身份错误、越桶、NaN、程序异常、坏产物和零真实更新仍停止。只通过本地检查或停止了坏轨迹，都不能算 T1-1 完成。

## 证据

- B02：[REPORT](../../infra/B02/REPORT.md)
- 已有轨迹摘要：`_audit/stack_probe/summary_2026-09-05_144252_exec.json`
- 已有 verl 源码检查：`_audit/stack_probe/summary_2026-09-05_143937_exec.json`
- 本轮 CPU 测量：`_audit/stack_probe/summary_2026-09-05_152208_exec.json`；Volume 原件 `/vol/_audit/mainline/T1-1/cpu_20260905a.json`。
- GPU 固定输入诊断：f 批双臂共 32 条，g 批单臂复验 16 条；后续真实 2-step RL 的工程验收见 [B03](../../infra/B03/REPORT.md)。

## 正常数据的实测长度

CPU 读取正式 SFT 的 1,222 行、5,349 个 assistant 轮，没有加载模型权重。

| 读数 | 中位数 | p99 | 最大值 |
|---|---:|---:|---:|
| 单轮 assistant token | 46 | 304 | 1,859 |
| 单轮思考 token | 0 | 269 | 1,771 |
| 首轮 prompt token | 6,883 | 7,063 | 7,124 |

未闭合思考为 0，缺失轮次结束符为 0。连续重复 token 最长 14；按词/字检查最长为 4。多数轮没有非空思考；这只是数据现状，不能据此认定它就是 RL 退化的原因。

数据 SHA256：`002fec9c06b470d8786340ab1f7c3997c8ede2567f13e1a973f2fdb43dc33966`。
源码快照 SHA256：`3f45760ed4bc2e1215f88d84e7f9766fc10ef984d5fcfffa3d1e8418e0ba9c6e`。
模板 SHA256：`e84f32a23fdda27689f868aa4a1a5621f41133e51a48d7f3efcbea2839574259`。

## 先修正测量路径

- 每轮保存实际输入位置、采样参数、结束原因和原始 token；缺失结束原因明确记为缺失。
- 明确向引擎传入剩余预算，预留模板结束符；不降低整条轨迹的 12,288 上限。
- 被裁掉的动作不得执行或用于判分。达到生成长度上限仍算不完整，不能借此报“修好”。
- 评测移除旧机器遗留的强制注意力后端，由当前 vLLM 选择。
- 本地预检补上 Jinja2 依赖及 Mac Bash 的变量边界兼容。源码先复制为不可变快照再上传，避免编辑 TASKS 时影响正在打包的实验。

## 输入、停止和奖励的核验

4 个题目的 parquet 消息均与当前共享构造函数相等。B02 dump 会去掉特殊 token；把重建 prompt 按同样规则解码后，4/4 与 dump 相等。首轮均只有一次隐式 `<think>` 开头，没有发现重复开头。旧 dump 不能还原完整的逐轮原始 token，新的完整 artifact 才负责这项证据。

verl 0.9 源码把 vLLM 的 `stop`、`length` 都写成 `completed`。此外，`FullyAsyncLLMServerClient` 续写转发层会重建结果，只保留策略版本字段，丢掉新增观察字段。因此要在服务端保存原始原因，并在续写层补回各分段读数；只增加读数，不改上游 token、logprob、重试或调度状态。两处都经过延迟导入，不能在 Ray 分卡前加载 CUDA。

对实际安装的 verl 方法做了正负对照：不装观察器时原始原因缺失；装上后 `stop`、`length` 和中途恢复的分段可区分；删去新增观察字段，整个结果与未安装时相等。实际 worker 钩子也已验证：安装时不导入 torch/verl/vLLM，随后导入真实上游时两处观察器都生效。

Modal CPU 最新专项测试 12 passed、0 skipped，源码 `7f3128ae525806392ec867a6df87aed1dd36f55f5b3264c258a533318a6d3ede`。证据：`_audit/stack_probe/summary_2026-09-05_170944_exec_7430c733.json`，Volume `/vol/_audit/mainline/T1-1/observer_cpu_20260905b.json`。此前 9 项检查保留在 `summary_2026-09-05_165930_exec_39e7ed00.json`。这些证明接口接线，不替代真实 RL artifact 的命中验收。

用完全不完成任务的固定输出复核了 4 题的奖励：两题 0.3 来自政策项 0.2（无需检查）+ 效率项 0.1；另两题只有效率项 0.1。任务结果与证据项都是 0。这解释了旧分数，不等于认可这种激励；本次暂不凭这一现象直接改奖励。

证据：`summary_2026-09-05_153438_exec.json`、`summary_2026-09-05_154014_exec_3f76a294.json`，均在 `_audit/stack_probe/`。Volume 的 `input_audit_20260905a.json` 和 `prompt_diff_20260905a.json` 保存完整数值。

## Mac 数据前置

从固定材料重建 2,030 条，EVAL/SFT/RL 为 401/597/1,032。独立重切三桶的三个 SHA256 与仓库冻结文件逐个相等，内容交集均为 0。没有重写云上的正式数据。

## 固定生成双臂诊断（跑前登记）

- 输入：B02 四道实际 RL 题目，重建 token；去掉特殊 token 后必须与旧 dump 相等。每题种子 1234、1235、1236、1237，共每臂 16 条。
- 权重：B02 合并 SFT 模型。不能声称重放了旧第 2 次更新后的策略；这是固定题目的重新采样。
- 两臂唯一差异：`full` 使用现行 1.0/1.0/-1；`model_defaults` 读取模型文件中的 temperature/top_p/top_k。不改项目默认值。
- 预算：仍为整条 12,288 token；没有新增重复惩罚或单轮短上限。连续重复 ≥128 token 只报警，不提前中断。正常 SFT 最大重复为 14。
- 资源：先 CPU 检查真实 verl/vLLM 接口和输入身份，通过后同时启动两个单卡 B200；各臂最多 45 分钟，合计最多 1.5 B200 卡时。每臂先验架构与 kernel。
- 输出：`/vol/_audit/mainline/T1-1/<run-id>/<arm>/gpu/`；完整 artifact 不离开 Volume；缓存、写者占位与记录各自独立。
- 读数：重复、未闭合、长度截断、停止原因缺失、工具错误、终答和逐题 reward。质量红项全部留到结尾；模型/输入身份和程序错误停止该臂。
- 若只有 full 明显退化：下一步核验截尾采样的概率/训练语义，不直接改默认。若两臂都退化：转查 SFT 数据与模型。若都不退化：明确记未复现，回到真实 verl 路径做带观察器的 2-step smoke，不能声称已修复。

本批命令：

```bash
modal run --detach modal_app/stack_probe.py --steps generation_probe \
  --probe-source-run b02_20260905a --probe-run-id t11_gen_20260905d --probe-mode gpu
```

该阶段 CPU 完整依赖回归：302 passed、3 skipped；当时本地相关回归：320 passed、10 skipped。10 项跳过中，7 项依赖 torch/verl/vLLM（已由对应 CPU 云检覆盖），3 项属于不适用的旧契约分支。后续观察器专项 CPU 测试为 12 passed、0 skipped；这些是不同范围、不同源码时点的检查，不能相加。

首批 `t11_gen_20260905a` 在 CPU 调度入口报 `No module named syncopate`，GPU 没有启动。原因是 Modal 外层 Python 与训练 venv 分开，新增共享调度组件在配置搜索路径之前被导入。已补标准库组件的路径设置及 `-I -S` 隔离导入测试；数据版本仍从训练 venv 读取，不在调度层复制常量。证据：`_audit/stack_probe/summary_2026-09-05_161603_generation_probe_310fd478.json`。

第二批 `t11_gen_20260905b` 的实际采样接口测试通过；上游观察器测试的假服务漏了 `_disaggregation_role` 字段，3 passed / 1 failed，因此没有申请 GPU。按已读到的上游字段补齐测试对象，不改上游实现，也不放宽断言。证据：`_audit/stack_probe/summary_2026-09-05_162253_generation_probe_3425980b.json`。

Modal 自动上传的调度入口另与镜像内冻结副本逐字节对拍，防止打包过程中编辑入口而混用源码。CPU 前置和 GPU 两臂都先启动全部独立调用，再等待结果。

第三批 `t11_gen_20260905c` 的两臂 CPU 接口和输入身份检查全部通过，源码为 `4b8aeb196684be702c152448e25285dc98dc37c20b7c31ff3a79098a2015c785`。两臂 GPU 同时发出，但 GPU 进程尚未开始，时间耗在复制 Volume 编译缓存：进程等待磁盘，实查一臂 B200 为 6 MiB / 0%。为避免空占显卡，在生成前停止此批；16:36:29 App 已 stopped、tasks=0，原始数据与缓存均未删除。该批没有生成结果，不能作 A/B 结论。证据：`summary_2026-09-05_163623_generation_probe_878221a6.json`。

准备流程已改为 CPU 完成缓存复制后写 `cache_ready.json`，GPU 只读该结果；复制中断不留完成标记。下一批维持相同题目、种子、采样对比和预算；CPU 准备两臂各最多 45 分钟，完成后 GPU 两臂各最多 45 分钟。生成跑完与形状通过分开记，质量红项不会被退出码 0 掩盖。

第四批 `t11_gen_20260905d` 源码为 `d6825126c7add6e4f7dc292a2dafa1e08387f475030475168c55136b99aa8c52`。两臂 CPU 全部通过，缓存复制分别花了 1,515.281 和 2,262.956 秒。两个单卡 GPU 同时启动，架构、FlashAttention 反向和 FLA 对拍均通过；但生成程序在启动 vLLM 时都报 `Cannot re-initialize CUDA in forked subprocess`，没有生成结果。17:26:27 App 已 stopped、tasks=0。证据：`_audit/stack_probe/summary_2026-09-05_172627_generation_probe_afa06f74.json`。不能把这次程序错误写成模型质量失败。

共享 Python 推理入口现显式要求 `spawn`，并在构造引擎前打印判据；不接受显式 `fork`。同时去掉了“模型 EOS 配置读失败就用空清单继续”的兜底。CPU 测试必须经过真实构造入口和上游 `get_mp_context()`，只把最后的模型分配替换成假工厂；缺失 EOS 必须在构造模型前失败。做法依据 [vLLM 进程启动说明](https://github.com/vllm-project/vllm/blob/main/docs/design/multiprocessing.md)；真正启动仍需 GPU 验证。

第五批 `t11_gen_20260905e` 维持原题目、种子、采样与 12,288 预算。先过包含新增启动测试的 CPU 前置，再并行申请两个单卡 B200，各最多 45 分钟。显式顺序复用已停止的 d 批各臂缓存，不复制两棵小文件树，也不复用任何旧实验结论。旧缓存来源保留，GPU 另对物理缓存目录占用原子写者键；共享缓存不能同时写。模型、数据、原始审计均不改。

e 批停在 CPU：38 passed、2 failed。两项真实构造测试在收尾时发现共享入口改写了 pytest 的日志收集器，报 `EncodedFile has no attribute getvalue`；没有申请 GPU。现只重定向普通 stdout handler，不接管第三方收集器，并加标准库本地测试。证据：`_audit/stack_probe/summary_2026-09-05_174005_generation_probe_c7698c29.json`。

第六批 `t11_gen_20260905f` 使用相同的双臂计划与时限，再次先过 CPU 前置；成功的 CPU 测试输出也会随记录保存。vLLM 实际使用的进程上下文必须等于登记值 `spawn`。缓存仍来自已停止的 d 批，不借用 e 批未完成的准备结果。

f 批源码为 `08b9b7b65a1d32eda70cdfc17aab4c5944910df81e70685f0befadd65d539a01`，镜像 `im-9iIf6e8Siu4aq8v2bu7FMb`。两臂 CPU 各 41 passed、0 skipped，输入身份和缓存复用通过，随后同时启动两个单卡 B200。GPU 架构、FlashAttention 反向和 FLA 对拍均通过，实际进程方式为 spawn。两臂均完成 16 条生成，但生成结束后引擎没有自行退出，因此不能叫正常完成的运行。

### f 批生成结果与退出问题

| 读数 | 现行采样 full | 模型默认采样 model_defaults |
|---|---:|---:|
| 完整结果行 | 16 | 16 |
| 长段重复的轨迹 | 1 | 0 |
| 未闭合思考的轨迹 | 3 | 0 |
| 长度截断 | 1 | 0 |
| 轮数耗尽 | 10 | 4 |
| 缺失原始停止原因 | 0 | 0 |
| 工具错误总次数 | 45 | 17 |
| 解析错误总次数 | 22 | 0 |

唯一长度截断发生在 full 的题目指纹 `62f1d6b236`、seed 1237：首轮真实生成 12,286 token，连续重复跨度 4,898 token，没有闭合思考。拼接的 2 个模板结束 token 使整条 response 达到 12,288。其 reward 仍为 0.3，但终答解析失败。另两条未闭合轨迹同时有解析错误。

“截断 11 对 4”大部分是轮数耗尽，不是长度截断。模型默认采样在这批小样本上的生成形状更好，但仍有工具错误和轮数耗尽，不能据此宣布质量通过、修改正式 RL 采样，或证明训练会更好。下一步先用 CPU 归因工具循环，并核验截尾采样与训练概率的契约。

按同题目、同 seed 比较 reward：7 条上升、7 条下降、2 条不变。均分分别为 0.1975 与 0.2078。这不是冻结业务评测，更不是显著收益。

生成结果已经落盘后，Python 与 EngineCore 仍占显存、GPU 利用率为 0。核对各自进程组后，仅对这两组已完成任务发出 SIGTERM；原始数据、缓存和 artifact 均保留。两臂退出码为 -15，说明需要外部回收，不能当作正常退出。App `ap-iZcvarZYXbo8u8UteaAiOo` 在 18:03:59（上海时间）停止；再次只读检查为 stopped、tasks=0。再次申请 GPU 前，必须补上显式引擎关闭、成功/异常两路回收测试，以及结果写完后仍不退出的短超时保护。

证据：`_audit/stack_probe/summary_2026-09-05_180356_generation_probe_e0ae0fd5.json`。匿名逐条数值在本目录的 `t11_gen_20260905f_full_result.json` 与 `t11_gen_20260905f_model_defaults_result.json`；原始 prompt、token 和轨迹只留 `/vol/_audit/mainline/T1-1/t11_gen_20260905f/<arm>/gpu/`。这些结果不是性能 baseline，也没有发生训练更新。

```bash
modal run --detach modal_app/stack_probe.py --steps generation_probe \
  --probe-source-run b02_20260905a --probe-run-id t11_gen_20260905f --probe-mode gpu \
  --probe-cache-from t11_gen_20260905d
```

## CPU 归因与接线修复

退出修复的 Modal CPU 检查为 53 passed、0 skipped，源码 `876834a9a08efe7dbb7944559b1ec749afb991ad2848187072de8f78b52cc12b`。覆盖真实 vLLM shutdown 方法、成功/异常路径、结果写完后的超时，以及只回收本任务的父子进程组。证据：`summary_2026-09-05_182645_exec_179ea5f5.json`；Volume `/vol/_audit/mainline/T1-1/cpu_lifecycle_20260905b.json`。这项只验接口和回收机制；真实 GPU 退出结果见后面的 g 批。

工具错误中，full 有 34 次 unknown_tool，涉及 6 个不存在于实际菜单的工具名；模型默认采样没有这类错误。另一臂有一条轨迹把同一参数、同一工具的“没有可用素材”错误重复了 11 次。原始名称、参数和输出仍只留 Volume。不能直接认定题库坏了，也不能靠加轮数掩盖。

另有两处已在真实 artifact 和本地负对照中定位的接线问题：

1. 204 个正常停止轮都已经返回 EOS，循环却漏补其后的换行。Gold 回放不带 EOS，因此原先只用假引擎的对拍没有覆盖真实返回形状。
2. 循环补入的模板 token 没有采样概率，却被 RL mask 标成 1。修复后真实生成与模板段分开；SFT 继续监督 Gold 结束符，不直接复用 RL mask。

本地先看到两项对应测试失败，再修改共享路径。修复后轮次 token 逐项相等、模板段 RL mask=0、模型 token 的 logprob 不变。6 个 Gold seed 的完整输入与 SFT mask 合并 SHA256 修复前后均为 `81b4771e90401565aceafdc21319030d8c375ec220807f40430359d9fff713ab`；没有重建或改写正式 1,222 行 SFT 数据。这项检查只覆盖这 6 个 seed，不能扩大成整份数据已重验。

注意：`cpu_lifecycle_20260905a.json` 的 `raw_eos_last_turns` 用了旧代 token ID，不能采用；b 版已从实际模型配置读取 `[248046, 248044]` 并纠正该读数。其他匿名工具计数不受影响。

## 修复后的单臂复验（g，跑前登记）

只复验 `full` 现行采样，不再额外分配模型默认采样臂。仍使用同一 B02 合并 SFT、4 题 × 4 seed、12,288 总预算，不训练、不改采样或奖励。

先在 Mac 通过 token、mask、SFT 不变性和退出测试；再由 CPU 前置用真实学生 tokenizer 与完整依赖重验。通过后申请 1×B200，最多 45 分钟，继续顺序复用 d 批 full 的独立缓存。观察长度/轮数截断、重复、未闭合、工具错误及真实 EOS 后的模板边界；质量 WARN 留到结尾。

程序验收另外要求：16 行结果与原始 artifact 齐全；每轮模型/模板 token 的 mask 正确；有 `shutdown=complete` 判据；进程正常退出、App 停止且任务数为 0。结果写完后仍挂住超过 60 秒即回收本进程组，记程序失败。重复或未闭合仍出现时，不跑 2-step RL 或 B03，先处理行为质量；不能用这轮诊断宣布 T1-1 关闭。

```bash
modal run --detach modal_app/stack_probe.py --steps generation_probe \
  --probe-source-run b02_20260905a --probe-run-id t11_gen_20260905g --probe-mode gpu \
  --probe-arms full --probe-cache-from t11_gen_20260905d
```

### g 批结果

源码 `14a50872cf557ca68ed44448468373f181efe53329ac06b0b9fadc74726ccd1a`，Git 底座仍为交接提交。先过 CPU 68 passed、0 skipped 和单卡 GPU/kernel 门禁，再完成全部 16 条生成。生成子进程用时 185.6 秒，退出码 0，没有触发超时保护。引擎显式 shutdown 从 18:46:07 到 18:46:10 完成；App 于 18:46:18 停止、tasks=0，无需人工回收。此耗时不是性能 baseline。

| 读数 | g 批现行采样 |
|---|---:|
| 结果与 artifact | 16 / 16 |
| 长段重复轨迹 | 1 |
| 未闭合思考轨迹 | 4 |
| 长度截断 / 轮数耗尽 | 2 / 4 |
| 工具错误 / 解析错误次数 | 16 / 22 |
| 平均 reward | 0.1461 |

长段重复仍发生在 `62f1d6b236 / seed 1237` 的首轮：12,286 个真实生成 token，重复跨度 5,139 token，reward 0.3。另一条长度截断来自多轮未闭合思考和解析重试，不是单次无限重复。轮次边界修复没有消除首轮退化；不能把修了接线写成模型问题已解决。

随后的 CPU artifact 检查通过：16 份记录、79 个生成轮、37,018 个真实模型 token 保持原样；81 个补入模板 token 的 RL mask 全为 0；77 个正常 EOS 后的换行逐 token 正确。其余两轮为长度结束。真实 RL 的 logprob、梯度、同步和 checkpoint 尚未验收，这轮没有训练。

证据：`summary_2026-09-05_184614_generation_probe_d2589f97.json`、`summary_2026-09-05_184921_exec_33ff8121.json`，均在 `_audit/stack_probe/`。原件为 `/vol/_audit/mainline/T1-1/t11_gen_20260905g/full/gpu/` 和 `/vol/_audit/mainline/T1-1/cpu_g_acceptance_20260905a.json`。

## 数据是否需要重建

CPU 再次只读检查正式 1,222 行 SFT：5,349 个 assistant 轮、4,213 次工具调用，无解析错误、无未登记工具名、无缺失结束符。四道异常题的 Gold 都可解；它们不需要素材表就能完成规定路径，不能把模型选错素材工具后的报错直接归为数据缺失。

其中 392 轮有非空思考，全部在 `cot_hard` 桶；其他桶没有非空思考。这个分布需要结合实际训练抽样、教师内容和恢复行为继续审阅，不能直接认定是重复思考的原因。当前没有整库重建的证据，也没有把 RL 题搬入 SFT 桶。

证据：`summary_2026-09-05_184341_exec_79335d79.json`；Volume `/vol/_audit/mainline/T1-1/cpu_training_content_20260905a.json`。正式 SFT SHA256 未改变。

## 既有诊断收尾与当前去向

- 本机最新相关回归 219 passed、13 skipped：10 项缺目标依赖，由同源码 Modal CPU 检查覆盖；3 项是当前 v15 模式不适用的旧契约分支。另在 v14/think-off 重放新增边界与 SFT 对拍，8 passed。runbook 全 19 段 dry-run、编译、Shell 语法与 `git diff --check` 通过。
- observe 收到了完整质量告警，没有改采样、奖励或 12,288 总预算来报绿。
- 用户随后决定冻结数据、停止语义清洗。原文回传受限不再是当前执行的前置；不继续索取原文或扩展样例审阅，也不把匿名统计说成逐条人工验收。
- 接线修复已在 [B03 c 批](../../infra/B03/REPORT.md) 的真实 2-step RL 完成复验：两个 rank 的 optimizer 状态均为 step 2；16 份 artifact 的模型 token、采样概率、模板 mask 通过；checkpoint 与导出的 adapter 身份通过。T1-1 已从任务队列移除，成套跨引擎概率/路由、恢复和性能重复仍由 B03 负责。
- c 批没有 token 长度截断，但有 3 条轮数耗尽；这不是对历史四题的根治证明，也不是能力提升结论。原有行为质量退出线已经撤销，不等于告警消失。上文 g 批的停止规则只描述那一批历史计划。
- 未提交、未推送；没有删除云端模型、checkpoint、原始审计或缓存。早期启动失败与回收问题的证据保留，不能只保留最后一次成功退出。
