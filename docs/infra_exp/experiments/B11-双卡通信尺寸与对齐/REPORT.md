# infra-probes/B11 · 双卡通信尺寸与对齐

状态：**停止主动下钻，保留局部基线与未解边界；真实训练通信画像需要时才重开。** Q07 / G7.1；2026-09-08。原生NCCL局部重复基线已建立；PyTorch Graph归因未完，公开direct接口受阻，不是全面通信正常声明。

## 问题与调查结论

为未来权重同步提供通信尺寸参照。消息来自Qwen3.6-35B-A3B revision `995ad96eacd98c81ed38be0c5b274b04031597b0` 最后一个full-attention层o_proj的实际header：BF16参数/二分片及rank8 LoRA A/B字节数，各取±2字节邻界。CPU/GPU绑定config/index/header SHA。没有测真实训练器发包频率，邻界不代表业务流量占比。

查询到NCCL stable `v2.31.2-1` / `7b83616df3ae082a1f32bb74c27458bfe8153a13`，master `fd168324a3dc0c9080fd4881b6c7f4bb252a95a2`；本次PyTorch配套实际运行库是 **NCCL2.29.7**。旧[issue413](https://github.com/NVIDIA/nccl/issues/413)不能证明当前B200问题；开放PR1452正文获取失败也不能当已合入修复。全部调查及失败范围见[research.json](research.json)。

## 已执行设置与判据

三独立分配02/03/04各2×B200、36案例；额外05同分配交错复核16MiB±2、9案例。三个collective为all-reduce、all-gather、reduce-scatter；BF16输出逐元素精确且有限，CPU破坏单元素负例必须失败。每窗口2次预热+10次计时、3窗口；05每轮循环移位尺寸顺序，三个尺寸分别占一次前/中/后位置；profile在全部测速后独立执行。进程硬限360秒/组60秒，单分配最多12GPU分钟，未扩大硬件。

主延迟是每窗口最慢rank总host时间/10，保留CUDA event与原始样本；输入复位/barrier排除。字节口径遵循[nccl-tests](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md)：all-reduce按单rank数组，gather/scatter按总数组，2卡bus因子分别1、1/2、1/2。邻界延迟>对齐1.2倍且差值>两者窗口std最大值3倍仅标复查线索，阈值未修改。三分配可能共物理主机，不以UUID不同推断三台机器。

## Q07 补充测量预注册（2026-09-08）

目标是解释旧测量中的发起/等待成本，当前新增入口为 `python -m syncopate.infra.communication_baseline_probe --mode cpu|gpu --out <独立目录> [--preflight <CPU目录>]`；资源编排由独立 Modal wrapper 负责。同一分配严格两张 B200，MPI 两进程、每进程一卡、每进程一个测试线程；PyTorch 同样两 rank，CPU 线程各 1。先 CPU 编译和身份检查，再 GPU；不读取模型权重，不产生 checkpoint。首个同卡诊断及有决定价值时的同源码重复由本批授权约束；本次单分配三个窗口不冒充三个独立实验。

官方 [nccl-tests v2.20.0 源码](https://github.com/NVIDIA/nccl-tests/tree/b4d5beebca8a76cf01335f724d154b9b9d394d96) 已于当日读取，查询 master SHA 为 `b4d5beebca8a76cf01335f724d154b9b9d394d96`。[common.cu](https://github.com/NVIDIA/nccl-tests/blob/b4d5beebca8a76cf01335f724d154b9b9d394d96/src/common.cu) 的批量计时包含发起和最终同步，`-a 3` 取最慢 rank；正确性另行重置数据后检查。它已提供 Graph 路径，故这里不将 Graph 当新优化，仅用 PyTorch Graph 解释 PyTorch 发起成本。CPU 仅编译三个官方目标，使用 PyTorch wheel 自带 NCCL 头和动态库，私有符号链接补齐 `-lnccl` 所需名字；CPU 的 torch `ldd`、GPU 的 native `ldd` 与 torch `/proc/self/maps` 逐一比对库 SHA。固定 NCCL 2.29.7 是为解释已有 B11，未升级至 NCCL 最新版。

消息口径明确为**总数组字节数** 16MiB±2。[AG](https://github.com/NVIDIA/nccl-tests/blob/b4d5beebca8a76cf01335f724d154b9b9d394d96/src/all_gather.cu) 与 [RS](https://github.com/NVIDIA/nccl-tests/blob/b4d5beebca8a76cf01335f724d154b9b9d394d96/src/reduce_scatter.cu) 会把每 rank 的 BF16 count 向下对齐到 16 字节，因此实际为 16MiB−32、16MiB、16MiB；重复实际尺寸合并，共七案例。这与旧 REPORT 的 `block_bytes` 口径不同，不能直接拿 AG/RS 的旧 32MiB 总数组时间相比。输出保留每个请求→实际字节映射，并核对官方输出 count。AR 比较 native in-place 与 torch in-place；AG/RS 比较各自 out-of-place。

每案例先 20 次预热、32 次 pilot；用最慢 rank pilot 选择约 100ms 的批量窗口，限制 128～4096 次。三个完整窗口保存原始时间，不逐次同步；native `-m 1 -N 3`，torch eager 每次发起、Graph 一次重放包含相同次数的图。torch 窗口交替 eager/Graph 顺序，native 先于 torch 的固定顺序是局限。记录 max-rank host 和 CUDA event 时间、每窗口空 Python 循环/空 event 控制、配对差异、stdev 和 max/min。**事先规定** max/min≤1.10 只表示本次分配窗口稳定；控制中位数须≤eager 的 5%，否则不能解析 10% 量级差异。噪声超限保存结果但不发布性能 baseline，不以放宽阈值或增加重复本身作因果解释；即使通过，单分配仍仅是局部诊断。

torch 计时使用全零 SUM 固定点，避免重复 in-place AR 溢出；所有元素须保持零。计时前后用逐元素 rank/位置变化的小整数做精确有限性验证，故意破坏一个元素必须被拒绝；另一个单调用 Graph 验证非零精确结果。官方保持原始数据生成和独立 datacheck，两个模式 `#wrong` 必须为 0；官方多次 in-place SUM 可能溢出，不能将其计时缓冲区称为非零数值工作负载对拍。各案例测速结束后单独采集该案例 PyTorch CUPTI trace（可能影响后续案例热态，非计时样本）；所有测速结束后才采集官方 NCCL TRACE，协议必须读 NCCL 日志，不能只读 kernel 名。

产物：CPU `communication-baseline-cpu.json`、编译/命令日志、源码和二进制 SHA；GPU `communication-baseline-result.json` 中 `summaries`、`native`、`ranks`、`diagnostics`，并保存原始 native 日志、逐 rank JSON 与独立 trace。只有 `ok=true` 且所有七案例/count/两 rank/正确性/库 SHA 完整才算工程完成。CPU 编译限 900 秒，每个 native 子进程 120 秒，torch 子进程 360 秒；任何程序错误立即保存证据，持续排查 20 分钟未解暂停本项。CPU wrapper 上限1200秒、4 CPU且不申请GPU；GPU wrapper上限1800秒、两张B200，具体身份登记在runs索引；未读取账单不报实际费用。

重复判据在补充GPU02结果前登记：若正确性通过且native/Graph对应案例窗口max/min≤1.10、空控制≤eager的5%，追加两次同源码双卡独立分配；各臂分别保留窗口和分配中位数，三分配中位数max/min≤1.10才称该尺寸/该路径的局部操作基线，不平均掩盖噪声。原生先跑、PyTorch复用进程组、每案例后profile以及Graph可能改变流依赖等限制仍在；Graph收益不能独归因Python。Graph批长由eager校准，实际窗口可能短于100ms。周期性非零模式不穷尽所有分片置换错误，长Graph计时只有零固定点检查和独立单次非零正确性，不宣称每个图节点已trace认证。

## 原生 Graph 三臂定位预注册（2026-09-08，已执行）

前一阶段出现官方 native 稳定、PyTorch Graph 稳定但更慢的待解释差异，因此只补齐官方 Graph 对照，不先宣称 Python 或链路有问题。[PyTorch 官方论坛的既有定位](https://discuss.pytorch.org/t/unexplained-gaps-in-execution-before-nccl-operations-when-using-cuda-graphs/197818/16) 已报告 NCCL 的 Graph 混合执行保护会增加 event record/wait 开销，串行工作负载可用 `NCCL_GRAPH_MIXING_SUPPORT=0` 消除；这是一项现有方案的有界确认，不是新上游 bug。

本次核对的 NCCL `v2.29.7-1` [init.cc](https://github.com/NVIDIA/nccl/blob/v2.29.7-1/src/init.cc) 将该环境变量 0 映射到 `graphUsageMode=0`，默认是 2；[strongstream.cc](https://github.com/NVIDIA/nccl/blob/v2.29.7-1/src/misc/strongstream.cc) 的 mixing 分支加入外部 event wait 和 graph event-record 节点（读取文件 SHA256 `42ad70660450e9c2fd87388b443217117f859909753eebfa36799502335c6080`）。当前[官方限制](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-graph-mixing-support)要求关闭后不能重叠启动 Graph 与普通调用；本测试每段最终同步、每臂独立 MPI 进程，符合边界。PyTorch `v2.13.0` 的 `asyncOp=false` 路径使用当前 stream，不能沿用旧版本额外通信 stream 的解释。

新模式 `native-graph` 复用同一 CPU 构建、库 SHA 和二进制验证，仍限两张 B200。只测 AR/AG/RS **总数组16MiB**，三臂是 `native_eager`（`-G 0`）、`native_graph`（`-G 1`）、`native_graph_mixing_off`（`-G 1` 加上述唯一开关）。固定 `-n 1024 -m 1 -N 1 -w 20 -a 3 -c 1`：官方捕获 1024 次调用后只 replay 一次，完成时间除以1024；`-G` 表示重放次数，不再乘一次1024。编译、capture、instantiate、warmup 不计稳态时间；默认 `-C 0` 使用完成墙钟而非仅CPU提交时间。native 原代码另捕获一次正确性调用并要求两个 in/out 模式 `#wrong=0`，不能将独立检查说成长图非零逐次正确性。

三窗口按三臂循环移位，每臂各占一次前/中/后位置，每臂/operation 启动独立 `MPI 2×1GPU`，共27次有界调用；禁止不同臂重叠、禁止额外NCCL调优。三臂统一开启仅 INIT/ENV 的 INFO 记录，写入每个窗口/臂/operation 独占的 `NCCL_DEBUG_FILE=...-%h-%p.log`，与benchmark stdout分离，避免INFO在数字行中间插入。子进程退出后必须找到该精确前缀的两份日志，并核对rank 0/1、文件SHA及mixing消费；stdout仍严格解析13列和真实 `graph` 头字段。不打开逐次COLL TRACE、不使用profiler测速；初始化期间仍可能有延迟INFO写盘，各臂采用相同日志级别。记录逐窗口原始行、实际count、参数、环境与官方输出；native-eager 控制臂及两Graph臂分别沿用 max/min≤1.10 稳定线，噪声超限只保留未定结果。比较配对绝对差和Graph/default、mixing-off/default比，不跑后增设收益阈值。

解释规则：默认原生Graph若复现较慢且OFF回到原生eager附近，支持已知混合执行保护开销；原生Graph无此差异，则该路径不能解释PyTorch差异，不扩大推断。若两个Graph臂相同，则本轮不支持mixing为主因。只有原生结果不支持既有解释且仍有决定价值，才另登记下一步，不能自动追加参数矩阵。单分配不是跨分配性能验收，开关不写入项目默认。GPU wrapper硬限1800秒、每次MPI120秒，程序错误遵循20分钟规则；结果仍写 `communication-baseline-result.json`，原始数据另写 `native-graph-windows.json`，保留资源/退出/费用边界。

## 结果

[CPU05](runs/batch6-b11-cpu-05.json)21项零跳过；[GPU07](runs/batch6-b11-gpu-07.json)27次三臂调用全部通过，两个rank实际读取mixing=0。AR/AG/RS中位数（µs）：原生eager52.37/37.24/36.95，原生Graph50.52/35.32/36.35，Graph关闭mixing50.42/34.82/36.33。原生Graph没有复现PyTorch的额外成本，关闭mixing也未解释差距；此既有修法不采用为项目默认。

[CPU04](runs/batch6-b11-cpu-04.json)19项零跳过。[GPU06](runs/batch6-b11-gpu-06.json)首臂因NCCL INFO插入benchmark数字行而触发自有严格解析失败；不算数值/通信失败。已把诊断输出分为精确命名、校验rank与SHA的独立文件，原13列校验不放宽；失败原文保留。

[补充GPU02](runs/batch6-b11-gpu-02.json)在两个rank写出七案例测量后360秒超时：原代码在进程组销毁之前写`ok:true`，Graph对象仍存活；本轮不能视为工程通过。与[PyTorch已知Graph/通信器生命周期问题](https://github.com/pytorch/pytorch/issues/115388#issuecomment-3009880966)吻合；修正自有探针收尾顺序后，[CPU03](runs/batch6-b11-cpu-03.json)15项零跳过，[GPU03](runs/batch6-b11-gpu-03.json)正常退出，torch子进程27.8秒。最终成功现在只在Graph释放、进程组销毁后写出。

补充轮[CPU01](runs/batch6-b11-cpu-01.json) 11项通过、零跳过，官方三个二进制编译和库身份通过。[GPU01](runs/batch6-b11-gpu-01.json)通过双B200门禁后，在MPI_Init失败：PMIx共享内存固定地址映射不可用，尚未进入NCCL计时。采用日志及[OpenPMIx维护者](https://github.com/openpmix/openpmix/issues/1747)已有的`PMIX_MCA_gds=hash`方案；CPU新增真实双进程MPI_Init/Allreduce/Finalize检查。它只改变MPI启动元数据存储，不修改NCCL载荷传输。这是环境启动错误，不是取消或通信数值失败；首次确认06:13:34 UTC；[CPU02](runs/batch6-b11-cpu-02.json)13项零跳过且真实双进程检查通过，GPU02随后已越过MPI进入七案例NCCL测量，此启动错误在20分钟内解决。

**补充GPU03/04/05三次同源码完整重复均工程通过。** 官方原生七案例每次窗口max/min均≤1.10，跨分配中位数最大max/min=1.0102；通过预登记的局部基线要求。约16MiB的各尺寸原生中位数范围：AR52.21～52.77µs、AG37.14～37.77µs、RS36.70～37.19µs。PyTorch eager没有案例满足全部重复稳定性；Graph六个案例满足、对齐AG未满足，并且没有普遍加速。原始窗口与逐臂判据见[summary](summary.json)的batch6；[GPU04](runs/batch6-b11-gpu-04.json)、[GPU05](runs/batch6-b11-gpu-05.json)保留独立身份。GPU03原生独立日志确认NVSwitch/P2P/CUMEM，不自动扩展为Graph协议认证。

目标CPU三轮分别62、64、66项通过，均零跳过；原配置、环境闸修正与focused调度各自验后晋级，见[CPU01](runs/batch5-communication-cpu-01.json)、[CPU02](runs/batch5-communication-cpu-02.json)、[CPU03](runs/batch5-communication-cpu-03.json)。

| run | 正确性/诊断 | 尺寸与时间结论 |
|---|---|---|
| [GPU01](runs/batch5-communication-gpu-01.json) | 运行前失败：自有环境闸误将镜像`NCCL_VERSION=2.28.3-1`标签当调优 | 已修为CPU/GPU共享验证；标签允许并单列，ALGO/PROTO仍拒绝；实际版本独立读取，原失败保留 |
| [GPU02](runs/batch5-communication-gpu-02.json) | 36案例逐元素通过 | 16MiB−2的AG1.566倍、RS1.211倍触发预登记旗标；单分配低噪声线索 |
| [GPU03](runs/batch5-communication-gpu-03.json) / [GPU04](runs/batch5-communication-gpu-04.json) | 各36案例通过；独立task/GPU对 | 无旗标，但16MiB区间整体更慢且窗口噪声明显；不能反证02尺寸影响 |
| [GPU05 focused](runs/batch5-communication-gpu-05.json) | 9案例通过；交错、日志、NVML与独立trace齐 | 无旗标；末轮AG(−2/0/+2)207.55/210.86/201.48µs，未复现稳定1.57倍；早期窗口仍有时变噪声 |

三次原矩阵源码、overlay、镜像及依赖锁身份相同。精确结果SHA、UUID和必要数值见[summary.json](summary.json)；完整复现和Volume身份见runs索引。原始证据保留于`syncopate-home:/vol/_audit/infra-probes/B11/<run-id>/<attempt>/`。没有checkpoint。

**05默认路径已量到：**日志9个payload组合各37次RING/**SIMPLE**、32通道，恰好覆盖3×(2预热+10计时)+1profile；334次4字节barrier才是LL。profiler符号带`RING_LL`不等于真实payload协议，已撤回按名字推断LL的说法。日志实际连接为`P2P/CUMEM`，图为GPU→NVS/0-0→GPU，`2/721.8/NVL`中的带宽是模型值，不是实测。

**拓扑解释：**实跑版本[源码topo.cc](https://github.com/NVIDIA/nccl/blob/v2.29.7-1/src/graph/topo.cc)将`0x068000`映射NVS；该分支创建/复用交换机节点，不解析全F target为GPU PCI地址。因此XML的全F target不能判为NCCL错连。NVML祖先和完整物理拓扑未暴露，与NCCL已建立NVSwitch路径分别表述；rank0拓扑dump存在，rank1未生成如实记录。

**计时边界：**05三个rank0 AG独立profile的kernel为84.116/790.506/356.009µs，对应CPU annotation434.893/1113.041/414.322µs。单次profile明显不稳定，也含profile扰动；kernel自身还可能等待对端。这些不与稳态样本混算。host与CUDA event区间都可能包含CPU发起/调度空档，二者同时变慢不能独自证明纯链路带宽下降。

## 最后一次有界调用路径定位预注册（2026-09-08）

GPU07的原生Graph/mixing对照未解释PyTorch Graph差距。最后增加`--mode torch-direct`：同一双B200分配、两进程各一卡、同一已锁定NCCL2.29.7库SHA，仅16MiB BF16原地AllReduce。对照ProcessGroupNCCL与公开`torch.cuda.nccl.all_reduce`显式world=2通信器/当前stream；Gloo仅交换UID与计时外同步，不在载荷Graph内。每臂Graph分别捕获128和1024次零输入归约，三个交错窗口，报告跨rank最大host/CUDA区间除以实际调用数及窗口max/min（≤1.10才称稳定）。两臂均在前后用非零rank相关输入的一次归约Graph逐元素验等，并验证刻意损坏能被发现；零输入计时后仍逐元素为零。此对照只能归因到调用路径组合，不能只凭差值断言某个event或CPU开销。

[PyTorch v2.13公开Python接口](https://github.com/pytorch/pytorch/blob/v2.13.0/torch/cuda/nccl.py)与[C++实现](https://github.com/pytorch/pytorch/blob/v2.13.0/torch/csrc/cuda/nccl.cpp)确认显式通信器、stream与BF16支持；CPU预检实际签名及源码SHA，不调用CUDA。Graph实例化后保存[官方get_graph_data](https://github.com/pytorch/pytorch/blob/v2.13.0/torch/cuda/graphs.py)原始拓扑、节点类型/核名计数，再独立采集计时外replay trace。该元数据API额外要求driver/compat≥13.1；若只缺该能力则保留确切错误及DOT，节点归因记未验，不能猜。接口源码返回字符串类型名与整数ID，不按CUDA handle猜JSON格式。

所有Graph先释放、NCCL/Gloo进程组显式销毁，最后才写rank成功，父进程要求torchrun正常退出。公开direct通信器capsule析构在此版本为no-op，因此只获准一次性进程生命周期诊断，由进程/context退出与single-use容器结束回收，不称长期训练可用的清理API。父结果沿用`communication-baseline-result.json`；`direct-rank-*.json`、graph JSON/DOT和trace保留于独立run目录。CPU重新验源码身份后才启动，双B200时限1800秒，不扩大消息/算子矩阵；完成本次两路径比较即停止，不把继续重复当根因证明。

## 最后封装层对照的实际状态

[CPU06](runs/batch6-b11-cpu-06.json)24项与[CPU07](runs/batch6-b11-cpu-07.json)25项目标CPU测试均零跳过。GPU08因本地新增driver查询漏传日志名，在通信初始化前失败；补上参数并增加调用签名回归后重新冻结源码。[GPU09](runs/batch6-b11-gpu-09.json)已越过此错误，但两rank均在`torch.cuda.nccl.init_rank`报`SystemError: PY_SSIZE_T_CLEAN macro must be defined for '#' formats`，尚未进入载荷/Graph计时。不能把CPU签名检查通过说成该GPU公开接口可用，也不能将此异常解释为通信数值或性能问题。精确torchrun日志在该run的`torch-direct.log`，失败JSON和身份已回读校验；本轮没有可报告的PG/direct性能比。

公开接口错误对应已有[issue #38019](https://github.com/pytorch/pytorch/issues/38019)与[修复 PR #127702](https://github.com/pytorch/pytorch/pull/127702)。这是编译时Python C API绑定要求，不能以运行时环境变量修复。本次不重编译PyTorch、不引入私有通信器绕路；辅助PG/direct对照登记受阻，待受支持公开接口可用后再决定复验。前一项本地参数错误已修；本项停止依据为确认已有接口缺陷和有界范围，不声称20分钟超时。最后GPU App已核对stopped、tasks=0；没有checkpoint，不删除仍用于复现的源码/日志/trace。

## 结论与停止条件

已确认本矩阵传对，并在05确认默认NVSwitch/P2P路径与SIMPLE协议。02线索没有变成跨分配或交错复核中的稳定尺寸因果；这不等于证明完全无尺寸影响。旧尺寸筛查未形成性能baseline；补充轮现已建立上述原生NCCL限定配置的局部基线，不是训练吞吐或所有调用路径的基线。没有新的上游问题声明。

此前按有限筛查关闭，非20分钟程序错误触发；原环境闸错误已修复。此前用户要求继续后恢复Q07并补测；现按真实训练优先裁定停止主动下钻，原始读数与未定结论不变。上述同GPU对、同NCCL库和消息口径的三路径对照已完成；原生Graph/mixing三臂已完成且不能解释PyTorch差距；最后的直接调用对照在初始化阶段失败，尚无两路径配对数字。不将增加重复次数本身当因果定位，也不把Graph收益直接称为链路带宽提升。JSON、trace和必要日志保留，运行时长见根wrapper与runs索引；未读取结算账单，不将进程时长当费用。没有上游移交。
