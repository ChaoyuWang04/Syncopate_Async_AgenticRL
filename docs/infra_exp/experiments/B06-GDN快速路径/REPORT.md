# infra-probes/B06 · GDN 快速路径为何未完整启用

状态：已结项；官方依赖方案在真实B200局部前后向确认，未证明加速。任务Q17；2026-09-08。

## 调查结论

现有CPU记录不能证明FLA损坏：实装 `import_utils.py` 的两个可用性函数都要求CUDA可用，CPU容器因此把FLA callable置空。实装模型的汇总 `is_fast_path_available` 仅控制警告，构造器分别选择卷积、delta核心和门控归一化；缺少 `causal_conv1d` 不会自动关闭FLA。旧GPU日志只能证明快速组件不齐，不能证明整个GDN全走torch。此前“全fallback”表述应收窄为“至少一个组件缺失”。

现成方案是补齐官方可选卷积依赖，不是自行修模型。2026-09-08刷新GitHub：Transformers稳定5.16.1、FLA稳定0.5.2、causal-conv1d稳定1.7.0；完整main SHA和检索见 [research.json](research.json)。本项保持实际5.10.4以隔离依赖变量；main已经改为Hub kernel接口，不能直接套回冻结版本。1.7.0资产列表没有torch2.13/cp312/x86_64匹配轮子；官方setup明确支持CUDA≥12.8的sm100，使用CUDA13在派生镜像从源码编译，仅添加该依赖，不能更换整个生产锁。

实装源码证据：Volume `syncopate-home` 的 `/vol/_audit/infra-probes/B06/source-inspection-01/attempt-065b284675c64d93bd3f7119a96c6a2e`；modeling SHA `40da264c51fcfadd7b87271c5485f029ee9efd05e6e54b34728c8ee252db4c9c`，import_utils SHA见research。CPU检查正常结束；GitHub的5.10.4标签404，因此以实装文件为冻结身份。

## 预注册设置与判据

只取已登记Qwen3.6-35B-A3B模型第0层GDN真实权重，沿真实embedding和input_layernorm产生B03固定输入第一条的64-token隐藏态。不加载35B完整反向；不使用cache、packing、LoRA。普通HF模块是算子机制参考，不能称为FSDP2训练或全模型性能。父入口复用B03模型完整身份闸、固定输入、冻结镜像及GPU门禁。

- 派生镜像仅添加 `causal-conv1d==1.7.0`；`CAUSAL_CONV1D_FORCE_BUILD=TRUE MAX_JOBS=4`，安装到已有目标venv且 `--no-deps --no-build-isolation`。保留源码包/生成wheel SHA和镜像身份，CPU必须直接导入causal_conv1d与fla.ops.gated_delta_rule均成功，核对全部冻结版本和modeling SHA，并留源文件；不在Mac安装训练依赖。
- 单B200，每次GPU硬上限30分钟；同一错误首次确认起20分钟仍未解决则暂停。独立attempt，禁止覆盖。父入口承担预算账与进程退出核验。
- 进入三臂之前先用原hybrid在inference_mode重放真实层输出，与B05封印capture的batch1_output逐元素相等。fastpath-native-identity.json先保存差值；不等即停止诊断输入、norm或核心差异，不能放宽身份阈值。
- 三臂同权重/同隐藏态/同随机cotangent：全torch仅为诊断参考；native重建原镜像的torch卷积+原FLA核心/归一化；fast仅相对native添加官方卷积。constructor实际FLA函数身份落盘，核心或卷积缺失直接失败。
- 每臂执行一次观测前向和向量积反向，计数真实卷积/核心调用；fast两类命中必须非零，全torch必须为零。完整输出、输入梯度、每个参数梯度均须有限、shape/dtype一致；相对全torch的每张量relative L2≤0.03作为BF16局部回归界，零参考只允许零差。逐张量max_abs同时报告，禁止只比较范数；缺梯度直接失败。这是局部回归，不是bitwise等价宣称。
- 再执行同shape前后向记录输出/输入梯度噪声；去掉观察包装后2次预热、3次CUDA Event计时，每个样本包含同一完整模块前后向。报告原始样本、中位数、离散度；同次分配3窗口只属初筛，不是3次独立实验。收益仅比较fast对native，必须大于两臂样本范围重叠才能称本次可分辨收益；全torch不得充当生产性能baseline。冷编译与稳态分开。

CPU：`python -m syncopate.infra.gdn_fastpath_probe --cpu --out <fresh-attempt>`；GPU：同入口去掉`--cpu`并传`--input <B03-input.json>`。`fastpath-dependencies.json`记录实装源码、CUDA与callable，`fastpath-partial.json`逐臂持久化，`fastpath-result.json`只有全部判据通过才`ok=true`。原始证据写 `syncopate-home:/_audit/infra-probes/B06/<run>/<attempt>/`；输入/config/index SHA写入终态，父索引绑定完整模型revision与权重manifest。

## 测试结果与结论

[运行索引](runs/batch4-fast-gpu-01.json)保留CPU/GPU命令、封印SHA及Volume位置。运行用冻结overlay `6330c0f980d43a51573db6371539f227538e6a34780968bcfa4fc91bd1c378f4`，不能以当前工作树冒名。

| 检查 | 结果 |
|---|---|
| 本机拒绝性测试 | 3 passed，CLI与语法通过 |
| 派生镜像构建 | 首次失败；Python sysconfig指定clang/clang++，环境无CC/CXX覆盖且二者不存在，PyTorch把检查失败的0.0.0哨兵值误呈为版本过低。显式CC=gcc/CXX=g++后以GCC13.3.0构建成功，628.69秒 |
| 目标CPU | batch4-fast-cpu-02通过，182.03秒；两个直接导入、冻结版本/源码及CUDA未初始化检查通过 |
| B200原路径身份 | native输出逐元素重现B05 batch1捕获；模型、输入、源码与CPU/GPU镜像闸通过 |
| 实际调用 | native卷积0/FLA核心1，fast卷积1/FLA核心1；native已使用FusedRMSNormGated，补依赖只新增快速卷积 |
| 局部前后向 | 每臂输出、输入梯度和9个参数梯度共11张量均有限且满足预注册relative L2≤0.03；native最大0.016684、fast最大0.009965，均为A_log梯度、参考为全torch；每臂输出/输入梯度重复噪声均为0 |
| 计时初筛 | native中位4.504ms、范围4.490–5.194ms；fast中位4.331ms、范围3.541–4.798ms，范围重叠，未通过可分辨收益判据 |

本项确认：之前不是整个GDN未加速，而是缺少可选卷积依赖，FLA核心已经启用；官方causal-conv1d安装方案在冻结栈可用。完整fast路径通过本次单层BF16前后向回归，但并非逐元素等价：其输出相对全torch的relative L2为0.002757，输入梯度为0.007761。两者都不能当作训练收敛、完整模型或FSDP2/LoRA回归。

GPU五个命令均rc=0、无超时，总阶段166.02秒，其中算子程序90.9秒。原始kernel JIT包含在阶段时间，未逐项拆出；表中计时为预热后的同次分配三个窗口，不是独立重复或全模型速度。未证明补齐卷积能在本64-token负载带来可分辨加速，不据此更改生产默认。

编译器现场在 `syncopate-home:/_audit/infra-probes/B06/compiler-inspection-01/attempt-d341f66667d64a9b9c45dd86c1f29285`；此证据确认sysconfig而非环境覆盖导致clang选择。镜像ID `im-GMKXfvLjPzdKfErOwKvwsi`及实装版本已记录，源码包/生成wheel SHA没有记录，不宣称可逐字节重建该wheel。实际费用未读取，阶段秒数不能替代账单。

已回读并重算终态、partial和native身份JSON的SHA，核对命中、33个比较项、噪声与计时范围；未下载完整梯度张量重新计算。保留本次JSON/日志/源码归档供复验，无checkpoint或大训练产物，无upstream移交：这是已有依赖方案确认，不是新上游缺陷。
