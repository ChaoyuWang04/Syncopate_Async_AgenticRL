# B13 · 双卡通信与 BF16 GEMM 初始画像

结论卡：**首轮真实 2×B200 数值与通信微测试通过，已取得初始带宽和 GEMM 读数；不改变默认配置。** 这些数字只属于本次机器分配，不能作为 B03 训练性能基线，也不能关闭完整 B13。

## 1. 问题与兑现物

现有双卡 smoke 只证明通信能运行；B03 还缺逐卡算子对拍、通信分算子字节口径及 kernel 证据。本探针只用合成张量，分别回答每张 GPU 的 BF16 前后向是否正确、四类通信是否传对内容、不同消息大小与 GEMM 形状的时间是多少。不读取模型、训练文本、checkpoint 或教师数据。

代码位于 `syncopate/infra/hardware_probe.py`；Modal 环境、源码快照、拓扑、资源和缓存由 `modal_app/stack_probe.py` 的统一入口负责。独立容器的数字不视为跨机器训练证据。

## 2. 跑前预测、资源与停止线

- CPU 前置：目标主镜像中用真实 PyTorch 做小型矩阵前向/反向、通信输入形状与闭式答案检查；读取实际安装的 distributed 签名和 torchrun 参数解析器。CPU 路径不查询或初始化 CUDA。`interfaces.nccl_compiled=true` 且所有数值检查通过，才可分配 GPU；本机缺 PyTorch 的跳过不能代替它。
- GPU：**同一容器 2×B200 / 8 CPU / 32 GiB，容器最多 20 分钟，预留 $7**。这是保守资源预算，不是账单；包含本次 CPU 前置的预留，主任务启动前计入 B03 总资源账。授权沿用 Compute 文档中已批准的 B03～B13、$300 和同时最多 6 张 B200 范围。
- 只接受可见 GPU 恰好两张、两张名称均为 B200、架构均为 sm_100。错误卡型、非有限值、内容不等、缺 rank、缺窗口或程序错误均失败。没有达到任意速度值也不失败；本次不设加速晋级线。
- 小矩阵 BF16 前向、dA、dB 都相对同输入的 CPU FP32 参考检查，三项各自 `relative_l2 <= 0.02`，实际值和参考值都必须有限。零参考只接受零误差。
- 所有通信使用精确可表示的 FP32 值，按 rank、分块和元素位置区分，接收内容必须逐元素完全相等且有限。每次 all-reduce 重新拷贝原始有限输入，不将上次求和结果继续求和。
- 每个 NCCL 进程组超时 90 秒；torchrun 整组子进程超时 480 秒，关闭弹性重试。退出时 `finally` 销毁进程组。外层留 105 秒让 torchrun 回收它创建的独立 worker 进程组，不在 5 秒后提前杀掉 launcher。保留失败结果和先前完成的单项证据。

## 3. 环境指纹

目标依赖、镜像与执行规则只认 [Compute](../../../docs/syncopate/05-COMPUTE.md)。探针记录实际 PyTorch/CUDA build、NCCL、GPU 名称、capability、UUID、显存、driver 与双向 peer access。字段不可读时明确记未测；不能由设备编号猜 UUID。父入口另保存源码 SHA、完整拓扑尝试、环境/缓存身份、GPU busy、显存与功耗采样。

CPU 使用相同源码和目标主镜像。GPU 使用父入口给本次 run 分配的独立 checkout、缓存和审计根目录，不改变 NCCL 算法/协议等默认值。现有共享模型和数据不参与计算。

## 4. 方法与字节口径

通信分别测 `all_reduce`、`all_gather_into_tensor`、`reduce_scatter_tensor`、`send_recv_0_to_1` 和 `send_recv_1_to_0`。点对点按两个单向 case 报告。每 rank 分块为 **12、16、20 字节、1 MiB、16 MiB、64 MiB**；MiB 为 2²⁰ 字节，通信 dtype 为 FP32。边界尺寸不做对齐补齐。

先完成两卡 BF16 小矩阵的前向/反向对拍，再完成全部通信正确性，之后开始计时。每个 case 固定 **3 个测量窗口，每窗口 2 次预热和 10 次测量**。每次测量前恢复输入、rank barrier 和 GPU synchronize 都在计时区间外；主时间包含 CPU 下发到 CUDA 完成的墙钟，另存 CUDA event 时间。每窗口后检查末次实际产物；正确性检查、构造和 profiler 不算稳态时间。

保留每 rank、每窗口、每次调用的原始时间。每窗口主读数为 `max(rank 的十次耗时之和) / 10`；再报三个窗口的中位数、最小/最大和样本标准差，同时保存逐次跨 rank 最大耗时的离散度。**三个窗口是同一次机器分配内的窗口，不是三个独立主机样本**。因此不报置信区间或加速比。

设每 rank 分块字节数为 B，rank 数为 n，单次耗时为 t：

| 算子 | 算法字节 S | 算法带宽 | bus 带宽 |
|---|---:|---:|---:|
| all-reduce | B | S/t | S/t × 2(n−1)/n |
| all-gather | nB，即输出总数组 | S/t | S/t × (n−1)/n |
| reduce-scatter | nB，即输入总数组 | S/t | S/t × (n−1)/n |
| 单向 send/recv | B | S/t | S/t |

带宽统一使用十进制 GB/s（10⁹ 字节/秒）；不把两个方向相加。公式按 [NVIDIA NCCL-tests 性能口径](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md)。本探针的墙钟含 Python 下发开销，不能将小消息读数等同于 nccl-tests 的纯 GPU 极限延迟。

每张 GPU 另测 BF16、方阵 1024/2048/4096 的前向 GEMM，每种形状也用上述三个窗口。一次矩阵乘法的 FLOPs 为 `2×M×N×K`，TFLOP/s 为该值除以秒和 10¹²。先在 37×43 乘 43×29 的小矩阵验 BF16 自动求导，参考 dA=`G×Bᵀ`、dB=`Aᵀ×G`；大矩阵只采有限性与时间，不重复算巨大 FP32 参考。

## 5. 正负对照与本机验证

父入口审查发现的证据覆盖、CPU 跳过放行和未接遥测已修复，真实入口体测试先 10 failed / 12 passed，修后连同遥测回收测试为 24 passed。独立规格审查已通过。硬件汇总器另补检查通信内容、卡型/架构与窗口身份，不只相信 `ok`：11 个新负例先失败，修后模块本机为 50 passed / 5 skipped。

GPU 前另登记 `launcher_cpu_20260906a`：4 CPU / 16 GiB、脚本最多 5 分钟，无 GPU，另预留 $1（总账 B03 累计 $94）。读取安装版 torchrun 关闭源码，并在独占目录中启动两条合成 CPU 子进程，验证超时后没有遗留进程；不操作模型、训练或已有服务。若子进程不在 launcher 进程组，不能依赖只杀父进程的短等待。目标 CPU 验证通过前不启动 GPU。

实测 torchrun 每个 worker 都创建独立 session。两条 CPU worker 故意忽略 SIGTERM；20 秒超时后，torchrun 等 30 秒再发 SIGKILL，50.3 秒返回，两条原 PID 都已消失。结果 `launcher_cpu_result_20260906a.json`、汇总 `_audit/stack_probe/summary_2026-09-06_100418_exec_bd6c7100.json`；App `ap-Xvw6vptHwUpYc7y2Qlhsnj`。当次允许退出等待 45 秒；依据安装源码中“全组 30 秒 + 每 worker 最多再等 30 秒”，正式双卡入口使用 105 秒。GPU 卡死在内核的情形仍未验证，不据 CPU 子进程结果声称硬件故障可恢复。

`hardware_cpu_20260906a` 实测 **77 passed / 0 skipped**，NCCL 已编译、torchrun 参数和全部 CPU 公式通过。App `ap-1P5jAyrXrXKXbDWpadNjKV`、镜像 `im-gxlkAzUwE8jQIYLZ79hbfC`、overlay `d8dc3fd57b9e982361f9d4511c24b49c8721da6921f1433d869aaf13ecb58b78`。汇总 `_audit/stack_probe/summary_2026-09-06_100342_infra_probe_8f36c7ed.json`。退出等待修改后，GPU 批次必须重新以新源码跑 CPU 前置，不复用这份旧源码通过标记。

单元测试先于实现，覆盖字节映射、单位/系数、非有限值、错数值、错形状、零参考、最慢 rank 汇总、目录防覆盖、CPU 导入安全、CPU 真公式和实际 torchrun 参数。CPU 真张量测试还刻意置换元素与注入 NaN，必须拒绝。目标云 CPU 测试要求零跳过。

本机首版空模块红测为 **32 failed / 4 skipped**；GPU 汇总与 kernel 解析后续红测为 2 failed，随后“缺 dB 证据但总标志为真”的负例也红测复现。首版定向测试为 39 passed / 5 skipped；后续汇总复核与退出等待补齐后，连同父入口为 **73 passed / 5 skipped**。5 个跳过都依赖 PyTorch，不代替目标 CPU 的实际结果。语法、CLI help 和禁用 torch 导入检查通过；独立规格和代码质量审查通过，没有提交或推送。

## 6. 性能、质量和成本

`hardware_20260906a` 实测完成。同源码 CPU 前置 **78 passed / 0 skipped**；GPU 子进程 10.2 秒正常结束，退出码 0、未超时，两 rank 都销毁了进程组。随后 Modal 列表确认 App stopped、tasks=0。这不是实际账单或训练 step time。

- App `ap-AuZ2wS5H02GNQzxoKKpazK`；镜像 `im-UdecN1nvW1kIdhJFxFp3yQ`。
- Git base `61e125a6aa05d6d75641aeaa7adf4af5d8b3ac90`；overlay `7997cecb63ca5ef80ec21c8a993121ce4b6fcd403add7cb0ab6658c4e940f6df`；编排源码 `254420162dc56e6ac44637dcf00bc325cda6abe9eaf57114a33701f7dec27387`。
- Chicago，task `ta-01M1T7HAR3JSFC0QE47D6Y2FNR`；两卡 B200/sm_100、各 148 SM，双向 peer access=true。GPU UUID 分别为 `GPU-270c44a3-7da6-9b13-3dd8-e9a7b95f0763`、`GPU-2b535200-f2f8-ede4-df6f-830d4ab8dcd5`。
- driver 580.95.05、PyTorch 2.13.0+cu130、NCCL 2.29.7。完整 PCI/拓扑矩阵未取得；NVLink 状态输出也不完整，不能据此宣称链路拓扑已全部验清。
- 两卡 BF16 小矩阵前向相对 L2 都是 `0.0012069948`，dA/dB 相对误差均为 0，参考梯度非零。30 个通信 case、共 60 份 rank 记录全部逐元素相等。198 个 rank/case 窗口完整，每个窗口都有 10 个有限正数墙钟和 CUDA event 样本。

以下为每 rank 分块 **64 MiB** 的三个窗口中位数，延迟包含主机下发与同步：

| 通信 | 延迟 μs | bus GB/s |
|---|---:|---:|
| all-reduce | 177.609 | 377.847 |
| all-gather | 199.582 | 336.247 |
| reduce-scatter | 205.974 | 325.812 |
| send 0→1 | 167.952 | 399.572 |
| send 1→0 | 170.831 | 392.838 |

4096³ BF16 GEMM：rank 0 中位数 `101.3625 μs / 1355.915 TFLOP/s`；rank 1 为 `102.1315 μs / 1345.706 TFLOP/s`。这些计算已由独立检查从原始窗口重新算出，没有调用生产汇总器。

噪声也要保留：16 字节 reduce-scatter 三窗为 `178.6268 / 70.4763 / 34.5412 μs`，2048 GEMM 首窗也偏慢；不能拿它宣布对齐悬崖。每卡只有 11 个逐秒遥测样本，rank 0 的 busy 都为 0%，rank 1 只出现一次 2%；短脉冲可能被采样漏掉，因此利用率和通信占比保持未测。采样峰值显存为 3665/3664 MiB、结束为 5/4 MiB；采样峰值功耗为 273.88/263.18 W，不是稳态功耗或能耗。

合成张量不衡量任务质量、学习效果或训练加速。本次没有跨主机重复，也没有默认配置采用项。

短独立 profiler 区间同时包含 GEMM 与 all-reduce；两 rank 都取得 CUDA kernel 事件，各 19 个名称，包含 `ncclDevKernel_AllReduce_Sum_f32_RING_LL` 与 `nvjet_sm100_tst_128x64_64x10_2x2_2cta_h_bz_NNT`。Chrome trace 留在 Volume；独立检查核对了结果中列出的名称，尚未逐文件核验原始 trace。**kernel 名称不等于 Tensor Core 指令利用率；该利用率保持未测。**

## 7. 边界与替代解释

小消息墙钟可能主要受 CPU 下发影响；预热窗口的离散度包含调度噪声，不代表跨主机稳定性。GPU 并行执行 GEMM 时共享整机功耗预算。拓扑缺项、profiler 缺项或当前消息形状不代表真实训练张量时，均不能据此提出训练默认配置更改。

## 8. 产物与复查

容器内固定调用：

```bash
python -m syncopate.infra.hardware_probe --mode cpu --output /vol/_audit/infra/B13/RUN/cpu
python -m syncopate.infra.hardware_probe --mode gpu --output /vol/_audit/infra/B13/RUN/gpu
```

父入口负责填入本次 run；`cpu`/`gpu` 叶目录由模块独占创建，存在时直接拒绝，不覆盖成功、失败或残留结果。重跑使用新 run id。

本次原始根目录 `/vol/_audit/infra/B13/hardware_20260906a/`；本地完整汇总 `_audit/stack_probe/summary_2026-09-06_100801_infra_probe_58790e8a.json`，遥测 `_audit/infra/B13/hardware_20260906a_gpu_telemetry.csv`。

| 产物 | 读什么 |
|---|---|
| `RUN/cpu/result.json` | `health_ok`、`formula_checks`、`interfaces.nccl_compiled`、实际接口与形状 |
| `RUN/gpu/plan.json` | 固定阈值、工作量、窗口、单位与停止线 |
| `RUN/gpu/result.json` | `health_ok`、所有 rank 原始结果、聚合带宽/TFLOP/s 与缺项 |
| `RUN/gpu/rank_N/` | 小矩阵对拍、每个通信 case、每个 GEMM case 和 rank 终态 |
| `RUN/gpu/rank_N/profile.trace.json` | 真正采到的 CUDA kernel 事件；缺失时查 profiler 原因 |
| 父入口同 run 的证据 | 源码/镜像/拓扑、逐秒 GPU 采样、进程退出、成本与 CPU→GPU 门禁 |

复查命令：`python -m pytest -q tests/infra/test_hardware_probe.py`；先检查 CPU `health_ok` 与 `nccl_compiled`，再逐项读取 GPU JSON。没有 result、rank、正确性或完整采样即失败；不能只看 stdout 最后一行。

## 9. 对其他文档和默认的影响

本次初始画像不关闭 B03 或完整 B13，不形成采用项，不更新简历。当前任务和全局资源账由主任务在唯一 TASKS/B03 REPORT 维护；后续深测须由真实瓶颈决定。
