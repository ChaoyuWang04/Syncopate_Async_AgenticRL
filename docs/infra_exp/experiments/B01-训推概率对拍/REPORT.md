# infra-probes/B01 · 公开 MoE 同 token 的训推概率

状态：已结论；Q02 完成真实 FSDP2/vLLM BF16 局部对拍；Q15 完成本批 CPU 执行前置；任务 Q02；分组 G2.1/G4。

## 问题与调查结论

比较未经过本项目训练的公开 Qwen3.6-35B-A3B checkpoint；这里 base 指实验起始权重，官方模型卡明确它已做 post-training，不冒称预训练版。官方 vLLM recipe 支持 Qwen3.5/3.6，当前 vLLM 稳定版0.28.0、main 392db567；目标安装 Transformers5.10.4/torch2.13/verl0.9，精确锁随运行归档。

当前默认批次浮点差并不承诺逐字节相等；batch invariance 是 beta，已验证列表不含 Qwen3.5/3.6。历史 #32992 是较旧 torch2.9 的 B200 compile 问题；#51187 是0.26/4060/dense/44并发，不能当成本项已复现或已有适用修复。先测存在性，不开 kernel 优化。来源与进一步调用源码记录在 research.json。

补充调查：[GDN支持PR #45819](https://github.com/vllm-project/vllm/pull/45819) 与 [#49827](https://github.com/vllm-project/vllm/pull/49827) 均未合并；覆盖的后端/混合批次机制不同，本轮不采用未合并方案。Transformers最新稳定已是5.16.1，保留5.10.4因verl0.9约束<5.11；该patch版本Git tag查询404，以实际安装源码SHA和锁为准。

## 设置与判据

CPU 先验证 Volume 原始权重相对于明确 HF revision 的 SHA、config/tokenizer 和原始 token 输入。只读 `/vol/models/Qwen3.6-35B-A3B`，不下载或重造权重、不用旧 adapter。原文只编码一次，后续两引擎使用相同 token ID；测 prompt 中第2个及之后 token 的条件对数概率，温度1，无采样过滤。

单 B200：依次独立进程运行 HF BF16 与 vLLM BF16，TP1、无 LoRA；固定两条不同内容等长短序列（每条64 token，位置0..63），先目标单条重复3次，再目标+邻居重复3次。vLLM 用 prompt_logprobs，prefix cache关闭以确保每次重算；默认 compile 保留，不是性能臂。HF走官方 SDPA，attention/experts 选择和真实类落盘。形状/原始 token 必须精确相等、概率有限；未同卡复验不作速度结论。

预注册初筛线：同引擎重复 max|Δlogp| >1e-5 记重复敏感；batch >0.02 或跨引擎 >0.1 记待定位（分别约2%/10.5%概率比变化），同时比较是否 >重复噪声10倍。这是定位触发线，不是框架精度保证或正确性通过阈值；差异不说明根因。超线只形成有界后续任务，本项不无限逐层深挖。出现 NaN、错身份、API错误停止该批。

预算：CPU 4核/16GiB，单次20分钟，最多5次（观测hook传输、三轴位置与官方批量入队接线重验，总$15预算不变）；GPU 1×B200/16 CPU/128GiB，单次30分钟硬上限，最多5次（前四次实际GPU函数合计1086.75秒，第五次最多1800秒，仍低于累计1卡时；第五次仅重试Modal取消的同源码批次），累计GPU≤1小时，整个B01费用上限$15（实际平台账单另列）。只有CPU全通过后启动GPU。原始证据 `syncopate-home:/_audit/infra-probes/B01/<run-id>/`，每次独立写者；无模型更新或 checkpoint。

## 当前补测预注册（2026-09-07 用户批准）

旧 HF 六次前向仅作机制参考。本轮在真实 verl FSDP2 `infer_batch` 计算 BF16 log-prob，双 B200、DP=2、forward-only、无 optimizer/LoRA；两 rank 重复相同输入以核对分片计算，不测 DP 吞吐。官方 forward-only 会启用 CPU offload，实际状态必须落盘。vLLM 用单 B200/TP1，同原始权重和 token，分别单条/同批三次，比较63个目标条件概率。沿用上述诊断触发线，不承诺引擎逐字节等价。拓扑查询不受平台支持时记录 unavailable；GPU 型号/数量仍硬校验，缺拓扑不做通信性能结论。同输入两 FSDP2 rank 的 max|Δlogp| 必须≤1e-5，否则保存差异并停止，不作为正确性通过。

新入口 `modal_app/probability_batch.py`，每次 Modal 投递创建独立 attempt，返回精确路径；结果 JSON 做终态 SHA，仍可能收到退出行的日志不再冒充不可变结果。CPU 4核/16GiB20分钟；FSDP2 双 B200/16核/192GiB30分钟；vLLM 单 B200/16核/128GiB30分钟。成功路径最多1.5卡时，取消重派按用户授权继续，实际费用逐次记录；实际程序错误第二轮仍失败暂停该项。旧五次预算属于历史批次，不倒改旧索引。

## 测试结果

完整身份、命令和证据SHA见各run索引；所有失败保留，秒数是函数墙钟，不是模型性能。

| run-id（点击索引） | 结果 | 秒 |
|---|---|---:|
| [cpu-01](runs/b01-cpu-01.json) | 3项测试及原始权重检查通过；复审后补齐shard完整性和CPU→GPU身份绑定，不作为GPU gate | 161.87 |
| [cpu-02](runs/b01-cpu-02.json) | 4项测试零跳过；完整模型/输入身份通过 | 157.48 |
| [gpu-01](runs/b01-gpu-01.json) | HF六次前向完成；vLLM自有hook的默认pickle限制拦截，未得到其概率 | 263.15 |
| [cpu-03](runs/b01-cpu-03.json) | 4项测试及真实MsgpackEncoder通过；仅本探针私有IPC允许传自有函数 | 134.73 |
| [gpu-02](runs/b01-gpu-02.json) | 真实token一致；三轴位置均正确，探针错误按一维比较而停止 | 417.65 |
| [cpu-04](runs/b01-cpu-04.json) | 5项测试零跳过，三轴位置正/负对照与身份通过 | 126.76 |
| [gpu-03](runs/b01-gpu-03.json) | 单序列三次root核对通过；两请求实际分开执行，未满足同批门槛 | 221.33 |
| [cpu-05](runs/b01-cpu-05.json) | 5项测试零跳过，最终批量入队代码与相同原始输入绑定通过 | 130.91 |
| [gpu-04](runs/b01-gpu-04.json) | Modal在vLLM预热时取消；原因未明确，finally保留终态；退出日志继续写入，回读SHA与终态清单不一致并单列，不能当通过证据。未进入概率测量 | 184.63 |
| [gpu-05](runs/b01-gpu-05.json) | 同源码重试再次被Modal取消；HF已完成并回读，vLLM启动阶段未得到概率。停止自动重试 | 112.38 |

模型revision `995ad96eacd98c81ed38be0c5b274b04031597b0`，71,916,703,545字节原始模型/元文件通过检查；各批input.json SHA均为`cf3fd1e9856ac560c441bfd42c96e573b47d3d8740d4330091482ac278636e28`。

为保证先入队后调度，第四GPU批按[官方API](https://github.com/vllm-project/vllm/blob/v0.28.0/vllm/entrypoints/llm.py)使用`sleep(level=0)`→`enqueue`→`wake_up(tags=["scheduling"])`。下层源码确认不卸载权重/缓存；仍要求真实128-token同批root命中，不以API列表长度代替。以上观测修正不改变概率容差，不包装成上游问题。

当前真实训练引擎补测：

| run-id | 结果 | 函数秒 |
|---|---|---:|
| [batch2-cpu-03](runs/batch2-cpu-03.json) | 20测试/9子测试零跳过；原始输入与真实 verl 准备接口通过，终态/结果 SHA 回读一致 | 150.49 |
| [batch2-fsdp-01](runs/batch2-fsdp-01.json) | 拓扑查询返回255，未进入模型；已将平台不可见性单列，不当数值门槛 | 见索引 |
| [batch2-fsdp-02](runs/batch2-fsdp-02.json) | 真 FSDP2、1026个分片参数、CPU offload，无optimizer/更新；两rank各6次结果完整，跨rank差0 | 189.25 |
| [batch2-vllm-01](runs/batch2-vllm-01.json) | 单B200/TP1，6次真实batch/token匹配，概率有限，结果 SHA 回读一致 | 205.42 |

## 结论与下一步

**真实 FSDP2 与 vLLM 的概率对拍已完成；两边均有 batch 敏感性。** 相同形状重复三次的差均为0，FSDP2两rank概率也完全相同；只加邻居仍改变结果。63个固定目标位置的读数见 [comparison.json](comparison.json)：

| 比较 | 平均绝对 Δlogp | P95 | 最大值 |
|---|---:|---:|---:|
| FSDP2 单条→双条 | 0.016706 | 0.106888 | 0.153227 |
| vLLM 单条→双条 | 0.016292 | 0.093725 | 0.111521 |
| FSDP2↔vLLM 单条 | 0.020832 | 0.101493 | 0.250417 |
| FSDP2↔vLLM 双条 | 0.015144 | 0.071531 | 0.132074 |

分布补充见 [distribution.json](distribution.json)：单条跨引擎P50=0.002200、P90=0.060665；63个位置中43个绝对Δlogp<0.01，4个≥0.1，1个≥0.2。最大差位置的固定token概率由12.22%变为9.52%；此前未保存全词表top-1，不能从单个token差推断选词翻转。

单条比较中 vLLM/FSDP2 的逐token概率比范围为0.7785～1.0625，不是概率百分点之差。真实FSDP2使用官方SDPA与GDN fallback，vLLM使用其优化路径；当前只证明该模型/输入/版本/摆放下的差，未分离每个kernel贡献，不能认定上游bug，也不能推断RL退化。HF机制层级定位独立交给 [B03](../B03-batch差异逐层定位/REPORT.md)。

这把尺子用于后续batch/低精度比较：先检查同引擎噪声，再报增量；不混用不同batch。各臂来自不同B200分配，拓扑不可获取；这是数值初筛，不是性能基线、全模型一致性认证或长期训练稳定性结果。Megatron未测试。

Q15前置：本机/目标CPU证明每次投递使用独立attempt，不覆盖旧证据；终态要求必需JSON完整、SHA可回读，退出日志不冒充不可变结果。未主动注入抢占，也未声称已实测平台重派恢复。FP8 CPU错误另见 [B04](../B04-FP8概率误差/REPORT.md)，不混入本项概率结论。

本轮GPU函数时间按索引累计，成功FSDP2约378.49卡秒、vLLM205.42卡秒，另有首次拓扑退出开销；CPU/内存费用与实际账单未测。旧五次GPU合计1199.13卡秒属于历史开销，未改写旧身份。未创建checkpoint；原始概率、源码、输入和日志留Volume，vLLM私有编译缓存暂留供已登记B04复验，不动共享模型。无疑似上游新bug，未移交DRAFT。

收尾：[本批存储盘点](../../storage/batch2-retention.json)、[自有App退出复核](../../storage/batch2-apps-closed.json)。
