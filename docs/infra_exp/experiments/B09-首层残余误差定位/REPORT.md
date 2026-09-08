# infra-probes/B09 · 首投影对齐后的首层残余误差

状态：Q20按用户方向裁定暂时收尾，移出活动队列；不是根因全部解决或严格一致验收通过。2026-09-08。GPU01局部证据及03:38–03:58重放排错失败保留，不再继续本模型逐层下钻。

## 问题与调查

[B07](../B07-GEMM误差与传播/REPORT.md) 已解释GDN输出投影的split-K选择。本项只追踪：目标首投影换为单条输出后，layer0为什么仍有322个差异坐标。实装源码显示后续路径为残差相加、post-norm、共享专家、router、专家、共享门控、合并和残差。不能事先宣判路由错误。

2026-09-08查阅官方Transformers主分支模型/专家实现、相关issue与PyTorch数值说明；未找到与当前322见证相同的已核实修法。checkpoint重计算、TP维度错误和A_log初始化各有不同触发条件，不能作为本题根因。固定torch2.13.0、Transformers5.10.4以及B07源码SHA以保持见证，最新源码只用于调查。经网络权限刷新，官方main固定到Transformers `8eaf75f…`、PyTorch `2e2b137…`；当前稳定版为5.16.1/2.14.0，旧实装版本只为保持B07见证而冻结，详见[research.json](research.json)。

## 开跑前判据与预算

1. CPU共用观察器/替换器正负对照：同目标输入人工制造一个batch2坐标差异，必须检出；假替换不变；真实替换消除目标差异且不动邻居；CPU不得初始化CUDA。
2. 输入revision、token SHA、模型manifest、B05捕获SHA、模型源码及实装版本逐项相等。实际全模型入口原始首投影必须逐元素复现B05；layer0自然差异必须精确为4819。
3. 从该真实root捕获layer0参数，独立layer调用必须逐元素等于全模型layer0；仅替换首投影后差异必须精确322。若不成立立即停止，不放宽门槛。观察hook和同shape假替换均须不改layer0全batch输出。
4. 对post-norm输入/输出、共享专家各Linear、router的logits/weights/indices、experts、shared gate、MLP合并和最终残差分别记录目标输入/输出差异。原始tensor随运行保存，不能把统计量相同当内容相同。
5. 每个出现差异的MLP边界独立换为单条原始输出，两次重复，记录322剩余坐标与邻居是否相等。真实原模块在原输入同batch shape重放，逐元素核对capture和profile输出；保存实际CUDA kernel trace。若首差在experts，实际F.linear插桩按精确权重地址识别专家gate_up/down，并按真实top-k的(slot,token)顺序映射目标行；同输入Linear逐个替换与trace，另存联合替换结果。其他多算子首差继续下钻，不能以模块名冒充最终根因。
6. `diagnostic_ok`只表示这些捕获/重放/干预门槛通过；`mechanism_complete`单独表示是否已有同输入Linear且该单变量替换完整恢复layer0。即使为true，闭源kernel内部累计指令仍未观察，报告必须保持边界。

固定入口：`python -m syncopate.infra.residual_probe --cpu --input INPUT --out FRESH`，GPU去掉`--cpu`。输出`residual-cpu.json`、`residual-result.json`、边界tensor与trace；运行由根任务统一冻结/调度。独立路径`/vol/_audit/infra-probes/B09/<run-id>/<attempt>/`。1×B200，每次GPU硬时限30分钟；同一程序错误从首次确认起连续20分钟未解暂停，不重置计时。无业务训练、其他Lab、提交或推送。

## 已执行验证

本机语法通过；本机无torch的数值fixture显式skip，不冒充数值通过。目标CPU三次运行通过，各自冻结源码与结果见[runs](runs/)。GPU01通过4819→322入口、观察/假替换、边界重放和单项干预，但`mechanism_complete=false`。GPU02/03均未越过追加入口闸；尚未执行联合恢复、固定M算法干预或精确点积参考。

## GPU01观察与有界追加

GPU01精确复现入口。router的logits/weights/indices、experts及shared_expert_gate均相等；共享专家gate_proj有21个、up_proj有30个因batch形状变化而在相同目标输入上新增的数值差异；这是两个共享专家Linear的见证，不是router异常。仅换gate后layer0剩149坐标，仅换up剩184；换down输出归零仅说明差异经过该处，不能归因为down。gate/up的M64实际是split-K主kernel加独立reduce，M128是不同tile的非split-K kernel。

追加`--followup`（CPU同加`--cpu`）硬绑定GPU01 tensor SHA `3aa32d…`，使用与GPU01共享的官方完整from_pretrained加载函数，然后仅调用layer0，使用真实root捕获的args/kwargs，先逐元素复现两臂以及322；联合替换gate/up目标输出、两次重复并保护邻居。两Linear各固定M64、输入/权重/32MiBworkspace/FP32compute，仅改变splitK/reduction=1/0；复用B07 host ABI桥，非干预五属性必须相等、AlgoCheck与实际无split-K trace必需，无限制臂先精确复现torchM64才可归因。保存M/N/K、两臂stride、权重/输入/输出SHA。FP64全目标参考与仅51个已观察差异坐标的Fraction精确点积比较单条/批次谁更近。若同算法干预不能复现M128或联合替换仍非零，如实保留未闭合，不放宽判据。CPU补联合干预正负fixture和host桥编译，不初始化CUDA。

GPU02在手工重建layer0的逐元素复现闸失败（首次确认03:38 UTC），没有进入归因或放宽判据。手工构造未复现官方loader路径；具体dtype或初始化差异尚未逐项证明，不将A_log猜测写成根因。修正统一首次capture与followup的官方loader函数，CPU验证固定加载参数及两入口复用；GPU记录真实参数dtype与experts配置，再重验原逐元素闸。程序错误20分钟暂停时钟不重置。

GPU03改用同一官方loader后，batch1逐元素重放通过，batch2全262144坐标有215差异，最大绝对差0.000244140625，仍在入口立即停止，见[运行索引](runs/batch5-residual-gpu-03.json)。实际A_log/dt_bias均为BF16，experts实现为grouped_mm；不能继续归咎于手工重建或声称A_log应保留FP32。215差异未分开目标/邻居，也未捕获当次首差边界，现有证据无法确定新增差异来源。GPU02/03是同一追加重放门禁错误，03:38 UTC首次确认后计时不重置。

仅在用户重新授权本题时，保留的最小诊断：在一次新官方root运行中同时保留即时layer调用与序列化后重载的layer调用，对比每个输入/kwargs的tensor内容、shape/stride/storage_offset及目标/邻居，记录原始projection/共享专家两Linear/experts首差边界和实际kernel。若即时调用能通过而重载失败，先定位序列化或执行上下文差别；若两者均漂移，比较root暖机和实际算法路径。以上是待验证假设与具体下一动作，不是已证根因。必须先重获完整入口相等，不能跳过215差异就对旧322下因果结论。

GPU03回读复核：failure、loader、replay-failure三份JSON的SHA均与运行索引封印相等；当前探针源码SHA与执行索引相等。overlay身份为`79c562b72028963075fda8930193a470690d2dc7f99dc65db7c526e27afc1b11`。未重新下载完整source archive；step日志按索引明确未封印，只作诊断，不取代JSON证据。

## 框架级调查与本线收尾

2026-09-08按用户要求阅读Miles官方main `db3d6b56c85dc9886f3e35de15177a319c417156` 的实际recipe、量化转换、true-on-policy契约、FSDP2前向适配、logprob与prefill重算代码，身份索引见[research.json](research.json)。本系列一直以同一公开模型作为数值见证，没有进行多模型逐层扫描；B07已给出一般性的batch形状→GEMM算法选择→归约差异因果证据，B09新增局部位置不能自动推广成所有模型都有同一缺陷。

Miles有两条不同路线：[Unified FP8 recipe](https://github.com/radixark/miles/blob/db3d6b56c85dc9886f3e35de15177a319c417156/examples/infra_features/low_precision/README.md)以E4M3/blockwise及scale/权重转换缩小量化失配，不承诺任意路径逐位相等；[true-on-policy](https://github.com/radixark/miles/blob/db3d6b56c85dc9886f3e35de15177a319c417156/examples/infra_features/true_on_policy/README.md)通过batch-invariant算子、相同attention及dtype/归约/logprob契约追求严格相等。正式profile当前只注册Qwen3 dense的一组模型，且包含模型级norm与权重同步精度适配，不能直接继承到本次Qwen3.6 MoE/GDN。

完全相同的输入/策略版本与确定性浮点执行路径可以构造严格一致参考；仅同框架、同dtype、同batch或关闭优化不构成保证。Batch-invariant算子能允许不同batch组合，并非必须逐条串行；吞吐代价依具体实现实测。当前Miles统一launch plan启用prefill重算并覆盖rollout_log_probs、标记来源，因此重评分一致与生成时实际采样概率一致须分开证明；量化一致也不消除async权重陈旧。另经实际loss与上游单测核对，`train_rollout_logprob_abs_diff`比较的是所选old/reference与rollout；`use_rollout_logprobs=True`时两者同源可自然为零，不能用该指标单独证明当前trainer前向相等。这不说明历史recipe采用了该设置。此次为只读源码调查，没有本地/Modal运行Miles recipe。

收尾决定：停止针对本模型322坐标的追加深挖，不宣称底层kernel有bug或问题已全面解决。未来仅在可泛化的框架契约、真实特性接线或性能回归出现具体缺口时另行立项；不自动复活本题，不将业务学习作为门槛。
