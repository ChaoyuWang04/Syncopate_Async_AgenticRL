# infra-probes/B07 · GDN 输出投影的split-K根因与下游传播

状态：Q19已结项，2026-09-08。已证明本次第0层 `out_proj` 的batch1/batch2差异由cuBLASLt的split-K选择造成；不等于整模型全部差异只有这一处原因，也没有生产修复或性能验收声明。

## 问题与调查

[B05](../B05-GDN投影边界定位/REPORT.md) 捕获完全相同目标输入，经同一BF16 `Linear` 因batch形状变化产生差异，但未确定实际算法。这里的投影是GDN第0层 `out_proj`，不是 `lm_head`。

官方[PyTorch数值说明](https://docs.pytorch.org/docs/main/notes/numerical_accuracy.html)不保证batch与slice逐位一致；[cuBLASLt](https://docs.nvidia.com/cuda/cublas/)提供split-K数量、归约方案与算法检查接口。split-K把一条点积的K维拆成多段并行计算，再合并部分和。当前问题有真实输入见证，需固定形状改变该选择才能确认因果，不能只引用浮点非结合性。检索、实装头文件SHA与冻结版本理由见 [research.json](research.json)。

## 设置与预注册判据

保持B05模型revision、64-token原输入、BF16权重、无cache、SDPA和GDN源码SHA；torch 2.13.0（git `cf30153c4c131c8164ee7798e5022d810682e2cb`）、Transformers 5.10.4、CUDA13.0、单B200。所有模型文件通过manifest核验。真实输入/输出捕获SHA `24c0face5106870ac0001256d7773f694eb63243b4c478251b40b0e88bffbe23`，精确路径见运行索引。各GPU调用硬上限30分钟，同一程序错误20分钟未解暂停；独立attempt写B07，无训练、checkpoint或Lab改动。

1. **CPU前置**：共享算子矩阵与tiny两层传播正对照；人工batch2首投影扰动必须被量到，假替换不变，真实替换恢复所有下游层与chosen-logprob。split-K追加CPU只编译host C++ ABI桥、加载torch实际cuBLASLt库，CUDA未初始化；不在Python猜枚举/结构体布局。
2. **形状/精度**：真实Linear原样重放必须精确等于B05捕获。batch1/2/4/8（M64/128/256/512，K4096/N2048）×真实/重复/零邻居×BF16归约精度ON/OFF/FP32 IEEE，共36臂。目标输入固定，每臂3次重复；记录FP64及舍回BF16参考、非零计数/分箱、2D展平对照、同M邻居对照。实际CUDA trace必需，profile不能改变输出。
3. **传播干预**：全模型batch1/2，各做观察、同shape假替换、首投影目标输出换成batch1原输出，每臂2次。原投影先核对B05捕获；替换必须精确命中且邻居不变。观察/假替换/batch1替换须保持完整logits哈希，记录40层与63个teacher-forced位置的chosen-logprob/top1。
4. **同M因果追加**：M固定64，BF16输入/权重固定，FP32compute、32MiBworkspace。比较无限制heuristic、禁归约preference、原算法只设splitK=1/reductionNONE。第三臂原始七项属性必须等于无限制臂，除splitK/reduction外的ID/tile/stages/swizzle/custom必须不变；AlgoCheck通过且trace无split-K。基线须精确重放torchM64才允许 `torch_attribution_ok=true`。保存完整输出、真实tensor SHA和两两相等判据，不能以相同统计量替代。

入口分别为 `python -m syncopate.infra.gemm_cause_probe --input <input.json> --out <fresh>` 和 `python -m syncopate.infra.splitk_probe --input <input.json> --out <fresh>`；CPU加`--cpu`。源快照、镜像、命令、输入与结果SHA只在下列运行索引展开。

## 已执行结果

| 运行 | 结果 |
|---|---|
| [主CPU](runs/batch4-gemm-cpu-01.json) / [主GPU](runs/batch4-gemm-gpu-01.json) | 共享正对照通过；36臂与12组传播完成，GPU调用225.98秒 |
| [追加CPU1](runs/batch4-splitk-cpu-01.json) | 编译/导入通过，但为补齐因果闸之前的CPU中间版本，未用于GPU |
| [追加CPU2](runs/batch4-splitk-cpu-02.json) / [同M GPU](runs/batch4-splitk-gpu-01.json) | 使用同一冻结源码`60b9b0…`，CPU通过；GPU162.02秒，diagnostic与torch attribution均通过 |
| [精确点积](exact-dot-summary.json) | CPU有理数逐项累加存储BF16乘积，覆盖全部269个差异坐标，CUDA未初始化 |

### 投影根因已闭合

原torch M64实际运行 `nvjet_sm100_tst_64x32_64x16_2x2_2cta_h_bz_splitK_TNT` 加独立 `cublasLt::splitKreduce_kernel`；M128运行对应非split-K kernel。只关BF16低精度归约仍看到split-K与相同报告指标，不能把这个开关叫作“禁split-K”。

同M64追加直接cuBLASLt无限制臂精确重放torch单条：algo66、tile13、stages35、swizzle0、custom3、splitK2/reduction2。保留五个非干预属性，仅把splitK/reduction改为1/0后，独立归约kernel消失；**全部131072个输出逐元素等于原torch M128目标输出**，与原M64恰有269个差异，最大绝对差 `0.000244140625`。输出SHA分别为`a14eb6…`和`12f0e1…`，完整值及2,362,409字节tensor产物SHA见同M运行索引/结果。禁归约preference另选algo21，也得到相同M128输出；因同时改变多个算法属性，该臂只作补充。

这使“batch改变M→split-K选择改变→该投影输出改变”的因果链得到固定M干预证明。七项公开属性已读回核对；没有独立读取额外inner/cluster属性，也没有声称掌握闭源kernel的每条累加指令。证据足以定位本算子选择机制，不是全栈所有数值问题的统一根因。

### 更大batch并不更准

36臂中同shape重复、profile、2D展平、同M邻居对照均逐元素相等。batch2/4/8及BF16 ON/OFF报告相同误差指标，但首轮未保存这些臂之间直接相等/hash，不把指标相同写成tensor逐元素相同。

相对舍回BF16的FP64参考，batch1有181个不匹配坐标，batch2有313个。全部269个跨batch差异坐标的精确有理数复算中，单条更接近精确值208个，batch2更接近61个。最大差示例 `(18,1229)` 的精确值约 `0.050415142250955114`，位于两结果中点旁；252个精确值在两结果之间，17个在区间外，不能把全部差异解释成恰好舍入平局。这里只评价相同存储BF16权重/输入的点积，不是原始未量化模型真值，更没有跨引擎大batch更接近的结论。

### 首投影修齐后，下游仍会产生差异

自然batch1/2的完整logits哈希与B05分别完全相同，所有63个chosen-logprob也一致；没有已证实的跨轮baseline漂移。观察/假替换保持OFF哈希，两次干预终态一致。

| 相对单条基线 | 自然batch2 | 替换首投影后batch2 |
|---|---:|---:|
| layer0输出relative L2 | 0.00047642 | 0.00009728 |
| layer0差异坐标 | 4819 | 322 |
| layer39输出relative L2 | 0.128819 | 0.138507 |
| chosen-logprob平均绝对差 | 0.017083 | 0.023918 |
| chosen-logprob最大绝对差 | 0.127478 | 0.169832 |
| 63位置top1变化数 | 0 | 0 |

自然batch2的40层relative L2有9次相邻下降，第31层峰值约0.205408，第39层回落到0.128819；隐藏态相对差不是概率误差百分比。首投影差异确实影响下游，但消除它没有使整模型差异归零，也没有单调改善终态；不能按局部L2比例分摊整模型误差。剩余layer0差异没有在本项继续逐算子定位，不冒充已经解释。top1不变仅限这63个teacher-forced位置，不是自由生成一致性。

## 验收边界

独立复核主结果与同M结果JSON封印、实际执行源码SHA、因果属性闸、输出SHA/直接相等、自然基线与假替换；原始trace/tensor留Volume，未在Mac重新下载张量计算。Q19按“投影形状差异的具体原因”结项。无生产开关修改、无更大batch准确性或吞吐收益承诺；第0层剩余差异作为有界追问Q20进入[TASKS](../../01-TASKS.md)，未扩大本批GPU范围。
