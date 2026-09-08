# infra-probes/B05 · GDN 输出投影输入与输出对拍

状态：已结项；目标CPU和单B200对拍完成。任务：Q18；日期：2026-09-07。

## 问题与调查

[B03](../B03-batch差异逐层定位/REPORT.md) 首次在第0层 GDN `out_proj` 输出看到 batch 差异，但缺少投影输入，不能认定 GEMM 是根因。本项补齐实际输入/输出，再用同一个真实投影重放完全相同的目标输入。HF 是机制参考，不是 trainer。

2026-09-07 查阅官方 main 模型源码、模型文档与 issue #46190；源码在 GDN 核心、门控归一化之后调用 `out_proj`。官方已有 Hub GDN kernel 入口，但切换后端属于另一个问题，本项保持 B03 路径。#46190 对比缓存解码与整段前向，不直接覆盖本项无缓存、只增加邻居的条件。检索和版本边界见 [research.json](research.json)。

## 设置与判据

保持 B03 的 Qwen3.6-35B-A3B、原始两条64-token输入、BF16、SDPA、无cache、无LoRA；安装模型源码必须精确等于 B03 SHA `40da264c51fcfadd7b87271c5485f029ee9efd05e6e54b34728c8ee252db4c9c`，否则停止重新登记。父入口负责相同模型权重/输入/镜像身份、GPU门禁和资源登记。

1. CPU先运行纯函数测试，再执行共享观测/重放实现的 tiny CPU 正对照，输出 `gdn-cpu.json`，必须 `ok=true`、CUDA未初始化。正对照明确给batch输入增加0.25，输入边界必须量到非零差。
2. 1×B200，单次GPU硬时限30分钟；CPU 4核/16GiB最长20分钟。每shape先3次OFF、再3次ON，全模型共12次前向；记录实际投影输入及输出的目标序列差、同shape重复差和完整logits哈希。OFF/ON哈希必须相等，缺少hook、错shape、非有限值立即停止。
3. 去除全部hook后，同一真实 `out_proj` 各重放3次：原始单条输入、单条原始目标加真实邻居的双条输入、原始双条输入。前两臂目标输入必须逐元素相同；自然输入重放输出必须与捕获值完全相等，否则本次重放不足以归因。对齐要求连续的三维张量，单个输入或输出上限16MiB。
4. `target_input_difference` 回答进入投影前是否已不同；`same_input_batch_difference` 回答输入完全相同后投影是否仍随batch形状变化。两者可以同时非零，不能强行二选一。自然双条与替换目标双条对拍单列，不能把其差异与shape差线性相加解释误差。
5. OFF全模型输出保存首条63个teacher-forced位置的argmax token ID、top2原始logit差与实际下一token的logprob；`top1_batch_difference`逐位置比较。并非自由生成，不作跨引擎或完整回答变化结论。并列最大值按argmax首索引选择，margin=0明确保留。

执行：`python -m syncopate.infra.gdn_boundary_probe --cpu --out <fresh-dir>`；GPU：`python -m syncopate.infra.gdn_boundary_probe --input <input.json> --out <fresh-dir>`。结果 `gdn-result.json`，过程 `gdn-partial.json`，首轮两种shape完整投影输入/输出 `gdn-captures.pt`；读取 `records[3]` 两种边界差、`replay_records`、`top1_batch_difference`。最终必须有6条全模型、3条重放记录与身份闸通过。同shape重复噪声如实报告，不放宽等价闸。

原始证据只写 `syncopate-home:/_audit/infra-probes/B05/<run-id>/<attempt>/`，每次独立attempt，不覆盖旧产物。遇未解决错误按用户本次裁定以连续20分钟墙钟计时后暂停汇报，替代旧两次错误门限；不通过扩大GPU预算或绕过身份闸解决。父任务负责精确run/镜像/命令与费用索引，不在本项启动GPU。

## 测试结果

| 检查 | 结果 |
|---|---|
| 本机纯函数测试与CLI导入 | `tests/infra/test_gdn_boundary_probe.py` 2 passed；无torch可导入/显示CLI |
| 目标CPU共享观测正对照 | [CPU记录](../B04-FP8概率误差/runs/batch3-cpu-01.json) 通过，CUDA未初始化，已知输入漂移被观测到 |
| B200原模型/投影重放 | [运行记录](runs/batch3-gdn-01.json) 248.06秒；6组OFF/ON完整logits一致，3组投影重放通过 |

## 结论与下一步

[数值摘要](summary.json)：投影目标输入跨batch逐元素相同；输出最大绝对差 `0.000244140625`，相对L2 `5.329844e-5`。完全相同输入送入同一真实BF16 Linear，只改变batch形状，三次都复现相同差值；自然重放与原捕获逐元素相同。同shape重复噪声为0。

因此本次最早可见差可定位到第0层输出投影的形状敏感计算，无需GDN核心输入漂移即可复现。尚未确定具体CUDA kernel/归约算法，也不能称为上游bug或概括其他层。63个固定teacher-forced位置的argmax变化为0；单条臂仍有一个top2 margin=0位置，不能说置信度或分布不变，更不是自由生成一致性证明。

独立复核了封印JSON、实际源码、OFF/ON、全部top1数组与重放结果。小型输入/输出tensor为2,361,807字节，留Volume供下一步最小算子复现；未另行下载张量重算。无checkpoint，无upstream移交。后续具体kernel/精度对照另列Q19，本项关项。
