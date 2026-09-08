# infra-probes/B03 · BF16 batch 敏感性的首个分歧层

状态：已结论，Q14关项。任务：Q14；日期：2026-09-07。

## 问题与调查

B01 普通 HF BF16 前向在固定 shape 重复时相同，增加一个邻居后 max Δlogp=0.127482。需要定位首次分歧，区分计算顺序、GDN 与 MoE 放大；不预设框架 bug。HF 在本项只作机制参考。

## 设置与判据

沿用 B01 精确公开模型 revision 与原始 2×64 token 输入；无项目训练/LoRA。单 B200，单次 GPU 硬时限 30 分钟；CPU 4 核/16 GiB 最长 20 分钟。每个 shape 重复三次，保存逐 token 原始概率、输入身份和实际执行路径。属于数值诊断，不宣称性能基线或学习收益。入口为 `modal_app/probability_batch.py --phase layers`；模块边界依官方模型源码，实际安装源码 SHA 留证，检索见 [research.json](research.json)。每个 shape 3 次 OFF 与 3 次 ON，共12次前向；保存完整 logits 哈希验证观察器未改变结果，activation 参考缓存上限512MiB。

观察器 OFF/ON 最终概率必须一致；保存层级差、同 shape 重复噪声与首个分歧，无法定位时明确未定位。

输入/权重错配、非有限值、程序异常立即停该次；实际程序错误第二轮仍失败暂停汇报。外部取消确认旧写者退出后用新 attempt 继续，不覆盖证据。用户已批准本批；预计每项成功路径不超过单卡 30 分钟，实际重复费用按每次时长累计，不把旧批预算移来。

## 测试结果

| run-id | 结果 | 函数秒 |
|---|---|---:|
| [batch2-layers-01](runs/batch2-layers-01.json) | 1×B200；每形状3次OFF、3次ON，完整logits全部相同；6次观测结果完整，SHA回读一致 | 140.78 |

原始结果 `syncopate-home:/_audit/infra-probes/B03/batch2-layers-01/attempt-cfaf2c0e9fa24b55aab2d9e8e43f20f2/`。安装模型源码SHA `40da264c51fcfadd7b87271c5485f029ee9efd05e6e54b34728c8ee252db4c9c`；精确来源/镜像/命令见run索引。

## 结论与下一步

**最早可观察的差异位于第0层 GDN `linear_attn.out_proj` 的输出。** 此前观测到的输入归一化与四个输入投影输出均相同；该输出最大差0.000244140625、relative L2=0.0000532984。后续层和路由出现差异，最终63个位置的max Δlogp=0.149146、mean=0.017295、P95=0.096732。同shape三次重复的690个已对齐模块边界全部相同。

第0层专家选择相同；第1层5个token的有序expert IDs变化，但仅1个token改变专家集合。后续最多44/64个token改变集合，最后一层29/64。排序与集合变化分别计算，不能把排序差当成选错专家。摘要见 [summary.json](summary.json)。

定位边界：60个输出因无法验证token轴未纳入对齐，这不是测试skip；功能式GDN内部算子没有逐点观测，也没有 `out_proj` 输入pre-hook。因此这里只能说“最早观察到的输出差”，不能断言out_proj GEMM是根因，或把MoE后续变化直接叫bug。观察器OFF/ON完整logits相等支持本次观测没有改变结果。

本项完成存在性与模块边界定位，不做性能/学习结论；下一步拆成Q17快速路径可用性、Q18该投影输入/输出最小对拍，见 [TASKS](../../01-TASKS.md)，本批不继续扩做。CPU参考缓存峰值约151MiB，仅进程内存在；未创建checkpoint。原始JSON/日志/源码留Volume，未移交upstream。

收尾：[本批存储盘点](../../storage/batch2-retention.json)、[自有App退出复核](../../storage/batch2-apps-closed.json)。
