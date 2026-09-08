# infra-probes/B04 · FP8 rollout 的增量概率误差

状态：已结项；目标CPU、单B200 FP8和同源码BF16对照完成。任务：Q16；日期：2026-09-07。

## 问题与调查

真实 RL 使用低精度 rollout 时，需要量到相对 BF16 rollout 的额外误差，而不是把所有跨引擎差归因于量化。固定训练侧 BF16，先调查本模型在当前 vLLM/B200 的官方 FP8 支持。

## 设置与判据

沿用 B01 精确公开模型 revision 与原始 2×64 token 输入；无项目训练/LoRA。单 B200，单次 GPU 硬时限 30 分钟；CPU 4 核/16 GiB 最长 20 分钟。每个 shape 重复三次，保存逐 token 原始概率、输入身份和实际执行路径。属于数值诊断，不宣称性能基线或学习收益。配置冻结为 vLLM0.28 `fp8_per_block`，`quantization_config.linear={weight:null,activation:null}`，dtype BF16/KV auto；仅 MoE 专家使用 online block FP8，不代表全模型 FP8。已有 per-tensor GDN 风险 #41022 不作为本项新发现；调查与版本 SHA 见 [research.json](research.json)。入口 `modal_app/probability_batch.py --phase fp8` 已按用户恢复授权重新启用；本轮CPU gate必须包含fp8-cpu.json。

BF16/FP8 使用相同原始权重、token、batch、采样参数、KV dtype；仅启用官方支持的 MoE-only block FP8，权重128×128/激活1×128 scale；shared expert 的融合归属以实际模块清单为准。记录实际量化模块、scale 与排除项；未命中 FP8 或输入不一致不能通过。

输入/权重错配、非有限值、程序异常立即停该次；同一实际程序错误持续排查20分钟仍未解决暂停汇报，正常运行遵守原硬超时。外部取消确认旧写者退出后用新 attempt 继续，不覆盖证据。用户已批准本批；预计每项成功路径不超过单卡 30 分钟，实际重复费用按每次时长累计，不把旧批预算移来。

## 测试结果

首轮 [CPU](runs/batch2-cpu-01.json)：权重与 FSDP2 接口通过，FP8 helper 错误要求显式 activation 非空；官方 online 配置由方法内部选择激活格式，activation 应为 None。随后短 CPU 只读诊断 App `ap-NgyoOmSyYUpoyNRKXvLstP` 因远端入口缺少模块路径失败（ModuleNotFoundError: modal_app）。两次都是自有探针/入口错误，不是模型/上游数值故障；按用户规则暂停，不进行第三次或 GPU 尝试。第二次原始日志和实际入口已上传本实验 Volume 独立目录，见 [第二轮索引](runs/batch2-config-diagnostic-02.json)；未获得远端模型结果。无 checkpoint 产出计划；原始证据存 syncopate-home 的 `/vol/_audit/infra-probes/B04/<run-id>/<attempt-id>/`。

本轮 [CPU前置](runs/batch3-cpu-01.json) 通过；[FP8](runs/batch3-fp8-01.json) 214.41秒、[BF16对照](runs/batch3-vllm-01.json) 264.73秒，均为包含加载/身份/诊断的阶段墙钟，不作性能比较。新BF16的完整结果JSON与B01旧BF16相同。40个routed MoE模块实际使用FP8（FLASHINFER_TRTLLM / TrtLlmFp8ExpertsMonolithic），120个GDN Linear与80个shared-expert投影保持BF16；权重/scale有限，真实root分别匹配3次单条与3次双条输入。结果SHA已回读验证。

| 绝对Δlogp，63个目标位置 | 平均 | 中位数 | P90 | 最大 |
|---|---:|---:|---:|---:|
| BF16 vs FP8，单条 | 0.020333 | 0.003531 | 0.055331 | 0.180062 |
| BF16 vs FP8，双条 | 0.022866 | 0.002523 | 0.094661 | 0.203149 |
| FP8自身，单条 vs 双条 | 0.030006 | 0.002638 | 0.117905 | 0.246139 |

完整分桶、概率比、FSDP2参考见[比较索引](comparison.json)；P90使用最近秩。BF16/FP8同shape重复噪声均为0。FSDP2对rollout平均差从BF16的0.020832/0.015144变为FP8的0.023276/0.022303，但单条最大差反而变小，不能宣称逐token单调恶化。短重复文本只提供局部证据，不是189个独立样本。

## 结论与下一步

MoE-only block FP8在本模型/当前栈/B200路径真实跑通，并测得新增概率差；不代表全模型FP8、性能收益或RL退化。未保存完整词表的FP8 top1，不能推断生成token已经改变。这是官方量化配置及其实际后端的合成差异，未拆成权重舍入、激活量化或单一kernel贡献。没有新上游故障可移交。独立复核了两臂身份、量化清单、分桶与统计；补齐索引的区间边界和概率比方向。

CPU也确认训练模型 `causal_conv1d` 未安装、GDN fast_path=false；这只解释待调查线索，快速路径支持/收益进入Q17，不在本项安装依赖。旧错误均是自有入口/断言错误，本轮未复发，20分钟规则未触发。无checkpoint；本批存储及App退出见[收尾记录](../../storage/batch3-retention.json)。
