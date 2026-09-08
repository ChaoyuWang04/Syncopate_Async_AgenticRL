> **2026-09-07 历史范围说明：**本文保留当时的设置、结果、授权和未完成项；正文中的“当前/下一步”、旧定位及资源限制只属于当时，不构成新实验排期或运行授权。原业务造数和完整学习计划已停止，原始证据与运行身份不倒写。 当前方向与任务只认 [Infra TASKS](../../../docs/infra_exp/01-TASKS.md)。

# B04 · SFT 双卡 DP 与双卡 TP

**前置**：B03；使用同一镜像、数据和 SFT 起点。当前训练器只实现了数据并行，固定 runbook 仍是 1×B200；文档调整不表示 TP 已经接通，也不表示默认值已经改完。

**问题**：同样占用 2×B200 时，是两张卡各跑一份模型、各吃不同样本的 `DP=2` 更快，还是两张卡共同切一份模型的 `TP=2` 能靠更大的 micro-batch 更快？用户已接受既有 DDP 扩展证据，不再做 1 卡与 2 卡 DP 的性能 A/B；B04 以 `DP=2` 为 reference。

**阶段 0 · 先证明 TP 值得接线**：先只读核对当前 Transformers、模型、PEFT 和 PyTorch 是否提供训练可用的官方 TP 路径。现有 SFT 会绕过模型根 `forward` 做稀疏词表投影，并手动归约 LoRA 梯度；TP 必须证明分片后的前向、反向、optimizer、adapter 保存/合并和现有 loss mask 都接在真实路径上。若官方路径不支持，或接线成本明显超过可预期收益，结论记为“不适用”，不为做实验临时手搓一套 TP。

**阶段 1 · 只比较并行方式**：`DP=2` 对 `TP=2` 使用相同模型、LoRA、数据顺序、seed、精度、有效 batch、每次更新的输入/监督 token、更新数和评测。两臂先固定相同的 global micro-batch；DP 把样本分给两个副本，TP 让两个 rank 共同处理同一批样本，梯度累积按同一个有效 batch 派生。这样测到的差异才主要来自并行方式。

**阶段 2 · 比较各自最佳实用配置**：两臂分别寻找不 OOM、数值稳定的最大 global micro-batch，但仍固定同一个有效 batch、每次更新 token 和总工作量；micro-batch 变大时同步减少梯度累积。若 TP 只在更大 micro-batch 下获胜，报告必须写成“TP 释放显存后带来的 batch 收益”，不能写成纯 TP 加速。不得用不同 global effective batch 做速度结论。

**正确性门槛**：DP 先通过当前栈的跨 rank 梯度/权重一致性；TP 再用 B03 噪声带对拍 token、loss、监督位置梯度、可训练参数与合并后模型输出，并证明没有整层意外复制、漏分片或错误重复计算。两臂的 adapter 都必须能保存、加载、合并并进入同一评测；故意改错 shard、mask 或 adapter 的负对照必须失败。

**性能与完成条件**：在两组独立 2×B200 容器中并行收集，每组记录拓扑并交叉运行两臂，至少取得三个预热后窗口，比较中位 step time、监督 tokens/s、峰值显存、GPU busy、通信占比、端到端墙钟和每固定工作量 GPU-hours，同时检查 loss、梯度和冻结任务质量。运行前用 B03 离散度预注册胜出线；差异落在噪声带内就算没有胜者。TP 只有在正确性全过、端到端显著更快且成本/质量不退化时才晋级，否则保留 `DP=2`。当前 runbook 从 1×B200 改为 `DP=2` 是单独的落地动作，至少先过一次当前源码的双卡正确性 smoke，不再为它另做 1×/2×速度实验。


## 当前执行状态

GPU 尚未运行。B03 前置期间完成了下面的只读兼容性检查；还没有 TP 模型反向证据。

### 官方路径与本地前置

[Transformers 官方训练文档](https://huggingface.co/docs/transformers/tensor_parallelism) 已提供 `tp_plan=auto` 的训练路径；[PEFT 0.20 文档](https://huggingface.co/docs/peft/v0.20.0/package_reference/lora#tensor-parallelism) 说明 LoRA TP 需要 Transformers ≥5.4，并能聚合 adapter 保存。当前版本满足该版本前提，但不等于 Qwen3.6 混合注意力、专家模块与自有 sparse-forward 已经兼容。下一步检查本模型实际 plan，再做微型 CPU/双卡前后向，不手搓完整 TP。

修复前，本仓库 `sft.py` 先对每个 micro-batch 的监督 token 求均值，再除以固定梯度累积数。读码可确认：监督长度不同或改变 micro-batch 切法时，token 的相对权重会变化；因此“有效 batch 相同”还不足以隔离 DP/TP 的纯性能差异。例子：两批 loss 分别为 `[1,1]` 和 `[3]`，均值的均值为 2，合并 token 后均值为 5/3。这是数学示例，不是模型实测。

此问题属于自有训练循环，不是上游 bug。正式 A/B 前必须用不同长度的小张量做梯度累积/跨 rank 对拍，明确统一每次更新的监督 token 分母；同时检查 epoch 最后不足一个累积窗口的归一化。没有重建数据，也没有修改本轮正在验证的 RL 源码。

CPU 前置批次 `b04_batch_weight_20260905a`：一容器 4 CPU/16 GiB，硬超时 10 分钟，预留 $1。用随机微型 Qwen 和 2 条人工张量（不是新增训练数据），调用真实 `token_losses`；比较整批梯度、现行逐 micro-batch 均值梯度、统一监督 token 分母梯度。统一分母相对 L2 误差须 ≤1e-4；现行分母若 >1e-3，确认切批改变目标。其余结果记无结论。只做 CPU，不改正式训练器或 B03 在跑源码。

初次微复现确认上述问题。整批梯度 L2=4.7855449；旧累积分母的相对梯度差=0.2094818；统一 token 分母的原型后=1.4714e-7。计算段 0.033 秒，不含容器和导入，不能作为性能数字。复现脚本为本目录 `batch_weight_probe.py`；机器结果在 `/vol/_audit/infra/B04/b04_batch_weight_20260905a/result.json`，Mac 汇总为 `_audit/stack_probe/summary_2026-09-05_202200_exec_e9e5909a.json`。当时还未改生产循环；正式修复的独立结果见下节，不混用原型数值。

接着用同一 CPU 预算做 `b04_tp_plan_20260905a`：只读当前安装的模型类和官方 TP plan，模型放在 meta 设备，不加载权重或申请 GPU。检查混合注意力和专家层是否有明确分片定义；这只证明计划存在，不能充当 TP 前向、反向或 LoRA 保存通过。结果落 `/vol/_audit/infra/B04/b04_tp_plan_20260905a/result.json`。不修改 B03 已冻结的源码。

结果：实际类为 `Qwen3_5MoeForCausalLM`，Transformers 5.10.4 / PEFT 0.20.0。模型的 `_tp_plan` 已覆盖普通 attention、路由专家和共享专家；30 层 linear attention 没有出现在该 plan 中，不能宣称所有层都分片，也不能据此直接断言有 bug。下一步需验证这些复制层的梯度语义与实际显存。`config.base_model_tp_plan` 为 null，但模型级 `_tp_plan` 非空，不能只查 config 的一个属性就误判不支持。CPU meta 检查没有权重、前向或反向，仍不算 TP 训练通过。证据为 `tp_plan_probe.py` 与 `_audit/stack_probe/summary_2026-09-05_203748_exec_72846fa8.json`。

## 梯度加权修复验收

已修改自有 SFT：累积 token loss 总和，在一次更新前用全局有效 token 数归一化，再 clip/更新；尾窗口使用实际分母。空本地批次仍参加窗口末尾的集合通信，只有全局零 token 才拒绝更新。日志同步改成全局 token 均值与吞吐。epoch 末比较全部可训练参数字节指纹，拒绝“范数一样但参数不同”的假通过。没有改数据、训练预算或采样。

批次 `b04_weight_fix_20260905a`：沿用 B04 的 CPU 预算，4 CPU/16 GiB，测试进程 600 秒上限；无 GPU。本机做语法和入口顺序检查，Torch 数值测试在 Modal CPU 补齐。判据预先固定：真实 sparse-forward 与整批参考相对梯度 L2 ≤1e-4；双 rank CPU 归约与 float64 整批参考误差 ≤1e-12；含短尾窗口的两次更新相等；零 token/非有限 loss 被拒绝；故意交换 rank 1 两个权重、保留范数时必须被完整指纹检查拒绝。结果写 `/vol/_audit/infra/B04/b04_weight_fix_20260905a/`。这不代替 NCCL、BF16 或全尺寸 B200 SFT 验收。

B03 c 批仍运行其不可变源码，不混入这次 SFT 修复；Mac 已另外保存 c 批完整源码副本并核对 overlay SHA，随后 RL adapter 导出仍使用原副本。

验收结果：Modal CPU 40 passed，1 条 pytest XML 格式警告；没有测试失败。真实生产帮助函数对独立整批参考的梯度相对 L2 误差为 `3.7788342410749465e-7`，低于预注册的 `1e-4`。两个 CPU rank 在不等 token 数及一个空 rank 的条件下均与 float64 整批参考通过 `atol/rtol=1e-12` 对拍；尾窗口两次更新、零/非有限 token 目标拒绝、保留范数但篡改参数的负例均通过。

App `ap-Wu8ffRns8lTlYnBWpNCm1J`；修复验收 overlay `1da8818bee13da53f9ee3accf346817493da48d6b041b1c95c081bec0b9694c9`。日志/XML 在 `/vol/_audit/infra/B04/b04_weight_fix_20260905a/`；本地为 `_audit/stack_probe/summary_2026-09-05_211459_exec_2dbaa4df.json`，误差 XML 属性由 `summary_2026-09-05_212237_exec_3c7a5d3e.json` 独立读回。测试总耗时 429.77 秒，不是 SFT 性能数据。

这一项只关闭 CPU 数值前置。全尺寸 BF16、B200 双卡 NCCL、官方 TP 前后向与 adapter 保存/合并仍未验收；B04 不能关闭，也还没有 DP/TP 加速结论。

## 继续核对实际 TP 接口

当前 Transformers TP 使用局部普通参数和模块 hook，不能只看 DTensor 的 placements 判断是否分片。现有模型实例的 plan 也不能由类属性猜测，尤其是 GDN 和输出层。PEFT 的分片加载、LoRA hook 与 adapter 聚合保存必须读取安装版源码，再写最小探针，不能让两 rank 同时覆盖同一文件。

登记 `tp_sources_20260906a`：4 CPU / 16 GiB、最多 5 分钟，无 GPU。只导入安装版接口并读取源码，记录实际对象、签名、文件 SHA 和版本；不读模型权重或数据。先本机 AST 检查，源文件归档到 `/vol/_audit/infra/B04/tp_sources_20260906a/`。关键对象缺失直接报告，不能替换成猜测路径。另预留 $1，批组累计保守预留 **$118**，不是实际账单；与 B03 c 可并行，独立写者和目录。

结果通过：PyTorch 2.13.0、Transformers 5.10.4、PEFT 0.20.0，CUDA 未初始化。实际取回 13 个源文件，包含 Transformers 的 tensor parallel、模型加载和 PEFT 集成，以及 PEFT 的 LoRA 层、TP 层、state dict 分片与保存/加载入口。归档 SHA256 为 `ad9c9dc79ef1f78103d7b6e6cd2bf9f09107aaf0ebb9305ba244d2d543d24243`；汇总为 `_audit/stack_probe/summary_2026-09-06_111543_exec_534aed1f.json`，Volume 目录为 `/vol/_audit/infra/B04/tp_sources_20260906a/`。App `ap-QJndOJliMVxARy6VI5bzQZ` 已停止、tasks=0。

只读结果说明当前 TP 通过普通局部参数和模块 hook 工作，不能只靠 `.placements` 判断分片。模型实例的实际 plan 也不能由类属性猜测，GDN 和 `lm_head` 等未列项要按复制层核对。PEFT 已有 `_maybe_shard_state_dict_for_tp` 等真实入口，但“入口存在”仍不代表 Qwen3.5 MoE、LoRA 和本项目稀疏监督前向已经正确。下一步是用小模型做 DP=2/TP=2 前向、反向、optimizer、adapter 保存/加载/合并的正确性探针；正确性通过前不开始性能 A/B。
