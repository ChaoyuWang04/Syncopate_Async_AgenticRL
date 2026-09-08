# 提交包 · 只为 attention 启用 LoRA 时，MoE 初始化缺少 context（infra-probes/B08）

> **状态：DRAFT。单B200真实启动失败已保存；官方默认wrapper配置已经通过完整五阶段交接终态；原过滤器组合仍保留真实启动失败见证。未修改上游包，未完成正式排重或提交。**
> 目标仓库：vllm-project/vllm。
> 类型：疑似 bug report；尚无上游补丁。
> 固定版本：vLLM 0.28.0，release commit `2cf0a6915ce544dc493a0990f2ea38d81601128a`；目标镜像固定torch2.13、Transformers5.10.4，锁及实际源码随运行归档。安装包不能只凭tag等同源码，CPU前置另存实装文件SHA。

## 一句话

公开Qwen3.6-35B-A3B在`enable_lora=True, lora_target_modules=['o_proj']`下，初始化profile_run即进入MoE LoRA核并报`AssertionError: LoRA context must be set`；尚未加载任何业务adapter，也没有进入A/B生成。只想给普通attention投影装LoRA仍能触发专家路径错误。

## 触发条件（最小配置）

下列是实际失败调用的核心配置摘录，**不是已经独立实跑的最小复现脚本**；正式独立复现仍需upstream负责人从归档脚本缩减并实际执行。

```yaml
model: Qwen/Qwen3.6-35B-A3B
revision: 995ad96eacd98c81ed38be0c5b274b04031597b0
dtype: bfloat16
tensor_parallel_size: 1
enable_lora: true
lora_target_modules: [o_proj]
max_loras: 1
max_cpu_loras: 1
max_lora_rank: 8
enforce_eager: true
enable_prefix_caching: false
max_model_len: 256
max_num_seqs: 1
max_num_batched_tokens: 256
gpu_memory_utilization: 0.70
limit_mm_per_prompt: {image: 0, video: 0}
```

真实硬件为1×B200。参数过滤是官方文档提供的能力；使用场景是MoE底座上仅训练/部署attention LoRA，限制额外wrapper和buffer。错误发生在profile_run，因此adapter文件内容、整数ID复用、LoRA名称和prefix cache重载均不是本次栈的必要触发因素。尚未在更小模型或其他卡型复现，不能声称所有MoE都会失败。

## 根因（带行号与版本）

以下行号对应v0.28.0官方源码，源文件指纹和检索在[B08 research](../../infra_exp/experiments/B08-LoRA加载与恢复/research.json)。

| 位置 | 读码事实 |
|---|---|
| `vllm/model_executor/layers/fused_moe/layer.py`，约L354 | `is_lora_enabled=vllm_config.lora_config is not None`，按全局配置判定，没有在这里检查experts是否被目标过滤器排除。 |
| `.../fused_moe/oracle/unquantized.py::select_unquantized_moe_backend`，L222–236 | 全局LoRA分支先检查支持条件并选择`TrtLlmBf16LoRAExperts`；该return早于L286之后的显式`moe_backend`处理。因此简单指定triton不能保证覆盖此分支。 |
| `vllm/lora/model_manager.py::_create_lora_modules`，L421；`_match_target_modules`，L704–725 | 被`target_modules`排除的模块直接跳过，不安装MoE LoRA wrapper。 |
| `vllm/lora/layers/fused_moe.py::FusedMoEWithLoRA.set_mapping`，L446–456 | wrapper构造`MoELoRAContext`并调用专家核的`set_lora_context`。 |
| `.../fused_moe/experts/trtllm_lora_moe.py::apply`，L246–247 | 在profile真实执行中读取`self._lora_context`并断言非None；本次就在此失败。 |

[推断，需独立对照确认] 全局后端选择与逐模块过滤的粒度不一致：没有安装专家wrapper，却选中依赖该wrapper设置context的专家核。错误栈、分支源码和过滤语义相互一致；默认wrapper复验用于进一步确认这条因果链，尚不能用读码代替修后真实行为。

## 判据（怎么知道它错了）

```text
原配置：LLM构造中的profile_run报 LoRA context must be set；handoff.json.ok=false，phases=[]。
官方默认wrapper配置：真实GPU初始化完成，两次base请求均返回token13914；原初始化断言不再出现。
后续adapter交接：本地generic RPC错误修正为官方typed API后，最终五阶段全部通过；这不属于本包context根因。
```

修后最低行为门槛是初始化完成并实际生成；B08进一步要求发送/接收adapter矩阵和GPU槽位相等、实际模块被调用、同输入A/B有可测差且A重载恢复。最终GPU复验已核对三个adapter阶段各40个专家wrapper的context存在且活动槽位全零，没有为规避错误增加专家adapter权重。完整门槛只有[B08 REPORT](../../infra_exp/experiments/B08-LoRA加载与恢复/REPORT.md)维护。

## 证据（指到具体产物）

Volume名称为`syncopate-home`。失败attempt的容器路径：

```text
/vol/_audit/infra-probes/B08/batch4-lora-gpu-01/attempt-19692aada6a4482ba9247dd71db1d4cd/
  step-4.log      EngineCore profile_run到trtllm_lora_moe.py:247的完整调用栈
  handoff.json    ok=false、phases为空、异常终态
  run.json        镜像、命令、来源前置、源码归档身份、函数墙钟
  source.tar.gz  该attempt原始代码快照，不能用后续工作树冒充
```

控制机已回读三份小文件：`/tmp/batch4-readback/lora01/{step-4.log,handoff.json,run.json}`；临时路径不是持久来源。

原始overlay SHA256为`ebfe1ede439d0dc34ef2d082eaf4b9124f176011b757e1c2455b3ab91c877943`；源码归档SHA256为`e4afcccd0ff16d27193196330ce6647fce4b98907c9b6dda598c641e7ea5d9ca`。镜像`im-J5jRGJOILryp9VikpHuQj8`。复现入口为归档中的`syncopate/infra/lora_handoff_probe.py::gpu`，依赖本仓库身份检查和冻结vLLM环境；现有脚本不是可脱离本仓库的正式上游最小复现。

## 官方配置规避与尚未实施的上游修法

未修改任何上游库。限定配置规避为撤掉部署`lora_target_modules=['o_proj']`，采用官方默认“所有受支持模块可装wrapper”；不可变A/B产物仍只有一个普通attention的`o_proj`。真实GPU已证实该配置可避开初始化错误，并通过全部预注册adapter交接判据。

目标CPU新增源码契约检查：对安装版`LoRAModelManager._match_target_modules`，旧过滤器应排除experts，默认None应包含experts；并记录实际oracle顺序和context设置代码SHA。这是局部机制验证，不能冒充修后初始化通过。

[推断，未实施] 上游最终修法可能应让每层`is_lora_enabled`与实际目标过滤相符，或让没有expert LoRA的后端选择普通专家核；需维护者确认MoE映射/打包模块语义。不能靠伪造非空context跳过断言，也不能在没有数值回归时直接删除断言。

## 影响面

已实测范围只包含本模型、BF16、B200/TP1和上述过滤器。结果是显式启动失败，不是静默权重错误。原失败批次未生成；最终官方默认配置已有生成和交接证据，但无训练更新或学习质量结论；投入和各次运行结果引用REPORT/运行索引，不把初始化墙钟当推理性能。普通dense、其他MoE backend、量化、TP/EP及其他负载的外推均未验证。

## 我们验过什么 / 没验过什么

- 已验证：原配置真实GPU失败；完整栈发生在adapter加载之前；官方v0.28调用链；当前读取main中oracle仍有同样分支顺序。
- 已验证：官方默认wrapper配置和typed API通过完整目标CPU及五阶段GPU终态；最终[B08 GPU3索引](../../infra_exp/experiments/B08-LoRA加载与恢复/runs/batch4-lora-gpu-03.json)保存结果SHA及独立回读检查。接收矩阵/GPU槽位逐元素相等，三个adapter阶段均有40个专家context且活动槽位全零，A与base卸载恢复差均为0。
- 未验证：独立最小上游复现、上游main GPU复现、修正后端选择的代码补丁、其他模型/硬件、完整正式排重与贡献流程。

## 开工背景调查（upstream 接手后重查）

调查细目只认[research.json](../../infra_exp/experiments/B08-LoRA加载与恢复/research.json)。初始live main为`51da0ca66c8065619c79e35dff97aa99aeaf5644`，稳定tag为上述`2cf0a691...`；本次追加读取main oracle仍见相同分支，未将其等同GPU重现。

追加API检索词为`repo:vllm-project/vllm "LoRA context"`、`repo:vllm-project/vllm lora_target_modules moe`和`repo:vllm-project/vllm trtllm lora context`，命中中没有识别出覆盖这个具体过滤器/专家context组合的适用已修方案。此前#42125/#48352是同名prefix-cache重载，#49525是多模态wrapper文本键映射，#53842是lm_head prompt-logprob分块；均不解释此次启动栈。#31452/#34984涉及target-modules功能本身，正式接手应继续核对其设计讨论和回归测试，不以本轮搜索代替正式排重。

本包只交付实验发现，不创建issue/PR、不联系维护者、不声明upstream负责人已接收。
