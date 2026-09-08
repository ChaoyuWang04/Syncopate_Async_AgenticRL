# infra-probes/B08 · LoRA 文件交接与 vLLM 实际消费

状态：**Q04已完成并可结项**；G7.2/G3.4。单B200/TP1/BF16、关闭prefix cache、仅普通attention `o_proj` adapter的加载→卸载→重载闭环通过。最初的部署过滤器组合存在初始化问题，已留[DRAFT](../../../upstream/DRAFT-infra-probes-B08-lora-moe-context/1-finding.md)；正式上游工作不占本实验队列。

## 问题与调查结论

RL更新后，引擎必须实际消费新adapter，不能只验加载返回。vLLM0.28官方提供LoRA加载/卸载与Qwen3.6支持，本项用确定性非零adapter确认其明确范围，不训练模型或预判新bug。稳定commit `2cf0a6915ce544dc493a0990f2ea38d81601128a`、main检索身份和实装源码指纹见[research.json](research.json)及运行索引。冻结torch2.13/Transformers5.10.4；后者受verl<5.11约束，不冒称最新。

已知#42125/#48352涉及同名重载的陈旧prefix cache，本项采用关闭cache的官方规避；#49525涉及裸`model.layers`键映射，本项使用原checkpoint完整前缀并验实际矩阵；#53842涉及lm_head分块，非本项o_proj路径。不对这些已知问题重复立项。

## 最终设置与预注册判据

只读公开`Qwen/Qwen3.6-35B-A3B`，revision `995ad96eacd98c81ed38be0c5b274b04031597b0`。CPU/GPU均复核B01权重清单；input.json SHA `cf3fd1e9856ac560c441bfd42c96e573b47d3d8740d4330091482ac278636e28`。第一条64-token输入原样消费，每请求生成1 token。两个合法rank8/alpha8 adapter只含第39层普通attention `o_proj`：共用随机LoRA-A矩阵，LoRA-B分别取同一非零矩阵及其负值；不是训练产物。

固定入口`python -m syncopate.infra.lora_handoff_probe cpu <out> <input.json>`及`gpu <out> <CPU目录>`，调度器绑定相同源码。CPU实际验证safetensors、PEFTHelper、模型mapper/LoRAModel加载、typed IPC及专家buffer形状；CPU-only fixture仅临时关闭loader的锁页内存分配标志并恢复，GPU代码不受影响。

GPU通过官方`llm.llm_engine.add_lora/remove_lora/list_loras`执行base→A→卸载→B→卸载→A_reload→卸载→base_restored，每阶段同形重复2次。使用官方默认支持模块wrappers，artifact仍只含o_proj；TP1、max_loras=max_cpu_loras=1、eager诊断、prefix cache关闭。每次模型root须出现相同64 token/位置；加载后接收矩阵和GPU槽位有效区域与文件逐元素相等，真实projection hook命中且有限；全部40个专家wrapper有context且活动槽位LoRA矩阵为零。

预注册终态：63项概率有限且对齐；同阶段重复max|Δlogp|≤1e-5；A/B和base/A差均>max(1e-4,10×重复噪声)；重载A恢复及卸载后base恢复差≤1e-5；卸载实际列表不含ID1。判据未放宽。相同整数ID1复用，不可变文件目录/SHA区分版本。

预算为每次1×B200/16核/128GiB、≤30分钟，CPU4核/16GiB；批次授权及实际调度以TASKS/索引为准。证据独占写`syncopate-home:/_audit/infra-probes/B08/<run>/<attempt>/`。

## 运行结果

| 运行索引 | 结果 | 函数墙钟秒 |
|---|---|---:|
| [batch4-lora-cpu-01](runs/batch4-lora-cpu-01.json) | CPU身份与adapter构造通过 | 219.34 |
| [batch4-lora-gpu-01](runs/batch4-lora-gpu-01.json) | 初始化context错误，尚无adapter阶段 | 456.84 |
| [batch4-lora-cpu-02](runs/batch4-lora-cpu-02.json) | CPU过滤器机制检查通过 | 472.70 |
| [batch4-lora-gpu-02](runs/batch4-lora-gpu-02.json) | 初始化及两次base通过；本地泛型RPC类型错误 | 361.30 |
| [batch4-lora-cpu-03](runs/batch4-lora-cpu-03.json) | CPU wire检查后，loader锁页分配因无NVIDIA驱动失败 | 233.02 |
| [batch4-lora-cpu-04](runs/batch4-lora-cpu-04.json) | 完整目标CPU检查及结果SHA回读通过 | 222.29 |
| [batch4-lora-gpu-03](runs/batch4-lora-gpu-03.json) | 五阶段终态通过，结果SHA回读一致 | 361.73 |

首次错误是全局LoRA专家核选择与逐模块过滤不一致，撤掉部署过滤器后真实初始化通过；源代码因果链只在DRAFT展开。随后泛型RPC将嵌套LoRARequest解码为list，属于本探针API误用，已改为官方typed utility并增加真实wire正/负对照；不为此创建上游问题。CPU3锁页分配错误以CPU fixture作用域修正，不申请GPU充当CPU前置。失败attempt和源码均保留。

最终[GPU3](runs/batch4-lora-gpu-03.json)五阶段各2次，读数：

| 判据 | max绝对Δlogp |
|---|---:|
| 同阶段重复 | 0 |
| A/B | 1.5623478889465332 |
| base/A | 0.8537850379943848 |
| A/重载A | 0 |
| base/卸载恢复base | 0 |

三次adapter阶段均在PID218读取实际接收矩阵和GPU槽位，40个专家wrapper的context/全零活动槽位均通过；真实模块为`language_model.model.layers.39.self_attn.o_proj`。A及重载A权重SHA相同，B不同，完整指纹见索引。独立回读核对10条root的64-token/三轴位置完全相同；8条projection记录中6条槽位ID1、2条恢复base槽位None，输出均有限。CPU4/GPU3必需结果JSON与调度器登记SHA一致；trace另记本机回读指纹，不冒充调度器封存JSON。

## 结论与边界

这证明当前范围内“文件版本→接收矩阵→GPU槽位→后续计算→卸载恢复”成立，Q04没有遗留执行动作。未覆盖GDN/专家adapter、训练更新、HF合并权重数值对拍、cache开启、并发在线请求、sleep/wake、TP/EP或长期稳定性。

打包、同机文件复制、卸载、加载及身份检查、请求等待分别记录；等待含计算，观测含额外拷贝，不是纯加载延迟、网络传输或干净性能重复。所有函数墙钟均非模型性能，无速度/学习收益声明。

未创建训练checkpoint；小adapter、输入、矩阵身份与trace留Volume供DRAFT/复验，共享模型只读。自有App退出与保留产物盘点已由[批次收尾](../../storage/batch4-retention.json)核对；不重复维护第二份资源账。
