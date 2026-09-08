# infra-probes/B02 · FSDP2 LoRA 变长累积与尾窗口

状态：已结论，Q03关项；任务Q03；分组G1.2。不以B01数值结论为gate。

## 问题与调查结论

合理场景是LoRA训练的样本长度不同、最后一次更新的样本数不足一个完整累积窗口。verl旧worker的分母问题 [#5625](https://github.com/verl-project/verl/issues/5625) 已随旧路径淘汰；当前engine在每个实际batch先归约有效token，再进入microbatch。critic修复 [#6957](https://github.com/verl-project/verl/pull/6957) 已合并，不能重复称为新发现。本项检查当前官方FSDP2 engine + PEFT + 官方sft_loss的实际接线；不写自己的loss代替被测路径。

目标verl0.9.0（483b8a0）与main（2f07745）、Transformers5.10.4（因verl<5.11约束保留）、torch2.13，精确源码归档。只重用仓库已有tiny构造/张量比较工具，不沿用旧B03的结果或gate。更完整来源索引在research.json。

## 设置与判据

两层原生Qwen3.5 Text MoE，vocab128/hidden64/4experts/top2，只有full attention，FP32/eager；明确不覆盖35B、GDN、专家LoRA、BF16/TP。随机种子137；PEFT q_proj/v_proj rank4、alpha8、dropout0、无bias；基础权重冻结。SFT全局有效token均值，loss_mask标记被预测token（首token不参与），官方函数负责左移，避免混用旧探针的prediction-position mask。

两次真实更新：窗口1四样本，监督token数[3,5,4,7]，每rank2个micro；尾窗口2两个样本，token数[2,6]，每rank1个micro。实际全局分母分别19/8，不能继承上次窗口或按micro均值。两rank样本不重复，梯度同步次数由真实最后micro决定。

对照是同初始权重、相同原始token/position/mask的单GPU非分片整批HF前向+独立torch交叉熵；各rank计算一份整批参考，避免CPU/GPU kernel差混入主要比较。先CPU验证整批/逐micro数学、官方sft_loss和adapter可保存加载，并以错误的micro均值作为检测能力负对照。

梯度：逐参数allclose atol1e-6/rtol1e-4且合并relative-L2≤1e-4；AdamW lr1e-3、betas(.9,.999)、eps1e-8、weight_decay0、foreach/fused关闭，参数对拍atol2e-6/rtol1e-4。核对完整梯度、optimizer参数组/step/一二阶矩、冻结权重不变、所有rank完整参数相等；每步必须梯度非零且adapter实际改变。默认B=0时第一步A梯度可合法为0，不能把非零要求套到每个参数。独立参考与同实际梯度的AdamW参考分别记录，不能用weight decay证明更新。

CPU4核/16GiB单次15分钟，最多3次（额外CPU仅重验metadata API接线）；GPU2×B200/16CPU/64GiB单次15分钟，最多2次，累计GPU≤1卡时，实验费用上限$15。进程组90秒通信超时，整批有硬超时；NaN/错模型/必需skip停止。只测正确性，不报性能baseline，不故障注入。

## 测试结果

| run-id（点击索引） | 结果 | 函数秒数 |
|---|---|---:|
| [cpu-01](runs/b02-cpu-01.json) | 探针试图修改只读model_config而停止；改用构造新配置 | 27.96 |
| [cpu-02](runs/b02-cpu-02.json) | 3项测试零跳过；两个窗口独立/官方loss梯度对拍通过，错误micro均值均被检出，adapter重载精确一致，未初始化CUDA | 23.18 |
| [gpu-01](runs/b02-gpu-01.json) | 两rank初始化与真实root前向已执行；探针metadata读取漏传必需default，在反向前停止，零更新 | 64.31 |
| [cpu-03](runs/b02-cpu-03.json) | 增加真实metadata API检查；已落盘数学/负对照/加载通过。随后Modal重派同一ID被目录保护拒绝，CLI最终1；原通过证据未覆盖 | 27.78 |
| [gpu-02](runs/b02-gpu-02.json) | 两rank两次真实更新、梯度/AdamW/冻结权重/跨rank身份全部通过，原始输入每rank三次命中 | 26.55 |

独立代码复审补齐逐坐标梯度allclose、真实root token/position断言；官方optimizer保留冻结参数是合法表示，完整参数组保留，独立AdamW只投影可训练项，并拒绝冻结项存在optimizer状态。本机相关纯CPU测试3 passed；本轮收尾回归23 passed，9个新增Python文件AST、21个JSON与45个相关本地链接检查通过。独立复审回读两rank SHA并重算HF读数，未发现阻塞性问题。以上接线修正不改模型、loss或数值阈值。

固定入口：`PYTHONPATH=. modal run --detach modal_app/lora_probe.py --mode cpu --run-id <new-id>`；CPU通过后同入口`--mode gpu --cpu-run <passed-cpu-id>`。每批原始证据独立落`syncopate-home:/_audit/infra-probes/B02/<run-id>/`；本项不读取旧tiny checkpoint。每个已结束批次回读并验证小型证据SHA。

## 结论与下一步

本配置未发现变长累积或尾窗口归一化错误。两次实际全局分母19/8，每rank micro数2/1；梯度relative-L2分别4.23e-7、4.02e-7，最大坐标误差2.37e-8、4.47e-8，全部满足预注册阈值。同实际梯度与独立梯度两种AdamW参考均通过，optimizer step确为1/2，一二阶矩通过，冻结base不变、adapter每步实际改变、两rank完整参数/optimizer指纹一致。结果与完整日志均从Volume回读并核对SHA。

只覆盖tiny full-attention MoE、FP32、q/v LoRA、官方sft_loss、FSDP2 DP2；不覆盖GDN、35B、专家LoRA、BF16/TP或训练稳定性长跑。不新增上游提案，不作速度baseline。两批GPU合计90.86函数秒（181.73卡秒），按[Modal公开B200单价](https://modal.com/pricing)GPU项约$0.32；CPU/内存/启动与实际账单未计，不能当最终费用。

收尾：保留源码、JSON与小型梯度/optimizer张量证据；确认五个B02 app均无活动任务后，已删除七个tiny/adapter临时目录，共3,530,728字节，重新列举确认不存在，见[清理索引](cleanup.json)。旧CPU预检的临时模型路径随关项退役；以后复验须重新跑CPU构造。Q03移出活动队列；未来不同模型/精度/并行仍需自己的正确性门槛。
