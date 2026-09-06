# B06 · OPD 角色摆放与有效更新率

**前置**：B02、B03。

**目标**：在两张 B200 上选择学生、教师、锚的摆放与 batch，并减少“生成了但没有可蒸 token”的空转。

**完成条件**：所有臂读取同一 RL adapter；`max_steps` 只数真实 optimizer update，另记 attempts/skips；比较有效更新/小时、KL/mask 数值、显存、通信和任务质量。SFT fallback 只能作为显式诊断臂，不能冒充全链 OPD。


## 当前执行状态

GPU 实验尚未运行。用户已批准依队列开展 smoke/observe 诊断；GPU 资源批次、超时和胜出线在前置通过后补齐，再启动。Candidate 另需批准。CPU 数学检查可与 B03 并行。

零 mask 对照已修复并通过目标 CPU 补验：原先固定拿批内第一条，回复为空时会在前向前返回，全部梯度为 None 仍打印“通过”。现在要求真实学生/辅助模型前向、学生反向和存在且有限的零梯度，空回复不得通过；使用同一训练批实际非空回复，不造新数据。另硬核对 attempts=updates+skips，记录的更新/跳步必须和 completion 相等。此项不是 OPD 效果或 GPU 验收。

## 只读接线检查

零对照补验 `controls_cpu_20260906a` 使用 4 CPU / 16 GiB，脚本最多 10 分钟，无 GPU，预留 $1（总账归 B03，累计 $85）。本地新负例在修复前 2 failed / 11 passed，修复后相关 43 passed / 1 skipped（本机缺 Torch）。CPU 要求空回复被拒绝、非空样本确实经过双模型前向/学生反向、去掉零 mask 后负例报错、辅助模型无梯度；任何跳过不能算通过。

实测 **24 passed / 0 skipped，16.21 秒**。包含上述正反例、均值目标和 completion 计数；归一化梯度误差仍为 `6.237553975552146e-7`。XML 格式警告保留。App `ap-bAocvx2CvSV1fAa5828ivr`，镜像 `im-T4rWSBzGLYegwVqlsFy0um`，overlay `9923d45c0a208e64f118e11972d849f12ab3608f785d4d91d771a0f581a5ec2a`，受测 `opd.py` SHA256 `eb7efb09c4bcdd1f71ef3e6de36981906d3078abeb6db691fed8622bc43ce561`。本机汇总 `_audit/stack_probe/summary_2026-09-06_095055_exec_cd244a13.json`，完整证据 `/vol/_audit/infra/B06/controls_cpu_20260906a/`。真实生成 token/mask、教师/anchor 调用账、B200 更新与新产物仍待验。

现行代码由学生采样，教师在学生采到的同一串 token 上提供分布，计算反向 KL；不是先让教师生成答案再做 SFT。目前 mask 只包含自然语言，不含 think，不能写成“已经蒸馏了思考”。text+think 需要作为独立目标比较，不修改数据来迎合输出。

已确认原实现累积 token KL 的总和、更新前没有除以全局有效 token 数，日志却报每 token KL。现已将 token 均值目标接入更新前；数值证据与验收状态见下。不能把目标尺度和 batch 同时变化后当成纯速度收益。

还必须补真实 B200 上的零 mask、有效更新和产物证据。原跨 rank 检查只比一个范数标量，已换成全部可训练参数的字节指纹，并核对日志与完成标记；CPU 检查不能冒充真实多卡训练通过。当前单学生布局不受跨 rank 副本差异影响，但新布局必须逐参数验证。

## 独立 OPD 诊断的输入接线计划

新的 B06 run 必须显式读取 B03 c 批已验证的 RL adapter，不能只改 SFT 路径、却从新 run 的空目录找 adapter。给唯一 runbook 增加 `--opd-input-run`，只允许独立 smoke 的 opd-train/opd-eval；与外部 SFT 参数互斥，不允许冒充完整链路或 candidate。

CPU 先核对源 run 的 RL 健康闸和 adapter PASS，再确认 export manifest 指向该 run 的确切 actor、分片和 adapter 文件。源 run 若继承了 B02 merge，就沿原 `rl_input.json` 找回同一底座；重新核对底座、分片和 adapter 内容哈希。新 `opd_input.json` 明记 `external_rl_for_opd_diagnostic`，不复制 SFT/RL 的 PASS。GPU 只消费已绑定输入，新增模型或坏产物不能自动 fallback。

本机先验坏 run、错模型/分片、同大小改内容、缺 CPU 校验、禁止全链/candidate 和真实 runbook `--check-inputs opd-train`。之后目标 CPU 再验真实 adapter 格式、冻结 prompts 结构及隔离。初批拟做两次真实更新、最多 16 次 attempts、每批两行；正式资源与命令在这些前置全部通过后登记。真实生成 token/mask 仍需接线核对，不能仅凭字符串渲染相同宣称 on-policy。

输入接线本机及独立审查已通过 83 项。新增共享模型文件选择检查，排除 HF 实际加载未绑定的单文件、缺分片、显式 `transformers_weights` 或额外 adapter；共享 adapter 闸同时拒绝 NaN/Inf 和非浮点权重，不能把它们的非零计数当训练成功。前置还核对实际 XML、每个必跑测试模块、源码、输入和教师完整 SHA；GPU 前只复核原始证据、小文件 SHA 和大文件身份，不在 GPU 重哈希全部大权重。

隔离仅做冻结 prompts 与考卷**全部轮次**逐字交集检查，当前本机交集为 0。这批旧 prompts 没有 `source_case_ids`，所以不声称新证明了三桶来源隔离；不修改或重新生成数据。数据可读、格式和身份检查与业务语义优劣分开。

独立复审发现 adapter 配置若同字节数改变，旧 GPU 前置只比大小会漏掉；已有失败反例，现输入与教师两侧的配置/tokenizer 均重新比小文件 SHA。含新云端薄入口的本机定向组合为 142 passed / 3 skipped；后者是实际 safetensors/Torch 检查，尚须目标 CPU。独立云入口只走固定 runbook，CPU失败不申请GPU，训练两卡、评测一卡；不使用旧的可清空输出目录的 OPD 临时入口。

## 原始生成 token 的接线计划

已用真实 Qwen tokenizer 找到机制反例：生成 `[64,65]` 解码为 `ab`，重新编码却是 `[365]`；中文和 NFC 归一化也有同类反例。这证明“转成文字再编码”不保证仍是学生实际采到的 token，尚未测历史轨迹中出现的比例。修复范围是 OPD 实现，不修改数据、采样预算、教师选择或 text-only 目标。

生成函数返回原始 prompt IDs、attention mask、response IDs 和展示文字。先验证返回序列的 prompt 前缀完全相同；保留首个 EOS，且只裁掉其后的 batch padding。KL 前向直接消费这份记录，不重编码回复，也不能丢掉左 padding 对应的 mask/位置。教师和学生必须读取相同 IDs、attention mask 和位置。

mask 复用现有字符级 think/tool/text 规则，用当前 ByteLevel 词表把原始 token 映射到整段 UTF-8 字节跨度。一个 token 的全部跨度都是 text 才蒸馏；跨格式边界的 token 保守屏蔽，不能用众数放入部分机器语法。特殊 token 按 HF 与 backend 的 special 元数据并集屏蔽；普通的 think/tool 标签仍保留用于解析。非法 UTF-8 只屏蔽可定位的异常范围并记 WARN；无法可靠定位则整条屏蔽并解释。未知词表/decoder/ID 或输入身份不一致直接失败。

先本机验证中文跨 token、NFC、两种 EOS、pad=EOS、合法与非法 replacement 字符、跨 think 结束边界、special 集合漏项以及生成前缀错位。真实小模型 CPU 再核对生成与两模型前向实际收到的 IDs/mask/位置相等、独立 KL/梯度公式仍相等、零 mask 真经过前后向。每次训练保存原始 token 与实际 mask 记录，门禁要求覆盖计数和对齐 WARN 计数出现。没有任何可蒸 token 的样本可跳过，但不能把不确定的对齐算作通过或继续在错误 token 上训练。

本机接线及审查反例已覆盖 96 项、0 skipped；真实数学两模块因 Mac 无 Torch 未执行。复审补了两类反例：原审计能自报错 ID→字节，以及同一坐标重复生成仍算覆盖。现门禁显式读取本轮底座 tokenizer、逐 ID 重新计算 bytes/special/mask，并验证样本回合唯一、顺序、family 和 final 边界。不能沿用旧 24 项 CPU 的通过记录当成新源码验收。

## 本轮真实更新证据

再补一层不改变训练目标的观察：每次 `opt.step()` 前后保存可训练参数和梯度的内容指纹、实际 optimizer 状态 step，以及内层调用完成次数。要区分“调用完成”“权重改变”“有非零目标梯度”；AdamW 仅因 weight decay 改变权重不能冒充蒸馏信号。当前两步诊断要求每步有有限非零梯度、实际一步状态前进和参数变化；零梯度可完整留下诊断记录，但最终健康闸不得将它列为有训练信号的更新。NaN/异常仍立即停止。

独立 CPU 用真实小模型和 AdamW 检查正常更新、空调用、同权重、非有限梯度，以及仅 weight decay 的零目标梯度负例。保存每 rank/attempt 的独占原始 JSONL，最终门禁按内容复算并核对 completion 和日志次数。这些额外拷贝属于正确性探针，不用于性能比较；本批不改 LR、weight decay、生成/蒸馏预算或样本顺序。

接线本机 142 passed / 20 skipped；定向的更新记录与最终门禁为 91 passed / 18 skipped。跳过项需要目标 CPU 的真实 Torch/AdamW。新增测试文件已加入独立 OPD CPU 必跑清单，不能凭本机跳过宣布真实更新已验。独立规格复审已通过：真实 `opt.step()`、内层 post-hook、参数/梯度字节指纹、None grad、step/attempt/update 账和 decay-only 负例均接到门禁，未发现改变训练数学。收尾时的最终代码质量复审也通过；独立组合实跑为 249 passed / 18 Torch skips，没有提交阻塞。下一步直接登记目标 CPU 零跳过，再按门禁进入双 B200 两次真实更新；当前没有启动这批 CPU/GPU。

## CPU 数学检查：objective_cpu_20260905a

4 CPU / 16 GiB，最多 20 分钟，无 GPU，另预留 $1；计入 B03 批组后累计保守预留 $82。只用小型随机 Qwen3 和固定合成 token，不加载业务数据或真实权重，不改训练数据。

检查安装版 Transformers 上的真实 `opd.kl_step`：与完整前向的独立反向 KL 公式对拍；确认 5 个有效 token 的梯度总和是否为均值的 5 倍；再用已有的 token 窗口帮助函数归一化，对照误差须低于 `1e-5`。零 mask 必须实际经过学生和教师前向、学生反向，再验证零梯度；相同学生/教师的 KL 必须接近 0。教师不得收到梯度。

这是数学和零对照检查，不代表真实 tokenizer、B200 或 OPD 质量验收。若通过，再将均值目标接入生产更新边界，做回归后登记 GPU；不提前改 batch 或同时改蒸馏内容。

结果：16/16 通过，0 skipped。原总和梯度相对均值目标的误差为 3.99999976，即本例 5 个有效 token 时是约 5 倍梯度；归一化后误差为 `6.23755398e-7`。这不是“模型更新幅度 5 倍”或性能结论。零 mask 确实执行了两模型前向和学生反向，教师没有梯度；同模型反向 KL 对照通过。证据 `_audit/stack_probe/summary_2026-09-05_230634_exec_443c7652.json`。XML 的 `record_property/xunit2` 警告保留，属性已从 XML 读回。

## 接入更新边界后的 CPU 回归

`objective_cpu_20260905b`：4 CPU / 16 GiB、最多 20 分钟、无 GPU，另预留 $1，批组累计保守预留 $83。检验生产 `kl_step`、共享 token 均值函数、OPD 完成门禁、SFT 回归、双 CPU rank 的不等 token/空 rank 对照，以及固定入口更新顺序。新门禁拒绝旧的范数“通过”、错指纹、缺均值目标或非有限 KL；不会为了沿用旧日志把它们算通过。

该修复没有改变蒸馏 mask、教师路由、样本次序或响应预算。各路由日志仍是 rank 0 的局部计数；新增 `opd/global_masked_tokens` 和 `opd/kl_per_token_global` 才是全局目标。未来多学生布局的完整路由账仍待补齐。

结果：59/59 通过，0 skipped，测试耗时 355.24 秒。OPD 归一化梯度误差为 `6.23755398e-7`；SFT 完整批与切批的梯度误差为 `3.77883424e-7`；双 CPU rank 的不等 token、空 rank 和同范数错参数负例均通过。两条 XML 格式警告保留，不影响已读回的数值属性。

证据为 `_audit/stack_probe/summary_2026-09-05_231807_exec_3bf0aa93.json`；完整日志、XML 和结果在 `/vol/_audit/infra/B06/objective_cpu_20260905b/`。App 为 `ap-OkRjMXKjxRg5gxrNBL1iBf`，overlay 为 `23ee5333c70160bf6448488fa64d1a9dc65bf0d7ffabc67f9d6c68b6dc6f548d`，镜像为 `im-mRqnykIm401zVyhFjA0nGE`；受测 `opd.py` SHA256 为 `fcefaf138c4a2c0c9090d3212ea7e87db9703c8081e1379f7881ea359abd3deb`。GPU、真实生成 mask、真实 optimizer 更新和新产物尚未验，B06 不关闭。
