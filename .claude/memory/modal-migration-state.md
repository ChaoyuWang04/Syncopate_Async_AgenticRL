---
name: modal-migration-state
description: ★09-03 Modal 搬家探针现场：镜像/Volume/git/权重已通；★冻结 v13 切分在 HEAD 代码下已生成不出来（裁定⑨ 使 3 对 case 题面同形被去重），不是 Modal 环境问题
metadata:
  type: project
---

**探针 = `modal_app/probe.py`（README 同目录）**，token profile spaemtuerl，Volume `syncopate-home`（/vol/repo git clone · /vol/models HF 权重 · /vol/data 重建产物 · /vol/_audit 判据）。
镜像 = nvidia/cuda 12.8 devel + python3.12 + `uv sync --frozen --all-extras --no-install-project`（一次通过，~15 min）。
09-03 读数：image ✅ · volume ✅ · git（HEAD 一致、CLI 通）· 权重 4B/0.6B 全落盘 · rebuild 0–4 步全 rc=0 但**切分 SHA 与 git 不同**。

**★ 切分不同的真相（三次本机基线复现，与 TZ/locale/PYTHONHASHSEED 无关）**：
当前 HEAD 生成 v11 会拒 8 条（git 里的是 5 条）：FRESH_0126≡FRESH_0019 · INJ_0050≡INJ_0010 · INJ_0059≡INJ_0019 · INJ_0057≡FAIL_0057。
根因 = 09-02 23:05 裁定⑨（account_id 不进 prompt）之后，只差 account_id 的 case 题面变成逐字节相同，
`prompt_fingerprint` 去重把它们刷掉 ⇒ 编号后移（FRESH_0150/INJ_0064/0065 顶上）。本机 09-02 14:58 的影子重建通过是因为早于该提交。
两对重复都在**同一切分内**（eval–eval / rl–rl），不是泄漏，但冻结 EVAL 343 在 v15 渲染下有 2 对重复题。
⇒ **★Chaoyu 09-03 裁定⑩（26 §6）：全部口径 v16，全部重新生成全部重来，忘掉一切旧的**；v13/v15 冻结读数不再是任何比较的一端。

**Why：** 「先量再动手」再次兑现——第一反应是 Modal 时区/locale，三个变量单独改都"复现"，其实基线本身就变了。
**How to apply：** 判"环境差异"之前先在本机不改任何变量重跑一次基线；派生数据与代码脱节时，判据要报的是"代码变了"而不是"机器变了"。
相关：[[v15-contract-refactor]] [[feedback-measure-dont-infer]] [[incremental-rebuild-freeze]]

**★09-03 裁定⑪ 换法三（26 §6）**：全家换 Qwen3.5（9B 学生 / 27B 思考教师 / 4B 语言教师 / 0.8B 测试），栈全新：vLLM 0.28.0 + torch 2.13 cu13 + verl 0.9.0 + transformers 5.10.x + FLA 0.5.2 + flash-attn 源码编 sm_120。
**Chaoyu 原话：首要目的是用新栈学新东西，模型本身 0 价值。** 新栈依赖表在 `modal_app/stack/`，探针 `modal_app/stack_probe.py`；Modal Volume 旧模型/旧数据已删。
硬事实：flash-attn 官方轮子只到 cu13torch2.10（v2.8.3）；**torch 2.13 的预编译轮子在 mjun0812/flash-attention-prebuild-wheels v0.9.47（cu130torch2.13 cp312，231 MB）**，Chaoyu 原话「不可能自己编译，肯定有的」——先搜社区仓库（mjun0812 / kingbri1 / alkemiik-coder），再谈编译；社区轮子必过卡上反向判据；Qwen3.5 全系是 `Qwen3_5ForConditionalGeneration`（带视觉编码器，model_type qwen3_5）；verl 0.9 要 vllm≥0.18、transformers ≥5.5.3 <5.11 ≠5.6.0。

**★09-03 晚 新栈在 PRO 6000 六步全绿**（08 §Modal 有读数）：社区 flash-attn 轮子反向过；FLA chunk 核梯度误差 ≤0.67%；vLLM 0.28 + Qwen3.5-9B **MTP 开反而 1.83× 快**（推翻 08 的 27B 先验）。
两个坑：venv/bin 必须在 PATH（FlashInfer JIT 找 ninja）；FLA naive 参数顺序 (q,k,v,beta,g) ≠ chunk (q,k,v,g,beta)，一律关键字传参。
**★Chaoyu 09-03 晚裁定方向：PRO 6000 也不用了（sm_120 无 TMEM 跑不了 FA4），一切环境在 B200（sm_100）上配**；B300 不建议（CUDA 13.1、库在追 sm_103、贵 $0.85/h）；
Modal 上没有更新的选择（GB200/GB300 不开放单函数）。等 Chaoyu 一句"B200"即改探针 GPU 标签 + 加 FA4 对拍步。

**★09-03 晚 B200 首轮读数（08 §Modal 有全表）**：新栈六步全绿（versions/gpu/fa4/nccl/models/verl）。FA4 在 sm_100 前向比 FA2 快 4.0×（≈1297 TFLOPS）；NVLink 双卡 all_gather 871 GB/s = 4×5090 PCIe 的 34×；P2P 直接通。
守则⑯ 机器判据首次就抓到 2 条未登记原因的非最新包。FA4 与 vllm 0.28 因 apache-tvm-ffi 钉冲突 ⇒ 独立 venv。vLLM 单卡/EP=2 读数待补。

**★守则⑰（00 §5，Chaoyu 09-03 立）：一切网络重活默认在 Modal 容器里做（解锁/拉权重/clone/编译产物全落 Volume），本机只改代码、提交、读判据；Modal 坑表在 00 §5 ⑰ 与 modal_app/README。** 最贵的一条：`modal app stop` 不带 --yes 静默不执行。

**★09-03 深夜 B200 新栈九步全绿**（含 vLLM 单卡 MTP 关/开 4.35/3.83 ms/token、EP=2 4.99 ms/token；核选择全记在 08 §Modal）。环境阶段收官。
下一步 = verl 0.9 训练冒烟（FSDP2 基线臂 + Megatron-Bridge EP=2/MXFP8 主课臂），按守则⑬先写逐步判据再动。

**★09-03 深夜 对齐地图（仓库测试在新栈上，Modal CPU）**：639 passed / 10 failed / 318 skipped（PG/Redis/GPU）。10 个失败**没有一个是"新栈把代码弄坏"**：
4 个要 data/batches/v13（v16 重建后消失）· 2 个要 models/Qwen3-4B（测试改指向新分词器）· 1 个要 PG · 1 个 vLLM 插件入口点未注册（--no-install-project 没装项目）· 1 个缺 prefix_grouper 包（E26 的 PG，新栈锁里没放，要决定留不留）· verl_patches 导入测试过了但 20 处 0.8 补丁的**运行时**有效性只有训练冒烟能判。
verl 官方镜像 verlai/verl:uv.cu130.dev3 在 Modal 展开后找不到 venv/site-packages（声明 VIRTUAL_ENV=/workspace/verl/.venv 但目录不在），暂不作 Megatron 臂基础镜像；Megatron 臂自建：megatron-core 0.19（cp312 轮子有）+ megatron-bridge 0.6 + TE 2.18（transformer_engine_torch 只有 sdist）。
pytest 判据坑：管道接 tail 吞退出码 ⇒ 输出落文件、`echo PYTEST_RC=$?`。

**★裁定⑬（09-03 晚）教师全用 Qwen3.8-27B（人话+思考同一端点），3.5-4B 退役；OPD 前核 tokenizer 一致。** wandb：Chaoyu 自己 `modal secret create wandb WANDB_API_KEY=…`，探针 `--steps wandb` 是接线判据；没建 secret 时 `STACK_NO_WANDB=1`。
B200 探索队列 18 条已进 `docs/infra_exp/01-TASKS.md §1`（P0 基线 → P1 训练 → P2 serving → P3 核/通信），纪律=先量默认基线再单变量改。

**★09-03 深夜 S1 对齐大半落地并已提交推送**：v16 口径一处（split.DATA_VERSION）· 模型路径一处（core/model_paths.py）· **Qwen3.5 工具调用是 XML 线格式**（`<function=…><parameter=…>`，不是 JSON）——解析两认、渲染默认 XML、schema 收型；
**Qwen3.5 模板渲单条 tool 消息会 raise "No user query found"** ⇒ 增量渲染用桩后缀法（rollout_loop.render_env_message_ids）。本机 v16 独立生成 SHA 记在 _audit/v16/。
坑：脚本批量插 import 会落进多行 `import (` 括号里 ⇒ 插完必 py_compile 全扫。
