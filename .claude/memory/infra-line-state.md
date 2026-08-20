---
name: infra-line-state
description: infra 线（多卡/异步/MoE）的已定决策与当前状态；入口是 docs/infra_exp/00-START.md
metadata: 
  node_type: memory
  type: project
  originSessionId: c3d425ff-4b6a-4dd8-a186-e21d060e01e9
  modified: 2026-08-19T13:34:45.625Z
---

> ⛔⛔ **2026-08-18：本条里所有「异步 / 陈旧度 / ESS」相关的结论作废** ——
> 两个基石级 bug（E21 梯度没跨 rank 同步 · E22 权重从没推给 rollout engine）。
> 见 [[two-foundational-bugs-2026-08-18]] 与 `docs/syncopate/21-invalidated-numbers.md`。
> ✅ **纯硬件/通信/kernel 的测量不受影响**（E18 对齐悬崖 · E16 FP8 · B11 拓扑 ·
> B20 dynamic_bsz · E01/A5 阶段归属 · E00 带宽）。

infra 线与主线**分开交接**：主线看 `docs/syncopate/00-START.md`，
infra 线看 **`docs/infra_exp/00-START.md`**（08-19 重组：00 导航/守则 · 01 队列 · 02 决策与作废 · TRACKS 兑现物；旧的 ONBOARDING/00-INFRA-HANDOFF/TRACK-A/TRACK-B 已并入，E12 归档）。

**组织方式**：E 编号是身份（永不重排），track 是叠加的索引视图：
`TRACKS.md`（负载稀疏 × 6.4GB/s 拓扑 ⇒ 该写什么算子）、
`TRACKS.md`（通用 RL 框架的假设在 agentic 负载上逐条失效）。
每个实验必须能答「服务哪条兑现物 / 需求由哪个测量指出」，答不上就显式停放（E04/E05/E06 已停）。

## ★★ 2026-08-17 的状态（第 1 批全部跑完 + 四条追加）

```
✅ A7  满载降频          4 卡满载单卡算力仅 −2.0% ⇒ 「会污染所有对照」被证伪
✅ A6  E02 三档稳态      DDP 7.97s / ZeRO-2 3.42× / ZeRO-3 6.02×（3 卡）
✅ A1  MoE go/no-go      GO：4bit 15.6GB、LoRA 30.1M、前反向通过、梯度全有限
⛔ E04 rollout TP=2      实测净负 20% ⇒ 再次停放（理由从推算升级成实测）
✅ A8  分算子带宽        all_gather 在 3 卡塌 12×（2卡51 / 4卡37.9 / 3卡3.2）
✅ A11 ZeRO-3 @4卡       1.54×（vs 3 卡 6.02×）
✅ A12 NCCL 旋钮         LL128 可治：47.94→14.40 s（3.33×）
✅ A13 刨到根            ★ **16 字节对齐悬崖** —— 见 [[collective-alignment-cliff]] 与 E18
```

⇒ **RL 三模式实测（v12 数据，Qwen3-4B+LoRA）**：
`fully_async 3+1 **14.3–17.6 s/步**（最快）· one_step_off 22.9 · colocate 3卡 29.2 · 单卡 67.2`
⚠️ **fully_async 的 timing 行覆盖 4 个 global step，绝对秒数要 ÷4** —— 我因此报错过一次。
⚠️ 三模式**尚未同尺子对照**（步数/配置不完全同源）⇒ 队列 B3。

## ★★ 2026-08-17 晚（A14 闭环 + 夜间批开跑）

```
✅ A14  真实 ZeRO-3 的分片确实错位：all_gather 335.5 GB 里 **99.9% 的字节** %16≠0
        主体 67,287,212 B（%16=12，每 rank **只差 4 个字节**）；同跑 Broadcast 0% 错位＝对照组
        ⇒ E18 因果链闭合（§12）。同尺子对：Simple 96.08 vs LL128 29.76 = **3.23×**（此前 3.33×）
        ⚠️ 绝对值不可引用 —— 这跑开着 NCCL_DEBUG=INFO（光 AllGather 71677 行）
🟡 A15  上游归属草稿（E18 §13）：**先提 PyTorch**（per-rank numel 补到 16/itemsize 的倍数），
        NCCL 那条引用它作下游实证。⛔ 更正：我们走 **FSDP1**（`_flat_param._get_shard`）不是 FSDP2
        ⚠️ 提之前先做 **A16**（在真实尺寸 67,287,212 上验「补 4 字节就恢复」）；提 issue 要 Chaoyu 点头
✅ E03  NCCL 旋钮层结案：只有 CUMEM_ENABLE=0 与（分片路径的）PROTO=LL128 有用；
        ALGO/Simple/BUFFSIZE(1–32MB)/NUMA绑定/切分次数 全部无效
✅ E01  白捡主线 nsys：trainer gemm 58%/elementwise 24%/attn 12%；**GEMM 全是 cutlass_80（Ampere 代）**；
        E13 的修复在 kernel 层被证实（每步 CPU 快照 8.31 GB → 0.26 GB）
        ⚠️ 阶段归属还差 NVTX —— verl 的 `marked_timer` 名字在、marker 一个没有 ⇒ 已写补丁 `--nvtx`
✅ B4写码 下发记账三类事件（dispatch/complete/abort）+ 修 Pool.ingest 把无 reward 行当 0 分的污染
```

⇒ **占空比的两个数各挪了一大截**（bucket 512 的白捡观测，待 B2/B3 补同尺子分母）：
权重同步占步 **18.8% → 6.5%**、三次前向 **72% → 84.9%** ⇒ **B12/E17 升、B1 降**。

⇒ **抢卡纪律（用户 2026-08-17 明令）**：主线训练/评测跑完之前一个 GPU 实验都不许起。
判据是**进程退出 + 显存归还 + 产物目录静默 10 分钟**三条一起查 ⇒ `scripts/gpu_gate.sh`。

⇒ 夜间批：`scripts/run_batch2_gpu.sh`（A14/B2/B3/A5/B12）→ `run_batch3_gpu.sh`（A16/B11/B10）。
⚠️ **正在跑的 bash 脚本一个字都不能改**（按字节偏移增量读）⇒ 后续项另开文件。

## 已定决策（别重新讨论）

- verl 不换；**DDP 必选**（`--fsdp-size 1`）；FA2 默认；~~dynamic_bsz 默认 True~~ ⛔E25 已翻案：**默认 False**（mb=1 等价完美打包，见 launch_rl 注释）
- 🆕 **MoE 用 `Qwen3-30B-A3B-Instruct-2507`**（已下 57 GB）。~~GLM-4.7-Flash~~ 的
  `Glm4MoeLiteForCausalLM` **当前栈不支持**（要 transformers 5.0rc）——
  ★ **「day-0 支持」必须落到 `architectures` 字段验证，不能引用新闻稿。**
- 🆕 **MoE 的 LoRA 绝不能用 `all-linear`**：98.7% 的 Linear 在专家里 ⇒
  参数 1696M（26×）、张量 37,346 个（74×）、每步同步 3.39 GB。用「注意力+router」30.1M。
  ★ **继承来的默认值要跟着模型结构重新审**（meta 设备数一下 Linear，两分钟、零显存）。
- 🔻 **E11 稀疏 logprob 降级，不写 kernel**：密度 4.17% 但 lm_head 只占前向 4.28%
  ⇒ 端到端仅 4.3%，而最笨的切片就有 4.0%。
  ★ **「浪费的比例」和「能拿回的收益」隔着一个分母。**

## ⚠️ 会撞的两个坑

1. **`--weight-sync-bucket-mb 2048` 会 OOM**（今天两跑死在这）：rollout 卡 vLLM 24.65 +
   CE worker 4.71，剩 1.99 要 2.00，**差 0.01 GB**。`gpu_util 0.75` 不是安全值，
   **解法是调小 bucket**（实际只推 132 MB）。one_step_off 也中招 ⇒ 不是 fully_async 特有。
2. `--save-freq 999` 挡不住收尾保存（见 [[machine-4x5090-constraints]]）。

## ★★ 2026-08-16 的两条重估（用面试官视角审了一遍）

**头号结论：两条 track 手上的数几乎全是 before，没有 after。**
「占空比 31%」「只快 1.59×」是**现状陈述不是成果陈述**，孤立放进简历反而像自曝短板。
⇒ **从此实验优先级只看一件事：它能不能把某个 before 变成 after。**

```
Track B  ~50%   诊断 85% / 优化 15% / 验收 0%      ← 「够撑一个项目」那句话要加限定词：够撑的是诊断
Track A  ~30%   论证 80% / 兑现 25% / 硬手艺 0%    ← 三条腿断了两条半
```
- **A 的病和 B 不一样**：B 是「故事没讲完」，A 是**四条兑现物只落地了一条**
  （①稀疏计算 ✅；②MoE 量化账算完了一次没跑；③E16 一次没做；④E14 门槛在 E01）。
  ★ ①的依据是「监督密度 ~4%」这个**与硬件无关的结构性特征**，最稳。
- **E02 抢素材已裁**：归 **A**（A 的论点就是「拓扑决定该做什么」），
  **B 里降级成背景句**，别一份素材写两个项目 —— 面试官会当成灌水。
- **A 有两条「证明了不该做」**（E11 降级、ostinato §4.0 因果链被推翻）：
  工程上是产出，**简历成果栏写不进去，只能进面试的故事** ⇒ 要主动讲，别等被问。
- 新文档 **`docs/infra_exp/NARRATIVE-AND-RESUME.md`**：完成态的故事线 + 简历。
  ★ **它只写终点**（所有实验做完的样子），没测的数字留 `〔 〕` 由实验填；
  **刻意不维护「现在能写的」那一版**——进度归 track 文档，两份并存最后两份都不准。
  欠的实验清单在 `TRACKS.md`（B1–B9）和 `TRACKS.md`（A1–A6）。

## 队列（全表见 `01-TASKS.md`）

**排序原则，按顺序应用**：① **短探测优先，尤其是能防止长实验白做的**
（30 分钟的 go/no-go 挡住 2–3 天的工作量，期望收益比任何优化都高）；
② 然后看**对核心数字的贡献力度**（能把某个 before 变成 after 的排前面）。

```
第0轨 不吃GPU  B4写码(仪器移位) · B6写码(分池patch) · E03成文
第1批 gate     B2 缓冲区A/B(25m,挡着B3) → A1 E07探针(30m,gate住A2的2-3天)
               → A7 E00降频+4卡曲线(1h,分母的分母) → B3 one_step_off(30m)
第2批 核心     B1 权重同步优化(1-2d) → B10 陈旧度曲线(顺带B7 η) → B11 配比
               ⚠️ B11 必须在 B1 之后：B1 改完时间构成变了，先测配比会白测
               → B5 任务级尺子(一次性验收 E13+B1+B10/B11)
第3批 A 兑现   A5 E01(🐟挂跑,零成本,是B12门槛) → A4 切片落地 → A6 E02补稳态
               → B12 三次前向(=E17) → A2 三摆法
第4批 长投入   A3 E16(~1周,唯一硬手艺,不gate也不被gate) → B6验证/B8/B9
```

★ **队列编号 B*/A* 是执行顺序，E 编号是报告身份**，映射写在 handoff §5 第三列；
新报告按 E 编号建文件（🆕 **E17 = 训练侧三次前向**）。
⚠️ `bitsandbytes` 未装（A2 要）。

相关：[[machine-4x5090-constraints]] [[syncopate-docs-map]] [[feedback-measure-dont-infer]]
[[project-mechanism-not-wired]] [[user-chaoyu-working-style]]

## ★★★ 2026-08-18：抓到并修好一个静默的正确性 bug（E21）

```
E21  三个 trainer rank **梯度从没同步过**（fsdp_size=1 ⇒ 网格(3,1) ⇒ HYBRID_SHARD
     ⇒ PyTorch 降级成 NO_SHARD 却把归约留在大小为 1 的组上 ⇒ 空操作）
     ⇒ 每次更新只用 1/3 的数据。**已修并复验**（三 rank 梯度逐位相同）
     ⇒ ⚠️ **此前所有位移/ESS 的绝对值都在坏基线上** —— 重测队列见 handoff §5
E20  RL 学不动：① 序列级 IS 在 694 token 上指数崩塌（chi2_seq 64.19 vs chi2_token 0.065）
     ② 一个 epoch 只更新 109 次。**token 级 IS 实测把 ESS 0.449→1.000，零吞吐代价**
     ⚠️ 判据错过一次：**位移是输入不是产出**（AdamW 下 ≈ lr×次数）⇒ 只能用任务级三计数
E19  FP8 在 sm_120 上是真的（真实形状 1.70–2.22×）。**ref 可换、rollout 先别换**
     （FP8 误差是 vLLM↔FSDP 数值地板的 316 倍，会喂大 E20 那个问题）
E01  三次前向占 kernel 时间 83.2%；但卡只忙 74.6–78.2%（我曾说过头成"卡是满的"）
```

⇒ **上游文档三份**（`docs/upstream/`）：16 字节对齐（PyTorch）、HYBRID_SHARD 静默不同步
（PyTorch + verl）。**都等 Chaoyu 点头再提。**

⇒ **排序原则已换**（Chaoyu 2026-08-18）：**影响正确性的 > 影响速度的**；
第二看**端到端**收益，不看组件收益。清单唯一来源是 `01-TASKS`。

## 🔴🔴🔴 2026-08-18 下午：第二个静默正确性 bug（E22），比 E21 影响更大

```
E22  disaggregated（fully_async / one_step_off）下 **LoRA 从没被推给 rollout**
     ⇒ 生成数据的策略恒为起点 π₀ ⇒ **我们从没跑过一次正确的异步 RL**
     判据：推出去的 ‖W‖ 与磁盘起点**逐位相同**；colocate **不受影响**（它调两次）
     止血 --lora-merge 已验，但是 bf16 合并 ⇒ 充分性待验（R0-b）
0-A  E21 的后续：归约口径**正确**（3 卡 = 1 卡，比值 1.000000，是求平均）
     白捡：verl「按全局 token 数归一 × dp_size」在变长序列下确实在保护我们
```

⇒ **重跑队列已整段重排**（`01-TASKS`），一句话判据：
**算了多少、搬了多少字节 → 不受影响；算得对不对、学到没有 → 全部作废。**
⇒ **上游草稿三份**（PyTorch 一条 + verl 两条），都等 Chaoyu 点头。
⇒ 相关记忆：[[silent-degradation-weight-sync]]

## ⭐ 2026-08-18 收尾：异步 RL 第一次真正跑通

```
E22 修法① **自己实现并默认开启**（SYNCOPATE_LORA_ADAPTER_SYNC）
    —— 两端能力本来都在，断的只是中间没有传参那一栏
    验证：list_loras()=[123] · 载荷 8,414→252 MiB · kl 回地板 · param_sync 6.25→0.974 s（6.4×）
    数值：两侧 scaling 都是 2.0 · log_ppl_diff 落在同版本地板 ~3.4e-4
⛔ --lora-merge 已否决（bf16 合并毁掉 adapter 一半作用），启动即报错拦住
🔻 FSDP1 留着不换（上游确认不修；FSDP2 另有张量形态问题）⇒ 退化网格补丁是**长期方案**
```

**三个默认值已改对**（"兜底必须是对的那个"）：
`--weight-sync-bucket-mb` 2048→**512** · `--rollout-is` sequence→**token** ·
`SYNCOPATE_LORA_ADAPTER_SYNC` **默认开**；并显式钉死 `ulysses_sp=1` / `top_p=1.0` / `top_k=-1`。

**一步的构成被改写**：`param_sync` 占 0.8%、**三次前向占 88.9%**
⇒ 吞吐线的靶子毫无争议是 E17/B12，B1（权重同步）可彻底停放。

⇒ **队列下一件：R1（E20 全套在修好的异步基线上重测）**，直接跑 fully_async。
⚠️ 但**读任务分之前要等主线 ⑥ 的重基线评测**（配对比较的合法基线，4 卡 15 分钟）。
⇒ 故事全文：`docs/infra_exp/STORY-async-lora-weight-sync.md`
⇒ 管线验证状态总表：`01-TASKS §5`

## ⭐⭐ 2026-08-18 夜 · 9 小时队列（19 项 / 7.1 h）的结论

```
✅ B5 任务尺子**第一次通过**：+0.101（t=9.3，MDE 0.022），且**不是 reward hacking**（cap 全线降）
✅ E24 合并损失实测 −0.025（MDE 0.016）⇒ 配对基线一律用 _audit/v13_sft_e1_merged.json
🔴 **否定结果**：token 级 vs 序列级 IS 在任务尺子上 **+0.000（MDE 0.016）**
   ⇒ 估计量改善了，任务分没有 ⇒ **真正让 RL 从"学不动"变成"+0.101"的是 E21+E22 的修复**
🔴🔴 **defer 崩塌**：lr 3e-5 该 defer 97%→83%；**lr 1e-4 →0%**，而总分仍 +0.063
   ⇒ 当前 reward 下 RL 会系统性学会"不拒绝"，lr 越大越彻底 ⇒ **reward 设计问题，归主线**
✅ B19/B10：sync_every 4→16（吞吐 **+11.4%**）· 陈旧阈值 0.1→0.5（陈旧样本 6×）
   两个旋钮在估计量上**都测不出代价** ⇒ 但**都没过任务尺子**
★★ 那 11.4% **全部省在 gen（trainer 等样本）上** —— param_sync 只占 4%
   ⇒ **同步的真实代价是"打断 rollout 流水线"，不是"搬权重"** ⇒ **B1 彻底死了**
```

**★ 今晚最该记的方法论（都是被自己的判据抓到的）**：
```
1. **总分连着两次盖住真实差异**（§7.5 defer 97→83；§7.9 总分打平而 defer 差 14 点）
   ⇒ 三计数比均值好，但**仍是"打包"判据**；要看"哪一类行为变了"
2. **训练分与任务分给出相反排序**：lr 1e-4 训练分更高（+0.123 vs +0.109），
   任务分更低（+0.063 vs +0.101，直接配对 **−0.039 显著**）
3. **「首值→末值」在噪声带宽 ≥ 变化量时会凭空造出趋势**（§7.11 把 30 个点全打出来才发现）
4. **五条判据被证伪，三条是我自己当晚写的**（探针假通过/假失败、打印粒度让答案落在缝里、
   分组口径 ≠ 主线 P4）⇒ **判据必须能对自己失败**
```

⇒ 新队首（`01-TASKS §1`）：**reward 设计（主线）→ 常驻行为判据 → 多种子 →
原因② 重新设计（固定 epoch 而非步数）→ 同步不打断 rollout → sync_every 过 B5**


## ★★★ 2026-08-19：一条大归因被推翻 + KL 定案 + PrefixGrouper 只到微基准

```
⛔⛔ defer 崩塌 = **prompt 截断**，不是 reward   同配置只改长度预算（3584→5120）：
     该 defer 97%→83% 变成 97%→**100%**，REJ −0.188 → **+0.203**，任务分 +0.101 → **+0.137**
     ⇒ 主线 R-1 从队首撤下，换成「5120 下重测 lr 1e-4」 ⇒ [[behavior-collapse-check-input-first]]
✅ E17 KL 两臂定案   砍 KL 省 **15.4%**，A vs B 任务分 −0.009 < MDE 0.015（无差异）、
     defer/REJ 双同 —— **B5 首次通过**。🟠 唯一反向信号 fabricated_safety_cap +2 ⇒ 多种子必盯
     ⛔ 连带 E19「ref 走 FP8」失效。⚠️ **默认值尚未改**
✅ E25 「trainer 没喂饱」证伪   micro_batch 拉高是**负收益**；关 GC 显存不够
     ⇒ 省时间只剩「让它少算」 ⇒ [[trainer-is-compute-bound-not-starved]]
🟡 E26 PrefixGrouper   微基准 **3.96×** + fp32 逐位等价；~~真实集成未通~~ → ✅ 下午已通（见文末）
     ⇒ [[integration-is-the-work-not-the-math]]
🔴 代修主线阻断 bug   `val_kwargs.seed` 是不存在的键 ⇒ launch_rl 100% 启动即死
```

**队首（`01-TASKS §1`）**：① 5120 下重测 lr 1e-4 → ② E26 同尺子吞吐 A/B → B5
→ ③ KL 多种子 → ④ token vs sequence 多种子。
**新增第四条常驻判据**：`prompt_length/clip_ratio` 必须 0.0000。
**上游第四包**（`docs/upstream/verl-prefix-grouper-not-wired/`）：掩码语义是我们独有的；
接线部分挂起（#7202 已被维护者关闭）。回信机制已废：⛔ 2026-08-19 起两线往来**只写根目录 `MAINLINE-INFRA.md`**（唯一文档，办完删行），旧信件全删。

## ★★★ 2026-08-19 下午：E26 集成收口 —— Adam dtype 的真身是 FSDP 归约竞态

```
✅ 根因定案（scripts/repro_pg_dtype.py，脱 Ray 单轮 30 s，12 轮 vs 此前 16 轮起训练无果）：
   打包前向绕过根 FSDP ⇒ 损失不流经根输出 ⇒ 根 pre-backward 永不触发 ⇒ 归约的
   final callback 没人排队 ⇒ fp32 all-reduce 挂在没人等的 stream 上 ⇒ optimizer 读竞态快照
   —— 有时**梯度不跨 rank 归约且不报错**（E21 形状），有时抓到半途 fp32（=那个 Adam 报错）
★ 显形条件是**组合**：state_dict()一次 × ≥2 micro-batch（单因素全不复现 ⇒ 全抄真实跑 + 留一法）
★ 「以前能跑」= 根没 lazy_init 时子单元自封根的**巧合**；adapter 同步的 state_dict 打破它
✅ 修法：前向走根 FSDP（CausalLM forward 临时换回 HF 原版 + logits_to_keep=1）+ hook 捕获
   hidden + log_probs 加 0×根输出的锚。验收：三 rank 梯度和与健康路径**逐位相同**、
   log_probs sum 分毫未动；真实冒烟四常驻判据全绿
⛔ 2.12× 那个方向数当天即作废（单行 timing 的覆盖数没实测过，坑 5 变体）
✅ 顺手修：launch_rl 数据默认值 v3→跟 DATA_VERSION 走 · --help 裸 % 崩溃
```
⇒ 教训进 00-START §5 坑 24–26：**FSDP1 下绕过根 forward = 静默不归约**；
单因素复现不了先全抄再留一法；盯长跑必须带进程退出兜底。

## ★★★ 2026-08-19 下午（续）：同尺子 A/B 定案 + 队列按 Chaoyu 重排

```
✅ A/B 四臂（20 gstep × seed 42，覆盖数实测=4，logs/queue_e26ab/AB.done）：
   off_mb1 34.52 · off_mb8 33.26 · **on_mb8 14.94** · on_mb16 15.79 s/gstep
   ⇒ **生产→PG 端到端 2.31×** · PG 净效果 2.23× · mb1→mb8 仅 +3.8% · mb16 慢 5.7%
   ⇒ **PG 生产配置 = mb8（一组一批）**；微基准 3.96× 兑现 ~70%
★ gen 占步 12%→26%：trainer 加速后瓶颈向 rollout 移 ⇒ **陈旧度剂量条件首次具备**
   （配比之谜的完整解释在 [[disaggregation-is-a-memory-decision]]）
🔄 队列重排（Chaoyu）：lr 1e-4 重测**降级**为可选上限基线——主因是**步数太少**
   （≤1 epoch）不是 lr 低；「固定 epoch 而非步数」的原因②验证进队列
⇒ 队首：**E26 B5 任务尺子**（过了才谈 PG 默认开）→ KL 多种子 → token/seq 多种子
   （三个都是 ~4 h 级，适合夜间队列；已向 Chaoyu 提议，待点头）
```


## ★★★ 2026-08-20（傍晚）：E14 批2 全线出数——graph/闸门/乒乓三案

```
✅ Test A · vLLM CUDA graph（--enforce-eager False 旗子已加）：同尺子 12.84→11.89 s/gstep
   （−7.4%），gen 等待 −33%；训练侧持平受控。晋级默认欠精度闸
⛔ Test B · 闸门放宽（sync16×staleness0.5）：吞吐 −24%、gen 28%→0.6% 接力赛消灭，
   但**精度闸拦下**——同 64 步 −0.030(t=−4.3)，★该 defer 92→64（−28pt 破红线）,
   伤集中判断类（FRESH −0.21/CLAR/CONF）；执行类反受益（BUD +0.05）
   ⇒ ★ 三层判据层层加狠：微观仪表(ESS 0.84)说没事→总分说小事→行为判据说大事
✅ R2 剂量曲线开张：两点（0→0.874 · sync16/0.5→0.844）；甜点扫描+等时臂夜跑中
✅ 乒乓元凶名单（torch-prof+栈对齐，912 sync/2.17s 每 update_actor）：
   ①AdamW step 张量 GPU 读回 ×1008（adam.py:544 _get_value）②PG 库 repeat_interleave
   缺 output_size ×584 ③自家 _to_jagged .item() ×96 ④verl padding ~150
   ⇒ 修理三件套已置队首（01 §1-1），每处独立 A/B
★ torch-prof 探针（SYNCOPATE_TORCH_PROF=N）+ 栈对齐脚本是本日新武器：
   nsys 答不了"谁调的"（对 Ray trainer 还丢事件），torch-prof 进程内记账直接指到行
```

## ★★ 2026-08-20（下午）：E29 定案 + E14 第一批定稿

```
✅ E29 ckpt 只存 LoRA：save 7.91→0.83s(9.5×)·ckpt 12×·续跑合成加载+逐位校验(504/504)常驻；
   默认开；上游第5包文档已备（verl-lora-only-checkpoint）
✅ E14 批1 定稿：vLLM 侧（完整数据,三/四采互证 33/35%）84.2s 忙仅 29.1s——
   ★ 10-100µs 微间隙 ×79.7万=32.4s **超过计算本身** ⇒ enforce_eager/CUDA graph=最大单点机会；
   另 >10ms 等待 17.7s=接力赛（流水线深度/闸门）。trainer 侧「update_actor 内大洞+
   streamSync×144 乒乓」低信心待 torch profiler 复核
⛔ 仪器三翻车（全靠"物理不可能"抓出）：nsys 窗口错位×2（启动段/尾巴段——nsys 拖慢启动
   ~2min、去 osrt 又提速一倍,窗口必须事后用日志时间戳四点验证）+
   nsys 对 Ray trainer 进程事件截断不可修（flush-interval 无效）⇒ trainer 侧禁用 nsys
★ 心智模型五层已写给 Chaoyu（L0 重叠成立性→L1 步构成→L2 CPU-GPU 协同→L3 kernel 间→L4 kernel 内,
   每层各有工具与判据;上层病不治,下层优化被等待淹没）
★ 主线两端点（GPU0 B-4 + GPU1 OPD 老师）已按 Chaoyu 指令 12:47 关闭让卡
```

## ★★ 2026-08-20（晚）：Track 换标 + JD 对齐重写

```
⚠️ 换标：Track A = 框架/异步 RL（原 B）· Track B = 算子/硬件（原 A）。
   执行编号（B5/B12/A2/A4/P4…）不变；08-20 前的记忆/git 里 "Track A/B" 按旧标读。
✅ NARRATIVE-AND-RESUME 全文重写：§1 五岗两家族 JD 压缩+叙事重心对照 ·
   §2 事实底账（简历数字唯一来源）· §3/4 四段式 · §5 训练版+推理版简历文本
✅ 队首 = JD 对齐组（01 §1-1..5：压测共建/量化推理/MoE/PD 探针/A3）；
   主线压测 before 基线已备（runtime_loadtest 24/25 达标，11 §5）
候补待入队：torch.compile A/B（训练 C 加分明写）· ckpt IO（save_ckpt 占步 19.5%）·
   上游四包提交（框架研发岗价值最高，等 Chaoyu 点头）
```

## ★★ 2026-08-20：candidate 兜底兑现 → PG/KL 切库默认 · 队首换 CoT

```
✅ cand_v13r2_e1 事后核查全绿：PG=1+mb8 ✓ · use_kl_loss=False ✓ · seq IS ✓
   ESS(rollout_is_eff_sample_size) 中位 0.92/最低 0.816 · rollout_corr/kl 中位 4e-4 地板
   步构成：update_actor 54.3% · gen 23.9% · olp 18.8% · param_sync 1.8% · save 0.6%
   ⛔ 曾误报 save 19.5%——解析器对稀疏键（3/100 行）没算出现率，E26「覆盖数」同款坑，
   解析器已修（share=Σ键/Σstep + 稀疏警告）；E29 价值主体改为字节（108GB→~1GB/跑）
✅ launch_rl 默认已切：SYNCOPATE_PREFIX_GROUPER setdefault=1（mb 联动默认 8）· KL 默认 False
🔄 Chaoyu 裁定撤销/停放：token-seq 多种子（ESS 健康无问题可答）· R2 陈旧度
   （partial_ratio 恒 0 = 条件不存在）· 同步不打断 rollout（轨迹从没被杀 abort=0，
   但 rollouter 每次 sync 暂停-排空-恢复 ×99 次实锤；当前不值钱，大头是 update_actor）
⇒ 三者复活条件统一 = CoT 后 rollout 变慢。队首 = CoT SFT/RL 训练支持（01 §1）
⛔ 「save_checkpoint 19.5%」已翻案（见上）——引用步构成一律用修正后解析器的输出
```

## ★ 2026-08-19 晚：E27 thinking 三臂定案

```
thinking 净效果 −0.057（t=−4.9，A vs B 单变量）：REJ/FRESH ↑、FAIL/ATTR/CHAT ↓
  （acted_when_should_not 0→14 —— thinking 会把自己说服到「动手」）
★ 但有梯度格子 170→233、卡死 109→60 ⇒ 不涨均分却把 RL 探索空间打开一半
SFT 完胜（A vs C +0.347）⇒ 吃 thinking 红利的路径 = 带思考的 SFT 数据，不是拨开关
永久基线：_audit/e27_base_off.json（base think-off @2048/轮）；
  ⚠️ 裸基座臂单轮上限必须 2048（256 的砍断与真实弱分不开，v13_base@256 已删）
fabricated_safety_line_cap 两处汇合（SFT +18 · E17 KL 臂 +2）⇒ 升常驻观察
开关：SYNCOPATE_THINK=1 只许评测（launch_rl 拦训练）；预算 on=5120/8192
```
