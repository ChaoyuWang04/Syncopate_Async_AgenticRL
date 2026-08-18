---
name: infra-line-state
description: infra 线（多卡/异步/MoE）的已定决策与当前状态；入口是 docs/infra_exp/00-INFRA-HANDOFF.md
metadata: 
  node_type: memory
  type: project
  originSessionId: c3d425ff-4b6a-4dd8-a186-e21d060e01e9
  modified: 2026-08-16T17:09:54.660Z
---

infra 线与主线**分开交接**：主线看 `docs/syncopate/05-handoff.md`，
infra 线看 **`docs/infra_exp/00-INFRA-HANDOFF.md`**（2026-08-14 晚更新，含明天的队列）。

**组织方式**：E 编号是身份（永不重排），track 是叠加的索引视图：
`TRACK-A-hardware-kernel.md`（负载稀疏 × 6.4GB/s 拓扑 ⇒ 该写什么算子）、
`TRACK-B-framework-async.md`（通用 RL 框架的假设在 agentic 负载上逐条失效）。
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

- verl 不换；**DDP 必选**（`--fsdp-size 1`，首步 FULL_SHARD×3 慢 5.97×）；FA2 默认；dynamic_bsz 默认 True
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
  欠的实验清单在 `TRACK-B §3.5`（B1–B9）和 `TRACK-A §7.5`（A1–A6）。

## 队列（全表见 `00-INFRA-HANDOFF.md` §5）

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
第二看**端到端**收益，不看组件收益。清单唯一来源是 `00-INFRA-HANDOFF §5`。
