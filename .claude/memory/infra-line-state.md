---
name: infra-line-state
description: infra 线（多卡/异步/MoE）的已定决策与当前状态；入口是 docs/infra_exp/00-INFRA-HANDOFF.md
metadata: 
  node_type: memory
  type: project
  originSessionId: c3d425ff-4b6a-4dd8-a186-e21d060e01e9
  modified: 2026-08-16T14:01:45.960Z
---

infra 线与主线**分开交接**：主线看 `docs/syncopate/05-handoff.md`，
infra 线看 **`docs/infra_exp/00-INFRA-HANDOFF.md`**（2026-08-14 晚更新，含明天的队列）。

**组织方式**：E 编号是身份（永不重排），track 是叠加的索引视图：
`TRACK-A-hardware-kernel.md`（负载稀疏 × 6.4GB/s 拓扑 ⇒ 该写什么算子）、
`TRACK-B-framework-async.md`（通用 RL 框架的假设在 agentic 负载上逐条失效）。
每个实验必须能答「服务哪条兑现物 / 需求由哪个测量指出」，答不上就显式停放（E04/E05/E06 已停）。

## ★ 2026-08-14 的头号结论（三个数互相印证）

```
① 用 4 张卡只换来 1.59× 加速   colocate 1卡 117.8 s/步 vs fully_async 3+1 74.1 s/步  [E08]
② 整机占空比只有 31%           trainer 空闲 54–57%，rollout 空闲 82.5% @ 47.7 W      [E08]
③ 权重同步 59.8 s 里 99.94%    不是传输（0.8 s）也不是编排（0.038 s），
   在「处理 132 MB 的 LoRA」上，只剩 send_weights 未排除                              [E12]
```
⇒ **动任何算子之前先搞清这 69% 的闲置。** 对照：Track A 全套自写 kernel 端到端仅 **4.3%**。

**Track B 已够撑一个项目；Track A 偏薄**（只有 SFT 稀疏投影一道兑现的菜）⇒ 重心要压到 A。

## 已落地的改动

- **E13**：`verl_patches.ddp_save_to_cpu` 加 `if param.requires_grad`
  （8.309 GB 里只有 3.18% 可训练，冻结基座跨版本逐字节相同）。
  `old_log_prob/ref` 比值 **1.941 → 1.069**，省 ≈8.5 s/步。3 条测试守着。
- `launch_rl` 新增 `--layered-summon`、`--target-modules`（都从写死改成显式参数）。

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

## 队列

见 `00-INFRA-HANDOFF.md` §5（2026-08-16 按「能不能把 before 变成 after」重排）。
**不吃 GPU 可并行推进的三件**：B4 仪器移位（写码）、B7 η 换算、B6 分池 patch。
⚠️ `bitsandbytes` 未装。

相关：[[machine-4x5090-constraints]] [[syncopate-docs-map]] [[feedback-measure-dont-infer]]
[[project-mechanism-not-wired]] [[user-chaoyu-working-style]]
