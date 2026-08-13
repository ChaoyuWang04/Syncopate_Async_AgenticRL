# Ostinato · 面向 Agentic 负载的推理 Infra 与「模型 × Harness × Infra」Codesign 设计文档 v0.2

> 2026-08-13。Syncopate 的姊妹项目：Syncopate 提供模型（Qwen3-4B SFT/RL）与 harness
> （沙盒、verifier、评测尺子），本项目补齐第三条腿——推理 infra——并让三者互相塑形。
>
> **v0.1 → v0.2 的变更**（2026-08-13 讨论定）：
> 1. **重心从「引擎集成」转向「手写算子」**：verl/vLLM 里现成的一行配置开关
>    （fp8 KV、`use_fused_kernels`、`use_dynamic_bsz`）全部**降级为对照组**，
>    我们的贡献是亲手写的 kernel + 三层仪表盘上的实测。
> 2. **新增硬件审计章**（sm_120 有什么、我们的栈用到没有）——发现 FP4 tensor core
>    在消费级 Blackwell 上几乎无人点亮（Triton 都会静默退化），是真正的探索缝隙。
> 3. **新增手写算子菜单 K1–K4** 与研究线 R1；里程碑从 P0–P6 重排为 H0–H5。
> 4. mini-sglang 从「底座」调整为「K2/K3 的宿主 + trace 重放台」。
>
> **v0.2.1 增补（2026-08-13 晚）**：新增 **§23 训练侧显存工程**——当天 6 次冒烟
> 把「显存优化」从简历项目**变成了关键路径**：现有配置的 RL **起不来**（wake_up OOM）。
> 章节内含实测账本、根因（同一冻结基座在卡上存在两份 + 每步搬运一次）、
> 三步解法（H1a 停搬运 / H1b 融合 kernel / H1c QLoRA），以及一条被证伪的猜测更正。
> 施工顺序按 Chaoyu 2026-08-13 决定：**先解决"底座重复"这个大问题，再写小 kernel，
> param_offload 最后再试**。
>
> **v0.2.2 增补（2026-08-13 深夜，搬到 4×5090 之后）**：这一版主要是**把被实测推翻的
> 结论就地记账**，而不是新增计划。**四条**：
> 1. 🔻 **§4 的因果链（本文档的心脏）被自己的测量推翻了一半** —— 见 §4.0。
>    「池子太小 ⇒ 前缀复用熄火 ⇒ 量化扩容救回来」的中间一环不成立：
>    实测命中率 96.7–97.5%、池子用不到一半、零 preemption。**K2 的容量红利归零，
>    只剩带宽红利**，H2 优先级下调。
> 2. 🔻 **§26.2「verl 自带的融合 kernel 用不了，必须自己写」不成立** —— 见 §26.2.1。
>    装上垫片开 `remove_padding` 之后它能用，实测 actor 峰值 −5.0 GB。
>    **K1 的 RL 变体从「关键路径」降级为「要证明能赢过 NVIDIA 那版」**。
> 3. ✅ **§23「RL 起不来」这个开工条件已经解除** —— H1a 落地并改成默认值，
>    单卡 91–99 秒/步能稳定跑。**本文档最强的那条 codesign 论据已经兑现完，
>    需要换一条**（见 §29.2）。
> 4. 🆕 **§29：搬到 4 卡之后，本文档的适用边界**。单卡显存账本（§3、§24）全部是
>    colocate 的产物；多卡的事去 `docs/distributed-training-design-v0.1.md`。
>
> ★ **本文档的记账纪律**：推翻的结论**不删**，就地加「⛔ 已推翻」块，
> 写清**原猜想 / 实测 / 推翻后 / 教训**四段。理由和 Syncopate 主线一样——
> **负结果是资产**，而「当初为什么会那样想」比结论本身更值钱。
>
> **命名**：Ostinato（音乐术语，固定反复音型）——一段**不变的乐句在变化的旋律下反复出现**，
> 正是 shared prefix 在一批 rollout 下反复复用的形状；与 Syncopate（切分音）同属音乐词根。
> 备选：Groundbass、Palimpsest。名字可换，机制隐喻不换。
>
> 阅读顺序：本文档 → `syncopate-project-design-v0.1.md`（负载来源）→
> `/home/samwang/code/projects/QuantizedKVCache/ForChaoyu.md`（算子资产）。

---

## 0 · 三个必须先钉死的前提

### 0.1 每个进简历的数字必须来自自己负载上的实测

这个项目的直接动机是简历上有 infra 内容，这没什么不好意思承认的。但**简历叙事的可辩护性
来自诚实分层**（Halftone 复盘的元认知 §6 原话）。草稿数字和已有实测的出入先摆到桌面上：

| 草稿里写的 | 实际已有的实测（Halftone） | 处理 |
|---|---|---|
| int8 attention kernel 提速 **1.9×** | **1.56×**（S=4096，gap 已定位到 occupancy） | K2 做完 split-K 后重测，**写实测值** |
| PPL 仅升 **3.2%** | KV int8 **+0.2%**；q+K+V 全 int8 **+1.2%** | 我们的实测**更好**，直接用 |
| W8A8 GEMM 接入引擎 | Halftone **没有** W8A8 GEMM | 由 K3「FP8 GEMM on sm_120」诚实复活 |
| 命中率 60%+ / prefill 减半 | 从未测过 | H0 先测——60% 在我们负载形状下**可推导**（§4.3） |
| 端到端 1.8× / 显存 −45% | 从未测过 | H4 用 trace 重放测，写实测值 |

**规则：设计文档里的目标数字全部标「目标」；简历只收「实测」。**

### 0.2 复用不重造，但手写优先：开关是对照组，kernel 是贡献

§三的调查证明 prefix caching / KV 量化 / overlap scheduling 在 SGLang / vLLM / slime
里都已是成熟或默认能力——这是许可证不是障碍。v0.2 在此基础上再进一步：

- **凡是一行配置能开的，都开，但只作为我们手写实现的对照组和 baseline**；
- **凡是我们声称的贡献，必须是亲手写的 kernel 或亲手做的测量**；
- 每个手写 kernel 都要和官方/社区实现（verl 自带、Liger、cuBLAS、vLLM 原生）**同台跑分**，
  赢了输了都进报告——「我们的版本 vs NVIDIA 工程师写的版本差 x%，差距归因到 y」
  比「我开了个开关」硬一个量级。

### 0.3 Codesign 的闭环定义：三层仪表盘，缺一层不算完成

每一条 infra 改动都要在三层留下数字：

```
微观 · kernel 层（ncu）    带宽利用率 % / tensor pipe 利用率 % / occupancy / 对拍误差
中观 · 引擎层             缓存命中率 / preemption 次数 / TTFT / tok/s / 显存峰值
宏观 · 任务层（harness）   EVAL 128×8 配对差(MDE 0.05) / cap 构成 / 决策位熵 / ESS
```

只报微观不报宏观 = 不知道有没有伤到模型；只报宏观不报微观 = 说不清收益从哪来。
**三层全绿才算一道菜上桌。**

---

# 一 · 背景与负载画像

## 1 · Agentic 负载的三个结构特征

| 特征 | 机制 | 佐证 |
|---|---|---|
| **超长共享前缀** | system 规则书 + 工具 schema 对同模板所有 case 逐字相同；GRPO n=8 对同 case 整条 prompt 逐字相同 | 工具菜单裁剪前 prompt 里 **78.7% 是工具说明书**（4889 token 中 3846） |
| **多轮 KV 只增不减** | 每轮 assistant 输出 + tool observation 追加，历史逐字保留 | Manus：生产 agent 输入:输出 ≈ **100:1**，KV 命中率是「生产 agent 唯一最重要指标」，cached/uncached 价差 **10×** |
| **decode 小 kernel 碎** | 逐 token 生成时 batch 小、kernel 启动/调度开销占比高 | Halftone：decode attention 只有 16 个 program 对 ~170 个 SM，有效带宽 <5% 峰值 |

## 2 · 我们自己负载的实测画像（infra 的输入条件）

全部来自 Syncopate 现有实测：

| 量 | 值 | 来源 |
|---|---|---|
| prompt 长度 | 中位 **3198** / max 3587 token（v9，裁剪后菜单 9–13 工具） | v9 生成报告 |
| response 预算 | 1536 token | launch_rl 配置 |
| 轨迹步数 | 平均 ~5 步，归因任务 8–10 步 | 交接文档 §12 |
| GRPO 组结构 | 每步 4 case × n=8 = 32 条序列，**每 8 条共享同一 prompt** | launch_rl 配置 |
| RL 步耗时 | **65 s/步**（50 步实测） | RL_v8_sync_e1b |
| straggler | 最慢/均 **1.37–2.75×** | 交接文档 §4.1 |
| SFT/RL 训练侧 | SFT batch=1（batch 2 OOM）；RL micro-batch=1 | 交接文档 §2.1 |
| 评测集 | 冻结 EVAL 128 × 8 采样，配对 MDE ≈ 0.05 | compare.py 实测 |

## 3 · ★★ 单卡 colocate 的显存账本（本项目一切工作的物理约束）

### 3.1 整卡的钱花在哪（sync colocate，sleep/wake 模式）

```
31.36 GB 整卡（RTX 5090）
├─ actor（FSDP 训练态）常驻 10.5 GB → 几十步后峰值 ~21 GB（碎片单调爬升，实测 19.2→21.1）
└─ vLLM 预算 gpu_util 0.30 × 31.36 ≈ 9.4 GB
   ├─ 推理权重 bf16   ~7.6 GB
   └─ KV 池           ~1.8 GB  ← 全卡最稀缺的资源
```

Qwen3-4B（按 36 层 · GQA 8 KV head × 128 dim 推算，H0 实测校准）每 token 的 KV：

$$2_{(K,V)} \times 36 \times 8 \times 128 \times 2\,\text{B} \approx 144\,\text{KB/token (bf16)}$$

⇒ KV 池装 **~12.5k token**（bf16）／ **~23k token**（int8，含 scale 开销按 0.54× 计）。

### 3.2 ★ 先把 OOM 的账算对：KV cache 不是 OOM 的凶手

vLLM 的 KV 是**预分配的固定池子**。prompt 再长也不会因 KV 而 OOM——池子不够时引擎做
**preemption**（掐掉跑到一半的序列、回头重算）或排队，表现是**变慢，不是爆卡**。

我们真实炸过的 OOM 全在训练侧：
- **wake_up OOM**：推理权重加载撞上 actor 峰值（10.49+13.2+7.6 ≈ 31.3 那笔账）；
- **logits OOM**：算 loss/logprob 时物化 `batch × 5600 × 151936` 的 logits
  （bf16 1.7 GB / fp32 3.4 GB，前向反向再翻倍）——SFT batch=1 的根因，K1 的靶子。

⇒ **量化 KV 不直接治 OOM**；它治的是下面这条链。

### 3.3 ★★★ 池子 12.5k、需求 ~60k：浪费的两种形态

> 🔻 **这一节的结论已被 §4.0 推翻**（实测命中率 96.7–97.5%、零 preemption）。
> 保留原文是因为**推算方法本身没错，错的是"最坏情况"当成了"实际情况"**——
> 下面两种浪费形态在 `gpu_util 0.30` 的旧配置下是真的，在 0.40 之后没有发生。

一个 RL 步名义上要装：4 条 prompt（组内 8 路共享后）≈ 12.8k token
+ 32 条 response ≤ 48k token ≈ **60k token**——是池子的 5 倍。硬扛的代价有两种形态，
都是「**算过的又算一遍**」：

1. **轮与轮之间**：多轮 agent 第 3 轮回来，第 2 轮的前缀 KV 已被 LRU 挤出池子
   ⇒ 整条重新 prefill。**prefix cache 开关开着，命中率静默掉向零。**
2. **一轮之内**：正在 decode 的序列被 preemption 掐掉腾地方，回头重算。

## 4 · ★★★ 因果链总图（为什么量化是核心、kernel 是抓手）

### 4.0 ⛔ 已推翻（2026-08-13）：中间那一环不成立

> **原猜想**（v0.1–v0.2 的心脏，§3.3 + §4.1 + §4.3）：
> KV 池只装得下 1.5 条满长序列，而一步要跑 32 条 ⇒ 前缀 KV 被 LRU 反复踢出 ⇒
> 每步重算 prefill、preemption 不断 ⇒ **量化是给油箱扩容，让 radix 这台发动机转起来**。
>
> **实测**（`gpu_util 0.40`、2 步、Qwen3-4B + LoRA r32、单卡 5090，
> 打开 `--vllm-log-level INFO` 之后才看得到的统计）：
> ```
> Prefix cache hit rate   96.7 – 97.5%      ← 命中率接近满分
> GPU KV cache usage      23.9% / 28.9% / 43.8%   ← 池子只用了不到一半
> Running: 26 reqs, Waiting: 0 reqs         ← 零排队
> Preemption              一次都没有
> ```
>
> **推翻后**：油箱根本没见底，发动机也没熄火。**当初的推算错在三处**：
> 1. 按最坏情况 32 条 × 5120 token 算，但**组内 8 路共享的 prompt 只存一份**
>    （radix cache 的本职工作）⇒ prompt 侧实际只占 4 份，不是 32 份；
> 2. 多轮 agent 的上下文是**逐步长出来的**，大部分时间在 3.5k 而不是满长 5120；
> 3. agent 的瓶颈是**工具调用的往返**，不是并发 —— 26 条在跑也不排队。
>
> ⇒ **K2 的两笔红利只剩一笔**：容量红利（§4.2 上半）**归零**；
> 带宽红利（decode 每 token 把全部 KV 从显存读一遍，int8 字节减半）**仍然成立**，
> 但它是一笔小得多的钱，H2 从「主菜」降级（见 §20 的状态列）。
> 连带 **QLoRA（H1c）的优先级也塌了** —— 它的价值本来就是"省显存换 KV 池"。
>
> **教训（两条，都比结论本身值钱）**：
> 1. **§4.4「诚实的天花板」里写的那个 if 分支真的发生了。** 原文写着「若 H0 测出
>    驱逐率本来就低，主战场退到 eval 提速 + 长上下文余量——负结果照样进报告」。
>    ⇒ **写设计文档时就把"如果我错了会怎样"写进去，是这次能干脆认账的唯一原因。**
>    没有那句话，人会倾向于把测量结果往假设上凑。
> 2. **这不是失败，是 H0 存在的全部意义**：先测再动手，省掉了 1–2 周去写一个
>    解决不存在问题的 kernel。**代价是几行日志开关，收益是两周。**
>
> ⚠️ 但注意这个结论**有条件**：它是在 `gpu_util 0.40` 下测的（H1a/H1b 省出显存
> 之后才给得起）。**在 0.30 的旧配置下池子只有 1.06 GB，当初的推算未必错**——
> 更准确的说法是：**我们已经用别的手段（停搬运 + 融合 kernel）把油箱加大了，
> 量化想解决的问题在路上被别人解决了。** 下一次引用这条结论，要连 `gpu_util` 一起引。

### 4.1 一句话版本

> **前缀共享（radix）是发动机**（prompt 侧理论省 8×）→ **单卡池子太小让它熄火**
> （存不住就没得复用）→ **量化是给油箱扩容**，让发动机真正转起来 →
> **kernel 的带宽红利是顺路白捡的**。

### 4.2 int8 KV 付两次红利

| 红利 | 机制 | 谁受益 |
|---|---|---|
| **容量红利** | 同一池子装 2× token → 驱逐/preemption 大约减半 → 命中率活下来、重算减少 | prefill（不重算）+ 多轮复用链 |
| **带宽红利** | decode 每生成一个 token 都要把全部 KV 从显存读一遍，int8 字节减半 | **每一个 decode step，就算缓存全命中也照拿**（Halftone 1.56× kernel 挣的就是这份钱） |

另外容量是**可兑换的**：不换吞吐，可以把 `gpu_util` 再往下压、把余量让给 actor
——治「碎片爬到 21 GB 后挂」的边际，给 M3/M4 的 8–10 步长轨迹留 prompt 空间。
**同一块字节花在吞吐还是安全边际，是我们自己选的旋钮。**

### 4.3 「命中率 60%+」是可推导的下界，不是拍的

prompt 占整条序列 ≈ 3.2k/4.7k ≈ 68%；组内 8 份共享 ⇒ 命中的 prefill token 比例
≥ 7/8 × 68% ≈ **60%**——还没算多轮内部的逐轮复用。前提是池子存得住——这正是量化的工作。

### 4.4 诚实的天花板

提速上限 = 65 秒/步里「重复 prefill」占的比例，这个数没人知道，**H0 第一件事就是量它**。
且容量 ×2 后池子 23k vs 需求 60k，驱逐仍会发生——收益是「减半」不是「消失」。
若 H0 测出驱逐率本来就低，主战场退到 eval 提速 + 长上下文余量——负结果照样进报告。

---

# 二 · 硬件画像：sm_120 审计（v0.2 新增）

## 5 · 5090 (sm_120) 有什么、没有什么

| 能力 | 有无 | 说明 |
|---|---|---|
| gen-5 tensor core：**FP4 原生**（NVFP4/MXFP4） | ✅ | 消费级 Blackwell 相对 Ada 唯一真正的新数据类型（Ada 止步 FP8） |
| FP8 / FP6 mma | ✅ | FP8 自 Ada 已有；sm_120 延续 |
| **TMEM / tcgen05**（数据中心 Blackwell 的自治 tensor core） | ❌ | sm_100 专属。**sm_120 = Ampere 式 mma.sync 编程模型 + 新数据类型** |
| GDDR7 **~1.79 TB/s** 带宽 | ✅ | decode 是带宽负载，这是本卡最大的资产 |
| 170 个 SM / 32 GB | ✅ | occupancy 是 Halftone 已实测的第一道墙 |

**对我们是好消息**：没有 TMEM 意味着不需要碰够不着的指令——Triton + inline PTX 全覆盖。

## 6 · ★ 我们的栈今天用到了什么：「睡觉能力」清单

| 硬件能力 | 我们的栈（vLLM 0.12 + torch 2.9 + Triton）现状 |
|---|---|
| FP4 tensor core | ❌ 完全没用。**Triton `tl.dot_scaled` 在 5090 上静默退化成 bf16 mma**（triton#7550）——生态里几乎没人真正点亮它；正确路径是 inline PTX `mma.sync.m16n8k32`（k 维数的是 8-bit 容器）。旁证：SageAttention3 在 5090 上跑出 >1000 TOPS 的 FP4 attention，证明硬件真能点亮 |
| FP8 GEMM | ⚠️ vLLM **0.17+ 才有 SM120 专用 FP8 GEMM**，我们的 0.12 没有；Triton `tl.dot` 的 fp8e4m3 路径可用（H3 第一天微基准验证） |
| fp8 KV cache | ⚠️ vLLM 0.12 有 `kv_cache_dtype=fp8_*` 选项，但 sm_120 上 attention 后端支持性未验（开放问题 A1，一个 flag 的实验） |
| GDDR7 带宽 | ⚠️ Halftone 实测 decode kernel 有效带宽 **<5% 峰值**（16 program vs 170 SM）——资产在睡觉，split-K 的主场 |
| CUDA graph | ❌ **被显存逼关的**：launch_rl 写死 `enforce_eager=True`（CUDA graph 显存池在 sleep/wake 循环里每轮重新申请，wake_up 死在这）⇒ decode 全程跑慢速 eager |

**H0 交付物之一就是把这张表的每个 ⚠️/❌ 换成实测数字**（ncu 的 tensor pipe 利用率、
DRAM 利用率、实际在跑的 attention backend）。

### 6.1 🆕 2026-08-13 更新：最后一行的**前提已经没了**（一个免费的实验）

`enforce_eager=True` 当初是被 **sleep/wake 循环**逼的——CUDA graph 的显存池在每轮
wake_up 时重新申请，OOM 就死在那。而 **H1a 之后 `free_cache_engine=False`，
sleep/wake 整个不发生了**（`sleep()` 开头就 return）。

⇒ **「CUDA graph 关着」这条今天是一笔没人查过的呆账**，不是物理约束。
原计划（§14、§20 的 H4）是「靠 K2 量化省出的显存赎回 CUDA graph」——
**现在不需要 K2 也可能已经赎回了**，只差一次实测。

⚠️ 但先别急着开，**它和 §4.0 是同一个形状的坑**：CUDA graph 的收益前提是
「decode 的 kernel 启动开销占比高」，而我们**还没做 H0**（91–99 秒的时间构成未知）。
⇒ **先 H0 再开开关**，别再表演一次「推断代替测量」。

## 7 · 可及性结论

- **Triton 够到的**：fp8 `tl.dot`、int8 load + register dequant、所有融合/分块/online 归约
  ⇒ K1、K2、K3 全部 Triton 可写；
- **必须 inline PTX 的**：FP4 mma（`m16n8k32`）⇒ 只有 K4 前哨战的第二步需要；
- **够不着的**：TMEM/tcgen05 ⇒ 明确不做，也不必做。

---

# 三 · 开源格局调查

## 8 · 总表：谁已经做了什么

| 能力 | SGLang | vLLM | slime | verl（我们在用） | mini-sglang |
|---|---|---|---|---|---|
| prefix cache | ★ RadixAttention 起家之作 | APC，v1 默认开 | 继承 SGLang | rollout 默认 `enable_prefix_caching: True` | ✅ radix cache 默认开 |
| 分层/持久缓存 | **HiCache**（GPU/CPU/存储三级） | LMCache/Mooncake connector | 继承 | ❌（colocate 每步清） | ❌ |
| KV 量化 | fp8 | fp8_e4m3/e5m2（CUDA 无 int8） | 继承 | 配置层存在，launch_rl 未暴露 | **❌ ← K2 落点** |
| 命中率指标 | ✅ | ✅ `prefix_cache_queries/hits`（本机 0.12 确认） | ✅ | 不上报到训练日志 | **❌ ← 可提 PR** |
| overlap scheduling / CUDA graph | ✅ | ✅ | — | 我们关着（§6） | ✅ |
| fused CE/logprob kernel | — | — | — | `use_fused_kernels`，**默认 false 我们没开**（NVIDIA 贡献的 linear_cross_entropy 实现，本机确认存在） | — |
| agentic RL rollout | — | — | ★ 核心场景，APRIL partial rollout（吞吐 +22.5%） | vLLM/SGLang 双后端 | — |

## 9 · 各家细节

### 9.1 SGLang：方向的原创者

RadixAttention（2023 论文）用 radix tree 管理 KV、跨请求自动复用共享前缀 + LRU 驱逐 +
cache-aware 调度——**问题「prefix 有没有人做」的直接答案：有，而且是它的立身之本**。
HiCache（2025-09）扩成三级缓存，coding agent 场景（8 轮 ~25k token）实测
**命中率 40%→80%、TTFT −56%、吞吐 ×2**；Tair/Mooncake/3FS 都做了它的 L3 后端。

### 9.2 vLLM：hash-block 版的同一件事

APC 以 block hash 链实现前缀复用，v1 默认开；`kv_cache_dtype ∈ {fp8, fp8_e4m3, fp8_e5m2}`；
v1 metrics 自带 `prefix_cache_queries/hits`（本机源码确认）——H0 直接可用。

### 9.3 slime（THUDM）：agentic RL 训练侧参照系

Megatron 训练 + SGLang rollout，GLM-4.5→5.2 背后的 RL 框架。APRIL（arXiv 2509.18521）
partial rollout：超发请求、够数即停、未完成轨迹跨步续写，rollout 吞吐平均 +22.5%。
「跨权重版本续写」正是 R1 要量化的 staleness 问题的工程先例。

### 9.4 verl（我们的栈）：能力都在，大半关着

本机 verl 0.8 源码确认：
- `enable_prefix_caching: True` 默认开 ✅；
- **但 colocate `free_cache_engine: True`（launch_rl.py:153）每步 sleep/wake 销毁重建 KV**
  ⇒ radix 复用只活在单步之内。这不是 bug：权重更新后旧 KV 属旧策略，async server 在
  wake_up 时也主动 `reset_prefix_cache()`。跨步复用 = 接受陈旧 KV = R1 的研究入口；
- `enforce_eager=True`（launch_rl.py:152，wake_up OOM 逼的）⇒ CUDA graph 关着；
- `use_dynamic_bsz=False`（launch_rl.py:136，显式关着）⇒ micro-batch 按序列数不按 token 数；
- `use_fused_kernels` 默认 false ⇒ RL 侧 logprob/熵计算物化整个 logits；
- `use_remove_padding=False`（flash-attn 垫片绕过，已知缺口 🟡）。

**这五个开关就是 K1/H4 的对照组军火库**（§14）。

#### 9.4.1 🆕 2026-08-13 更新：五个开关里有三个已经翻了（并且成了默认值）

| 开关 | v0.2 时 | 现在 | 实测效果 |
|---|---|---|---|
| `free_cache_engine` | True（每步搬 7.6 GB） | **False** | wake_up OOM 的触发点消失（H1a） |
| `use_remove_padding` | False（缺 flash-attn） | **True** | ~~靠垫片~~ → **2026-08-13 晚已换真 flash-attn 2.8.3**（预编译轮子零编译装上，sm_120 kernel cuobjdump 验证；`attn_implementation` 默认切 `flash_attention_2`）。垫片退役——它对正确性够用，但 sdpa 路径恒物化 mask（连单序列都物化），dynamic_bsz 打包更是 2.2× 倒退，详见 launch_rl 注释与交接文档 §7.3 后记 |
| `use_fused_kernels` | False | **True** | actor 峰值 **−5.0 GB**（和上一条必须一起开，见 §26.2.1） |
| `enforce_eager` | True | True（**但前提没了**，见 §6.1） | 未测 |
| `use_dynamic_bsz` | False | False | 未测（§14 说的"免费红利"还没去拿） |

★ **改的是默认值而不是文档里的推荐命令**——理由是 Syncopate 主线的老教训：
**默认值必须是能跑通的那套**（交接文档 §9.1）。军火库里的开关一旦验证过，
就该进默认值；留在文档里当"记得加上"的参数，迟早有一轮忘了加。

### 9.5 mini-sglang：K2/K3 的宿主 + trace 重放台

sgl-project 官方教学实现（2025-12），~5000 行 Python：radix cache / chunked prefill /
overlap scheduling / CUDA graph / TP / FlashAttention+FlashInfer，**支持 Qwen3**。
没有的：KV 量化、prefix cache 指标、投机解码。
⇒ v0.2 的定位：**不是「我们的底座」，是手写 kernel 的宿主**——K2 的 int8 KV 装进它的
paged/radix 布局，K3 的 fp8 prefill 换掉它的 prefill 路径，我们的 trace 在它上面重放出
中观层数字。改一个 5000 行的教学引擎，比改 vLLM（40 万行）的收益/成本比高一个量级。

### 9.6 量化 KV 的更广格局（知识背景，标注待核）

工程侧：LMDeploy TurboMind 有 int8/int4 KV；TensorRT-LLM 有 int8/fp8 KV（版本细节待核）。
研究侧：KIVI（2bit，K per-channel / V per-token 共识出处，Halftone 独立复现）、KVQuant、
QServe（W4A8KV4）。**缝隙**：所有这些工作的精度验证都是 PPL / 通用 benchmark——
「量化 KV 对**多轮工具调用任务正确率**和 **RL 采样分布**（ESS、rollout↔训练 logprob 差）
的影响」没有公开的系统测量。**FP4 KV 在 agent 负载下的精度更是完全空白**（K4 前哨的价值）。

### 9.7 Manus 的生产结论：拼装规范是被验证过的工程纪律

KV 命中率是「生产 agent 唯一最重要指标」；cached/uncached 价差 10×；三条规则——
**前缀稳定**（system prompt 开头一个时间戳毁掉整条缓存）、**append-only**（不改历史、
序列化确定性）、**工具 mask 而非移除**。三条会进我们的拼装规范（§13），
而我们 harness 已满足两条半。

## 10 · 结论：三类缝隙 = 我们的全部定位

1. **手写实现 + 硬件探索**：教学底座没有量化、生态没人点亮 sm_120 的 FP4、
   decode kernel 的带宽资产在睡觉 ⇒ K1–K4；
2. **agent 任务级 + RL 分布级验证**：全生态的量化验证止步 PPL ⇒ 我们的三层仪表盘；
3. **观测与规范**：agent 训练负载下的命中率画像 + 训练/线上统一拼装规范是
   workload-specific 的活，没人替你做 ⇒ H0 + §13。

---

# 四 · 已有资产盘点

## 11 · Halftone（QuantizedKVCache）：可移植的算子资产

实测结论（RTX 5090 / Qwen3-0.6B / Triton 3.6）：

| 资产 | 实测 | 移植去向 |
|---|---|---|
| 量化方案：K per-channel · V per-token · 对称 int8（SQNR 数据驱动） | PPL 1.002×（KV int8） | K2 直接沿用，换 4B 重验 |
| 物理 int8 cache（HF DynamicCache 子类） | 显存 0.538× → 0.50× | K2 重写为 paged/radix 布局 |
| fused quantize-on-write kernel | V 2.8–3.3×；K 有 coalescing 病灶（S=4096 退化到 0.45×） | K2 修 2D tiling |
| fused int8 flash-style decode attention | 1.56×（gap 归因 occupancy：16 program vs 170 SM，反解固定开销 R≈47μs） | K2 split-K 后有望 →2× |
| SQNR 分析器（真实 K/V 采集 + 四粒度打分） | K per-channel 46.2 dB 领先 +10.5 dB | **K4 前哨战直接复用，零成本起步** |

Halftone roadmap 里三条「还没做」恰好是移植必需件：block-wise per-channel K
（production-PagedAttention 的答案）、batched/split-K 占用率、静态量化省 findmax。
⇒ **Ostinato 不是新开一摊，是把 Halftone 的 roadmap 放进有真实负载的引擎里做完。**

## 12 · Syncopate harness：现成的宏观层尺子

| 尺子 | 量什么 | 对 Ostinato 的用途 |
|---|---|---|
| 冻结 EVAL 128×8 + `compare.py` | 配对差，自报 MDE ≈ 0.05 | 一切量化/缓存改动的**任务级回归门** |
| 26 个 cap + defer/恢复动作双向 | 错误构成 | 回归不止看均值，看**错误方向**变没变 |
| `entropy.py` | 决策位熵 | 量化是否让输出分布变形 |
| `staleness.py` | σ²(k)（已有实测点：ESS/N=0.846，σ²(0)≈2.0e-4/token） | R1 的核心尺子 |
| rollout 记账 + rl_report | 三段耗时 / 分布漂移 / wandb 补报 | 吞吐收益的分解归因 |
| 65s/步 + straggler 1.37–2.75× | 端到端基线 | 所有加速比的分母 |

## 13 · ★★ Harness 已天然 cache-friendly——「训练纪律 = 缓存纪律」（codesign 核心论点）

本机源码确认，逐条对照 Manus 规则：

| Manus 规则 | 我们的现状 | 当初为什么这么做（跟缓存无关！） |
|---|---|---|
| append-only、不改历史 | ✅ `rollout_loop._append_message` 增量拼 token，单代码路径 | 修 Qwen3 模板不对称逼出来的 |
| 序列化确定性 | ✅ `step_user.txt` 里 `context \| dictsort` | 修「prompt 取决于 dict 插入顺序」的去重泄漏 |
| 前缀稳定、易变字段后置 | ✅ system 规则书最前（模板内共享），`reference_now` 在 user 轮首行（共享块之后） | M2 决定 reference_now 进 prompt 时顺手放对了 |
| 工具 mask 而非移除 | ⚠️ 半符合：菜单 per-case 静态裁剪（轨迹内不变），但跨模板共享前缀被缩短 | 为把 prompt −29% 让 RL 跑得通 |

**论点**：训练一致性纪律（同一路径、确定性序列化、基线可比）和缓存友好纪律是**同一条纪律**
——都是「字节级稳定的前缀」。训练要可复现的输入分布，缓存要可复用的输入前缀，
共同的敌人都是「不确定的字节」。这就是「反向制定训练与线上统一拼装规范」的理论根据。

最后一行的 ⚠️ 是一道真 codesign 权衡题：**裁剪省下的 prefill token vs 缓存损失的复用
token，在什么 KV 预算下谁赢**——两边都有数，H0 顺手量（开放问题 A5）。

产出物：`docs/ostinato/prompt-assembly-spec.md` + harness CI 校验器（挂在 prompt_hash
处，违反规范当场报错——守卫长在主路径上，Syncopate 的老规矩）。

## 14 · verl 里躺着的开关 = 对照组军火库（v0.2 新增）

这些一行配置的实验**全都要做**（便宜、快、有数），但记账记为「baseline/对照组」：

| 开关 | 治什么 | 角色 |
|---|---|---|
| `kv_cache_dtype=fp8_e4m3`（vLLM 现成） | KV 容量 ×2 | **K2 int8 实现的对照组**（fp8-flag vs int8-手写：容量近似、精度/带宽路径不同） |
| `use_fused_kernels=True`（verl 自带 NVIDIA 实现） | RL 侧 logits 物化 | **K1 的对照组之一**（另两个：naive eager、Liger） |
| `use_dynamic_bsz=True` | micro-batch padding 浪费 | 免费红利，先开先赚，进 H0 基线 |
| CUDA graph（关掉 `enforce_eager`） | decode eager 慢 | **H4 主角**：靠 K2 省出的显存赎回——「量化买回 CUDA graph」是杠杆串联的样板 |
| LoRA-only 权重同步 | wake_up 推 7.6 GB 全量权重的峰值与耗时 | **H0 查证**：`--lora-rank 32` 时 verl 同步的是全量还是 adapter（几十 MB）；若全量，改 adapter-only 直接治 wake_up OOM |
| `use_remove_padding` 修复 | 训练侧 padding token 白算 | 已知缺口 🟡，排 H4 之后 |
| APRIL 式超发早停 | straggler 尾巴 1.37–2.75× | **研究性，慎动**：先完成的偏短 ⇒ advantage 长度偏差，动它=动实验有效性，做之前设计对照 |

### 14.1 🆕 2026-08-13 结算：军火库打完之后剩下什么

| 开关 | 结算 |
|---|---|
| `use_fused_kernels` | ✅ **已开、已量**（−5.0 GB）。⇒ 它不再是 K1 的"对照组"，**它是现在的 baseline**，K1 要赢的是它 |
| `free_cache_engine=False` | ✅ 已开（H1a），wake_up OOM 消失 |
| `kv_cache_dtype=fp8` | 🔻 **动机塌了**（§4.0：池子用不到一半）。留着当 K2 的对照组，但不再有"扩容"的故事 |
| CUDA graph | 🆕 **前提没了、还没测**（§6.1）——现在是最便宜的一个未拆盲盒 |
| `use_dynamic_bsz` | ⬜ 仍未测。「免费红利」写了两版了还没去拿，说明它其实不在关键路径上 |
| LoRA-only 权重同步 | ✅ 已答（§25.1：本来就是 adapter-only）。⚠️ 但 **4 卡分卡之后它长出了新的显存代价**，见 §29.3 |
| `use_remove_padding` | ✅ 已修（垫片） |

⇒ **军火库基本清空了，而且清空它拿到的收益（−5.0 GB + 不再搬运）比原计划里
任何一道手写菜都大。** 这件事本身要记进方法论：**「一行开关」的对照组不是陪跑的，
它经常就是答案**；手写 kernel 的价值要建立在"开关已经开完了还不够"之上。

---

# 五 · 手写算子菜单（v0.2 的心脏）

每道菜五要素：写什么 / 治什么 / 亲眼看到什么 / 对照组 / 风险与周期。

## 15 · K1 · Fused Linear-CE kernel（第一道菜）

> **状态（2026-08-13）**：**SFT 侧 ✅ 已上线**（§27 稀疏投影：16.94→10.40 GB，
> T=12288 从 OOM 变能跑，4 条对拍测试守着）——**这一半仍然是真贡献**，
> 因为 `sft.py` 走 HF 路径，verl 的融合 kernel 插不进去。
> **RL 侧 🔻 降级**：verl 自带的能用了（§26.2.1），K1 的 RL 变体从「现成的用不了，
> 必须自己写」变成「要证明能赢过 NVIDIA 那版」。**做与不做取决于 H0**：
> 如果 91–99 秒里 logprob 那段占比小，写它就是自娱自乐。

**治什么**：SFT batch=1、RL micro-batch=1 的根因——算 loss/logprob 要物化
`batch × 5600 × 151936` 的 logits（bf16 1.7 GB / fp32 3.4 GB，反向再翻倍）。
**关键：我们的 sft.py 走 HF 路径，verl 自带的 fused kernel 插不进去——手写的两边通吃。**

**写什么**（Triton）：
- **前向**：对每个 token 的 hidden `x`（d=2560），沿词表维分块循环 `W_j`（tile × d）：
  `logits_j = x @ W_jᵀ`，做 **online logsumexp**（flash attention 的在线 softmax 思想
  搬到 15 万词表维：维护 running max `m` 和 running sum `s`，跨 tile 修正），
  同时在 target 所在 tile 抓出 `z_target`。loss = `(m + log s) − z_target`。
  **整个 logits 从不落地**，每 token 只留 `(m, s, z_target)` 三个标量。
- **反向**：`∂loss/∂z_j = softmax_j − 1[j=target]`，分块重算 logits tile，
  边算边累积 `dX += (p_j − δ_j) @ W_j` 和 `dW_j += (p_j − δ_j)ᵀ x`（fp32 累加器）。
- **RL 变体**：同一个 kernel 顺手输出 per-token logprob 和熵
  （`H = m + log s − Σ p·z`，online 可算）——GRPO 的 old_log_prob/entropy 路径直接用。

**亲眼看到**：
- 微观：对拍 eager fp32，loss 差 <1e-4、grad allclose(rtol 1e-3)；
- 中观：`torch.cuda.max_memory_allocated` 峰值 −〔目标 5–7 GB〕⇒ **SFT batch 1→4/8**，
  tokens/s ×〔实测〕，一个 epoch 的墙钟时间前后对比；
- 宏观：换 kernel 重训一轮 SFT，loss 曲线与 eager 版重叠 + EVAL 128 配对无回归。

**对照组**：naive eager / verl 自带（NVIDIA 写的 linear_cross_entropy）/ Liger。
三方同台跑分表——赢了输了都是报告素材，输了就归因（他们用了什么我们没用的技巧）。

**风险与周期**：数值稳定性（online logsumexp 的经典坑：跨 tile max 修正、fp32 累加）；
dW 的原子累加 vs split 归约的选型。参照物充分，风险低。**3–5 天。**

## 16 · K2 · int8 KV 三件套 + split-K flash-decoding（主菜）

> 🔻 **状态（2026-08-13）：从「主菜」降级，理由见 §4.0。**
> 三笔理由里**第一笔没了**：§3.3 的重算浪费实测没有发生（命中率 96.7–97.5%、
> 零 preemption）。剩下两笔仍然成立，但要重新称重：
> - **带宽红利**（decode 每 token 读全部 KV，int8 字节减半）：**只有在 decode
>   确实是瓶颈时才值钱** ⇒ **必须等 H0 的 91–99 秒拆解**。
> - **occupancy 墙**（Halftone 1.56× → 2×）：这是 kernel 自身的功课，
>   和我们的负载是否需要它无关 —— **它是"简历/手艺"价值，不是"项目关键路径"价值**。
>   这两种价值可以同时承认，但**不能混在一起给它排优先级**。
> ⇒ 现在的诚实排法：K2 排在 H0 之后，且**只有 H0 指向 decode 带宽时才提前**。
> 🆕 4 卡上它多了一个新的兑现场景：**D5（多 rollout 副本会不会打碎前缀缓存）**
> —— 如果命中率在分片后真的掉下来，「容量红利」会以另一种形式复活，见 §29.4。

**治什么**：§3.3 的两种重算浪费（容量红利）+ §4.2 的 decode 读带宽（带宽红利）
+ Halftone 遗留的 occupancy 墙（1.56× → 2× 的最后一段路）。

**写什么**（Triton，装进 mini-sglang 的 paged/radix 布局）：
1. **quantize-on-write**：K/V 写入 cache 时融合 findmax + 量化 + 写回。
   K 用 **block-wise per-channel** scale（每个 paged block 内 per-channel——
   Halftone roadmap 点名的 production 方案，同时修掉 S=4096 退化 0.45× 的
   coalescing 病灶：2D tiling，最内维保持连续）；V per-token。
2. **int8 paged decode attention**：flash-style online softmax，int8 load →
   register dequant → fp32 计算，读 block table 间接寻址。
3. **split-K / Flash-Decoding**：把 S 维切段，每段独立算部分 `(m, s, acc)`，
   第二个 kernel 归约合并。**grid 从 batch×heads=16 涨到 batch×heads×splits（上千）**，
   填满 170 个 SM。

**亲眼看到**：
- 微观：ncu `dram__throughput` 从 **<5% → 50%+**；kernel vs fp16 孪生版
  1.56× → 〔目标 ~2×〕@S=4096；对拍误差 ≤0.1%；
- 中观：同一 trace 重放，命中率上升〔实测〕、preemption 次数下降〔实测〕、tok/s ×〔实测〕；
  **命中率–池子容量曲线**（人为掐小池子扫一遍——「油箱多小发动机熄火、扩容多少救回来」
  这条曲线是全项目的招牌图）；
- 宏观：EVAL 128 配对差 <MDE + cap 构成不变 + 决策位熵不变；4B PPL 复验 0.6B 的 1.002×。

**对照组**：vLLM `kv_cache_dtype=fp8` flag（容量近似的官方路径）/ bf16 基线。

**风险与周期**：paged 布局的 block table 间接寻址是新增复杂度；mini-sglang 在 sm_120
的可用性要 H2 第一天验（A3）。**1.5–2 周。**

## 17 · K3 · FP8 GEMM / prefill attention on sm_120（tensor core 探索）

**治什么**：prefill 是 compute-bound（和 decode 的带宽负载互补）——FP8 mma 理论吞吐
≈ bf16 的 2×。这是 Halftone roadmap「int8 tensor-core prefill」的 Blackwell 版本，
也是简历草稿 W8A8 GEMM 那条的**诚实复活**（我们栈上 vLLM 0.12 没有 SM120 FP8 GEMM，
0.17 才有——自己写正好填自己的洞）。

**写什么**（Triton）：
1. **第一天微基准**：`tl.dot` fp8e4m3 × fp8e4m3 → fp32 在 sm_120 上是否真发 fp8 mma
   （用 ncu 看指令；防 triton#7550 式静默退化）；
2. **FP8 GEMM**：per-block scaling 的 W8A8 式 GEMM，M/N/K 扫描出 roofline 曲线，
   对比 cuBLAS bf16（torch.matmul）；
3. **（可选）FP8 prefill attention**：QKᵀ 和 PV 两个矩阵乘 fp8 化（SageAttention 式
   per-block 量化），在我们的 shape（d=2560、seq 3.2k）上测。

**亲眼看到**：roofline 图（实测 TFLOPS vs 理论 2× bf16）；ncu tensor pipe 利用率；
我们 shape 上的实际加速比。精度侧：GEMM 数值误差谱 + 若接进 prefill 则走三层仪表盘。

**对照组**：cuBLAS bf16 / torch._scaled_mm（若在本 torch 版本可用）。

**风险与周期**：Triton fp8 在 sm_120 的成熟度是最大变数（所以第一天先微基准，
不行就降级为 inline PTX 或改测 CUTLASS 路径并如实报告）。**1 周。**

## 18 · K4 · FP4 KV 前哨战（探索性，可选）

**为什么值得**：FP4 是 sm_120 唯一真正新的数据类型，而生态几乎没人点亮
（Triton 会静默退化）；**agent 负载下的 FP4 KV 精度没有任何公开数据**——
这一格做出来无论结论正负都是独一份。

**怎么做**（分两步，第一步零 kernel 成本）：
1. **SQNR 前哨**：拿 Halftone 现成的分析器，把采集好的真实 K/V 量到 NVFP4
   （e2m1 + 16 元素 block scale），和 int8 的 SQNR 表并排——一个下午出数。
   **决策门**：若 K per-channel 在 FP4 下掉到 ~30 dB 以下就止步（结论：FP4 只配给
   冷前缀或 V，写进报告）；若可用，考虑**分级量化**（热 token int8、冷前缀 FP4）。
2. **inline PTX 微基准**：`mma.sync.m16n8k32`（k 维数 8-bit 容器）的 TOPS 实测
   ——Triton 指不上，这正是「探知 sm_120」的动手乐趣所在。

**亲眼看到**：SQNR 对比表 → go/no-go；PTX 微基准 TOPS vs 标称。

**周期**：前哨 1 天；后续开放。

## 19 · R1 · 跨权重版本的 KV 复用 × σ²(k)（研究线，可选亮点）

**问题**：所有 RL 框架在权重更新后清 prefix cache（正确性优先）。但 agent 前缀的大头是
system 规则 + 工具文档——**这部分 KV 对 LoRA 小步更新的敏感度到底多大？**
若很小，跨步保留缓存就能省掉「每步重算 12.8k token prefill」。

**方法**（单卡离线可做，工具已有）：第 t−k 步 ckpt 算 prompt KV，第 t 步权重接着
decode，和全新计算比 logprob 漂移 → 「KV 陈旧度」的 σ²(k) 曲线，与已有的
「策略陈旧度」σ²(k) 并排。**决策规则可直接写**：KV-σ²(k=1) ≪ 策略-σ²(k=1) ⇒
跨步保留划算，报告给出开/关判据。

**顺手的一问**：量化 KV 的分布偏移（ESS 量）和陈旧一步的偏移谁大？
「量化的代价 ≈ 陈旧 x 步」这个换算关系本身就是可发表级的小发现。

---

# 六 · 里程碑与执行

## 20 · 路线表 H0–H5

| # | 里程碑 | 内容 | 验收（亲眼看到的东西） | 预估 |
|---|---|---|---|---|
| **H0** | 全身 CT | nsys 拆一个真实 RL 步（65s 分解：prefill 重算/decode/wake_up/straggler/训练）；KV 池三数字（容量/命中率/驱逐次数，vLLM 计数器）；ncu 抽查 decode/prefill kernel（DRAM%、tensor pipe%、在跑的 backend）；§6 睡觉清单换成实测；查证 LoRA 同步粒度（A4）与 eval 后端（A7）；开 `use_dynamic_bsz` 白拿红利 | **一张「钱花在哪、哪些硬件在睡觉」的表** + K1–K4 的真实优先级排序 | 1–2 天 |
| **H1** | K1 fused CE | §15 全项。**SFT 侧已完成**（§27：稀疏投影，16.94→10.40 GB、T=12288 从 OOM 到能跑） | 三方跑分表；峰值显存 −〔实测〕；EVAL 配对无回归 | 3–5 天 |
| **H1a** 🔴 | **停止搬运** | `free_cache_engine=False`：vLLM 权重常驻不再 sleep/wake（§26.1）。**这是 RL 的开工条件，优先级最高** | RL 能跑完 ≥3 步不 OOM；actor 峰值实测 vs 21.9 GB 上限 | 1 天 |
| **H1b** 🔴 | **融合 logprob kernel** | K1 的 RL 变体（§26.2）：不物化 `[1×5120×151936]`。verl 自带的用不了（§9.4） | actor 峰值 −〔实测〕；logprob 与朴素路径对拍；TIS ratio 不异常 | 3–5 天 |
| **H1c** | **QLoRA 基座量化** | actor 的冻结基座 int8（§26.3）。先做可行性探针（A13） | actor 常驻 10.5→〔实测〕；EVAL 128 配对回归 | 1 周 |
| **H2** | K2 int8 KV | mini-sglang 跑通 Qwen3-4B（A3）+ 加命中率指标（可 PR）+ trace 重放 → int8 三件套 + split-K | **命中率–池子容量曲线**（招牌图）；kernel 1.56×→〔实测〕；三层全绿 | 1.5–2 周 |
| **H3** | K3 FP8 | 微基准验路 → FP8 GEMM roofline →（可选）prefill attention | roofline 图 + tensor pipe 利用率 + 我们 shape 的加速比 | 1 周 |
| **H4** | 消泡 + 端到端 | 拿 K2 省出的显存赎回 CUDA graph（关 `enforce_eager`）；nsys 逐项开关归因（overlap/graph/chunked prefill）；remove_padding 修复排此处；同一 trace 重放优化前后 | 消泡归因表 + **端到端吞吐/显存的诚实版数字**（替换草稿 1.8×/−45%） | 1 周 |
| **H5**（可选） | 探索双响 | K4 FP4 前哨 + R1 σ²(k) 研究 | SQNR 决策门数据；KV-σ²(k) vs 策略-σ²(k) 并排图 + 开/关判据 | 开放 |

依赖：H0 → H1（独立于引擎，最快见效）→ H2 → H3/H4（可并行）→ H5 随时可插（离线）。
合计 **5–7 周**的 part-time 工程量。

**与 Syncopate 主线的资源协调**：H0 的 nsys 挂在下一次真实 RL 训练上跑（v9 重训那轮），
**一鱼两吃**；H1 kernel 开发用小规模验证，冲突小；H2+ 大部分是写代码，
避开重训窗口即可。

### 20.1 🆕 状态盘点（2026-08-13 深夜）与重排

| # | 状态 | 说明 |
|---|---|---|
| **H0 全身 CT** | ⬜ **一次都没做，而且现在是第一优先级** | 这是所有降级/升级的裁判。**目前唯一确定的事实是"时间不在我们以为的地方"**：KV 池空着、零排队、命中率 97%，**91–99 秒仍然花掉了** |
| H1 K1（SFT 侧） | ✅ 完成 | §27，实测在案 |
| **H1a 停搬运** | ✅ **完成并进默认值** | wake_up OOM 消失；这是"RL 起不来"的解药 |
| **H1b 融合 logprob** | ✅ **完成，但不是我们写的** | 用 verl 自带的（§26.2.1），−5.0 GB。**自己写降级为可选** |
| H1c QLoRA | 🔻 **降级** | 它的价值是"省显存换 KV 池"，而池子用不满（§4.0）。**4 卡上还多了一个替代方案：显存不够就换并行策略，不必压模型** |
| H2 K2 int8 KV | 🔻 **降级**，见 §16 | 等 H0；或等 D5 让容量红利以"分片打碎缓存"的形式复活 |
| H3 K3 FP8 GEMM | ⬜ 不变 | 它的价值一直是"探知 sm_120 + 手艺"，**不依赖上面那条因果链**，所以没被这次推翻影响。可能是现在**最健壮**的一道菜 |
| H4 消泡 + 端到端 | ⬜ 部分前提变了 | CUDA graph 不必等 K2 赎回了（§6.1）；`remove_padding` 已修 |
| H5 K4/R1 | ⬜ 不变 | R1（KV 陈旧度 σ²(k)）在 4 卡上和 D3 异步线**天然接得上**，见 §29.4 |

★ **重排后的一句话**：**H0 → 看结果再决定 H2/H3/K1-RL 的死活。**
在 H0 之前排任何 kernel 的优先级，都是在重复 §4.0 那个错误。

## 21 · 评测协议（口径钉死，防数字注水）

| 数字 | 口径 |
|---|---|
| 命中率 | **token 级**：`prefix_cache_hits / queries`，分「RL 步内」「eval 重放」两场景报，不报请求级（虚高） |
| prefill 节省 | 命中 token / 总 prompt token，同 trace 对比 |
| 端到端吞吐 | **固定 trace 重放**：从 Syncopate 导出（RL 步 32 序列组结构 + eval 104×8），优化前后各跑三遍取中位；不用合成负载 |
| 显存 | 同 trace 峰值 reserved，按几十步后的峰值算（wake_up OOM 的教训） |
| 精度回归 | EVAL 128×8 配对差 ± CI，报 MDE；cap 构成同列；「无显著差异」必须写成「差值 x ± y，MDE z」 |
| kernel 加速比 | 对拍孪生 kernel + 理论上界 + gap 归因（Halftone 规矩：1.56× 不丢人，解释不了才丢人） |
| 对照组 | 每个手写 kernel 至少一个官方/社区实现同台，输赢都归因 |

## 22 · 交付物清单

1. **Ostinato 仓库**：K1–K4 kernel + mini-sglang fork（int8 KV + 指标）+（可选）上游 PR；
2. **agent-trace replay benchmark**：多轮工具调用 + GRPO 组结构的可复放负载
   （生态里只有 ShareGPT 式负载，没有这个——本身是可开源的小件）；
3. **拼装规范文档 + harness CI 校验器**（落回 Syncopate 仓库）；
4. **测量报告**：H0 全身 CT / 三方 CE 跑分 / 命中率–容量曲线 / FP8 roofline /
   消泡归因表 /（可选）FP4 SQNR + KV-staleness；
5. **简历 bullet 实测版**（附录 B 模板填数）。

---

# 七 · 训练侧显存工程（v0.2.1 增补，2026-08-13）

## 23 · ★★★ 结论先行：这不再是优化，是开工条件

> ✅ **状态更新（2026-08-13 当晚）：这个开工条件已经解除。**
> H1a（停搬运）+ verl 自带融合 kernel（§26.2.1）+ `gpu_util 0.40` 三件之后，
> 单卡 RL **稳定跑到 91–99 秒/步**，actor 峰值 13.92 GB（余量 4.9 GB）、
> KV 池 4.19 GB / 30,528 token。**下面这张"6 次冒烟没有一次跑完一步"的表是历史记录。**
>
> ⚠️ **随之而来的代价要记账**：本文档最强的那条 codesign 论据
> （「不是 infra 让训练更快，是不做 infra 训练跑不了」）**已经兑现完了，现在过期了**。
> 再引用它就是拿已解决的问题当动机。**新的动机必须从 H0 的时间构成里长出来**
> ——见 §29.2。

2026-08-13 为了验证「计数器有没有挂上」跑了 6 次 RL 冒烟，**没有一次跑完一个 step**。
现有配置（交接文档 §5 那套跑通过 v8 50 步的参数）在 v11 上**起不来**。

| # | 配置 | 结果 |
|---|---|---|
| 1 | 0.30 + fused kernel | 过 wake_up ✅，死在 padding 契约（§9.4） |
| 2 | 0.30 + fused kernel（修完预算 bug） | 过 wake_up ✅，死在 padding 契约 |
| 3 | **0.30 标准配置** | **wake_up OOM** |
| 4 | 0.28 | **起不来**：KV 池 0.43 GiB < 一条序列要的 0.70 GiB |
| 5 | 0.30 + `max_num_seqs 16` | **wake_up OOM** |
| 6 | 0.30 + `param_offload=True` | 过 wake_up ✅，死在 `sleep()`（主机 30 GB 内存装不下两份权重） |

⇒ **§五的 kernel 工作不再是「为了简历/研究」，它是 RL 能不能开工的前提。**
这是本项目最强的 codesign 论据：不是「infra 优化让训练更快」，是「不做 infra，训练跑不了」。

## 24 · 实测账本（数字全部来自当天日志，不是推算）

```
31.36 GB 整卡
├─ actor（FSDP 建完）      10.49 GB   allocated 7.75 / reserved 8.69（日志 "After FSDP"）
│                                    跑到 step 24 时 reserved 爬到 21.08（碎片，v8 实测）
└─ vLLM（gpu_util 0.30）    9.41 GB
   ├─ 推理权重 bf16         7.6  GB   ← **吃掉 vLLM 预算的 81%**
   └─ KV 池                ~1.06 GB   ≈ 7.4k token ≈ **1.5 条满长序列**
                                      而每步要跑 **32 条**（4 case × 8 rollout）
每步 sleep/wake：权重 7.6 GB 在 GPU↔CPU 之间**搬来搬去一个来回**
```

**新校准的常数（推翻/确认了 §3.1 的推算）**：

- 一条 5120 token 的序列需要 **0.70 GiB** KV ⇒ **143 KB/token**
  —— §3.1 按 36 层 × 8 KV 头 × 128 dim 推的 144 KB/token，**分毫不差** ✅
- 但 §3.1 里「KV 池 ~1.8 GB / 12.5k token」**偏乐观了 70%**：真实是 1.06 GB / 7.4k token
  （漏算了 vLLM 自己的激活工作区 ~0.75 GB）

**⇒ preemption 不是「会不会发生」，是每一步都在发生。** §4 那条因果链
（前缀共享是发动机 → 池子太小让它熄火）现在有了硬数字：**油箱只有 1.5 条序列的容量。**

## 25 · ★ 根因：同一个冻结基座在卡上存在两份，还每步搬一次

LoRA 下 **98.4% 的参数是冻结的**，actor 和 vLLM 两边**逐字相同、永不改变**；
真正在变的只有 66M adapter（**132 MB**）。而现状是：

```
actor 持有 bf16 基座    7.75 GB  ┐
vLLM  持有 bf16 基座    7.6  GB  ┘ 同一份权重的两个副本 ≈ 半张卡
每步 sleep：vLLM 的 7.6 GB → CPU
每步 wake_up：7.6 GB CPU → GPU    ← **OOM 就发生在这一刻**
```

### 25.1 ⚠️ 一条要更正的猜测（原 A4 的答案）

**猜测**（2026-08-13 白天写进 §14 的）：「verl 每步同步的是全量 7.6 GB 权重」。
**查证结果：错了。** verl 0.8 的权重同步**本来就是 LoRA-only**：

```python
# verl/workers/rollout/vllm_rollout/utils.py:262  _update_weights()
if peft_config and base_sync_done:      # 首次之后
    TensorLoRARequest(..., lora_tensors=weights) → add_lora()   # ★ 只推 adapter
else:
    ...全量加载...                                              # 只在第一次
```

而且 `lora_as_adapter`（vllm_async_server.py:186）对我们的配置**已经是 True**，
生成时走 `LoRARequest`、sleep level 也因此自动降到 1。

**⇒ 搬运 7.6 GB 的不是「权重同步」，是 `sleep`/`wake_up` 本身。**
两者是独立的机制，我之前把它们混为一谈了。**教训：定位到"某个东西很贵"之后，
还要再问一次"贵在哪个动作上"——否则会去优化一个已经优化过的地方。**

### 25.2 verl 现有的三种显存模式

读 `vllm_async_server.py` 得到的地图：

| 模式 | 开关 | 每步行为 | 我们的处境 |
|---|---|---|---|
| **A. 睡权重+睡KV**（现状） | `free_cache_engine=True` | sleep level 1：权重→CPU、KV 丢弃；wake_up 全搬回 | **wake_up 时 OOM** |
| **B. 完全不睡** | `free_cache_engine=False` | 无 sleep/wake（`sleep()` 开头就 return） | **搬运消失**，但 vLLM 9.41 GB 常驻 ⇒ actor 只剩 ~21.9 GB |
| **C. 只睡 KV、留权重** | `release_kv_cache()` | 理想形态：权重常驻，只释放 KV | ⚠️ **verl 0.8 里是空壳**（源码注释 `# TODO: support true release of kv_cache`）⇒ **可上游贡献点** |

★ 注意模式 C 对我们收益有限（KV 池才 1.06 GB，释放它省不了多少），
**但它对大池子场景是刚需**——等 H1c 之后池子涨到 10 GB+ 时，C 才真正值钱。

## 26 · 三步解法（施工顺序已定：先大后小）

> Chaoyu 2026-08-13 定的顺序：**H1a 先做**（拆掉"底座重复搬运"这个大问题）→
> H1b 融合 kernel（小 kernel）→ H1c QLoRA → param_offload 最后再试。

### 26.1 H1a · 停止搬运（一个开关，最先测）

**做什么**：`free_cache_engine=False`，让 vLLM 的权重**常驻 GPU、永不搬运**。

**为什么应该成**：wake_up OOM 的直接触发点（一次性映射 7.6 GB 物理页）整个消失。
`sleep()` 在 `free_cache_engine=False` 时开头就 return（源码确认）。

**代价与风险**：
- vLLM 的 9.41 GB 变成常驻 ⇒ actor 的可用空间从「训练时独占 ~21 GB」变成**硬上限 21.9 GB**；
- 而 v8 实测 actor 的 reserved 在 step 24 爬到 **21.08 GB** ⇒ **贴边**，
  这正是为什么 H1b 必须紧跟着做（它把 actor 峰值砍下来）；
- `wake_up()` 未被 `free_cache_engine` 门控，会对一个没睡过的引擎调用 wake_up
  —— 预期是 no-op，**但必须实测确认**。

### 26.2 H1b · 融合 logprob kernel（把 actor 峰值砍下来）

**做什么**：§15 的 K1 kernel 的 RL 变体——算 old_log_prob / ref logprob 时不物化
`[1 × 5120 × 151936]`（fp32 **3.1 GB**，一次前向一份，actor 一步内要算两次）。

**为什么必须自己写**（§9.4 已实测）：verl 自带的融合路径假定 `use_remove_padding=True`，
而 sm_120 没装 flash-attn ⇒ padding 契约对不上直接 AssertionError。
**⇒ 这不是「自己写更好玩」，是「现成的用不了」。**

**预期**：actor 峰值 −3~6 GB ⇒ 从「贴着 21.9 GB 的边」变成「留 5 GB 余量」。
SFT 侧的同一件事已经做完并实测（§27），迁移过来的把握很大。

#### 26.2.1 ⛔ 已推翻（2026-08-13 当晚）：verl 自带的**能用**，我们不必自己写

> **原猜想**（上面这一节的立论）：verl 自带的融合路径假定 `use_remove_padding=True`，
> 而 sm_120 装不了 flash-attn ⇒ padding 契约对不上直接 AssertionError
> ⇒ **「这不是自己写更好玩，是现成的用不了」**。
>
> **实测**：链条上**两个环节都不成立**——
> 1. `use_remove_padding` 不需要真的 flash-attn。verl 从 `flash_attn.bert_padding`
>    导入的**只是四个纯 PyTorch 的 gather/scatter 函数**（`index_first_axis` 那一族），
>    本仓库的垫片（`scripts/install_flash_attn_shim.py`）秒装就够。
>    ⇒ 一度以为要在 sm_120 上编译一两个小时的真 flash-attn，**实测证伪**。
> 2. 垫片装上、`remove_padding=True` + `fused_kernels=True` **一起开**之后，
>    verl 的融合路径正常工作：**actor 峰值 −5.0 GB**（18.76 → 13.92 GB reserved）。
>
> **推翻后**：H1b ✅ 完成，但**不是我们写的**。K1 的 RL 变体降级为"可选、且要赢过它"。
>
> **教训（这条最贵）**：
> **"现成的用不了"这句话，我们是从一次 AssertionError 推出来的，没有再往下查一层。**
> 真正的因果是「A 需要 B，B 我们没有」——而**没去问"B 到底需要它的哪一部分"**。
> 答案是只需要四个纯 Python 函数。
> ⇒ **看到依赖缺失时，先问"缺的是哪几个符号"，再决定是绕过、垫片、还是硬编译。**
> 这和 §25.1 的教训是同一句话的两种说法：**定位到"某个东西不行"之后，
> 还要再问一次"不行在哪个具体动作上"。**
>
> ⚠️ 附带一个必须一起记住的坑（§9.4.1 那张表里也有）：
> **这两个开关必须同时开**。只开 `fused_kernels` 会 AssertionError
> （融合路径返回 padded `[B,T]`，而 `_compute_old_log_prob` 无条件按 unpadded 还原）。
> **"只开一个等于没开"这个形状，当天撞了两次**（另一次是 vLLM 统计的两个日志开关）。

### 26.3 H1c · QLoRA：冻结基座量化（把两份变成一份半）

**做什么**：actor 那 7.75 GB 里 98.4% 是冻结的——**冻结的参数不需要 bf16**。
int8 存、前向时反量化 ⇒ 7.75 → **~4 GB**（4-bit 则 ~2.2 GB）。

**和 LoRA-only 同步是否兼容**：✅ 完全兼容，两者作用在不同对象上——
QLoRA 压的是 **actor 自己持有的基座副本**（训练侧），
LoRA-only 同步管的是**推给 vLLM 的东西**（adapter 依然是 bf16 的 132 MB，不受影响）。
QLoRA 本来就是「4-bit 冻结基座 + 高精度 adapter」这个组合的标准做法。

**风险（必须先验证再投入）**：
- verl 的 FSDP 封装能不能接受量化后的基座权重（bitsandbytes 的 `Linear4bit`/`Linear8bit`
  和 FSDP 的分片语义有已知摩擦）——**这是最大的未知数，要先做可行性探针**；
- 数值影响要过 EVAL 128 配对回归 + TIS 诊断（rollout 和 training 的策略差）。

**⇒ 三步做完的账**：

```
actor       10.5 → ~5 GB      （H1b 去掉 logits 物化 + H1c 基座 int8）
vLLM 权重    7.6 → 7.6 GB     （常驻，不再搬运）
wake_up     +7.6 → 0          （H1a：不搬了）
────────────────────────────────────────
省出         ~13 GB → 全部可以还给 KV 池
KV 池        1.06 GB → 10+ GB ≈ 70k token ≈ **15 条满长序列**（现在是 1.5 条）
```

**这就是 §4「给油箱扩容让发动机转起来」在训练侧的完整兑现**，
而且动机从「让 rollout 快一点」升级成「让 RL 能启动」。

### 26.4 暂缓：param_offload（实测过，暂时不走）

`param_offload=True` **确实解决了 wake_up**（冒烟 #6 过了），但把压力转移到主机内存：
sleep 时 vLLM 也要往 CPU 搬 7.6 GB，加上 actor 的 7.75 GB，
**30 GB 主机内存装不下两份权重** ⇒ EngineCore 被杀（"sleep failed: cancelled"）。
⇒ 交接文档 §5 记的「Ray 杀 worker，内存爆」在新配置下**复现**。
**H1a 做完之后 vLLM 不再往 CPU 搬，那时这条路的压力会小一半，值得重试。**

> 🆕 **2026-08-13 更新：这一整节的物理前提在新机器上没了。**
> 上面每一句话的分母都是「**主机 30 GB 内存**」。新机器 **944 GB**。
> ⇒ `param_offload` / `--object-store-gb 2`（Ray 按 RAM 的 30% 预留对象存储，
> 曾经是杀 worker 的真凶）这两条**都必须重测，不能照抄，也不能照着"已知结论"排除**。
>
> **教训**：显存/内存类的结论**永远要连着硬件一起引用**。
> 本项目已经有一条「信 commit message 的结论、没查那次真正跑通的配置」的教训
> （交接文档 §8-16），这是它的孪生形态：**信了结论，没查结论成立的机器。**

## 27 · 已完成：SFT 侧的稀疏投影（2026-08-13）

`sft.py::token_losses` —— 只对**被监督的位置**做 lm_head + CE。

**实测的负载画像**（这是它值得做的全部理由）：

```
v11 SFT 数据：平均每条 5402 token，**只有 204 个进 loss ⇒ 监督占比 3.8–4.9%**
⇒ 朴素路径 96% 的 logits 算完就被 ignore_index=-100 扔掉
```

| | 朴素 | 稀疏 |
|---|---|---|
| bs=1 峰值 | 16.94 GB | **10.40 GB** |
| bs=2 | 25.54 GB | 12.61 GB |
| bs=4 | **OOM** | 17.23 GB |
| T=8192 | 24.99 GB | 12.26 GB |
| T=12288 | **OOM** | 22.30 GB |
| T=16384 | **OOM** | 16.14 GB |

数学等价（loss 逐位相同），4 条测试守着（含与 HF `model(**batch).loss` 对拍）。

★ **反直觉的实测结论**：跑完同样 16 条样本，**bs=1 最快**（17.4s，bs=4 是 32.7s，
bs=8 是 29.2s，padding 浪费已排除）——4400 token 的序列在 bs=1 时就已经喂饱 170 个 SM，
加 batch 只增显存压力不增并行度。
⇒ **省下的显存该换「更长的序列」（M3/M4 的 8–14 步长轨迹），不是「更大的 batch」。**

★ 为什么上游框架不默认做：通用 SFT 的监督占比通常 30–50%，收益远没这么夸张。
**我们 3.8% 的极端值来自 agent 负载的形状**（prompt 3.2k + 工具返回一大堆，
模型自己说的话很少）——又一条「负载画像决定该优化什么」的 codesign 证据。

## 28 · 顺带修掉的一个致命 bug（不属于 infra，但同一次冒烟测出来的）

`rollout_loop` 每轮无条件发生成请求，不检查剩余预算。response 吃满 1536 时
送进引擎的上下文正好 = `max_model_len` 5120 ⇒ vLLM 抛
`leaves no room to generate` ⇒ **Ray 把整个训练任务杀掉**。

v8 从没炸是因为轨迹短；**v11 的 GEO(14 步)/ATTR(8–10 步) 第一次把预算真正吃满**。
已修（剩余 <`MIN_GENERATION_HEADROOM`=64 token 就标 truncated 收工）+ 守卫测试。
⇒ **教训：长任务会把所有"刚好够用"的边界一次性引爆**，M3/M4 之后这类边界还会有。

---

# 八 · 搬到 4×5090 之后（v0.2.2 新增）

## 29 · 本文档的适用边界，和新长出来的问题

### 29.1 ⚠️ 哪些结论是「单卡 colocate 的产物」

**本文档 §3（显存账本）、§23–26（训练侧显存工程）的每一个数字，
分母都是"一张 31.36 GB 的卡上同时住着 actor 和 vLLM"。** 分卡之后语义全变：

| 本文档的结论 | 4 卡上还成立吗 |
|---|---|
| §3.1 整卡账本 31.36 GB 三分 | ❌ rollout 卡上只有 vLLM，trainer 卡上只有 actor |
| §25「同一基座在卡上存在两份」 | ⚠️ **仍然成立，但从"半张卡的浪费"变成"两张卡各持一份"**——分卡之后它不再是显存压力，而是**权重同步的通信量**（性质变了，见 §29.3） |
| §26.1 H1a 停搬运 | ⚠️ **不再需要**：不共卡就没有 sleep/wake 争抢 |
| §26.2 融合 kernel / `remove_padding` | ✅ **跨机器都成立**（省的是训练侧显存，和卡数无关），继续开 |
| §27 SFT 稀疏投影 | ✅ 成立（单卡训练的事） |
| §4.0 命中率 96.7–97.5% | ⚠️ **只对"单 rollout 副本"成立**。多副本之后要重测——这正是 D5 |

⇒ **多卡的一切去 `docs/distributed-training-design-v0.1.md`。**
本文档从 v0.2.2 起定位收窄为：**单卡/单副本的推理与训练侧优化 + 手写算子**。

### 29.2 ★★ 需要一条新的 codesign 动机（旧的已经兑现完了）

§23 那条论据（「不做 infra，RL 跑不了」）**已经过期**。诚实地说，现在的处境是：

```
RL 能跑了            ✅（91–99 秒/步，稳定）
KV 池空着一半        ✅（不是瓶颈）
命中率 97%           ✅（不是瓶颈）
GPU 零排队           ✅（不是瓶颈）
——那 91–99 秒到底花在哪？   ❓ 不知道
```

⇒ **这个问号就是新动机的唯一来源。** 在回答它之前，任何一道手写菜的优先级都是猜的。
**H0 从"第一步"升级为"唯一能做的下一步"。**

★ 而且现在做 H0 比原计划**更值**：4 卡上可以顺便做
「单卡 colocate vs DP=4 vs 分卡异步」的**时间构成 diff**——
同一份 trace，三种摆法，看时间从哪一格挪到了哪一格。**这是单卡时代做不到的。**

### 29.3 🆕 分卡长出来的新显存账：权重同步的 bucket 在 **rollout 卡**上

> **原预期**（分布式文档 §6 表格）：「`--rollout-gpu-util 0.40` 是和 actor 抢同一张卡
> 算出来的 ⇒ **分卡后 rollout 卡上没人抢，可以给到 0.8+**」。
>
> **实测（2026-08-13，one_step_off，trainer 3 卡 / rollout 1 卡）：不成立。**
> `--rollout-gpu-util 0.85` 跑到**第一次权重同步**时 OOM：
> ```
> vLLM 0.85 × 31.37          = 26.66 GB 常驻（进程实占 27.55 GB）
> CheckpointEngineWorker 自身  2.58 GB
> NCCL bucket                 2.00 GB   ← nccl_checkpoint_engine.py:142 prepare
> 31.37 − 26.66 = 4.71 GB 可用          ⇒ 差 0.13 GB，贴着墙，挂
> ```
> **推翻后**：rollout 卡上**不是只有 vLLM**。trainer 每隔几步要把权重推过来，
> NCCL checkpoint engine 会在**接收端**现开 bucket 暂存区
> （`update_weights_bucket_megabytes` 默认 **2048 MB**，而且源码注释写明
> **send/recv 双缓冲 ⇒ 开销是 2×bucket**）。
> ⇒ 默认降到 0.75，并把 bucket 做成显式参数 `--weight-sync-bucket-mb`。
>
> **教训（本项目第 8 次同一形状）**：
> **"分卡之后没人和它抢"是一个想当然。** 抢的东西换了个名字（从 actor 换成
> checkpoint engine），就没被算进预算里。
> ⇒ **凡是说"某某资源现在独占"，先列一遍"还有谁会在这张卡上分配内存"。**
> ⚠️ 而且它炸得很贵：**发生在第一步训练之后**——前面模型加载、vLLM 建池、
> rollout 全跑完了才炸。**贴着墙的配置会把失败推迟到最贵的时刻。**

★ 顺带修正 §25.1 的一个说法：那里说「LoRA-only 同步只推 132 MB，很便宜」。
**在 colocate 下是对的**（同卡，几乎无成本）；**分卡之后它有了两笔新成本**：
① 接收端 2×bucket 的显存；② 经 PCIe/主机内存的实际传输时间（**5090 没有 P2P**）。
⇒ 「LoRA 让权重同步很便宜」这句话，**分卡之后要重新量**（分布式文档 D7）。

### 29.4 ostinato 的三条线在 4 卡上的接口

| 本文档的线 | 4 卡上的对应实验 | 为什么接得上 |
|---|---|---|
| **前缀缓存**（§4、K2 的容量红利） | **D5 · 多 rollout 副本会不会打碎前缀缓存** | 单副本命中率 97%（§4.0）。**分片之后掉多少没人测过** ⇒ 容量红利可能以"分片损失"的形式复活，而且这次是**真问题**（同组 8 路被分到不同副本 = prefill 算两遍） |
| **R1 · 跨权重版本的 KV 复用 σ²(k)** | **D3 · 异步三模式** | 异步天然产生"用旧权重的 KV 接着 decode"的样本，**不用再人为构造 k** |
| **H0 全身 CT / 通信画像** | **D0 硬件画像 + D7 通信占比** | nsys 的同一条时间线上，把 NCCL kernel 单独拉出来就是 D7 |

⇒ **两个项目不是并行的两摊，是同一台机器上的同一条时间线的两种读法。**

---

# 附录

## A · 开放问题

| # | 问题 | 何时必须回答 |
|---|---|---|
| A1 | sm_120 上 vLLM 0.12 的 fp8 KV attention 后端支持性 | H0/H1 期间，一个 flag 的实验 |
| A2 | Triton `tl.dot` fp8 在 sm_120 是否真发 fp8 mma（防 #7550 式退化） | H3 第一天微基准 + ncu 看指令 |
| A3 | mini-sglang 在 sm_120 的可用性（FlashInfer/FA 依赖；按 TRELLIS 环境仗的经验走社区配方） | H2 第一天 |
| A4 | ~~verl 每步同步全量还是 adapter~~ **已答（2026-08-13，源码查证）**：**本来就是 adapter-only**（首次推基座，之后 `TensorLoRARequest` 只推 132 MB）。真正搬 7.6 GB 的是 `sleep`/`wake_up` 本身，两者是独立机制 —— 详见 §25.1 | ✅ |
| A11 | ~~`free_cache_engine=False` 时 `wake_up()` 对没睡过的引擎是否 no-op~~ **已答（2026-08-13，实测）**：是，H1a 上线后 RL 稳定跑满 2 步冒烟与更长的训练，未见异常 | ✅ |
| A12 | ~~actor 峰值在 vLLM 常驻后是否还装得下~~ **已答（2026-08-13，实测）**：装得下，而且比预期宽松——融合 kernel 把 actor 峰值从 18.76 压到 **13.92 GB**（余量 4.9 GB），不必贴着 21.9 GB 的边 | ✅ |
| A13 | verl 的 FSDP 能否接受 bitsandbytes 量化后的基座（分片语义摩擦） | H1c 可行性探针，投入前必须先答 |
| A5 | 菜单裁剪 vs 跨模板缓存共享：什么 KV 预算下谁赢 | H0 顺手量；「小池子裁剪赢、大池子共享赢」本身就是好结论 |
| A6 | FP4 KV 的 SQNR 决策门（K per-channel 掉到多少 dB 止步） | H5 前哨当天 |
| A7 | ~~eval_local 的生成后端~~ **已答并修复（2026-08-13）**：原是 HF 逐条 generate、每轮重 prefill 整条历史。已实现 `--backend vllm`（AsyncLLMEngine + 组内并发 + prefix caching；采样参数逐项对齐，含 generation_config 隐式的 top_k=20）。冒烟对 HF 约 **11×**；温度 0 并发 4 路轨迹逐字相同 = per-rollout 隔离的端到端证明 | ✅ |
| A8 | int8 KV 在 4B 上是否保持 0.6B 的近无损 | H2 验收 |
| A9 | APRIL 式超发早停的长度偏差如何设对照（研究性，慎动） | 若做，单独设计 |
| A10 | 项目名最终定夺（Ostinato / Groundbass / Palimpsest） | H2 建仓库前 |
| **A14** 🔴 | **每步 91–99 秒到底花在哪**（KV 池空着、零排队、命中率 97%，时间显然在别处） | **H0，现在是第一优先级**。它是 §4.0 之后唯一能重排优先级的依据 |
| A15 | CUDA graph 现在能不能开（`enforce_eager=True` 的前提已随 H1a 消失，见 §6.1） | H0 之后（**先知道 decode 占比，再决定值不值**） |
| A16 | `use_dynamic_bsz=True` 的"免费红利"到底有多少（写了两版没去拿） | H0 顺手 |
| A17 | 4 卡上 `param_offload` / `--object-store-gb` 的新最优值（944 GB 内存，旧结论的物理前提没了，§26.4） | 分卡冒烟稳定之后 |
| A18 | 权重同步 bucket（默认 2048 MB × **双缓冲**）在分卡模式下的最优值：调小省显存 vs 调大省同步次数 | D3 异步实验前，见 §29.3 |

## B · 简历措辞：草稿 → 实测版模板

> 规则：〔 〕内为 H0–H5 实测后填入；没做的线整条删除。

- **前缀缓存与拼装协同**：在多轮工具调用 agent 负载（中位 prompt 3.2k token、GRPO 组内
  8 路共享）上量化 radix/prefix cache 收益，命中率〔H0/H2 实测〕、prefill 计算量下降〔实测〕，
  给出**命中率–缓存容量曲线**；据此反向制定训练与线上统一的提示拼装规范
  （append-only、确定性序列化、易变字段后置），以 CI 校验器落进训练 harness。
- **手写算子（Triton/PTX，RTX 5090 sm_120）**：
  ① fused linear-CE（15 万词表维 online logsumexp，logits 不落地）：训练峰值显存
  −〔实测〕GB，SFT batch 1→〔实测〕，与 verl 内置/Liger 同台跑分〔结果与归因〕；
  ② int8 KV paged cache（block-wise per-channel K + per-token V，SQNR 数据驱动选型）
  + split-K flash-decoding 填满 170 SM：kernel〔实测〕×（理论 2×，gap 归因），
  带宽利用率〔前→后〕，KV 容量 ×2 → 命中率/并发〔实测〕；
  ③ FP8 GEMM on sm_120 roofline〔实测 TFLOPS〕；（可选）④ FP4 KV 精度前哨
  〔SQNR 结论〕——agent 负载下无公开数据的第一份测量。
- **执行层消泡**：nsys/ncu 定位 decode 空转根因，CUDA graph（被显存逼关、由量化容量
  赎回）/overlap/chunked prefill 逐项归因〔H4 表〕，端到端吞吐〔实测〕、显存〔实测〕。
- **任务级验收**（区别于全生态 PPL-only 的做法）：128 条冻结 agent 评测集配对回归
  差值〔实测〕（MDE 0.05）+ 错误构成不变 + RL 采样分布偏移（ESS）〔实测〕。
- （可选，R1 做了）**RL × infra 交叉发现**：权重更新下 prefix KV 陈旧度的 σ²(k) 测量
  与跨步缓存开/关判据；「量化代价 ≈ 陈旧 x 步」换算关系。

## C · 本次调查的主要来源

- sgl-project/mini-sglang（GitHub + LMSYS 博客 2025-12-17）：~5k 行、radix cache、
  overlap scheduling、CUDA graph、Qwen3 支持；无 KV 量化、无缓存指标
- LMSYS 博客 2025-09-10 SGLang HiCache：coding agent 命中率 40%→80%、TTFT −56%
- THUDM/slime + APRIL（arXiv 2509.18521）：partial rollout 吞吐 +22.5%
- Manus《Context Engineering for AI Agents》：KV 命中率第一指标、10× 价差、三条拼装规则
- Blackwell GPU wiki（SM100 vs SM120）+ zartbot sm_120 微架构测量：gen-5 TC 原生
  FP4/FP6/FP8、无 TMEM/tcgen05、mma.sync 编程模型
- triton-lang/triton#7550：`tl.dot_scaled` 在 5090 静默退化 bf16 mma；
  FP4 mma 正确形状 m16n8k32（k 维数 8-bit 容器）
- SageAttention3：sm_120 上 >1000 TOPS FP4 attention 的可行性旁证
- vllm-project/vllm#31085 + 发布记录：SM120 NVFP4/FP8 GEMM 支持在 0.17+（我们在 0.12）
- 本机源码确认：verl 0.8 `enable_prefix_caching: True` / wake_up `reset_prefix_cache` /
  `use_fused_kernels` 默认 false（含 NVIDIA 贡献的 linear_cross_entropy 实现）/
  launch_rl.py 的 `enforce_eager=True`、`free_cache_engine=True`、`use_dynamic_bsz=False`；
  vLLM 0.12 `kv_cache_dtype` 选项与 `prefix_cache_queries/hits` 计数器
- Halftone 仓库（README + ForChaoyu.md）：全部实测数字
