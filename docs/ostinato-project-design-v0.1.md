# Ostinato · 面向 Agentic 负载的推理 Infra 与「模型 × Harness × Infra」Codesign 设计文档 v0.1

> 2026-08-12。Syncopate 的姊妹项目：Syncopate 提供模型（Qwen3-4B SFT/RL）与 harness（沙盒、
> verifier、评测尺子），本项目补齐第三条腿——推理 infra——并让三者互相塑形（codesign）。
>
> **命名**：Ostinato（音乐术语，固定反复音型）——一段**不变的乐句在变化的旋律下反复出现**，
> 正是 shared prefix 在一批 rollout 下反复复用的形状；与 Syncopate（切分音）同属音乐词根。
> 备选：Groundbass（固定低音）、Palimpsest（重写本羊皮纸——擦而不净的复用）。名字可换，机制隐喻不换。
>
> 阅读顺序：先本文档 → `syncopate-project-design-v0.1.md`（负载来源）→
> `/home/samwang/code/projects/QuantizedKVCache/ForChaoyu.md`（算子资产）。

---

## 0 · 三个必须先钉死的前提

### 0.1 每个进简历的数字必须来自自己负载上的实测

这个项目的直接动机是简历上有 infra 内容，这没什么不好意思承认的。但**简历叙事的可辩护性
来自诚实分层**（Halftone 的元认知 §6 原话）。现在草稿里的数字和已有实测存在出入，先摆到桌面上：

| 草稿里写的 | 实际已有的实测（Halftone） | 处理 |
|---|---|---|
| int8 attention kernel 提速 **1.9×** | **1.56×**（S=4096，差距已定位到 occupancy） | 做完 split-K / batched 之后重测，**写实测值** |
| PPL 仅升 **3.2%** | KV int8 **+0.2%**；q+K+V 全 int8 **+1.2%** | 我们的实测**更好**，直接用 |
| W8A8 GEMM 接入引擎 | Halftone **没有** W8A8 GEMM（那是更早 L40S профiling 的工作） | 列为可选线，做了才写 |
| 命中率 60%+ / prefill 减半 | 从未测过 | P0 先测基线——好消息是 60% **在我们负载形状下可推导**（见 §3.3） |
| 端到端 1.8× / 显存 −45% | 从未测过 | P5 用 trace 重放测，写实测值 |

**规则：设计文档里的目标数字全部标「目标」；简历只收「实测」。**

### 0.2 复用不重造：轮子已经存在，我们做的是「复刻 + 三个增量」

§二的调查会证明 prefix caching / KV 量化 / overlap scheduling 在 SGLang / vLLM / slime 里
都已是成熟或默认能力。**这不是坏消息，是许可证**：我们光明正大地以官方教学实现
**mini-sglang**（sgl-project 出品，~5000 行，支持 Qwen3，自带 radix cache + overlap
scheduling + CUDA graph）为底座做复刻，把力气花在它**明确没有**的三件事上：

1. **KV 量化**（fp8/int8）——mini-sglang 未实现，Halftone 的 kernel 与量化方案直接移植；
2. **prefix cache 可观测性**——mini-sglang 连命中率指标都没有，这是第一个可以往上游提 PR 的点；
3. **agentic 负载的量化验证**——所有开源框架的量化验证都是 PPL / 通用 benchmark，
   没有人拿「多轮工具调用 agent 的任务级正确率 + RL 采样分布」当验收尺子。我们有现成的。

### 0.3 Codesign 的闭环定义：不过 harness 尺子的优化不算完成

「模型 × harness × infra codesign」不是三个项目摆在一起，而是**每一条 infra 改动都要在
另外两侧闭环**：

```
infra 改动（量化 KV / 拼装顺序 / 缓存策略）
   → 模型侧验收：冻结 EVAL 128 × 8 采样配对比较（MDE 0.05）+ cap 分解 + 决策位熵
   → harness/RL 侧验收：rollout 吞吐（65s/步基线）+ rollout↔训练 logprob 差（ESS）+ σ²(k)
   → 反向输出：训练与线上统一的提示拼装规范（infra 的约束反过来塑形 harness）
```

只报吞吐不报任务精度的优化，等于把开源结论搬运一遍，一律不算完成。

---

# 一 · 背景与负载画像

## 1 · Agentic 负载的三个结构特征

| 特征 | 机制 | 佐证 |
|---|---|---|
| **超长共享前缀** | system 规则书 + 工具 schema 对同模板所有 case 逐字相同；GRPO n=8 对同 case 整条 prompt 逐字相同 | 工具菜单裁剪前 prompt 里 **78.7% 是工具说明书**（4889 token 中 3846，commit a2f753） |
| **多轮 KV 只增不减** | 每轮 assistant 输出 + tool observation 追加到上下文，历史逐字保留 | Manus：生产 agent 输入:输出 ≈ **100:1**，KV 命中率是「生产 agent 唯一最重要指标」，cached/uncached 价差 **10×** |
| **decode 小 kernel 碎** | 逐 token 生成时 batch 小、kernel 启动/调度开销占比高，GPU 空转 | Halftone：S=4096 时 decode attention 只有 16 个 program 对 ~170 个 SM，有效带宽 <5% 峰值 |

## 2 · 我们自己负载的实测画像（infra 的输入条件）

全部来自 Syncopate 现有实测，不是估计：

| 量 | 值 | 来源 |
|---|---|---|
| prompt 长度 | 中位 **3198** / max 3587 token（v9，裁剪后菜单 9–13 工具） | v9 生成报告 |
| response 预算 | 1536 token | launch_rl 配置 |
| 轨迹步数 | 平均 ~5 步，归因任务 8–10 步 | 交接文档 §12 |
| GRPO 组结构 | 每步 4 case × n=8 = 32 条序列，**每 8 条共享同一 prompt** | launch_rl 配置 |
| RL 步耗时 | **65 s/步**（50 步实测） | RL_v8_sync_e1b |
| straggler | 最慢/均 **1.37–2.75×**（sync barrier 的浪费上限） | 交接文档 §4.1 |
| 单卡 vLLM 预算 | `gpu_util 0.30` ≈ 9.4 GB（权重 bf16 ~7.6 GB → **KV 只剩 ~1.8 GB**） | 交接文档 §5 |
| 评测集 | 冻结 EVAL 128 × 8 采样，配对 MDE ≈ 0.05 | compare.py 实测 |

## 3 · 单卡 colocate 的 KV 显存账（推算，P0 校准）

Qwen3-4B（GQA 8 KV head × 128 dim，36 层）每 token 的 KV：

$$2_{(K,V)} \times 36 \times 8 \times 128 \times 2\,\text{B} \approx 144\,\text{KB/token (bf16)}$$

### 3.1 KV 预算只够 ~12k token——量化是容量问题不是速度问题

1.8 GB ÷ 144 KB ≈ **12.5k token**。而一个 RL 步名义上需要：
32 条序列 × (3.2k prompt + ≤1.5k response) ≈ **150k token**——差 12 倍。

### 3.2 前缀共享是现在能跑起来的（一半）原因

开着 `enable_prefix_caching`（verl 默认 True）时，8 条同 prompt 的 rollout 共享 KV block：
prompt 侧只需 4 × 3.2k ≈ **12.8k** token——恰好把 KV 预算吃满，response 靠调度器
preemption/排队周转。⇒ **prefix caching 在我们的训练循环里不是锦上添花，是可行性条件**。
这笔账 P0 要实测校准（vLLM v1 自带 `prefix_cache_queries/hits` 计数器，本机 0.12 已确认存在）。

### 3.3 草稿里的「命中率 60%+」在我们负载形状下可推导

prompt 占整条序列 ≈ 3.2k/4.7k ≈ 68%；组内 8 份共享 ⇒ 命中的 prefill token 比例
≈ 7/8 × 68% ≈ **60%**——还没算多轮内部的逐轮复用。所以 60% 不是拍的，是我们负载
结构的下界推算；**实测大概率更高**（多轮追加时每轮都复用全部历史）。

---

# 二 · 开源格局调查（问题一的答案）

## 4 · 总表：谁已经做了什么

| 能力 | SGLang | vLLM | slime | verl（我们在用） | mini-sglang |
|---|---|---|---|---|---|
| prefix cache | ★ RadixAttention 起家之作 | Automatic Prefix Caching，v1 默认开 | 继承 SGLang | rollout 默认 `enable_prefix_caching: True` | ✅ radix cache 默认开 |
| 分层/持久缓存 | **HiCache**（GPU/CPU/存储三级） | LMCache/Mooncake connector | 继承 | ❌（colocate 每步清） | ❌ |
| KV 量化 | fp8 | fp8_e4m3/e5m2（CUDA 无 int8） | 继承 | 未暴露给 rollout 配置 | **❌ ← 我们的增量** |
| 命中率指标 | ✅ | ✅ `prefix_cache_queries/hits` | ✅ | 不上报到训练日志 | **❌ ← 我们的增量** |
| overlap scheduling | ✅ | ✅ | — | — | ✅ |
| agentic RL rollout | — | — | ★ 核心场景，APRIL partial rollout（吞吐 +22.5%） | vLLM/SGLang 双后端 | — |

## 5 · 各家细节

### 5.1 SGLang：这个方向的原创者，也是天花板

- **RadixAttention**（2023 论文）用 radix tree 管理 KV，跨请求自动复用共享前缀 + LRU 驱逐
  + cache-aware 调度。**SGLang 本身就是靠这个起家的**——问题一的直接答案是：是，而且是它的立身之本。
- **HiCache**（2025-09 LMSYS 博客）把 radix 扩成三级（L1 GPU / L2 host / L3 分布式存储）。
  关键实测：Qwen3-Coder-480B 的 coding agent 场景（会话 8 轮 ~25k token），
  **命中率 40% → 80%，TTFT −56%，吞吐 ×2**；通用场景 up to 6× 吞吐。
  阿里 Tair、Mooncake、DeepSeek 3FS 都做了它的 L3 后端——「agentic inference 缓存」已是产业共识。

### 5.2 vLLM：hash-block 版本的同一件事

- APC 用 block hash 链（父块 hash + 本块 token）实现前缀复用，v1 引擎默认开。
- `kv_cache_dtype ∈ {fp8, fp8_e4m3, fp8_e5m2}`（本机 0.12 源码确认；**CUDA 路径没有 int8**）。
- v1 metrics 自带 `prefix_cache_queries/hits`（本机确认）——P0 直接可用。

### 5.3 slime（THUDM）：agentic RL 训练侧的参照系

Megatron 训练 + SGLang rollout，GLM-4.5→GLM-5.2 背后的 RL 框架。多轮工具调用、沙盒交互
是一等公民；**APRIL**（arXiv 2509.18521）的 partial rollout——超发请求、够数即停、
未完成的轨迹跨步续写——rollout 吞吐平均 +22.5%（上限 44%）。
跨权重版本续写轨迹这件事，正是 §12（C3）要量化的 staleness 问题的工程侧先例。

### 5.4 verl（我们的栈）：默认开了，但 colocate 每步全清

- rollout.yaml 默认 `enable_prefix_caching: True` ✅
- **但** colocate 模式 `free_cache_engine: True`（我们 launch_rl.py:153 也显式写死）——
  每个训练步 sleep/wake 一次，**KV cache 整体销毁再重建** ⇒ radix 复用只活在单步之内。
  这不是 bug：权重更新后旧 KV 属于旧策略，async server 在 wake_up 时也主动
  `reset_prefix_cache()`（本机 verl 0.8 源码确认）。**跨步复用 = 接受陈旧 KV**，
  这正是 C3 的研究入口。
- `kv_cache_dtype` verl 配置层存在，但我们的 launch_rl **没暴露** ⇒ P1 加一个透传 flag 就能做实验。

### 5.5 mini-sglang：为我们量身定做的底座

sgl-project 官方教学实现（2025-12 LMSYS 博客），~5000 行 Python：
radix cache / chunked prefill / overlap scheduling / CUDA graph / TP / FlashAttention+FlashInfer，
**支持 Qwen3 系列**（正好是我们的模型）。
**没有的**：KV 量化、投机解码、prefix cache 指标、benchmark 数字。
⇒ 底座有骨架无血肉，我们的三个增量（§0.2）每个都有明确的落点文件。

### 5.6 量化 KV 的更广格局（知识背景，标注待核）

- 工程侧：LMDeploy TurboMind 有 int8/int4 KV 在线量化；TensorRT-LLM 有 int8/fp8 KV（待核实版本细节）。
- 研究侧：KIVI（2bit，K per-channel / V per-token 共识的出处——Halftone 独立复现过）、
  KVQuant、QServe（W4A8KV4）。
- **缝隙**：以上所有工作的精度验证都是 PPL / logits 误差 / 通用 benchmark。
  「量化 KV 对 **多轮工具调用任务正确率** 和 **RL 采样分布**（rollout↔训练 logprob 差、ESS）
  的影响」没有公开的系统测量。这是我们真正能补的东西，而且尺子已经造好了。

### 5.7 Manus 的生产结论：拼装规范不是玄学，是被验证过的工程纪律

Manus 博客（Context Engineering for AI Agents）：KV 命中率是「生产 agent 唯一最重要指标」；
cached/uncached 价差 10×；三条规则——**前缀稳定**（system prompt 开头一个时间戳就毁掉整条缓存）、
**append-only**（不改历史、序列化确定性、JSON key 排序）、**工具 mask 而非移除**（动态删工具
会击穿缓存）。这三条会直接进我们的拼装规范（§10），而且我们 harness 已经满足了两条半（§9）。

## 6 · ★ 结论：三类缝隙，也就是我们全部的工作

1. **观测与规范**：框架有能力、有指标，但「agent 训练负载下的命中率画像 + 训练/线上统一
   拼装规范」没人替你做——这是 workload-specific 的活。
2. **量化 × 任务级验证**：引擎有 fp8 KV，教学底座连量化都没有；int8 方案 + agent 任务级
   + RL 分布级验收，三者的交集为空——Halftone 的 kernel + Syncopate 的尺子正好填上。
3. **跨权重版本的 KV 复用**：所有 RL 框架都在权重更新时清缓存（正确性优先），
   partial rollout 已经开了「跨版本轨迹」的口子，但**陈旧 prefix KV 的分布代价没人量过**
   ——我们有 σ²(k) 的尺子和离线合成方法，别人没有。

---

# 三 · 已有资产盘点

## 7 · Halftone（QuantizedKVCache）：可移植的算子资产

实测结论（RTX 5090 / Qwen3-0.6B / Triton 3.6）：

| 资产 | 实测 | 移植去向 |
|---|---|---|
| 量化方案：K per-channel · V per-token · 对称 int8（SQNR 数据驱动选出） | PPL 1.002×（KV int8） | 直接沿用，换 4B 重验 |
| 物理 int8 cache（HF DynamicCache 子类） | 显存 0.538× → 0.50× | 重写为 paged/radix 布局（mini-sglang 的 cache 结构） |
| fused quantize-on-write kernel | V 2.8–3.3×；K 有 coalescing 病灶待修 | 写入路径，修 2D tiling |
| fused int8 flash-style decode attention | 1.56×（gap 已归因 occupancy） | decode 路径；split-K/batched 后有望 →2× |

Roadmap 里三条「还没做」恰好就是移植必需件：**block-wise per-channel K**（原话：
production-PagedAttention 的答案）、**batched 占用率验证**、**静态量化省 findmax**。
⇒ Ostinato 不是新开一摊，是把 Halftone 的 roadmap 放进一个有真实负载的引擎里做完。

## 8 · Syncopate harness：现成的验收尺子

| 尺子 | 量什么 | 对 Ostinato 的用途 |
|---|---|---|
| 冻结 EVAL 128 × 8 + `compare.py` | 任务分配对差，自报 MDE ≈ 0.05 | 量化/缓存改动的**任务级回归门**（草稿里「任务级精度回归无显著差异」的实测化） |
| 26 个 cap + 恢复动作/defer 双向 | 错误构成 | 精度回归不止看均值，看**错误方向**有没有变 |
| `entropy.py` | 决策位熵 | 量化是否让输出分布变尖/变钝 |
| `staleness.py` | σ²(k)（已有实测点：ESS/N=0.846，σ²(0)≈2.0e-4/token） | C3 的核心尺子 |
| rollout 记账（record_dispatch + rl_report） | 三段耗时 / 分布漂移 | 吞吐收益的分解归因 |
| 65s/步 + straggler 1.37–2.75× | 端到端基线 | 所有加速比的分母 |

## 9 · ★★ Harness 已经天然 cache-friendly——而且是「为别的 bug 立的纪律」顺手做到的

这是整个 codesign 叙事的核心证据。检查现有 harness（本机源码确认）：

| Manus 规则 | 我们的现状 | 当初为什么这么做（跟缓存无关！） |
|---|---|---|
| append-only、不改历史 | ✅ `rollout_loop._append_message` 增量拼 token，单代码路径 | 修 Qwen3 模板不对称（整段渲染 ≠ 增量拼接）逼出来的 |
| 序列化确定性 | ✅ `step_user.txt` 里 `context \| dictsort` | 修「prompt 内容取决于 dict 插入顺序」导致的去重泄漏 |
| 前缀稳定、易变字段后置 | ✅ system 规则书在最前（模板内共享），`reference_now` 在 user 轮首行（共享块之后） | M2 决定 reference_now 必须进 prompt 时顺手放对了位置 |
| 工具 mask 而非移除 | ⚠️ **半符合**：菜单是 per-case 静态裁剪（轨迹内不变，不击穿轨迹内缓存），但跨模板的共享前缀因此缩短 | 为了把 prompt −29% 让 RL 跑得通 |

**论点**：训练一致性纪律（同一路径、确定性序列化、基线可比）和缓存友好性纪律是**同一条纪律**
——都是「字节级稳定的前缀」。这不是巧合：训练要的是可复现的输入分布，缓存要的是可复用的
输入前缀，两者的敌人都是「不确定的字节」。把这条讲透，就是「反向制定训练与线上统一的
提示拼装规范」的理论根据。

最后一行的 ⚠️ 是 C1 里最有意思的实验：**裁剪省下的 prefill token vs 缓存损失的复用 token**，
在什么 KV 预算下谁赢——这是一道真正的 codesign 权衡题，两边都有数。

---

# 四 · 四条工作线

## 10 · C1 · 前缀缓存与拼装协同（观测 → 规范）

**改什么**：不改引擎，先把观测接通——launch_rl/eval 采集 vLLM `prefix_cache_queries/hits`、
prefill token 数、TTFT，进 rl_report 的 wandb 补报；给 `prompt_hash` 机制加「共享前缀长度」
审计（同批 case 两两 LCP 分布）。

**量什么**：RL 步内命中率（组内 8 份共享 + 多轮逐轮复用）、eval 重放命中率、
菜单裁剪对跨 case 共享块的削减量。

**产出**：`docs/ostinato/prompt-assembly-spec.md`——训练（SFT 构造 + RL rollout）与
线上（eval/serving）统一的拼装规范：字段排序、易变字段后置、append-only、静态菜单、
禁止时间戳进 system 段。**harness 侧加一个 CI 校验器**：拼装结果违反规范直接报错
（和 prompt_hash 同一处，守卫长在主路径上）。

**目标读数**（P0 实测替换）：命中率 ≥60%（§3.3 推出的下界）、prefill 计算量 −50%+。

## 11 · C2 · 量化 KV：容量 → 命中率/并发，精度用任务级尺子验收

**两步走，先免费后自研**：

1. **fp8 KV（免费杠杆，一周内出数）**：launch_rl 透传 `kv_cache_dtype=fp8_e4m3`
   （sm_120 后端支持性是第一个要试的风险点）。KV 容量 ×2 ⇒ 12.5k → 25k token，
   组内可同时 decode 的序列数翻倍，preemption 减少。验收：EVAL 128 配对差在 MDE 内
   + cap 构成不变 + rollout↔训练 logprob 差（verl 已记录）不恶化 + 65s/步的变化。
2. **int8 KV（自研主菜，进 mini-sglang）**：Halftone 方案重写为 paged 布局
   （block-wise per-channel K），quantize-on-write 融合进 radix cache 写路径，
   int8 decode attention 换掉 fp16 路径。**验收三层**：kernel 级（对拍 + 带宽利用率）、
   PPL 级（4B 复验 0.6B 的 1.002×）、**任务级**（EVAL 128 配对 + 决策位熵 + 错误构成）。

**为什么值得**（对训练侧的直接收益）：§3.1 的账——单卡 colocate 的 KV 预算只有 ~1.8 GB，
量化等于把这个最稀缺的资源翻倍；多轮长会话显存减半、并发接近翻倍这两句草稿话
在这个负载下是**同一件事的两个读数**。

**RL 特有的一问**（没人测过）：量化 KV 让 rollout 的采样分布偏移多少？
用 ESS/logprob-gap 量——如果偏移和 σ²(k) 同量级，说明「量化的分布代价 ≈ 多陈旧一步」，
这个换算关系本身就是一个可写进报告的发现。

## 12 · C3 · 跨权重版本的 KV 复用 × staleness（研究亮点，可选）

**问题**：所有 RL 框架在权重更新后清 prefix cache（verl wake_up 时 reset，本机源码确认）。
但 agent 负载的前缀大头是 system 规则 + 工具文档——**这部分 KV 对权重更新的敏感度
到底多大？** 如果 σ² 很小，跨步保留 prefix cache 就能把「每步重算 12.8k token prefill」
省掉，这是 colocate 单卡最贵的重复劳动之一。

**方法**（不需要双卡，全部离线可做，工具已有）：拿第 t−k 步 ckpt 算 prompt 的 KV，
第 t 步权重接着 decode，和全新计算比 logprob 漂移 → 得到「KV 陈旧度版本」的 σ²(k) 曲线，
和已有的「策略陈旧度」σ²(k) 对照。**决策规则可以直接写**：若 KV-σ²(k=1) ≪ 策略-σ²(k=1)，
则「LoRA 小步更新下跨步保留 prefix cache」是账面上划算的，报告给出开/关判据。

这条线把 Syncopate 的第二目标（异步 RL 研究）和 infra 焊在一起，是三者 codesign 最强的一格。

## 13 · C4 · 执行层消泡（工程线，草稿第三条的实测化）

mini-sglang 已有 overlap scheduling 和 CUDA graph——所以这条线不是从零写，
是**测量驱动的 ablation**：nsys 拆解 5090 上 Qwen3-4B agent decode 的时间线
（kernel 间隙 / CPU 调度 / 采样开销），逐项开关 overlap、CUDA graph、chunked prefill，
把每一项的贡献率钉死。产出一张「泡从哪来、被谁消掉」的归因表，
替换草稿里的「1.4–1.7×」为分项实测。投机解码列为远期可选（mini-sglang 也没有，工作量大）。

---

# 五 · 落地路线

## 14 · 里程碑（P0–P6）

原则沿用 Syncopate：每个里程碑有可验收的数字；先测量后动手；基线跑之前把预期写死。

| # | 里程碑 | 内容 | 验收 | 预估 |
|---|---|---|---|---|
| **P0** | 基线画像 | 现有 verl+vLLM 栈接通 `prefix_cache_*` 指标；跑一轮 eval 重放 + 一段 RL，记录命中率/prefill 节省/TTFT/显存/65s 分解；§3 的显存账实测校准 | 一份基线报告，所有后续加速比的分母 | 1–2 天 |
| **P1** | 免费杠杆 | launch_rl 透传 `kv_cache_dtype`；fp8 KV 的 EVAL 128 配对回归 + 吞吐/并发变化；菜单裁剪的缓存代价量化（§9 权衡题） | fp8 可用性结论 + 配对差 < MDE 或明确超出 | 2–3 天 |
| **P2** | 拼装规范 v1 | `prompt-assembly-spec.md` + harness CI 校验器（挂在 prompt_hash 处） | 规范落库，校验器有测试 | 2 天 |
| **P3** | mini-sglang 起步 | 跑通 Qwen3-4B；**加 prefix cache 命中率指标**（上游没有 → 可提 PR）；用 Syncopate trace 重放复现 radix 收益 | 我们 trace 上的命中率/TTFT 复现表 | 3–5 天 |
| **P4** | int8 KV 移植 | Halftone → paged 布局（block-wise per-channel K）+ 写路径融合 + int8 decode attention 接入；顺手做 split-K/batched 把 1.56× 往 2× 推 | kernel 对拍 + 4B PPL + **EVAL 128 任务级配对回归** | 1–2 周 |
| **P5** | 消泡 ablation + 端到端 | nsys 归因表；overlap/CUDA graph/chunked prefill 开关矩阵；trace 重放的端到端吞吐/显存对比（草稿 1.8×/−45% 的实测替换） | 归因表 + 端到端数字（诚实版） | 1 周 |
| **P6**（可选） | 跨版本 KV 复用 | KV-σ²(k) 曲线 vs 策略-σ²(k)；跨步保留 prefix cache 的开/关判据 | 一份研究小报告 | 开放 |

依赖关系：P0→P1→P2 在现有栈上串行（一周内）；P3 起切换到 mini-sglang 底座；
P4 依赖 P3；P5 依赖 P3（P4 可并行）；P6 随时可插（纯离线）。

**与 Syncopate 主线的资源冲突**：P0/P1 需要 GPU 各半天级别，避开 M2+M3 重生成后的
重训窗口即可；P3+ 大部分是写代码和小模型调试，冲突小。

## 15 · 评测协议（口径先钉死，防止数字注水）

| 数字 | 口径 |
|---|---|
| 命中率 | **token 级**：`prefix_cache_hits / prefix_cache_queries`（vLLM v1 计数器），分「RL 步内」和「eval 重放」两个场景报，不报请求级（虚高） |
| prefill 节省 | 命中 token 数 / 总 prompt token 数，同 trace 对比 |
| 端到端吞吐 | **固定 trace 重放**：从 Syncopate 导出（RL 步 32 序列组结构 + eval 104×8），同一 trace 在优化前后各跑三遍取中位；不用合成随机负载 |
| 显存 | 同 trace 峰值 reserved（按几十步后的峰值算，不按第一步——wake_up OOM 的教训） |
| 精度回归 | EVAL 128 × 8 配对差 ± CI，报 MDE；cap 构成表同列；「无显著差异」必须写成「差值 x ± y，MDE z」 |
| kernel 加速比 | 和 fp16 孪生 kernel 对比 + 理论上界 + gap 归因（Halftone 的规矩） |

## 16 · 交付物清单

1. **Ostinato 仓库**：mini-sglang fork + int8 KV + 指标 +（可选）上游 PR；
2. **agent-trace replay benchmark**：Syncopate 负载导出的可复放 trace + 跑分脚本——
   本身是个可开源的小件（别人只有 ShareGPT 式负载，没有多轮工具调用 + GRPO 组结构的）；
3. **拼装规范文档 + harness 校验器**（落回 Syncopate 仓库）；
4. **测量报告**：基线画像 / fp8 / int8 / 消泡归因 / （可选）KV-staleness；
5. **简历 bullet 实测版**（附录 B 模板填数）。

---

# 附录

## A · 开放问题

| # | 问题 | 什么时候必须回答 |
|---|---|---|
| A1 | sm_120 上 vLLM fp8 KV 的 attention 后端支持性（FA3/FlashInfer 对 Blackwell 消费卡的 fp8 KV 路径） | P1 第一天，试了就知道 |
| A2 | mini-sglang 在 sm_120 的可用性（它依赖 FlashInfer/FA——TRELLIS 环境仗的经验：新硬件按社区配方走） | P3 第一天 |
| A3 | int8 KV 在 4B 上是否保持 0.6B 的近无损（Halftone roadmap 原有疑问） | P4 验收 |
| A4 | 菜单裁剪 vs 缓存共享的权衡最终判决（可能的结论：KV 预算小时裁剪赢，预算大时共享赢——这本身就是个好结论） | P1 |
| A5 | C3 的结论如果是「KV 陈旧代价不可忽略」怎么办 → 也是合法结论：解释了为什么所有框架都清缓存，负结果照样进报告 | P6 |
| A6 | W8A8 GEMM 线（草稿第二条前半）做不做：Halftone 没有现成件，纯新增 1–2 周。倾向 **先不做**，int8 KV + 消泡的故事已完整；做的话挂 P5 之后 | P5 结束时 |
| A7 | 项目名最终定夺（Ostinato / Groundbass / Palimpsest） | P3 建仓库前 |

## B · 简历措辞：草稿 → 实测版模板

> 规则：〔 〕内为 P0–P5 实测后填入；没做的线整条删除，不留含糊表述。

- 前缀缓存与拼装协同：在多轮工具调用 agent 负载（中位 prompt 3.2k token、GRPO 组内 8 路共享）上
  量化 radix/prefix cache 收益，命中率〔P0/P3 实测〕、prefill 计算量下降〔实测〕；
  据此反向制定训练与线上统一的提示拼装规范（append-only、确定性序列化、易变字段后置），
  并以 CI 校验器落进训练 harness。
- 量化 KV cache：基于实测 SQNR 选型（K per-channel / V per-token / 对称 int8），
  int8 物理缓存显存 0.5×、PPL +0.2%〔4B 复验后更新〕；fused int8 decode attention
  〔split-K 后实测〕×（理论上界 2×，gap 归因到 occupancy）；
  **任务级验收**：128 条冻结评测集配对回归差值〔实测〕（MDE 0.05）。
- 执行层消泡：nsys/ncu 定位 decode 空转根因，overlap scheduling / CUDA graph /
  chunked prefill 逐项 ablation，贡献归因〔P5 表〕，端到端吞吐〔实测〕、显存〔实测〕。
- （可选，若 P6 做了）RL 特有发现：权重更新下 prefix KV 陈旧度的分布代价 σ²(k) 测量，
  给出跨步保留缓存的开/关判据——推理缓存策略第一次由 RL 训练理论的尺子来定。

## C · 本次调查的主要来源

- sgl-project/mini-sglang（GitHub + LMSYS 博客 2025-12-17）：~5k 行、radix cache、
  overlap scheduling、CUDA graph、Qwen3 支持；无 KV 量化、无缓存指标
- LMSYS 博客 2025-09-10 SGLang HiCache：coding agent 命中率 40%→80%、TTFT −56%
- THUDM/slime + APRIL（arXiv 2509.18521）：partial rollout 吞吐 +22.5%
- Manus《Context Engineering for AI Agents》：KV 命中率第一指标、10× 价差、三条拼装规则
- 本机源码确认：verl 0.8 `rollout.yaml enable_prefix_caching: True` / async server
  wake_up 时 `reset_prefix_cache` / vLLM 0.12 `kv_cache_dtype` 选项与 `prefix_cache_*` 计数器
- Halftone 仓库（README + ForChaoyu.md）：全部实测数字
