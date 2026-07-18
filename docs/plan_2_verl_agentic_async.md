# 计划书 Task 2：verl Agentic RL — 同步 colocate vs Fully-Async 全对比

> 目标硬件：本地 1×RTX 5090 (32GB) 做 smoke test → 云端 4×5090 正式训练（必要时升 8 卡）
> 核心目标：在同一个多轮 agentic 任务上，亲手跑同步 colocate 模式和 fully_async_policy 模式，从机制层面理解「长尾 rollout → GPU 空转 → 异步化」这条因果链；同时备好 slime 逃生通道
> 前置：Task 1（VeRL-Omni）完成后开始，verl 环境可部分复用
> 预计周期：3–4 周（含 buffer）

---

## 0. 全局认知：先想清楚 fully-async 到底在解决什么

三个先验判断，整个项目就是用实验去验证/修正它们：

1. **同步 RL 的根本矛盾是 batch barrier × 长尾。**
   一个 batch 里最慢的轨迹（工具调用多、环境响应慢、生成长）决定整个 batch 的 rollout 时间，其余 GPU 全在等。agentic 场景下轨迹长度方差天然巨大，所以矛盾被放大。数学上：同步模式的 rollout 耗时 ≈ max(t_i)，异步流式 ≈ mean(t_i)，差距由长尾分布的尾部决定。

2. **fully-async 用「数据陈旧性 (staleness)」换吞吐。**
   Trainer 和 Rollouter 解耦后，rollout 用的策略落后于当前策略 1~k 个版本，这是 off-policy 偏差。参数同步频率是核心旋钮：同步越频繁 → 越接近 on-policy 但吞吐收益越小。verl 官方在 128 卡 Qwen2.5-7B 上报 2.35–2.67× 加速。

3. **预判：4 卡小规模下加速比可能 ≤ 1。**
   类比你 MoE-DeepEP 的教训（单机 NVLink 下通信-计算 overlap 是负优化）：异步化的收益是长尾严重度和规模的函数。4 卡分离式部署（如 3 train + 1 rollout）会牺牲 rollout 算力，参数同步还有额外开销。**但这不影响学习目标**——我们要的是亲手观察机制（staleness、同步频率、partial rollout），并定量回答"收益从什么规模/什么长尾形态开始为正"。这个判断框架本身就是产出。

**社区已知风险**：有人在跑 fully-async 时遇到 verl 框架支持问题转投 slime。所以本计划内置 slime 逃生通道（Phase 4），且 Phase 1 起所有任务定义/数据/reward 都做成框架无关的格式，保证可迁移。

---

## Phase 0：单卡 smoke test（本地 5090，2–3 天）

**目标**：用最小模型在本地把「多轮 agentic rollout + GRPO」的完整代码路径打通，确认 sm_120 上 verl + sglang/vllm 栈可用。

### 0.1 环境

- [ ] verl main 分支 + sglang（rollout 默认引擎，vllm 做备选）；确认 sm_120 wheel 可用性（sglang 对 Blackwell 的支持状态实测为准）
- [ ] 镜像策略复用 Task 1 的 setup 脚本

### 0.2 任务与模型选型

- [ ] 模型：**Qwen2.5-1.5B-Instruct**（单卡 smoke 专用；32GB 上 1.5B full-param GRPO + colocate rollout 可以跑）
- [ ] 任务：verl 官方 agentic 入口任务 **GSM8K + calculator/code 工具调用**（tool_agent_loop）。理由：官方例子 = 路径最稳，先排除任务侧变量
- [ ] 跑通 colocate 同步 GRPO 50 step，确认 reward 上升

### 0.3 读码（带问题清单）

- [ ] Agent Loop 架构：server-based rollout 中 inference engine（server）和 agent（client）如何分离？asyncio 协程怎样避免等工具返回时 GPU 空转？
- [ ] 权重同步：colocate 模式下 FSDP → sglang 的 reshard + 更新链路在哪个文件、走什么通信？
- [ ] 画一张 verl 同步模式数据流 Mermaid 图（之后和 fully-async 的图对照，差异即本质）

**学习点**：verl 的 HybridFlow 编程模型（single-controller + multi-worker）；agent loop 的协程调度；多轮轨迹的 loss masking（工具返回 token 不算 loss——这是 agentic RL 最常见的正确性坑，亲手检查 mask）。

**验收标准**：1.5B 本地训练曲线正常 + 两个机制问题能笔头回答 + 数据流图完成。

---

## Phase 1：多卡同步基线（云端 4×5090，4–6 天）

**目标**：建立一个"值得被异步化"的同步基线——任务要有真实长尾，否则 Phase 2 的对比没有意义。

### 1.1 任务升级：从工具调用到真 agentic

- [ ] 模型：**Qwen2.5-7B-Instruct + LoRA**（4×32GB 上 7B full-param 的 AdamW 状态算不过账：fp32 master+m+v ≈ 84GB，FSDP 分片后叠加 rollout 引擎太紧；LoRA 是预算内正解。若后续升 8 卡可改 full-param）
- [ ] 任务二选一（按长尾强度排序）：
  - **A. Search-R1 式多轮检索问答**（你复现过 TinyZero/读过 Search-R1，迁移成本最低；检索延迟天然制造长尾）
  - **B. verl-agent 的 ALFWorld**（最长 50 步的长 horizon 任务，长尾最极端，但环境部署成本高）
  - 建议 A 起步，B 作为 Phase 3 加压选项
- [ ] reward/数据格式写成框架无关的独立模块（slime 逃生通道的前置投资）

### 1.2 基线测量（Phase 2 的对照组，认真做）

- [ ] 跑 200+ step 同步训练，记录：reward 曲线、每 step 耗时分解（rollout / old_logprob / update / weight sync）
- [ ] **关键测量：rollout 时长分布**。导出每条轨迹的生成耗时，画直方图，量化长尾（P50 vs P99）。这是后面解释异步收益的核心证据
- [ ] nsys 抓 2–3 个 step 的 timeline，标出 GPU 空转段及其原因（等最慢轨迹？等工具返回？权重同步？）

**学习点**：colocate 模式的资源切换机制（训练和推理共享 GPU，offload/reload 的开销有多大）；长尾的真实形态；agentic RL 的训练不稳定性（多轮任务 reward 方差远大于单轮）。

**验收标准**：同步基线 reward 收敛 + 长尾分布图 + GPU 空转归因表。

---

## Phase 2：fully_async_policy 切换与对照实验（4×5090，5–7 天）

**目标**：核心阶段。同任务切换到 fully-async recipe，定量对照，理解每个新旋钮。

### 2.1 架构切换

- [ ] 部署形态：4 卡拆分 **3 trainer + 1 rollouter**（备选 2+2，对比一次资源配比敏感性）
- [ ] 读 fully_async_policy recipe 源码，重点回答：
  - Trainer 和 Rollouter 之间的样本队列实现（streaming 粒度是什么？单样本还是 micro-batch？）
  - 参数从 Trainer 推到 Rollouter 的同步机制（NCCL broadcast？checkpoint 文件？sglang 的 update_weights 接口？）
  - **partial rollout**：参数更新到来时未完成的轨迹如何处理（中断重续？用旧策略跑完？两种选择的 off-policy 含义不同）
- [ ] 更新 Phase 0 的数据流图 → 同步 vs 异步对照图

### 2.2 对照实验矩阵（控制预算，每格 100–150 step）

| 实验 | 变量 | 看什么 |
|---|---|---|
| E1 | 同步基线（Phase 1 已有） | 对照组 |
| E2 | async，默认同步频率 | 吞吐 vs E1；reward 曲线是否劣化 |
| E3 | async，同步频率调稀 2–4× | staleness 加大后训练质量的退化形态 |
| E4 | async，partial rollout 开/关 | 长尾消除效果 vs off-policy 代价 |
| E5（选做） | 3+1 vs 2+2 资源配比 | rollout/train 算力配比对吞吐瓶颈的影响 |

- [ ] 统一指标：samples/sec（端到端）、达到同一 reward 水平的 wall-clock、staleness 分布（每条轨迹的策略版本差）
- [ ] 写结论：**在本规模 + 本任务长尾形态下，fully-async 的收益是正是负？拐点估计在哪？**（即使结论是"4 卡下是负优化"，这也是高质量结论——和你 MoE 项目的 overlap 结论同款）

### 预期坑

1. fully-async recipe 是较新代码路径，配置项文档滞后，参数含义要读源码确认
2. 异步模式下 wandb 指标的 step 语义会乱（trainer step ≠ rollout 版本），先想清楚怎么对齐再画对比图
3. 权重同步与 sglang 推理并发时可能死锁/显存尖刺（社区转投 slime 的常见触发点，遇到先翻 verl issue 区）
4. 1 张 5090 做 rollouter 服务 7B 模型 + 高并发多轮请求，KV cache 容量可能成为新瓶颈（长尾从"计算慢"变成"排队等 KV"——瓶颈迁移本身就是值得记录的现象）

**学习点**：异步 RL 的全部核心机制（staleness / 同步频率 / partial rollout / 资源配比）+ 一套"异步化收益判断框架"。

**验收标准**：E1–E4 完成 + 对照报告 + 收益判断结论。

---

## Phase 3：深挖与加压（选做模块，按兴趣和预算挑 1–2 个，3–5 天）

- [ ] **A. 长尾加压**：换 ALFWorld（50 步 horizon）重跑 E1/E2 对照——验证"长尾越重异步收益越大"的预测曲线
- [ ] **B. Megatron 后端**：同任务把训练后端从 FSDP 切到 Megatron（verl 双后端），对比吞吐/显存/配置复杂度；顺便踩 HF ↔ Megatron checkpoint 转换的坑（对 Task 2.5 的 slime 也是前置投资）
- [ ] **C. 升 8 卡复测**：用 8×5090 重跑 E2，验证规模对异步收益的影响（拐点估计的实证）
- [ ] **D. VLM agentic**：换 Qwen2.5-VL-7B + GUI/视觉任务（verl-agent 支持视觉环境），体验多模态 rollout 在 agentic 场景的额外坑（图像 token 的 mask、显存）

---

## Phase 4：slime 逃生通道 / 对比通道（条件触发或主动选择，4–6 天）

**触发条件**（满足其一）：
- Phase 2 遇到 fully-async blocker，verl issue 区无解且自己修不动（卡 > 3 天）
- Phase 1–3 顺利完成，主动追加框架对比（推荐：这是最好的 blog 素材）

### 步骤

- [ ] slime 环境：官方 docker 镜像起步（slimerl/slime），Megatron + SGLang 原生栈
- [ ] checkpoint 转换：HF → Megatron torch_dist（著名坑区，预留一整天）
- [ ] 用 Phase 1 准备好的框架无关任务模块接入 slime 的 Data Buffer / custom rollout 接口，复跑同步 + 异步两组
- [ ] 架构对比笔记：verl 的 HybridFlow 抽象 vs slime 的 native pass-through（参数直传、不加中间层）——两种框架哲学在「改起来顺不顺手」「出 bug 好不好查」上的真实差异，用你自己的踩坑记录做证据

**学习点**：第二个框架学到的永远不是用法而是设计空间——同一问题的两种解法对照，才知道哪些复杂度是本质的、哪些是框架自找的。

---

## 预算估算

| 项 | 配置 | 估时 | 估价 |
|---|---|---|---|
| Phase 1 | 4×5090 | ~40 GPU·hr×4 | $100–180 |
| Phase 2 | 4×5090 | ~60 GPU·hr×4 | $150–300 |
| Phase 3 | 4×5090（C 选项 8 卡） | 时间盒 | $100–250 |
| Phase 4 | 4×5090 | ~40 GPU·hr×4 | $100–180 |
| **合计** | | | **$450–900（按选做范围浮动）** |

省钱原则同 Task 1：本地调通再上云；用 1.5B/3B 验证所有 pipeline 改动后才换 7B；E2–E4 每格控制在 150 step 内（看趋势不看收敛）。

---

## 风险与退出条件

| 风险 | 信号 | 应对 |
|---|---|---|
| sglang sm_120 支持不全 | rollout 引擎报错 | 切 vllm rollout（verl 双引擎，一行配置）；再不行云端换 A100/H100 |
| fully-async blocker | 卡 > 3 天 | 触发 Phase 4 slime 通道，verl 留 issue |
| 7B LoRA 在 4 卡上 rollout 吞吐过低 | E1 单 step > 15 min | 降 3B full-param（机制学习不受影响）或升 8 卡 |
| agentic 任务训练不收敛 | 200 step reward 无趋势 | 先回退单轮任务验证 RL 链路正确性，再排查 mask/reward/模板（90% 的"不收敛"是正确性 bug 不是调参问题） |

---

## 总学习清单（自查表）

- [ ] 能画出 verl 同步 colocate 与 fully-async 两种架构的数据流对照图
- [ ] 能解释 staleness、参数同步频率、partial rollout 三个旋钮各自的吞吐/质量权衡
- [ ] 能基于长尾分布 + 规模给出"是否值得异步化"的定量判断框架
- [ ] 理解 agentic RL 的 loss masking、多轮 credit assignment 的工程实现
- [ ] （若做 Phase 4）能从设计哲学层面对比 verl 与 slime
- [ ] 产出：1 篇对照实验博客 + 长尾/吞吐数据集 + ≥1 个上游 issue
