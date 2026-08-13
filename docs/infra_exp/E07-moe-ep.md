# E07 · MoE 与专家并行

> 状态：🟡（决策完成 2026-08-13，探针未跑）   最后更新：2026-08-13

## 0 · 结论卡片

| | |
|---|---|
| **问题** | 在 4×5090（无 P2P，卡间 6.4 GB/s）上训一个 MoE 的最佳「模型×框架×并行」组合是什么 |
| **答案（决策，待实测验证）** | **GLM-4.7-Flash（30B-A3B，MIT）+ verl + LoRA + GSPO**；训推分离：trainer FSDP2×3 + rollout 4bit 量化×1。EP 先用手写 toy 台架感受，Megatron 探针通过后上真的 |
| **信心** | 中。显存账是推算的，探针 P1–P6 一个没跑；量化 rollout 的 mismatch 和 MoE 路由问题是已知风险 |
| **决策日期** | 2026-08-13，Chaoyu 批准（「研究 MoE 是主目的，框架是其次」） |
| **下一步** | 跑探针 P1–P3 |

---

## 1 · 决策记录（为什么是这个组合）

### 1.1 框架：verl（抛开沉没成本重选，仍然是它）

三条硬筛子：① 训练后端要在 sm_120 上活下来；② 训推分离+异步是必选项；③ 重型多轮 agentic rollout 要有成熟接口。

| 框架 | 淘汰/保留理由 |
|---|---|
| **verl** ✅ | 唯一三条全过且**本机已实证**（FSDP 纯 PyTorch 路径 + vLLM 0.12 + 自制垫片全通）；三档异步全有；FSDP2 与 Megatron 双后端留了后路 |
| slime / Miles | 训练侧 Megatron-only。**保留为教材**（读源码）+ Miles 的 R3 是 MoE 线必读。见 §1.2 的风险修正 |
| AReaL | 训练侧 Megatron-only（lite 版可能有 FSDP，探针 P5）；decoupled PPO 是异步研究线的必读论文 |
| NeMo-RL | 备胎 #2：DTensor+LoRA GRPO 都有，但整套 NV 容器生态没在消费卡上验证过 |
| ROLL / TorchForge / TRL | 借鉴思想（环境级异步 / TorchStore / 默认 TIS），不作为底座 |

### 1.2 ⚠️ 一条风险评估的修正（2026-08-13，来自 Chaoyu）

> 原评估把「Megatron/TE 在消费卡上跑不通」列为高风险。
> **用户信息：老师有 4×4090 跑通 slime/Megatron 的实际案例。**
> 记两点：① 这显著降低了 Megatron 路线的风险预期——消费卡上这条路**有人走通过**；
> ② 但 4090 是 **sm_89（Ada）**，TE 对它的支持是成熟的；5090 是 **sm_120**，
> TE 的 FP8 示例在 5090 上仍有 open issue。⇒ 探针 P4（TE bf16 编译）照跑，
> 成本半天，通过就解锁真 EP。**风险从「大概率不通」下调为「待验证、有先例」。**

### 1.3 模型：GLM-4.7-Flash（30B-A3B）

| 候选 | 判断 |
|---|---|
| **GLM-4.7-Flash**（2026-01，MIT，31B 总/~3B 激活） | ★ 选它：唯一为 agentic 调过的这个体量 MoE（τ²-Bench 79.5 / SWE-bench 59.2 / 原生工具调用），vLLM day-0，社区有 GPTQ-4bit |
| Qwen3-30B-A3B-2507 | fallback：verl 有现成 megatron recipe，最稳妥 |
| Ling-mini-2.0 / ERNIE-4.5-21B-A3B（16–21B） | 跳板：若 31B 全链太痛苦，先拿小 MoE 跑通 |
| Qwen3-Next-80B-A3B | ❌ bf16 160GB 装不下 |
| gpt-oss-20b | ❌ 暂缓：MXFP4 在 sm_120 上回退 Marlin，训练路径最不成熟 |

### 1.4 算法：GSPO 而不是 GRPO

MoE 的 RL 有一个专属问题（Miles 的 R3）：**推理和训练两侧的 router 可能选不同的专家**——
稠密模型的 mismatch 只是数值偏差，MoE 是走了不同的计算路径。verl 没有路由重放；
**GSPO 的序列级重要性比率对 token 级路由抖动远不敏感**（Qwen 团队设计它的动机之一
就是 MoE RL 稳定性），verl 已支持。⇒ MoE 线一律 GSPO + rollout-IS，ESS 每步盯。

---

## 2 · 三种训练摆法（本实验组的核心对照）

> 洞见（待验证）：这台机器卡间 6.4 GB/s 且所有流量本来就经主机内存 ⇒
> **「量化复制」可能赢「分片聚合」**——通信最贵的机器上，最好的并行可能是不通信。

| 配置 | 摆法 | 每步训练侧通信（推算） | 预测 |
|---|---|---|---|
| **A · FSDP2 分片** | trainer 3 卡 FULL_SHARD，bf16 基座 62GB÷3≈21GB/卡 + 激活~3GB ✓ | 每次前向+反向都要 all-gather 冻结基座：每 micro-batch ~90GB 跨卡 ⇒ **15–20 s/micro-batch 纯通信** | 能跑但慢；grad-accum 会线性放大 |
| **B · CPU offload** | 权重驻主机内存（944GB），用时搬上卡 | 与 A 同量级——**反正都过主机**，这是本机特有的等价性 | 和 A 差距远小于正常机器；值得实测证明 |
| **C · QLoRA 复制** ★ | 基座 4bit ~17GB **每卡完整一份**，纯 DP，LoRA 梯度 all-reduce | **~132MB/步** ——比 A 少三个数量级 | **本机最快的 MoE 训练摆法**；代价是 4bit 基座质量 + 训练侧也引入量化 |
| rollout（共用） | 1 卡 GPTQ-4bit（~18GB 权重 + ~8GB KV）；FP8 要 vLLM 0.17+ 的 SM120 GEMM，暂不动 | 权重同步 LoRA-only 132MB | 2+2（bf16 TP=2 rollout）作为 mismatch 对照组 |

**EP 单列**：真 EP 需要 Megatron（P4 探针后）。在此之前用 **E01 microbench 里手写
~200 行 toy MoE + 显式 all-to-all** 感受 dispatch/combine/负载不均。
推算：20k token × top-8 × hidden 2048 × 3/4 跨卡 ≈ 每层 ~1GB 往返 ⇒ 40+ 层 ⇒
**每步十几秒级纯 a2a**。「EP 在无 P2P 消费卡上是不是必然负收益、曲线什么形状」
是本实验组最有独特价值的产出（数据中心论文里没有这条曲线）。

## 3 · 预测（跑之前写死）

1. 配置 C（QLoRA 复制）训练侧吞吐 ≥ 配置 A 的 **3 倍**；若没有，说明瓶颈不在通信（回 E01 查）。
2. 配置 A 与 B 的差距 < 30%（「反正都过主机」假说）；若 B 慢得多，说明 H2D 和卡间 SHM 走的不是同一条瓶颈路。
3. 4bit rollout + bf16 训练的 ESS 会**明显低于** dense 线（路由抖动叠加量化），GSPO 下仍可训（reward 不崩）；若 ESS < ~0.5，降级到 2+2 bf16 rollout。
4. toy EP 的 a2a 占比 > 50% 每步时间。

## 4 · 探针清单（半天以内/个，按序）

| # | 探针 | 回答什么 |
|---|---|---|
| P1 | vLLM 0.12.0 加载 GLM-4.7-Flash GPTQ-4bit | day-0 支持在不在我们这个版本；不在则量 vLLM 升级和 verl 0.8 的兼容风险 |
| P2 | verl FSDP 路径加载 GLM-4.7-Flash + LoRA r32 前向一步 | HF Glm4MoE 类和 verl actor 的兼容性 |
| P3 | 4bit rollout vs bf16 ref 的逐 token logprob 差（用现有 ESS 尺子） | mismatch 是「可修正」还是「不可用」级 |
| P4 | TE bf16-only 在 sm_120 编译（后台挂着） | 真 EP / verl-Megatron 的生死；有 4×4090 先例（§1.2） |
| P5 | AReaL-lite 有没有 FSDP 后端（读仓库） | 异步研究线的第二条腿 |
| P6 | bitsandbytes 4bit 在 sm_120 + verl FSDP 下能不能构建 actor | 配置 C 的生死 |

## 5 · 已知风险

- **R3（路由不一致）**：无路由重放，靠 GSPO + ESS 监控兜底；ESS 崩了升级 rollout 精度。
- **量化 mismatch 双重来源**：rollout 4bit ≠ 训练 bf16（配置 A/B），或训推**同为** 4bit 但 kernel 不同（配置 C）。每个配置都要单独量 ESS，不能互相引用。
- **显存账全是推算**：GLM-4.7-Flash 的 hidden/层数没核对，激活值按 dense 经验估的。P2 探针第一件事就是校准这张表。

## 6 · 环境指纹（决策时点）

```
2026-08-13 · 4×RTX 5090 32GB sm_120 · 无 P2P · 卡间 6.4 GB/s (NCCL_CUMEM_ENABLE=0)
verl 0.8.0 / torch 2.9.0+cu128 / vllm 0.12.0 / flash_attn 2.8.3 真轮子（2026-08-13 晚装，/workspace/wheels/）
megatron/torchtitan/TE/bnb 均未装（P4/P6 探针对象）
```
