# E07 · MoE 与专家并行

> 状态：🟡（决策完成 2026-08-13，探针未跑）   最后更新：2026-08-13

## 0 · 结论卡片

| | |
|---|---|
| **问题** | 在 4×5090（无 P2P，卡间 6.4 GB/s）上训一个 MoE 的最佳「模型×框架×并行」组合是什么 |
| **答案（决策，2026-08-14 更正）** | ~~GLM-4.7-Flash~~ → **`Qwen3-30B-A3B` + verl + LoRA + GSPO**（GLM-4.7-Flash 的 `Glm4MoeLiteForCausalLM` 当前栈不支持，需 transformers 5.0rc，见 §4.5.1）；训推分离：trainer×3 + rollout 4bit 量化×1 |
| **信心** | 中→**中高**。显存/通信账**已用真 config 算过**（§4.5.2）：bf16 61.1 GB、4bit 16.8 GB、分片每 micro-batch gather 61.1 GB 而只用 10%。探针 P1–P6 仍未跑；量化 mismatch 与 MoE 路由仍是已知风险 |
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

## 4.5 ⛔ 选型更正 + 显存账校准（2026-08-14，实测）

### 4.5.1 GLM-4.7-Flash **在当前栈上跑不了** —— 决策要改

只下了 `config.json`（几 KB）就查实：

```
GLM-4.7-Flash 的 architectures   ['Glm4MoeLiteForCausalLM']
                 model_type       glm4_moe_lite
                 config 自述      transformers_version: 5.0.0rc0
本机 transformers 4.57.6          models/ 下只有 glm4_moe，❌ 无 glm4_moe_lite
本机 vLLM 0.12.0 registry         只有 Glm4MoeForCausalLM，❌ 无 Lite
```

⇒ **探针 P1（vLLM 加载）与 P2（verl FSDP 建 actor）会当场失败。**
§1.3 里写的「vLLM day-0 支持」对的是更早的 GLM MoE，**4.7-Flash 用的是全新架构类**。

⇒ 要用它就得升 transformers 到 5.0rc + 换 vLLM ——
而这套栈是硬啃下来的（sm_120 + FA2 真轮子 + vLLM 0.12），**动它风险远大于收益**。

★ **改用本文档 §1.3 自己写的备选 `Qwen3-30B-A3B`**（当时评价"最稳妥"）：
`Qwen3MoeForCausalLM` / `qwen3_moe` 在 **transformers 4.57.6 与 vLLM 0.12.0 中均原生支持**（已验证）。
已下载 `models/Qwen3-30B-A3B-Instruct-2507`（26 文件 / 16 safetensors / 57 GB）。

**教训**：**「day-0 支持」这类说法必须落到 `architectures` 字段上验证，不能引用新闻稿。**
成本：下一个 config.json，10 分钟；省下的：几天的白工。

### 4.5.2 显存与通信账（§5 说"全是推算"，现在用真 config 算了）

`Qwen3-30B-A3B`：hidden 2048 · 48 层 · **128 专家 top-8** · moe_inter 768 · vocab 151936（不共享 embedding）

```
每层   attention 18.9M  +  MoE 604.0M   ⇒  623.1M
总参数 30.53 B     bf16 61.1 GB     4bit(含 scale) ≈ 16.8 GB
激活/token 3.04 B  ⇒ 名副其实的 A3B（10%）
```

| 摆法 | 每次通信 | @6.4 GB/s |
|---|---|---|
| **A · FSDP 分片** | 每 micro-batch all-gather **全部专家 61.1 GB** | **9.5 s 纯通信/micro-batch** |
| | ★ 但只有 **10%（3.04B）** 真被用到——top-8/128，**其余 90% 是白 gather** | |
| **C · 4bit 复制** | 每步只同步 LoRA ≈ **0.13 GB** | **20 ms** |
| | | ⇒ **通信量差 470×** |

**单卡能否装下（32 GB）**：bf16 61.1 GB ❌ / **4bit 16.8 GB ✅**（余 15.2 GB 给激活+KV+LoRA）

⇒ **§2 的核心洞见被真数字加强了**：MoE + 无 P2P 的组合下，
分片不只是"通信贵"，而是**贵在 gather 了 90% 用不到的专家**。
这正是「通信最贵的机器上，最好的并行是不通信」最锋利的例证。

⚠️ 原 §2 表里「~90GB 跨卡 ⇒ 15–20 s」是按 GLM 推的，**现在的实测口径是 61.1 GB ⇒ 9.5 s**。

### 4.5.3 ★★★ `target_modules=all-linear` 在 MoE 上是灾难（纯 CPU 探针，2 分钟）

meta 设备实例化骨架（不读权重、不占显存）后数 Linear：

```
Linear 总数 18,673
  gate_proj / up_proj / down_proj  各 6,144   ⇒ 专家里 18,432（**98.7%**）
  q/k/v/o_proj                     各    48   ⇒ 注意力      192
  router(gate)                          48
  lm_head                                1
```

`launch_rl` 目前对 dense 写死 `all-linear`（注释说明了理由：只挂注意力容量差 2.8 倍）。
**但那条推理只对 dense 成立**：

| 方案 | 可训练参数 | bf16 体积 = 每步同步量 | LoRA 张量个数 |
|---|---|---|---|
| **all-linear** | **1696 M** | **3.39 GB** | **37,346** |
| 仅注意力 q/k/v/o | 26.7 M | 0.05 GB | 384 |
| **注意力 + router** ★ | **30.1 M** | **0.06 GB** | **480** |
| 对照：dense Qwen3-4B all-linear | 66 M | 0.13 GB | 504 |

⇒ **MoE 上 all-linear 是 dense 的 26×（参数）/ 74×（张量数）。**

★★ **和 E13 叠加起来更糟**：E13 已证明**逐张量 CPU 拷贝是 proximal anchor 的瓶颈**
（504 个张量 0.037 s）。线性外推 **37,346 个 ≈ 2.7 s**，
而每步要拷三趟（save + 2×restore）⇒ **光快照就约 8 s/步**，
还没算 `collect_lora_params` 同样逐张量的遍历。

★ **更根本的问题**：top-8/128 ⇒ **每个专家平均只见到 1/16 的 token**，
1696 M 可训练参数里绝大多数梯度极稀疏 —— 花 26× 的代价买到的是稀疏更新。

⇒ **决策：MoE 线用「注意力 + router」（30.1 M，与 dense 的 66 M 同量级）。**
router 尤其要挂——它直接决定专家选择，正是 §5 里 R3（训推路由不一致）的作用点。
⚠️ 但「注意力 vs all-linear 的容量差」在 MoE 上是多少，**没人测过**，
所以这是**基于成本的默认选择，不是已验证的最优**；真要比容量，得单独设对照。

⇒ 已把 `target_modules` 从写死改成显式参数 `--target-modules`（dense 默认不变）。

**教训**：**继承来的默认值要跟着模型结构一起重新审。**
`all-linear` 这四个字在 dense 上是最佳实践，在 MoE 上是 26× 的账 ——
而这个差别，**用 meta 设备数一下 Linear 就能发现，两分钟、零显存。**

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

---

## 9 · ⛔ A9（2026-08-18）：**预量化存盘救不了碎片** —— 这条路是死的

**背景**：A1 证明 4bit MoE 能跑，但**加载时 bnb 逐层量化造成严重碎片**
（权重 13.32 GB，却有 **17.43 GB reserved-but-unallocated** ⇒ 直接 OOM）。
当时靠 `expandable_segments:True` 解掉，**但它在真训练路径上用不了**
（与 vLLM colocate 的内存池冲突，pytorch#147851，`launch_rl` 专门 pop 掉它）。

**A9 的三阶段实验**（尺子 `scripts/infra/probe_moe_4bit_load.py`，判据写死在探针里）：

| 阶段 | 结果 |
|---|---|
| ① 在线 4bit 量化 | 🔴 **复现**：已分配 **13.32 GB** / 碎片 **17.43 GB** ⇒ OOM（与 A1 的数字**逐位相同**） |
| ② 量化一次并存盘 | ✅ 成功（开了 `expandable_segments`，离线一次性动作），产出 `models/Qwen3-30B-A3B-nf4`（16 GB） |
| ③ **从预量化的盘上加载** | 🔴 **也 OOM**：已分配 13.40 GB / 碎片 **17.46 GB** —— **和在线量化几乎一模一样** |

> **原预测 P2**：预量化加载的碎片 < 1 GB ⇒ A2 的加载路径就用它。
> **实测**：碎片 17.46 GB，**和在线量化没区别**。
> **推翻后（P3 成立）**：**碎片不来自"量化这个过程"，来自 bnb 4bit 的权重布局本身** ——
> Qwen3-30B-A3B 有 **18,432 个专家 Linear**，即上万个小张量，
> 每个都要一块小显存 ⇒ 分配器被打得稀碎。**换个时间点量化，张量个数不变，碎片就不变。**
> **教训**：**"提前算好存起来"只能省掉计算，省不掉数据结构本身带来的代价。**

### 9.0 ⛔ 先认一个错：**"expandable_segments 用不了"这个前提，被我过度推广了**

A1 当时写的是「`expandable_segments` 在真训练路径上用不了（与 vLLM colocate 内存池冲突）」，
我后来一路把它当成"**所有模式都用不了**"，A9 的整个设计（找替代加载路径）就建立在这上面。

**Chaoyu 2026-08-18 指出的问题成立**：

```
冲突的原文是             AssertionError: Expandable segments are not compatible with memory pool
                        —— 这个 memory pool 是 **vLLM 在同一个进程里**建的
而我们的 MoE 配置是      fully_async 3+1：**3 张训练卡上根本没有 vLLM**
                        vLLM 只活在 rollout 那一张卡的**另一个进程**里
⇒ **冲突的前提是"同进程"，而分卡模式下它们本来就不同进程**
```

⇒ **所以最直接的解法一直摆在那儿：只给 trainer 的 worker 进程开 `expandable_segments`。**
⇒ A9 三阶段的**测量本身是对的**（碎片确实来自权重布局，预量化确实救不了），
**但"必须换加载路径"这个动机是错的** —— 我们不需要换路径，只需要**分进程设环境变量**。

★ **教训**：一个限制条件被记下来的时候带着它的**成立范围**（"colocate 内存池"），
而在后续引用中范围被丢掉了，只剩下结论。
⇒ **凡是"某某用不了"的结论，引用时必须把它的成立条件一起搬过来。**
⚠️ 具体到代码：`launch_rl` 现在是**无条件** `env.pop("PYTORCH_CUDA_ALLOC_CONF")`，
要改成**只在 colocate 下 pop**（排成 **A19**，改动 3 行）。

### 9.05 那 17 GB 到底是什么（回答"是不是为反量化预留的"）

**不是。** 它不是为反量化预留的空间，而是 **PyTorch 分配器手里"要不回去也用不上"的碎块**：

```
bnb 的 4bit 加载是**逐层做**的：读一层 fp16 权重 → 量化成 4bit → 释放 fp16 那块
⇒ 一层一层地 申请大块/释放大块/申请小块，来回上万次（Qwen3-30B-A3B 有 18,432 个专家 Linear）
⇒ 分配器的空闲块被切得又碎又不规则，后面来一个 2 MB 的请求，
   明明总共空着 17 GB，却**没有一块连续的 2 MB** ⇒ OOM
```

⇒ 你问的「能不能规定一把固定的尺子、用到哪块反量化哪块」——
**这正是 `expandable_segments` 做的事**：它把显存池变成一个**可增长的连续虚拟区间**，
碎块可以被重新拼起来用。所以答案是：**能，而且不用我们自己写**，
只是要**在没有 vLLM 的那个进程里**开（见 §9.0）。

⚠️ 而**推理时的反量化是另一回事**：bnb 在**每次 matmul 时临时**把用到的那块权重解回 bf16，
用完就丢，占用是**瞬态的、按 tile 大小的**，不是这 17 GB。**两件事别混。**

### 9.06 量化后端的区别（回答"为什么是 bnb，AWQ/GPTQ 有什么不同"）

| 后端 | 什么时候量化 | 怎么存 | **能不能训练** | 典型用途 |
|---|---|---|---|---|
| **bnb NF4**（我们用的） | **加载时在线量化** | 4bit 打包 + 每 64 个数一个 scale | ✅ **能** —— QLoRA 就是它，梯度经反量化后的权重流回 LoRA | **训练** |
| **GPTQ** | **离线**，用校准集逐层最小化误差 | int4 + group scale | 🔴 基本**只做推理**（kernel 不为反向设计） | 推理 |
| **AWQ** | **离线**，按激活重要性保护显著通道 | int4 + per-channel scale | 🔴 同上 | 推理 |

⇒ **我们选 bnb 不是因为"只有它能 4bit"，是因为「**只有它支持训练**」** ——
AWQ/GPTQ 是推理量化，拿来做 LoRA 训练在这个栈里走不通。
⇒ **所以"换后端"并不能解决碎片问题**（它们连训练都不支持），
真正能解决的是 §9.0 的分进程 `expandable_segments`，或者更根本的「融合专家权重」。

⚠️ 一个**成立范围**：AWQ/GPTQ 用在 **rollout 那张卡的 vLLM 上**是完全可行的
（vLLM 原生支持）——那属于 E19 §3-③ 的范畴，和训练侧的量化是**两件独立的事**。

### 9.1 ⇒ A2 的加载路径还剩哪些选项

```
❌ 预量化存盘                     本节证伪
❌ expandable_segments（同进程）   与 vLLM colocate 的内存池冲突
🟡 **分进程**：fully_async 下 trainer 与 rollout **本来就是两个进程**
   ⇒ trainer 侧开 expandable_segments、vLLM 那个进程不开 —— **冲突的前提是同进程**
   ★ 这是目前最有希望的一条，而且改动只是"给 trainer 的 worker 加环境变量"
🟡 换量化后端（张量更少/更大）     bnb 之外还有 AWQ/GPTQ/compressed-tensors，但要先确认 verl+LoRA 支持
🟡 融合专家权重（一个大张量而不是 18432 个小的）  改动最大，但直击根因
```

⇒ **下一步（A9b）**：验「分进程开 expandable_segments」这条 —— 成本很低，
判据是 **trainer 进程里 4bit 加载不 OOM，且 vLLM 那个进程照常起来**。
