---
name: trainer-is-compute-bound-not-starved
description: trainer 慢是「活多 11.6 倍」不是「没喂饱」；喂饱 GPU 的单位是 token 不是序列条数
metadata:
  type: project
---

**E25（2026-08-19）实测证伪了「trainer 没喂饱」**：`micro_batch` 1→2 是**负收益**
（定长慢 1.0% / 变长慢 6.3%，多花 4.2 GB；mb=4 OOM），关 `gradient_checkpointing`
在 **mb=1 就 OOM**。

**Why**：一条序列已有 **~4850 token**（4196 题面 + 654 回答），一次前向就是
`[4850 × 2560]` 量级的 GEMM —— GPU 早就吃饱了。
⇒ ★ **喂饱 GPU 的单位是 token，不是序列条数。「batch=1」在 LLM 训练里不代表批次小。**
变长负载上 `mb=1` 还等价于**完美打包**（mb>1 要 pad 到 max(lens)）。

**How to apply**：
- `micro_batch=1` / `dynamic_bsz=False` / `gradient_checkpointing=True` 三条**别再动**，
  理由已写进 `launch_rl.py` 的参数 help 与 `00-INFRA-HANDOFF §2`。
- 省时间只剩「**让它少算**」：prefix grouper（8 条样本共享题面只算一次，上界 4.1×）·
  砍 ref（12.7%）· ref 走 FP8。
- ⚠️ `use_prefix_grouper` verl 内置但**不是即插即用**：包没装，且 verl 把 `response_mask`
  （我们的**梯度**掩码）当成「token 存不存在」⇒ 多轮下会把**工具返回从输入里删掉**且不报错。
  立项判据必须是「开/关两条路 logprob 逐位相同」。
- ★ 教训：我把「峰值 15.55/32 GB ⇒ 还剩 16 GB」当成余量，而那 15.55 **正是 GC 省出来的**
  ⇒ [[feedback-measure-dont-infer]] 第三次兑现：**一个指标换个前提就不是同一件事**。

相关：[[machine-4x5090-constraints]] [[infra-line-state]] [[project-mechanism-not-wired]]

## 🆕 2026-08-19 续：prefix grouper 兑现了（E26）

**三次前向 3.96×**（10.211 → 2.577 s），显存 11.88 → 12.31 GB（+3.6%），
纸面上界 4.12× **兑现 96%**。等价性 fp32 + 噪声地板全过。
⚠️⚠️ **但这只是微基准** —— 真实训练集成 16 次尝试、13 处接线，**至今未通**
（卡在 Adam dtype，E26 §6.3）⇒ **端到端吞吐数一个都没有，3.96× 不许当端到端引用**。
⇒ 接线那本账单独记在 [[integration-is-the-work-not-the-math]]。
⇒ 「让它少算」这条路是对的，而且比预期兑现得更彻底。

**两个只有实测才能发现的坑**：
- `position_ids` 透传给子调用 ⇒ FA2 当成变长序列 ⇒ **非法访存**（崩溃点报在别处，
  CUDA 异步错误的假位置；flash_attn 本身所有形状都正常）
- 只有 SDPA 能跑通的那一版，净收益只有 **1.22×**；修好 FA2 之后是 **3.96×**
  ⇒ ★ **「唯一能跑通的配置」≠「最优配置」** —— 前者会让 4× 的优化看起来像 1.2×

★★ **最贵的一课**：我写的「等价判据 = logprob 逐位相同」在 **bf16 下无法执行** ——
噪声地板本身就有 mean 1.28e-2 / max 1.0。因此在噪声里追了三轮假根因。
⇒ **判据太严和太松一样糟**；救命的动作是**加一个噪声地板对照**
（同数据同数学、只改一个无关变量）。见 [[blank-thresholds-are-not-passes]]
