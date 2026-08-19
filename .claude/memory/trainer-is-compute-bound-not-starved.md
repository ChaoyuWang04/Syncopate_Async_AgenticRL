---
name: trainer-is-compute-bound-not-starved
description: trainer 慢是「活多 11.6 倍」不是「没喂饱」；喂饱 GPU 的单位是 token 不是序列条数
metadata: 
  node_type: memory
  type: project
  originSessionId: 8fda7c79-7275-5040-8ca9-2552dddaa97f
  modified: 2026-08-19T13:34:02.557Z
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

微基准 **3.96×**（纸面上界的 96%）；~~真实集成未通~~ → ✅ **当天下午集成通了 + 同尺子 A/B 定案**：
**端到端 2.31×**（34.52→14.94 s/gstep）、PG 净效果 2.23×、微基准兑现 ~70%。
⇒ 接线那本账（含集成卡点的真身 = 绕过根 FSDP 的归约竞态）在 [[integration-is-the-work-not-the-math]]。

**「token 饱和」的第三次验证（A/B 白捡的两条）**：
- 不开 PG 时 mb1→mb8 只 +3.8%（且显存 29–31/32 GB 贴顶——E25 定长探针的 mb=4 OOM
  没迁移到变长真实批，但这余量不能当生产配置）
- 开 PG 时 mb8→mb16（两组/批，组构成 [8,8] 正常）反而**慢 5.7%** ——
  打包后 ~9400 token/前向已饱和，再翻倍只多付组间补齐
⇒ **PG 的生产配置定格：mb=8（一组一批），别再调**。

**两个只有实测才能发现的坑**：
- `position_ids` 透传给子调用 ⇒ FA2 当成变长序列 ⇒ **非法访存**（崩溃点报在别处，
  CUDA 异步错误的假位置；flash_attn 本身所有形状都正常）
- 只有 SDPA 能跑通的那一版，净收益只有 **1.22×**；修好 FA2 之后是 **3.96×**
  ⇒ ★ **「唯一能跑通的配置」≠「最优配置」** —— 前者会让 4× 的优化看起来像 1.2×

★★ **最贵的一课**：我写的「等价判据 = logprob 逐位相同」在 **bf16 下无法执行** ——
噪声地板本身就有 mean 1.28e-2 / max 1.0。因此在噪声里追了三轮假根因。
⇒ **判据太严和太松一样糟**；救命的动作是**加一个噪声地板对照**
（同数据同数学、只改一个无关变量）。见 [[blank-thresholds-are-not-passes]]
