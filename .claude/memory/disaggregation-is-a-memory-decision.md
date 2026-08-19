---
name: disaggregation-is-a-memory-decision
description: 我们的训推分离是显存逼出来的，不是为了吃异步的重叠收益；负载配比与业界相反
metadata: 
  node_type: memory
  type: project
  originSessionId: 8fda7c79-7275-5040-8ca9-2552dddaa97f
  modified: 2026-08-19T13:33:43.206Z
---

**我们的 3 trainer + 1 rollout 不是「为了异步」，是显存逼出来的**：
一张 32 GB 的 5090 上 trainer 峰值就 15.55 GB，vLLM 还要 KV cache ⇒ colocate 时
`gpu_memory_utilization` 只能给 0.5，四卡全 colocate 会更挤。

**而这个配比与业界完全相反**（2026-08-19 检索）：
AReaL 把 **3/4 的卡给推理**；verl 自己的 fully_async 文档是 rollout 96 : trainer 32；
SemiAnalysis 的四个真实案例是 1:3 / 1:1 / 1.3~2:1 / 1:2（trainer:generator）。
**原因是输出长度**：他们是 32K token 的 rollout，我们 response 均值只有 **654**（差 40–50 倍）。

**实测的失衡**（E25 + E08）：
```
生成 48 条：1 卡 ×  9.5 s =  9.5 GPU-秒
训练 48 条：3 卡 × 27.7 s = 83   GPU-秒     ⇒ 训练一条的成本是生成它的 8.7 倍
rollout 卡 idle_ratio 0.70，且**每次都是被陈旧度上限叫停的**（14/14），不是没活干
把上限放大 4 倍（26→105），idle 几乎不动（0.70→0.72）
```

**How to apply**：
- **别再把「异步」当成吞吐结论的来源**。在这个负载上，训推分离买到的主要是
  「显存放得下」，不是「重叠省时间」——重叠早就吃满了，rollout 干完就在等。
- ⇒ 陈旧度旋钮测不出代价，是因为**剂量本来就极低**（轨迹跨版本 span 实测 **0/21312**，
  `log_ppl_diff` 中位 3.3e-4 ≈ vLLM↔FSDP 的数值地板）。别读成「异步没有代价」。
- ⇒ **「fully_async 对任务无损」目前没有证据** —— B3 三模式的学习类对比已作废，
  干净基线上**从没在任务尺子上比过** colocate vs fully_async。空白 ≠ 通过。
- ⇒ 要让配比合理，只有让 trainer **少算**（见 [[trainer-is-compute-bound-not-starved]]）。

## ★★ 2026-08-19 下午：配比之谜的完整解释（Chaoyu 问「为什么我们和大厂反着」）

**逐 token 的账（普适）**：生成 1 token ≈ 2N FLOPs;训练 1 token = 3 次前向 + 反向 + GC 重算
≈ **10–12N** ⇒ trainer 天生干 5–6 倍的活。

**把配比反过来的是负载形状**：
- **我们**：prompt 4100 / response 650、组内 8 条共享题面。rollout 侧 vLLM prefix cache
  实测命中 **98.7%** ⇒ 题面只 prefill 一次;trainer（PG 前）把题面**×8 再 ×10N** 地算
  ⇒ FLOPs 比 ≈ **20:1**（380k·N vs 19k·N）⇒ 3:1 给 trainer 都不够。
- **大厂**：long-CoT，response 16k+、prompt 短 ⇒ 没有大题面可共享;decode 串行且带宽受限
  （MFU 个位数 vs 训练 40–60%）+ 长尾 ⇒ rollout 占端到端 **70–85%**（SortedRL/RollPacker/
  Laminar/MIT News, 2026-08-19 检索）⇒ 卡往 rollout 堆 + 异步吃长尾（AReaL 等 2.2–2.7×）。

**与 DDP/消费卡通讯无关，TP/PP 会更糟**（全有实测）：LoRA 下 all-reduce 占步 <0.1%;
E25 GPU 算力已饱和;TP=2 rollout 净负 20%（E04）;ZeRO-3 慢 6×（E18）。4B+LoRA 单卡放得下
⇒ **DP 就是正确答案，4D 里另外三个 D 在这个规模不需要**。

**解法与后果**：让 trainer 少算（= E26 PG，端到端 2.31×）⇒ gen 占步 12%→**26%**
⇒ **trainer 越快，配比越向业界靠，陈旧度的剂量条件第一次真正具备**——
加速 trainer 不只是省时间，是「异步的代价」这个研究问题的**前置条件**。
⇒ 上面那条「rollout 干完就在等」的旧图景已开始失效，B11 配比实验要按新形状重想。

相关：[[infra-line-state]] [[machine-4x5090-constraints]] [[blank-thresholds-are-not-passes]]
[[trainer-is-compute-bound-not-starved]]
