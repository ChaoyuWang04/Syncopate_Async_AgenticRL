---
name: disaggregation-is-a-memory-decision
description: 我们的训推分离是显存逼出来的，不是为了吃异步的重叠收益；负载配比与业界相反
metadata:
  type: project
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

相关：[[infra-line-state]] [[machine-4x5090-constraints]] [[blank-thresholds-are-not-passes]]
