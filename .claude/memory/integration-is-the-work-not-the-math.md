---
name: integration-is-the-work-not-the-math
description: 一个被证明正确的优化，接进真实系统花了 13 处修复；数学零错误，全错在接缝
metadata:
  type: project
---

**E26（2026-08-19）**：PrefixGrouper（论文保证训练等价）在我们这里的账 ——

```
数学部分   等价性(fp32 逐位) · 因果对齐 · 掩码语义 · response-only 投影   **零错误、零改动**
微基准     三次前向 **3.96×**（纸面上界 4.12× 的 96%），显存仅 +3.6%
真实集成   **16 次尝试 / 13 处接线 / 至今未通**（卡在 Adam dtype）
分布       主线 1 · verl 缺陷 3 · **我们接错 9**
```

**Why**：我们改的是**两个系统之间的接口**，而接口错误的特征是
**上游产出「形状对、语义错」的东西，下游照单全收继续走，直到某层碰上处理不了的
类型/长度/dtype 才炸** —— 中间每一层都没有契约检查。
⇒ 实测 **13 处里 5 处报错在别人家里**（verl postprocess / verl padding / NCCL / torch Adam），
**没有一次报在我们改的那几行上**。

**How to apply**：
- 接缝上工作时，**报错位置的信息接近误导性**。有效手段只有两个：
  ① 在接缝处**自己打判据行**（如「组构成 [8]」「grouped 9311 → 投影 5776」——
     这两条直接抓出了碎片化和 padding 混入）；② 把问题**降维成秒级观测**。
- ⛔ **别用「起训练」当调试循环**：最近 6 轮里有 3 轮在修诊断工具本身。
  正确姿势是脱 Ray 的最小复现。
- 多 actor（Ray）架构里，补丁要问「装在**哪个进程**」——
  `_compute_old_log_prob` 在 trainer driver、模型构建在 WorkerDict；装错的表现是
  「补丁装了、也打印了、现象一字不变」。
- `setup_worker` 里**绝不能 import 会碰 CUDA 的模块**（它早于 Ray 分卡 ⇒ 三 rank 挤一张卡）。
  加 import 前先用 `torch.cuda.is_initialized()` 对照一行。
- ★ **13 处里只有 2 处是被判据主动抓住的**，其余靠崩溃暴露 ⇒ 如果它们碰巧不崩，
  拿到的就是**跑得飞快、结果全错**的训练。这就是等价性判据必须排在吞吐之前的原因。

相关：[[trainer-is-compute-bound-not-starved]] [[blank-thresholds-are-not-passes]]
[[project-mechanism-not-wired]] [[feedback-measure-dont-infer]]
