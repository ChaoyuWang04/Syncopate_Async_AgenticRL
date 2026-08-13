---
name: feedback-measure-dont-infer
description: 从正确的观察推出修法然后直接改代码，一天里错了两次；要先用小探针证伪，且一次只变一个变量
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 254d8707-7512-4e9b-bd89-6e1eeec39011
  modified: 2026-08-13T17:23:44.178Z
---

2026-08-13 一天里犯了两次同一个错，**都是从一个正确的观察推出一个错误的修法，而且代码改完了才去测**：

| 正确的观察 | 我推出的修法 | 实测 |
|---|---|---|
| 5090 的 P2P 全关 | 设 `NCCL_P2P_DISABLE=1` | ❌ 完全无效。真根因是 P2P 缺失 **×** Ray 只给每个 worker 开一张卡 |
| 序列才 4k token 喂不饱 5090 | 开 `use_dynamic_bsz` 打包成 16k | ❌ 慢 2.2×。因为 flash-attn 是垫片，打包后注意力退化成 O(总长²) |

**Why**：这个项目的历史教训里，"用推理代替测量"已经反复出现（交接文档记着
"看到 bf16 让内存降一半就推断可以开 param_offload —— 没测，爆了"）。
推理链看起来越顺，越容易跳过验证。

**How to apply**：
1. **改代码之前先想：能不能用五分钟证伪它？** 四行探针脚本、两分钟，
   抵得上几轮"改一改再跑跑看"。
2. **把条件拆开，一次只变一个。** 调优时我同时改了 DDP 和 dynamic_bsz，
   1.36× 的收益分不出是谁的；补了 2×2 对照才发现一个 ÷3.0、一个 ×2.2，**方向相反**——
   差点把倒退当收益收下。
3. **结论要带成立范围。**「flash_attn 垫片就够了」对正确性够用，对序列打包不够用。

相关：[[machine-4x5090-constraints]] [[project-mechanism-not-wired]]
