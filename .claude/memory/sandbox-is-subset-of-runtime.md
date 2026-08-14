---
name: sandbox-is-subset-of-runtime
description: 沙盒是 runtime 的子集且契约由 runtime 定义；两边行为不一致，训出来的策略在线上就不成立
metadata: 
  node_type: memory
  type: project
  originSessionId: 957ae9f2-2820-4a54-ab6d-75be32051e25
  modified: 2026-08-14T17:10:18.236Z
---

**M9（2026-08-14）定下的一条贯穿性纪律**：

    沙盒（syncopate/core）可以**简化实现**，但**不能有 runtime 没有的行为**。
    ⇒ 新增工具行为时，**先在 runtime 这边定契约，再让沙盒去满足它**。

**Why**：沙盒的工具是纯函数、无网络、确定性；runtime 要处理超时、重试、限流、部分失败。
**同一个工具两边行为不一致，训出来的策略在线上就不成立** —— 这是训推不一致里最贵的一种，
因为它不是"能力没学到"，是**学了个错的**。

典型反例（都刻意避免了）：
- 沙盒里"重试一定成功" ⇒ 模型学不到"失败之后怎么办"
- 沙盒里错误文本能区分两种超时 ⇒ 模型学会读文本，而真平台没有那个信号

**最锋利的一个例子：超时分两种**

    请求没发出去    重试是安全的
    到了但回包丢了  **重试 = 重复扣款**

现象**一模一样**，所以 `platform.TIMEOUT_MESSAGE` **只有一份**（错误文本逐字相同）。
唯一的分辨手段是**幂等键**。沙盒里这条靠 `EnvSnapshot.failures.side_effect_applied`
建模，runtime 里靠 `tool_calls` 的唯一索引兑现。
⚠️ 实查过：**Meta Marketing API 本身没有幂等机制**，所以这层保证是我们自己给的。

**同源的一条**：M8 的 RAG 也有这个缺口 —— 线上是真向量库 + rerank，沙盒是确定性词法。
处理办法**不是把它藏起来**，而是让沙盒实现**同一份契约**（会返回空、会返回过期的、
会返回只沾边的）。按「沙盒不要比真实世界友好」的原则，把不完美如实建模。

相关：[[project-mechanism-not-wired]] [[syncopate-docs-map]]
