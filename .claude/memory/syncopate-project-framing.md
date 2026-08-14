---
name: syncopate-project-framing
description: Syncopate 的第一目标是手游买量的业务闭环 agent，异步 agentic RL 是并行的第二目标——这个主次曾被搞反
metadata: 
  node_type: memory
  type: project
  originSessionId: 254d8707-7512-4e9b-bd89-6e1eeec39011
  modified: 2026-08-13T17:24:09.211Z
---

**第一目标（真实业务）**：手游买量投放的**全链路闭环** agent。
价值指标是 **span of control** —— 一个优化师能管住的 平台×产品×地域×素材 组合数。

**第二目标（并行，不阻塞第一目标）**：异步 agentic RL 研究
（sync colocate vs one_step_off vs fully_async；σ²(k)、ESS、分布漂移）。

⚠️ **这个主次被搞反过，导致给出过错误建议。** 讨论取舍时先确认是在为哪个目标服务：
比如"要不要为了梯度多样性牺牲正确性" —— 答案是不牺牲，因为第一目标是业务正确性。

**三条线（2026-08-14 起并行）**：
- **训练线** M0–M8：数据 / 沙盒 / 判据 / 模型。当前 v12，M8 施工完待验收
- **Runtime 线** M9：真服务（FastAPI + PG + 幂等 + 审批网关）。几乎不复用训练线代码，
  但**契约同源**（见 [[sandbox-is-subset-of-runtime]]）。M9.1–M9.6 完，压测未做
- **infra 线**：多卡 / 异步 / MoE，另一个窗口负责（见 [[infra-line-state]]）

**三条钉死的前提**：
1. 会变的进 RAG，不变的进权重，绝不能错的进代码
2. 沙盒里只有过程奖励 ⇒ 灰度上线不是验收，是训练的第二阶段
3. 归因延迟是第一性约束（D7 才知对错，D1 数据极易被误当结论）

相关：[[syncopate-docs-map]]
