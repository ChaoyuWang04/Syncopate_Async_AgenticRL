---
name: infra-line-state
description: infra 线（多卡/异步/MoE）的已定决策与当前状态；入口是 docs/infra_exp/00-INFRA-HANDOFF.md
metadata: 
  node_type: memory
  type: project
  originSessionId: d8054c42-ce87-481d-a266-b7806a058358
  modified: 2026-08-13T17:35:10.675Z
---

infra 线与主线训练**分开交接**：主线看 `docs/syncopate/05-handoff.md`，
infra 线看 **`docs/infra_exp/00-INFRA-HANDOFF.md`**（2026-08-13 关机前写，含下一步排序）。

**已定决策（2026-08-13，Chaoyu 批准，别重新讨论）**：
- 框架 **verl 不换**（抛开沉没成本重选仍是它；论证在 E07 §1）
- MoE 线：**GLM-4.7-Flash 30B-A3B + LoRA + GSPO**，三摆法对照
  （FSDP 分片 / QLoRA 4bit 复制 / EP toy→Megatron 探针），先跑探针 P1–P6
- 实验以 E 编号报告组织（`docs/infra_exp/`），**按问题编号不按框架**，
  预测跑之前写死、推翻不删记四段

**当前状态关键三条**：
1. one_step_off ✅ 跑通+调优；**fully_async ❌ 崩在 verl 上游 bug**
   （`fully_async_policy/detach_utils.py:153` param_version None 相减）——M7 被挡，
   修 bug 或 fallback one_step_off
2. 权重同步 13.3–24 s/次**未查因**（LoRA 仅 132MB，时间不在传输上）
3. FA2 三点对照待跑（sdpa 静态 84.5s 已有基线；dynamic_bsz 大概率翻正）

相关：[[machine-4x5090-constraints]] [[syncopate-docs-map]] [[user-chaoyu-working-style]]
