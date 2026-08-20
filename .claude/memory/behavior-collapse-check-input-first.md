---
name: behavior-collapse-check-input-first
description: defer 崩塌是 prompt 被截断、不是 reward 教的；行为异常先查输入再查激励
metadata:
  type: project
---

**2026-08-19 翻案（E20 §7.12）**：夜跑最严重的结论「当前 reward 会系统性教模型**不拒绝**」
**被推翻**。E17 A 臂与 `r1_tokenis` 同配置、**只差长度预算**（3584/1536 → 5120/2048）：

```
                    3584（100% 截断）    5120（0% 截断）
该 defer            97% → **83%** 🟠     97% → **100%** ✅
REJ（8 条）         **−0.188** 🔴         **+0.203** ✅
fabricated_safety   **+3** 🔴             **−2** ✅
任务分              +0.101                **+0.137**（t=11.6）
```

**Why**：3584 下 100% 的 prompt 被**左截断**（中位砍 573 token，砍的是规则书**开头**）——
「调查先于任何结论（含**拒绝**、反问）」截断后存活 **0/659**、「数据不够时用 defer」存活 46%。
⇒ **模型训练时从没见过"可以拒绝"这个选项，评测时却见得到。**
RL 只强化模型在**它实际看到的那个 prompt** 下的行为。

**How to apply**：
- ★ **「行为异常」先查输入，再查激励。** 模型没做 X，第一候选是「它看不看得见 X 这个选项」，
  不是「它被教成不做 X」。
- `prompt_length/clip_ratio` **必须是 0.0000**，已提为第四条常驻判据 ——
  它两个月来一直在日志里打着 1.0，没人看（同 [[silent-degradation-fsdp-nosync]] 的形状）。
- ⚠️ 边界三条：① 严格讲变的是**长度预算这一组**（response 截断率仅 2.14%→0.03%，
  prompt 是主因属强推断）；② `lr 1e-4 → defer 0%` **同样量在截断之下**，
  **5120 下重测之前不许动 reward**；③ 夜跑各臂**之间**的相对比较仍是受控对照 ——
  作废的是**解释**，不是排序。
- ⇒ 主线 R-1 已从队首撤下，换成「5120 下重测 lr 1e-4」。回信：
  ~~INFRA-TO-MAINLINE-2026-08-19b.md~~（信件已删；⛔ 铁律：往来只走根目录 `MAINLINE-INFRA.md`）

相关：[[infra-line-state]] [[blank-thresholds-are-not-passes]] [[observed-needs-an-owner]]
