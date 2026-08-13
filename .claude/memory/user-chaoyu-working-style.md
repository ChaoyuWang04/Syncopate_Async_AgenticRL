---
name: user-chaoyu-working-style
description: Chaoyu 是这个项目的作者，愿意大胆多改快迭代，但要求每个结论可验证、并且要用通俗语言解释基础设施概念
metadata: 
  node_type: memory
  type: user
  originSessionId: 254d8707-7512-4e9b-bd89-6e1eeec39011
  modified: 2026-08-13T17:24:38.589Z
---

Chaoyu（RunPod 上的开发者，中文交流）：

- **对分布式/多卡是新手**，明确说过"我第一次做多卡的训练"。问到 NCCL、FSDP、MFU、
  显存占用这类概念时，**要用最简单直白、通俗易懂的语言解释**，可以打比方，
  但不要因此省掉真实数字。
- **愿意大胆改、快迭代**：「没关系的，遇到没预料到的问题再修改就行，这是一次大胆的
  多修改的尝试」。不需要为每个小改动请示。
- **但要求可验证**：会主动追问"这个前提检查了吗"。例如开 fully_async 之前，
  明确要求先确认 staleness 的修正系数在不在 —— **不要跳过前提校验去追速度**。
- **对"多卡没起作用"这类浪费很敏感**，会主动指出并要求先调优再跑长任务。
- 文档要**短**，反对增量堆积（见 [[syncopate-docs-map]]）。

相关：[[syncopate-project-framing]] [[feedback-measure-dont-infer]]
