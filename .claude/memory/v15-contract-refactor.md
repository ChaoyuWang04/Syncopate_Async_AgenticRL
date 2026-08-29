---
name: v15-contract-refactor
description: ★08-29 起唯一队首=v15 契约重构（行为进强通道，施工图 25）；U 路 P3/P4 并入 R6/R5
metadata: 
  node_type: memory
  type: project
  originSessionId: f5bed7da-2659-5ec6-bac9-73049c9ac8d6
  modified: 2026-08-29T14:51:38.101Z
---

**v15 = 2026-08-29 Chaoyu 立项的业务大版本**（真人 dev mode 三模型实测后的裁定）：把行为标签从自研终答 JSON 壳（`{"behavior":"defer",…}`——预训练零先验的弱通道，RL 后标签整体漂移实证在案）搬进 **function calling 强通道**（session.defer/clarify/reject/report 信令工具族），终答变纯自然语言。目标「Claude Code 一般自然」已定成 N1–N5 可验收性质。

- 施工图 = `docs/syncopate/25-v15-contract.md`：七阶段 R0–R6 全量化门槛。**R0 假说验证先行**（120 行双契约对照微调，分布外行为正确率 ≥+15pp 才动全身）；R5 核心承诺=行为形态正确率 ≥97%；R6 成败判据=RL 400 步后形态跌幅 ≤3pp（对照 v14 壳通道 RL 后 defer 100→0）。
- 判分/编排需求**同构保持**：verifier_engine.py:322 行为闸逻辑不动、只换 trajectory.behavior 的推导来源（从形态推导：调 session.*→对应行为、纯文本终答→answer、有业务工具→tool_call）。
- 保留资产：S1 题库/S2 句式库/S3 revision 库、OOV held-out 词表、盲评口径、OPD prompts、pool/守卫、四卡 DDP、dev mode 三模型栈（v14.5 base/sft-e3/rl-s12 留作 R3 桥测锚点）。
- R3 起旧基线跨版本可比性全作废（21 号登记；三锚点桥测差值只作换尺参考）。
- [[u-route-unified-training]] 的 P3/P4 并入 R6/R5；defer 塌陷、标签漂移、summary 污染、CoT 压制的完整证据链在 24 §4-P3/§7。
