---
name: modal-migration-state
description: Modal 2×B200 当前现场；v16 全链机械 smoke 已通过，质量收口与固定源码复验是队首
metadata:
  node_type: memory
  type: project
  originSessionId: 3ed1b9c6-3ae1-544f-b9a5-207777e46c52
  modified: 2026-09-04T00:00:00.000Z
---

# Modal / B200 当前记忆

现行权威：

- 当前任务：`docs/syncopate/01-TASKS.md`
- 数据：`docs/syncopate/03-DATA.md`
- 训练：`docs/syncopate/04-TRAINING.md`
- 机器：`docs/syncopate/05-COMPUTE.md`
- 操作入口：`modal_app/README.md`
- 历史施工：`docs/archive/syncopate/pre-consolidation-v16/26-repair-rulers-and-data.md` 与 `31-modal-and-new-stack.md`

当前事实：

- 家在 Modal 的 `syncopate-home` Volume，训练机器为 B200×2。
- 当前主栈是 PyTorch 2.13、vLLM 0.28、verl 0.9、Transformers 5.10。
- v16 题库 2030 条；切分 EVAL/SFT/RL 为 401/597/1032，跨机器重建一致。
- v16 SFT 数据 1222 行、18 桶；现行结构闸和三桶隔离可供 smoke，candidate 严格带宽尚未冻结。
- B01 上云前认证通过；本机缺环境的 5 项已在 Modal CPU 目标镜像 5/5 通过。
- B02 run `b02_20260905a` 已证明本轮 SFT、合并模型、双卡 RL、RL adapter 和 OPD adapter 连续传递；SFT/RL/OPD 都有真实更新。
- B02 是边跑边修的 smoke：`pipeline_ok=true`、`all_passed=false`。Exam 有质量 WARN，RL 每步 25% rollout 撞响应上限，OPD 一更新短评测质量待解；不是 candidate 或性能 baseline。
- 完整证据在 `_audit/infra/B02/REPORT.md`；当前进度仍只认两条线各自 `01-TASKS.md`。
- Serving 主体已施工，但当前环境正式验收未结束。

当前队首：

1. 主线 T1 先处理 RL 长回复截断、Exam 和 OPD 质量欠账。
2. infra B03 用固定源码建立可重复 B200 reference，并补齐训推/adapter 身份正负对照。
3. B04 以 SFT `DP=2` 为 reference，对比同样 2×B200 的 `TP=2`；不再测 1 卡与 2 卡 DP 的速度。当前 runbook 仍是单卡，改成 DP=2 前只补当前源码的双卡正确性 smoke。
4. 再做 RL/OPD 单因素实验和固定源码 clean smoke；付费运行仍需逐次授权。

不能误写：

- B02 机械全链通过不等于质量全绿，也不能当固定源码性能 baseline。
- 全链 smoke 不等于 candidate 质量验收。
- B200 结果不能直接套到 B300。
- 旧 5090、PRO 6000、旧依赖和 run17–27 过程只在历史档案中使用。
