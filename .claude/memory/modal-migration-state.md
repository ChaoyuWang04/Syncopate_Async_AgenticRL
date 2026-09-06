---
name: modal-migration-state
description: Modal B200 当前现场；数据冻结，B03 小模型数值与恢复通过，35B 概率、路由及重复性仍待验
metadata:
  node_type: memory
  type: project
  originSessionId: 3ed1b9c6-3ae1-544f-b9a5-207777e46c52
  modified: 2026-09-06T16:00:00.000Z
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

- 持久资源在 Modal 的 `syncopate-home` Volume；只使用 Modal CPU / B200，按任务申请 CPU、单卡或双卡，独立实验尽量并行。
- 当前主栈是 PyTorch 2.13、vLLM 0.28、verl 0.9、Transformers 5.10。
- v16 题库 2030 条；切分 EVAL/SFT/RL 为 401/597/1032，跨机器重建一致。
- v16 SFT 数据 1222 行、18 桶；现行结构闸和三桶隔离可供 smoke，candidate 严格带宽尚未冻结。
- B01 上云前认证通过；本机缺环境的 5 项已在 Modal CPU 目标镜像 5/5 通过。
- B02 run `b02_20260905a` 已证明本轮 SFT、合并模型、双卡 RL、RL adapter 和 OPD adapter 连续传递；SFT/RL/OPD 都有真实更新。
- B02 是边跑边修的 smoke：`pipeline_ok=true`、`all_passed=false`。Exam 有质量 WARN，RL 每步 25% rollout 撞响应上限，OPD 一更新短评测质量待解；不是 candidate 或性能 baseline。
- 完整证据在 `_audit/infra/B02/REPORT.md`；当前进度仍只认两条线各自 `01-TASKS.md`。
- Serving 主体已施工，但当前环境正式验收未结束。

当前队首：

1. 数据冻结，不再语义清洗；质量 WARN 不阻塞 infra。T1-1 的真实更新复验已由 B03 c 完成；没有新的完整全链或质量提升结论。
2. B03 c/d 已补 35B 的同步载荷、逐轮版本、原始 token 和真实更新。`training_20260906c` 又用两层 full-attention Text MoE 小模型验过双卡独立梯度、同状态 AdamW、连续两步和正常恢复。原汇总只因 PyTorch 返回裸 UUID 被误判，保存证据用修正后的格式检查重新聚合为通过；没有重跑 GPU，也没有改数学门槛。
3. B03 仍缺 35B trainer/vLLM 同 token 概率解释、GDN/MoE 逐层路由和至少三次固定源码重复。小模型结果不能替代 35B、GDN、LoRA、rollout 或性能基线。
4. B04 以 SFT `DP=2` 对比同样 2×B200 的 `TP=2`；目标 CPU 已取回 Transformers/PEFT 的真实 TP 源码，但还没有执行 tiny TP 前后向、LoRA、optimizer 和保存/加载。
5. B06 已接入原始 token/mask 和真实 optimizer 更新证据，规格与代码质量复审均通过；目标 CPU 零跳过和双 B200 两次真实更新仍待做。
6. B13 首轮 2×B200 通信与 BF16 GEMM 微测试通过；64 MiB bus 带宽约为 all-reduce 377.847、all-gather 336.247、reduce-scatter 325.812 GB/s，4096 方阵约 1345.7～1355.9 TFLOP/s。拓扑、可靠利用率、能耗和跨机器重复未完成，不能当训练 baseline。
7. 正确性通过后再并行做训练、推理和 kernel 单因素；采用项组合 smoke 后完成学习运行。资源与时限只认 Compute 和各 B 报告。

不能误写：

- B02 机械全链通过不等于质量全绿，也不能当固定源码性能 baseline。
- 全链 smoke 不等于 candidate 质量验收。
