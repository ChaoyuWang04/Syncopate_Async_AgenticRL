---
name: infra-line-state
description: Modal 2×B200 上的 infra 当前入口、B 系列队列与旧 5090 归档边界
metadata:
  node_type: memory
  type: project
  modified: 2026-09-04T00:00:00.000Z
---

# Infra 当前记忆

现行权威：

- 导航：`docs/infra_exp/00-START.md`
- 当前任务：`docs/infra_exp/01-TASKS.md`
- 边界与证据流：`docs/infra_exp/02-SYSTEM.md`
- 训练：`docs/infra_exp/03-TRAINING.md`
- Rollout/Serving：`docs/infra_exp/04-SERVING.md`
- B200/B300、通信和 kernel：`docs/infra_exp/05-COMPUTE-AND-KERNELS.md`
- 实验协议：`docs/infra_exp/06-EXPERIMENTS.md`
- 当前对外材料：`docs/narrative/infra-resume.md`
- 旧 E 系列和 4×5090：`docs/archive/infra_exp/legacy-4x5090/`

当前事实：

- 当前主场是 Modal 2×B200；精确环境只认 `docs/syncopate/05-COMPUTE.md`。
- B200 环境、双卡通信和模型启动可工作；B01 上云前认证已通过。
- B02 已完成真实 v16 机械全链，SFT、RL、OPD 均有真实更新和可加载产物；质量仍有 WARN。
- B02 跨多次源码修复，只能证明线路接通，不能当性能 before 或简历成果；可重复 baseline 从 B03 开始。
- 新实验从 B01 开始；施工报告写 `_audit/infra/Bxx/REPORT.md`，原始证据放同目录各机器臂，验收后完整报告归档。
- 队首是 B03 固定源码重复性与训推身份尺子，然后是 B04 在 2×B200 上比较 SFT `DP=2`/`TP=2`；不再测 1 卡与 2 卡 DP 的速度。B05、B06 才做 RL/OPD 单因素实验。
- B300 只在 B200 胜出配置稳定后复核，所有数字重新测。

跨线规则：

- `MAINLINE-INFRA.md` 已退役并归档。
- 未完成事项只进入唯一负责方的 TASKS，另一条线只链接依赖。
- 负责人之间通过 Codex 独立任务消息工具直接沟通，并读取对方实际回复；不新建 HANDOFF/REPLY/信件文档。

不能误写：

- 旧 E01～E33、FSDP1、verl 0.8、vLLM 0.12、sm_120 和 4×5090 结论只属于历史。
- B02 机械全链不等于质量全绿或性能 baseline，组件加速不等于端到端收益，付费 GPU 运行仍需用户逐次授权。
