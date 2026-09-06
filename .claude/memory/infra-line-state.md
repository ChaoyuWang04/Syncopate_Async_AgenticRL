---
name: infra-line-state
description: Modal 2×B200 上的 infra 当前入口、B 系列队列与历史证据边界
metadata:
  node_type: memory
  type: project
  modified: 2026-09-06T16:00:00.000Z
---

# Infra 当前记忆

现行权威：

- 导航：`docs/infra_exp/00-START.md`
- 当前任务：`docs/infra_exp/01-TASKS.md`
- 边界与证据流：`docs/infra_exp/02-SYSTEM.md`
- 训练：`docs/infra_exp/03-TRAINING.md`
- Rollout/Serving：`docs/infra_exp/04-SERVING.md`
- B200、通信和 kernel：`docs/infra_exp/05-COMPUTE-AND-KERNELS.md`
- 实验协议：`docs/infra_exp/06-EXPERIMENTS.md`
- 当前对外材料：`docs/narrative/infra-resume.md`

当前事实：

- 唯一云端是 Modal CPU / B200，按需分配 CPU、单卡或双卡，独立实验尽量并行；精确环境只认 `docs/syncopate/05-COMPUTE.md`。
- B200 环境、双卡通信和模型启动可工作；B01 上云前认证已通过。
- B02 已完成真实 v16 机械全链，SFT、RL、OPD 均有真实更新和可加载产物；质量仍有 WARN。
- B02 跨多次源码修复，只能证明线路接通，不能当性能 before 或简历成果；可重复 baseline 从 B03 开始。
- 新实验从 B01 开始；施工报告写 `_audit/infra/Bxx/REPORT.md`，原始证据放同目录各机器臂，验收后完整报告归档。
- B03 c/d 已验 35B 真实更新、adapter、同步载荷、版本和原始 token；trainer 重算相同，跨引擎概率差仍待解释。小模型 `training_20260906c` 已验双卡独立梯度、同状态 AdamW、连续两步和正常 checkpoint 恢复；UUID 格式误报只用保存证据重新聚合修正，没有 GPU 重跑。35B 的 GDN/MoE 路由和三次固定源码重复仍在队首。
- B04 比 `DP=2`/`TP=2`，不重测 1/2 卡 DDP。目标 CPU 已取回真实 Transformers/PEFT TP 源码；tiny TP 前后向、LoRA、optimizer、保存/加载及正式对比尚未执行。
- B06 已把原始 token/mask 和真实 optimizer 更新记录接到门禁，规格与代码质量复审均通过；目标 CPU 零跳过和双 B200 两次更新尚未执行。
- B13 首轮双 B200 通信和 BF16 GEMM 微测试通过，已有 64 MiB 通信与 4096 方阵读数；拓扑、可靠利用率、能耗和跨机器重复仍未完成，因此不改变默认。
- 进度与数值只认 TASKS 和 REPORT。正确性前置通过后独立实验尽量并行；遇到通用问题留 upstream DRAFT，不强求提交。

跨线规则：

- `MAINLINE-INFRA.md` 已退役并归档。
- 未完成事项只进入唯一负责方的 TASKS，另一条线只链接依赖。
- 负责人之间通过 Codex 独立任务消息工具直接沟通，并读取对方实际回复；不新建 HANDOFF/REPLY/信件文档。

不能误写：

- B02 机械全链不等于质量全绿或性能 baseline，组件加速不等于端到端收益，在用户已批准的范围内登记各批资源、时限和证据。
