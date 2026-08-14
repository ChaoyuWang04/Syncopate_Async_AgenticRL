---
name: syncopate-docs-map
description: Syncopate 五份文档各放什么、该先读哪份；文档刻意保持精简，别做增量堆积
metadata: 
  node_type: memory
  type: project
  originSessionId: 254d8707-7512-4e9b-bd89-6e1eeec39011
  modified: 2026-08-14T05:08:54.367Z
---

```
docs/syncopate/05-handoff.md              ★ 先读：现在在哪 / 下一步 / 已定的决策 / 反复栽的坑
docs/syncopate/08-machine-and-environment.md  怎么搭环境、怎么跑命令、参数为什么是那个值
docs/syncopate/06-rl-run-protocol.md      RL 跑之前的预期与停止条件（预期写在跑之前）
docs/syncopate/07-toolbox-and-runtime-design.md  沙盒设计（§1.1 = M8 的 RAG 语料层）
docs/syncopate/09-runtime-handoff.md      ★ M9 Runtime 交接（真服务；PG 起法见 08）
docs/syncopate-project-design-v0.1.md     权威设计（里程碑 M0–M12）
docs/ostinato-project-design-v0.2.md      单卡 infra 优化
docs/distributed-training-design-v0.1.md  多卡实验设计（异步 / 并行策略 / 通信画像）
docs/infra_exp/00-INFRA-HANDOFF.md        ★ infra 线交接（与主线 05-handoff 平行的另一条线）
docs/infra_exp/TRACK-A-hardware-kernel.md ★ 软硬结合线：负载稀疏结构 × 6.4GB/s 拓扑 决定写什么算子
docs/infra_exp/TRACK-B-framework-async.md ★ 框架线：通用 RL 框架的假设在 agentic 负载上逐条失效
docs/infra_exp/README.md                  实验索引 E00–E15 / 报告模板 / 编号规则 / 无主实验的停放理由
docs/infra_exp/E07-moe-ep.md              MoE 决策全文（GLM-4.7-Flash + verl + GSPO + 探针 P1–P6）
docs/llm-rl-framework.md                  RL 框架全景调查（Chaoyu 写的，选型背景）
```

**约定（2026-08-13 Chaoyu 明确要求）**：
- **交接文档要短**，不能一直增量更新堆长。只保证下一个窗口能理解
  「遇到过什么问题 / 关键决策怎么做的 / 下一步做什么」。
- **环境配置类内容不进交接文档**，单独放 08。
- 设计文档里**推翻的预期不删**，就地写「原猜想 / 实测 / 推翻后 / 教训」四段 ——
  记录"上机之前我们以为会怎样"本身就是价值的一半。

**infra 线的三层分工（2026-08-14 改组）**：
`E 编号是身份、永不重排`（别的文档引用「E02 的结论」）；`track 是叠加的索引视图`。
⇒ **track 文档答「这条线要兑现什么、现在在哪」，E 报告答「量到了什么」，README 是索引。三者不重复。**
新纪律：每个实验必须能答「服务哪条 track 的哪条兑现物 / 需求由哪个测量指出」，
答不上就**显式停放**（E04/E05/E06 已停，理由写在 README §2.1，不删）。

方法论问题先查 `核心手册/AgenticRL/sft-finetune-takeaways.md`，别凭通用经验答。

相关：[[syncopate-project-framing]]
