---
name: syncopate-docs-map
description: Syncopate 五份文档各放什么、该先读哪份；文档刻意保持精简，别做增量堆积
metadata: 
  node_type: memory
  type: project
  originSessionId: 254d8707-7512-4e9b-bd89-6e1eeec39011
  modified: 2026-08-16T17:19:33.538Z
---

```
docs/focus-migration-2026-08.md           ★ 焦点怎么定下来的（唯一记录迁移历史处）
docs/syncopate/05-handoff.md              ★ 先读：现在在哪 / 下一步 / 已定的决策 / 反复栽的坑
docs/syncopate/08-machine-and-environment.md  怎么搭环境、怎么跑命令、参数为什么是那个值
docs/syncopate/06-rl-run-protocol.md      RL 跑之前的预期与停止条件（预期写在跑之前）
docs/syncopate/07-toolbox-and-runtime-design.md  沙盒设计（§1.1 已缩成指针，RAG 细节去 10）
docs/syncopate/09-runtime-handoff.md      ★ M9 Runtime 交接（真服务；PG 起法见 08）
docs/syncopate/11-runtime-acceptance.md   ★ M9 验收权威文档（2026-08-17 新建）：
                                          40 条判据逐条 · 五个确认缺口 F1–F5 ·
                                          压测前先做两件（填【待定】+ 场景②④没有被测对象）
                                          ⚠️ 09 只写「怎么起/施工抓到什么」，验收结论一律在这
docs/syncopate/10-rag-retrieval.md        ★ RAG/检索权威文档（2026-08-16 新建）：
                                            **我们没做向量化** —— BM25「是排序器不是判定器」、
                                            Qwen3-0.6B 向量「没有分离带」，两条都是实测淘汰；
                                            阈值 0.35 标定 / 两项验收怎么落成 cap / 那次审查
docs/syncopate-project-design-v0.1.md     权威设计（里程碑 M0–M12）
docs/ostinato-project-design-v0.2.md      单卡 infra 优化
docs/distributed-training-design-v0.1.md  多卡实验设计（异步 / 并行策略 / 通信画像）
docs/infra_exp/00-INFRA-HANDOFF.md        ★ infra 线交接（与主线 05-handoff 平行的另一条线）
docs/infra_exp/TRACK-A-hardware-kernel.md ★ 软硬结合线：负载稀疏结构 × 6.4GB/s 拓扑 决定写什么算子
docs/infra_exp/TRACK-B-framework-async.md ★ 框架线：通用 RL 框架的假设在 agentic 负载上逐条失效
docs/infra_exp/README.md                  实验索引 E00–E16 / 报告模板 / 编号规则 / 无主实验的停放理由
docs/infra_exp/E07-moe-ep.md              MoE 决策（★ 2026-08-14 改用 Qwen3-30B-A3B；GLM-4.7-Flash 当前栈不支持）
docs/infra_exp/E08-async-rl.md            异步三模式 + 分布漂移 + ★ 占空比 31% + 同机分母 1.59×
docs/infra_exp/E12-weight-sync.md         ★ 权重同步查因（方法论样板：两点反解→微基准→分步计时→A/B 排除）
docs/infra_exp/E13-proximal-anchor-snapshot.md  一行 `if requires_grad` 的全过程（含三次自我更正）
docs/infra_exp/E11-sparse-logprob.md      稀疏 logprob：做了调研后主动降级不写 kernel
docs/infra_exp/E02-data-parallel.md       DDP vs FSDP（"慢 6 倍"的口径边界）
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

**章节验收的做法**（M8 立的，M9 照做）：**先立判据再看代码**（反过来是自证）→
**每条缺口要有复现**（不能只写"读代码看着不对"）→ **缺口变成机器可见的东西**
（M8 是 cap，M9 是 `xfail(strict=True)`）→ **未修的债显式挂账 + 写「谁在打」** →
收尾写「别再重新讨论的」。范本：`10-rag-retrieval.md` 与 `11-runtime-acceptance.md`。

方法论问题先查 `核心手册/AgenticRL/sft-finetune-takeaways.md`，别凭通用经验答。

相关：[[syncopate-project-framing]]
