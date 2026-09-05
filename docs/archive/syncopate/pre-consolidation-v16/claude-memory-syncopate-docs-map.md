---
name: syncopate-docs-map
description: Syncopate 文档地图（2026-08-19 压缩后主线 13 份）；该读哪份、空号清单、章节验收五步
metadata:
  node_type: memory
  type: project
  originSessionId: 254d8707-7512-4e9b-bd89-6e1eeec39011
  modified: 2026-08-19T14:09:15.833Z
---

> 📦 **历史 Claude 记忆快照。** 这是文档整合前的地图，不是当前入口；现行地图见
> [`docs/syncopate/00-START.md`](../../../syncopate/00-START.md)。

```
docs/focus-migration-2026-08.md           ★ 焦点怎么定下来的（唯一记录迁移历史处）
docs/syncopate/00-START.md              ★ 先读：现在在哪 / 下一步 / 已定的决策 / 反复栽的坑
docs/syncopate/08-machine-and-environment.md  怎么搭环境、怎么跑命令、参数为什么是那个值
docs/syncopate/06-rl-run-protocol.md      ★ 训练协议：**§1 训练前自查清单（每次必过）** · 停机/完成判据 · H 部分=原 14
                                          （指标红线/选点；首跑预期段已抽到 archive）
docs/syncopate/07-toolbox-and-runtime-design.md  沙盒设计（§1.1 已缩成指针，RAG 细节去 10）
docs/syncopate/09-runtime-handoff.md      ★ M9 Runtime 交接（真服务；PG 起法见 08）
docs/syncopate/21-invalidated-numbers.md ⛔★★★ **先读这份**：哪些数字不能引用
                                          （2026-08-18 两个基石 bug 污染了所有 RL 实测）
                                          判据强制：check_pipeline_invariants --only quarantine
docs/syncopate/01-TASKS.md ★★ 主线执行顺序的**唯一来源**，每条带「谁在打」
~~19 复盘~~ 已归档（共同形状已成 00-START 守则；存活结论在 22 §H）
⛔ **2026-08-19 压缩令**：主线 19 份 → 13 份；空号永不复用 = 02 05 12 14 15 17 19 20
/MAINLINE-INFRA.md（仓库根目录）        ⛔★ 两线往来的**唯一**文档（铁律：禁止再写信件；办完删行）
docs/syncopate/18-pipeline-assumption-probes.md 管线前提探针审计（E21 之后的同族排查）
~~17 ESS/lr 方案~~ 已归档（活的部分各归其位；§6.5 白名单问题捞进 01-TASKS C-5）
docs/archive/16-m7b-rl-run.md ⛔结论作废（归档；当前状态看 05/20/21）
~~docs/syncopate/16~~          ★ M7-b（2026-08-17）：lr 被夹在两堵墙之间 ——
                                          3e-5 时 ESS 跌破 0.3 停机、1e-5 时位移只有 0.0487%
                                          ⇒ 下一跑 colocate+3e-5 对照（异步的代价）
docs/syncopate/13-diversity-gates.md      ★ 数据门禁一份全：D 族多样性 D1–D11（重建前必跑，
                                          D5「句式不能预测档位」唯一真抓过 bug）
                                          + L 部分=原 15（三桶泄露 L1/L2，重切后必跑）
                                          `python -m syncopate.pipeline.data_gates --batch <批次> [--split-dir]`
docs/syncopate/11-runtime-acceptance.md   ★ M9 验收权威文档（2026-08-17 新建）：
                                          40 条判据逐条 · 五个确认缺口 F1–F5 ·
                                          压测前先做两件（填【待定】+ 场景②④没有被测对象）
                                          ⚠️ 09 只写「怎么起/施工抓到什么」，验收结论一律在这
docs/syncopate/10-rag-retrieval.md        ★ RAG 唯一权威（R 部分=原 12 runtime 三态契约）：
                                            **我们没做向量化** —— BM25「是排序器不是判定器」、
                                            向量「没有分离带」均实测淘汰；阈值 0.35（沙盒）/
                                            0.53（runtime，操作点必然不同）；向量门槛已写死
docs/syncopate-project-design-v0.1.md     权威设计（里程碑 M0–M12）
docs/ostinato-project-design-v0.2.md      单卡 infra 优化
docs/distributed-training-design-v0.1.md  多卡实验设计（异步 / 并行策略 / 通信画像）
docs/infra_exp/00-START.md              ★★ infra 新窗口从这里开始（导航+守则；08-19 重组：
                                          原 ONBOARDING/00-INFRA-HANDOFF 已并入 00/01/02）
docs/infra_exp/01-TASKS.md                ★ infra 队列唯一来源（办完删行）
docs/infra_exp/02-DECISIONS.md            ★ infra 已定决策 · ⛔作废登记 · 已落地改动
docs/infra_exp/TRACKS.md                  两条 track 的兑现物与完成度（A+B 合并版）
docs/infra_exp/README.md                  实验索引 E00–E16 / 报告模板 / 编号规则 / 无主实验的停放理由
docs/infra_exp/E07-moe-ep.md              MoE 决策（★ 2026-08-14 改用 Qwen3-30B-A3B；GLM-4.7-Flash 当前栈不支持）
docs/infra_exp/E08-async-rl.md            异步三模式 + 分布漂移 + ★ 占空比 31% + 同机分母 1.59×
docs/archive/E12-weight-sync.md         ★ 权重同步查因（方法论样板：两点反解→微基准→分步计时→A/B 排除）
docs/infra_exp/E13-proximal-anchor-snapshot.md  一行 `if requires_grad` 的全过程（含三次自我更正）
docs/infra_exp/E11-sparse-logprob.md      稀疏 logprob：做了调研后主动降级不写 kernel
docs/infra_exp/E02-data-parallel.md       DDP vs FSDP（"慢 6 倍"的口径边界）
docs/archive/                             📦 归档：历史记录，**不是当前状态**（含 01/03/04/16 与
                                          llm-rl-framework；为什么归档见 archive/README.md）
```

**约定（2026-08-13 Chaoyu 明确要求）**：
- **交接文档要短**，不能一直增量更新堆长。只保证下一个窗口能理解
  「遇到过什么问题 / 关键决策怎么做的 / 下一步做什么」。
- **环境配置类内容不进交接文档**，单独放 08。
- 设计文档里**推翻的预期不删**，就地写「原猜想 / 实测 / 推翻后 / 教训」四段 ——
  记录"上机之前我们以为会怎样"本身就是价值的一半。
- ★★ **更新即改写，不是追加**（守则⑪，Chaoyu 2026-08-19，两线通用）：
  动笔前先通读那份；新内容并入已有章节（错了就地改、过时删或归档）；
  追加只对登记型表格合法且**办完要删行**；更新完行数不该默认变大；
  新开文档前先问能不能进现有的某一份。

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
