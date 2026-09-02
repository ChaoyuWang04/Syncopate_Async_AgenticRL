---
name: serving-harness-k-line
description: ★K 线（27 serving harness 生产化）09-02 K0–K11 全部落地收官；交付=30 上线清单/runbook 六卡/复盘/灾备 RTO 1.5s；挂账=压测重测 S-05 · 前端 build · 权重灾备 · 备份策略 · C-15 负向认证 · 28 O-01..O-10 优化；§16 四件已裁；坑表 28 · 对照 29
metadata: 
  node_type: memory
  type: project
  originSessionId: d41319ab-b22c-5d75-a1c9-8bcb12bcbd24
  modified: 2026-09-02T05:01:11.943Z
---

**K 线 = `docs/syncopate/27`（施工记录 K0–K11）+ `28`（坑表）+ `30`（上线清单/runbook）；★交接入口 = `09 §0.0`（训练机负责人从这进）。**
2026-09-02 Chaoyu 放行 K0，并在读完课件 CH3/CH2/CH5/CH8 相关章后先拍了 §16 四件
（原计划"盘点后再拍"改为"先拍、K0 只核实"）：

- 队列 = **Celery + Redis**（Chaoyu 明确要"熟悉队列软件的坑并做优化"，工业对齐优先于简单）
- 数据库 = PostgreSQL 不换 + **Alembic** 版本化 migration；不引 SQLAlchemy ORM
- 恢复 = **课件快照式**（每 append 一条存一次 checkpoint，带 last/completed_tool_calls）；
  run_events 只做回放展示。这推翻了 27 原预设"重放为主"
- id 不迁移（run_id/conversation_id 已是随机 hex）

**Why**：机制一律以课件（`docs/reference/`，不入库）为准，标准=工业级容灾；
Chaoyu 要求坑表与后续排查/优化项**单独文档集**，不污染 27 的施工纯洁性。

**进度（09-02）**：K0 ✅（29）· K2 ✅（迁移链 `syncopate/runtime/migrations/`，schema.sql 已退役成
`schema.snapshot.txt`；`db.next_seq/append_event` 领号器；创建事务写 run.created；usage 粒度=每次执行一行）
· K1 ✅（cancel/resume/trace、幂等三态 409、错误信封 13 码、run_type、resume_token 一次性；取消只接了 ActionGate 入口这一个安全点，其余 K5-5）· K3 ✅（0003 outbox/DLQ/门铃；dispatcher.py；celery_app.py；定向 claim 只认 queued；LeaseHeartbeat；transient→queued+outbox 延迟；集成测试起真 celery 子进程验 ack 前崩溃）· K4 ✅（transition_run 唯一入口 + 10 条边（含改造边 waiting→succeeded）+ 触发者矩阵；事件 run.succeeded→**run.completed** 全仓改名；轮询 claim 内联 sweeper 三分支全走状态机；rerun 端点）· K5 ✅（一轮两档快照 + last；恢复两路读意图日志：succeeded 回填/response_lost 转对账 awaiting_reconciliation；任何执行都 resume；record_tool_call 五态；幂等键去 run_id；安全点 loop 顶+收口入口）· K6 ✅（`tool_governance.py` 30 工具治理表 + 导入时对 REGISTRY 完整性断言，**没动** tool_registry；WRITE_TOOLS 派生；0004 error_json/blocked_by/五态唯一索引；四闸拦下也落库；写工具带键可重试是登记的改造）· K7 ✅（event_layer 分层注册表，未登记默认不推 + 结构测试；after 查询路；retry 前导；Cookie 与 Bearer 同表；只读 AST 断言；门槛④"静默零查询"改造为 2s 兜底轮询）· K8 ✅（sweeper.py：A/B/C/D/E 五类扫描 + reconcile_once；`db.sweep_expired_run` 一份实现供 sweeper 与轮询 claim；platform_ledger（0005）假平台写穿；repair 端点；11 条联测）· K9 ✅（budget.py + 0006；loop 记账/预算经 gate；/metrics /alerts；log.py；checkpoint v；SYNCOPATE_DISABLED_TOOLS；drain 真演练；停队列/回滚只写命令）· K10 ✅（flywheel.py；导出=考卷 v4 题形落 data/feedback_exports/，吸入=考卷 v5 由 26 线定；label 复用 cap 名/behavior/六族，reason_code 是 K 线的病因枚举；前端 👍👎 未做）· K11 ✅（`30-serving-release-checklist-and-runbook.md`：42 条清单 ✓36/挂账 6、六张卡 + 传导图、复盘回填 08-20「常驻 worker 抢走测试租户 run」、`scripts/dr_drill.sh` 数据层重建 RTO 1.5s、`scripts/runbook_queries.py` 六卡查询真跑、SLO 基线 `_audit/serving_k11/`）。**K 线收官**：runtime 372 passed / 全量绿；27 §14 列全线挂账。
**下一轮的料**（未开工）：28 O-01..O-10 中间件优化实验 · C-15 负向认证 · S-05 压测重测（需训练机端点）· 前端 👍👎 + build（需 node）· 生产备份策略（灰测放真人前）。
⚠️ 轮询 worker 入口（`python -m syncopate.runtime.worker`）必须保留：26 线考场链、b4_stack、start_worker.sh 都在用；Celery 是新增的生产投递路径，两者共用 `Worker.execute_claimed`。
Chaoyu 09-02 追加四裁：K1/K2 交错 · Alembic 唯一真相 · 终态事件改名 run.completed（K4-2 时三处同步含前端）· 引入 run_type。

**How to apply**：
- 每阶段开工前对 28 该阶段的行；每条坑要负向认证（人为制造 → 判据必红）才算 ✅。
- 本机（samwang-X870I，单 5090，**无 sudo**）：PG 16 + Redis 8 装在 conda env
  `syncopate-infra`（conda-forge）；`scripts/pg_bootstrap.sh` 已加非 root 分支
  （`PG_HOME/PG_SHARE/PG_LIB` 指向 conda env，PGDATA 在 `~/.local/share/syncopate/pgdata/16`）；
  `scripts/redis_bootstrap.sh` 新建（requirepass/AOF/noeviction，判据行 `[redis-config]`）。
  runtime 依赖用 `uv sync --inexact --extra runtime --extra dev`（⛔ 不带 `--inexact` 会拆掉 torch/vllm）。
- 与 26 线（verl-22 会话）并行：K 线只动 runtime 目录与 27/28；不碰 tool_registry、
  decider.build_messages、contract.py、scripts/u_*（26 的 W2 目标）；动共用件先写 MAINLINE-INFRA。
- 已知会撞的第一个坑：asyncio worker × Celery prefork（28 C-09）；B-5 goodput 数字 Celery 化后作废（S-05）。

相关：[[v15-contract-refactor]] [[registered-is-not-implemented]] [[project-mechanism-not-wired]] [[clean-machine-only-gaps]]
