---
name: serving-harness-k-line
description: ★K 线（27 serving harness 生产化）现场：09-02 K0–K6 ✅（治理表+注册断言、四闸落库、分诊 error_json）；下一步 K7 SSE 事件流；§16 四件已裁（Celery+Redis · PG+Alembic · 快照式恢复 · id 不迁）；坑表 28 · 对照 29；下一步 K1
metadata: 
  node_type: memory
  type: project
  originSessionId: d41319ab-b22c-5d75-a1c9-8bcb12bcbd24
  modified: 2026-09-02T05:01:11.943Z
---

**K 线 = `docs/syncopate/27`（施工计划 K0–K11）+ `28`（中间件坑表，刻意独立）。**
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
· K1 ✅（cancel/resume/trace、幂等三态 409、错误信封 13 码、run_type、resume_token 一次性；取消只接了 ActionGate 入口这一个安全点，其余 K5-5）· K3 ✅（0003 outbox/DLQ/门铃；dispatcher.py；celery_app.py；定向 claim 只认 queued；LeaseHeartbeat；transient→queued+outbox 延迟；集成测试起真 celery 子进程验 ack 前崩溃）· K4 ✅（transition_run 唯一入口 + 10 条边（含改造边 waiting→succeeded）+ 触发者矩阵；事件 run.succeeded→**run.completed** 全仓改名；轮询 claim 内联 sweeper 三分支全走状态机；rerun 端点）· K5 ✅（一轮两档快照 + last；恢复两路读意图日志：succeeded 回填/response_lost 转对账 awaiting_reconciliation；任何执行都 resume；record_tool_call 五态；幂等键去 run_id；安全点 loop 顶+收口入口）· K6 ✅（`tool_governance.py` 30 工具治理表 + 导入时对 REGISTRY 完整性断言，**没动** tool_registry；WRITE_TOOLS 派生；0004 error_json/blocked_by/五态唯一索引；四闸拦下也落库；写工具带键可重试是登记的改造）· 下一步 K7 SSE（`after` 查询路、事件分层 public/internal 过滤、同域 Cookie 鉴权、只读纪律断言；进程内门铃已有）。
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
