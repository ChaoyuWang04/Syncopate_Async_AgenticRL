# Syncopate · 28 · Serving 中间件坑表与排查清单（Celery / Redis / PostgreSQL / Alembic + 我们的接缝）

> 立项：Chaoyu 2026-09-02，随 `27 §16` 四件裁定（Celery+Redis · PG+Alembic · 快照式恢复 · id 不迁）。
> **本文是登记型清单，刻意独立于 27**：27 只放施工步骤与门槛；每条坑的来源、后果、
> 防法、负向认证、归属阶段与状态都在这里。施工每个阶段开工前对一遍本表该阶段的行；
> 做中间件优化实验时从 §5 取题。
>
> 纪律：① 每条坑必须写"怎么人为制造它 → 哪条判据必红"（负向认证，`27 §0` 准则二）；
> ② 状态只有四种：⬜ 未排查 · 🔶 已复现（判据会红）· ✅ 已防住（判据绿且负向认证过）· ⛔ 裁剪（写复活条件）；
> ③ 排查完的行**不删**（这是知识，不是待办），但结论就地改写、不追加；
> ④ 课件坑用 H 编号引用（CH3 H20–H30/H102–H108，CH2 H11–H19，CH9 H85 等），课件原文在 `docs/reference/`。

---

## 0 · 一句话总纲

课件 CH3 §4："**装个 Celery ≠ 有了 Agent 后端**"。框架只管"任务怎么投递和执行"；
run 状态机、事务、幂等、lease、工具安全、可观测全部还是我们的活。本表 §1/§2 是
"框架和中间件自己会怎么坑你"，§3 是数据库层，§4 是"我们现有代码和它们的接缝"，§5 是优化候选。

---

## 1 · Celery（框架层）

| # | 坑 | 来源 | 后果（不修会怎样） | 防法 / 判据 | 负向认证 | 归属 | 状态 |
|---|---|---|---|---|---|---|---|
| C-01 | `task_acks_late` 默认 **False** = 先 ack 再执行 | 课件 H29 | worker 执行中被杀 ⇒ 队列认为完成、库里 run 还在 running ⇒ **无人再救，永久卡死** | 全局 `task_acks_late=True`；判据：kill -9 执行中的 worker，消息必须被重投 | 改回 False，kill 后 run 停在 running 且队列空 ⇒ K3 门槛③必红 | K3 | 🔶 配置已写（celery_app.py），负向认证 ⬜ |
| C-02 | Redis broker 的 **visibility_timeout**（默认 3600s）= 伪 ack 语义 | Celery Redis transport 文档；课件 H30 同族 | 任务执行超过它 ⇒ **被重投给第二个 worker**（at-least-once 第四个来源）；设太长 ⇒ 崩溃后要等一小时才重投 | 与 K3-7 lease TTL 联动定死：`visibility_timeout ≥ 2×lease TTL`，且 run 级恢复以 lease/sweeper 为准、不靠 visibility 重投；判据行 `[celery-cfg] visibility_timeout=<n> lease_ttl=<m>` | 把 visibility_timeout 设成 5s 跑一条 30s 的 run ⇒ K3 门槛②"副作用计数=1"仍须绿（五道闸兜住），但 `duplicate_prevented_total` 必 +1 | K3 | 🔶 visibility_timeout=900≥2×TTL 已配；定向 claim 只认 queued ⇒ 重投无害已由集成测试③证明；负向认证 ⬜ |
| C-03 | `worker_prefetch_multiplier` 默认 4 | Celery 文档 | 每个进程预取 4×并发条消息攥在手里；进程死 ⇒ 那批全部等 visibility_timeout；长任务场景吞吐反而变差 | 设 1（长任务 + acks_late 的标准组合） | 设回 4，起 1 个进程投 8 条，观察 `oldest_job_age` 比 prefetch=1 高一个数量级 | K3 | 🔶 配置已写（celery_app.py），负向认证 ⬜ |
| C-04 | `task_reject_on_worker_lost` 默认 False | Celery 文档 | 子进程被 OOM/SIGKILL（不是异常）⇒ 即使 acks_late，任务标 WorkerLostError **不重投** | 设 True；安全前提=五道闸（终态检查/claim/幂等键/checkpoint/唯一约束）已就位，否则重投即重复 | 设 False，kill -9 子进程 ⇒ run 永远不再被领取 | K3 | 🔶 配置已写（celery_app.py），负向认证 ⬜ |
| C-05 | 结果后端（result backend）把 Redis 填满 | Celery 文档 | 每个任务的返回值/状态写进 Redis，永不过期或占内存；我们的事实在 PG，结果后端是冗余的第二事实 | `task_ignore_result=True`，不配 result backend；判据：跑 100 条 run 后 Redis `dbsize` 不增长 | 打开 result backend 跑 100 条 ⇒ Redis key 数 +100 | K3 | 🔶 配置已写（celery_app.py），负向认证 ⬜ |
| C-06 | **三个 attempts 同名不同义**：Celery `request.retries` · outbox `attempts` · `agent_runs.attempts` | 课件 H25/H28 | 监控把投递失败率和执行失败率混算 ⇒ 扩容决策错 | 三个数分别出指标，命名区分（`outbox_dispatch_attempts` / `celery_retries` / `run_attempts`）；`agent_runs.attempts` 只在 claim 时 +1 | 故意让 dispatcher 退避 3 次 ⇒ `run_attempts` 必须仍是 0 | K3/K9 | ✅ 09-02：三个计数各自独立（test_outbox_attempts_do_not_touch_run_attempts） |
| C-07 | 序列化默认允许 pickle（老版本）/ 未锁死 JSON | 课件 H85 同族 | Redis 密码泄漏 + pickle = 远程代码执行 | `task_serializer=accept_content=['json']`；task 体只有 `run_id`（H23） | 投一条 pickle 消息 ⇒ worker 必须拒收（ContentDisallowed） | K3 | 🔶 配置已写（celery_app.py），负向认证 ⬜ |
| C-08 | `time_limit`（硬）= SIGKILL 子进程 = **强杀** | 课件 H105 | 崩在"写工具已执行、未记账"窗口 ⇒ 重复来源 B | 只用 `soft_time_limit`（抛 SoftTimeLimitExceeded，在安全点协作式停）；硬上限留 2× 余量作最后兜底并接 C-04 | 只设硬上限，在写工具执行后 sleep 超限 ⇒ tool_calls 留下 running 行（意图日志能看见） | K3/K5 | ⬜ |
| C-09 | prefork + **asyncio**：事件循环与连接池跨 fork | 我们特有（worker 是 asyncio + asyncpg） | 池在父进程建、子进程继承 ⇒ 连接被多进程共用 ⇒ `another operation is in progress` 一族（09 §4 ⑤ 同病）；每任务 `asyncio.run` 反复建池 = 连接风暴 | 池在 `worker_process_init` 信号里建、`worker_process_shutdown` 里关；每个子进程一个长活事件循环（任务里 `loop.run_until_complete`）；判据行 `[worker-init] pid=<p> pool=<n>` 每子进程一行 | 在父进程建池 ⇒ 并发 4 个任务必现 InterfaceError | K3 | ✅ 09-02：worker_process_init 建池，`[worker-init] pid=…` 每子进程一行（集成测试断言） |
| C-10 | Redis broker **没有真优先级队列** | Celery 文档（priority 在 Redis 上是分桶模拟，最多 10 级且不严格） | 交互式 run 被批处理淹没 | 用**分队列**（interactive / batch / maintenance）+ worker 按队列订阅，不靠 priority 参数（课件 §12.3 同款） | 只用 priority：灌 100 条 batch 再投 1 条 interactive ⇒ 其 P95 破 K3 SLO | K3 | 🔶 配置已写（celery_app.py），负向认证 ⬜ |
| C-11 | Celery Beat 只能**单实例**跑 | Celery 文档 | 起两个 beat ⇒ sweeper/对账每周期跑两遍（sweeper 抢跑 = H102 同族） | sweeper/对账不用 beat；用我们自己的常驻循环 + PG 行锁做单飞（`27` K8 已定单机线程）；若用 beat，判据 = `pgrep -f "celery beat"` 恰 1 | 起两个 beat ⇒ `run.requeued_by_sweeper` 事件成对出现 | K8 | ⬜ |
| C-12 | 任务名即契约：改函数名/模块路径 ⇒ 老消息 `Received unregistered task` | Celery 文档 | 队列里积压的消息全部变死信 | 任务名显式固定（`@app.task(name="syncopate.execute_run")`），进 K9 schema_version 纪律 | 改名后投旧消息 ⇒ worker 报 unregistered、消息丢失 | K3/K9 | ⬜ |
| C-13 | 关停语义：warm（等当前任务）vs cold（立刻）vs SIGKILL | Celery 文档 | 发布时 cold 关停 = 强杀（H105） | `drain` = warm shutdown + 放 lease；K9 发布五能力之一要演练 | cold 关停正在写工具的 worker ⇒ 意图日志出现 running 行 | K9 | ⬜ |
| C-14 | 用 `countdown/eta` 做退避重试 | Celery Redis transport 文档：eta > visibility_timeout 的任务会被**多次执行** | 退避 15min 的任务被投两次 | 退避不用 eta：transient ⇒ run 回 `queued` + outbox `next_attempt_at`（课件 §9/§14.3），Celery 层零重试逻辑 | 用 eta=20min 且 visibility_timeout=10min ⇒ 任务执行两次 | K3 | ✅ 09-02：退避全走 outbox.next_attempt_at；Celery 侧只有基础设施错误 5/15/45s 短重投 |
| C-15 | 启动时 broker 不可用：`broker_connection_retry_on_startup` | Celery ≥5.3 行为变更 | worker 起在 Redis 之前 ⇒ 直接退出，systemd 反复拉起 | 设 True + 判据行 `[celery-broker] connected`；起服务顺序进 runbook | 先起 worker 后起 Redis ⇒ 无判据行则退出 | K3/K11 | 🔶 配置已写（celery_app.py），负向认证 ⬜ |

---

## 2 · Redis（中间件层）

| # | 坑 | 来源 | 后果 | 防法 / 判据 | 负向认证 | 归属 | 状态 |
|---|---|---|---|---|---|---|---|
| R-01 | 默认只有 RDB 快照 ⇒ 重启丢最近几分钟消息 | 课件 CH3 §4.4"要认真配置持久化"；附录 A §3.3 | 已 dispatched 但未消费的 job 消失；outbox 已标 dispatched ⇒ **没有人再投**（僵尸 queued，H37） | `appendonly yes` + `appendfsync everysec`（`scripts/redis_bootstrap.sh` 已写死并有判据行）；且 sweeper 扫"queued 超龄 ∧ outbox 已 dispatched" ⇒ 告警+人工重投（27 K8 已定） | 关 AOF，投 10 条后 `kill -9 redis` 再起 ⇒ 队列空、10 条 run 停在 queued；sweeper 告警必响 | K3/K8 | 🔶 配置已写，负向认证 ⬜ |
| R-02 | `maxmemory` 未设 = 主机 OOM；设了 + 默认淘汰策略 = **静默丢队列消息** | Redis 文档 | LRU 把 job 当缓存淘汰，谁都不报错 | `maxmemory-policy noeviction`（写入报错是正确行为）；四职分库（broker 单独 db，见 R-07） | 改成 allkeys-lru + maxmemory 1mb，灌 1000 条 ⇒ 消息数 < 1000 且无错 | K3 | 🔶 配置已写，负向认证 ⬜ |
| R-03 | 无密码 / 监听 0.0.0.0 | 课件 H85 | 密码泄漏 = 直接往队列塞 `execute_run` job，**绕过 API 六道入口闸** | `bind 127.0.0.1` + `requirepass` + `protected-mode`；密码从 env/secret 注入，⛔ 不进 URL/日志（K7-4 CI grep 同款） | 去掉 requirepass ⇒ 匿名 `redis-cli LPUSH` 能造一条被执行的 job | K3/K9 | 🔶 配置已写，负向认证 ⬜ |
| R-04 | 单线程：大 payload / 大 key 阻塞所有客户端 | 课件 H23 | 一条塞了完整 context 的消息拖慢限流器和信号量 | task 体只放 run_id；判据：消息体 < 1KB（CI 测试断言） | 投一条 5MB 消息 ⇒ 限流器 P99 飙升 | K3 | ⬜ |
| R-05 | Redis 重启时 **in-flight（已领未 ack）消息**的去向 | Celery Redis transport：unacked 存在 Redis 的 `unacked` 有序集里 | AOF 未 fsync 的那 1s 内丢失；worker 完成后 ack 一个不存在的消息（无害） | 接受 at-least-once + 依赖 lease/sweeper 而不是 Redis 恢复 | 同 R-01 实验 | K3/K8 | ⬜ |
| R-06 | Pub/Sub 不持久（订阅者不在线就丢） | 附录 A §5.2 | 若用它做 SSE 唤醒或 outbox nudge，把它当事实就丢事件 | 只当"唤醒"，事实永远回库查（27 K7-3 已定进程内 channel，不用 Redis Pub/Sub；outbox nudge 若用 Redis 也只是优化） | 唤醒通道断开 ⇒ SSE 仍能靠轮询/after 补齐（K7 门槛①） | K7 | ⬜ |
| R-07 | 四职共用一个 keyspace：broker / 限流计数 / 信号量 / 缓存 | 课件 CH9 §13"身兼四职" | `FLUSHALL` 清缓存顺手清空队列；key 名撞车 | 分 db：0 broker · 1 限流 · 2 信号量 · 3 缓存（`redis_bootstrap.sh` 约定）+ key 前缀；判据：各 db 的 key 前缀集合互不相交 | 在 db0 `FLUSHDB` ⇒ 限流计数不受影响 | K9 | ⬜ |
| R-08 | 信号量/锁的 key 无 TTL ⇒ worker 死后**永久占位** | 课件 CH9 §5.4 Redis semaphore | 写工具并发额度被死进程吃光 | 信号量 key 必带 TTL 且 TTL 与 lease 联动；判据：kill 持有者后 TTL 内额度自动回收 | 去掉 TTL，kill 持有者 ⇒ 额度永不回收 | K9 | ⬜ |
| R-09 | 时钟与固定窗口限流的边界效应 | 课件 CH9 §4.4 第一版固定窗口 | 窗口交界处 2× 突发 | 第一版接受；复活条件=真实流量出现边界突发 ⇒ 换滑动窗口 | — | K9 | ⛔ 裁剪 |

---

## 3 · PostgreSQL / Alembic（数据库层）

| # | 坑 | 来源 | 后果 | 防法 / 判据 | 负向认证 | 归属 | 状态 |
|---|---|---|---|---|---|---|---|
| P-01 | `now()` = 事务开始时间，同事务多条事件时间戳相同 | 课件 CH2 C7 | 按 `created_at` 排序随机 | 排序只认 `seq`；CI grep：`ORDER BY created_at` 不出现在事件查询 | 用 created_at 排序回放 ⇒ 顺序抖动测试必红 | K2 | ⬜ |
| P-02 | `updated_at DEFAULT now()` 只在 INSERT 生效 | 课件 H14 | 永远停在创建时间 | 触发器 `trg_agent_runs_touch`（0002）；判据 UPDATE 后变化 | ✅ `test_schema_migrations::test_updated_at_moves_on_update_and_not_without_trigger`（DROP TRIGGER 必红） | K2 | ✅ 09-02 |
| P-03 | `seq` 用 `MAX(seq)+1` 分配 | 课件 H13；曾在 finish_run / worker.emit / park_run_for_user / close_parked_clarify_runs 四处 | 两个写者必撞；SSE 补发出空洞 | `db.next_seq`/`append_event`（`agent_runs.last_seq` 同事务领号）；四处全部收编，全仓 `grep "max(seq)"` 只剩禁令注释 | ✅ `test_seq_allocator_two_writers_no_gaps_no_collisions` + 负向 `test_negative_max_plus_one_collides_under_two_writers`（确定性构造两事务撞号） | K2 | ✅ 09-02 |
| P-04 | Alembic 与 `schema.sql` **两份真相** | 27 准则〇 | 新库走 schema.sql、老库走 migration ⇒ 两条路建出的表不同形（schema.sql 里已有"加列路径"注释就是苗头） | 裁定：Alembic 迁移链是唯一真相，`pg_bootstrap.sh` 改成 `alembic upgrade head`；schema.sql 退役或改成由 `pg_dump --schema-only` 生成的只读快照；判据：干净库 `upgrade head` 后与快照逐字节一致 | 手改 schema.sql 加一列 ⇒ 一致性判据必红 | K2 | ✅ 09-02：迁移链 `syncopate/runtime/migrations/`（0001 baseline 内嵌旧 schema.sql · 0002 K2 地基）；schema.sql 已 git rm；快照 `schema.snapshot.txt`；pg_bootstrap 末尾 `--check` |
| P-05 | 无 ORM ⇒ Alembic **没有 autogenerate**，漂移没人报 | 我们特有（不引 SQLAlchemy ORM） | 有人直接 `ALTER TABLE` 改库，迁移链不知道 | 测试：live schema（information_schema 导出）== 迁移链建出的 schema（"两个东西应当相同"型） | 库上手动加列 ⇒ 判据必红 | K2 | ✅ 09-02 `test_fresh_upgrade_head_matches_committed_snapshot` + bootstrap `schema_snapshot.py --check` |
| P-06 | Alembic 默认把整个 migration 包在一个事务里；`CREATE INDEX CONCURRENTLY` **不能在事务内** | PG 文档 | 大表建索引锁表；或 migration 直接报错 | 需要 CONCURRENTLY 的 migration 标 `transaction_per_migration` / `autocommit_block`；expand/contract 纪律进 K9-5 | 在事务内跑 CONCURRENTLY ⇒ 报错 | K2/K9 | ⬜ |
| P-07 | 连接数：Celery prefork N 进程 × 每进程池大小 vs `max_connections`（默认 100；训练机曾调 300） | 09 §0 / MAINLINE | 起满 worker 就 `too many connections`，API 跟着挂 | 连接预算写死在配置并有判据行 `[db-pool] procs=<n> per=<m> total=<n*m> max_conn=<k>`；total ≤ 0.8×max | 把 per 调到 64 起 8 进程 ⇒ 判据行必红/连接报错 | K3 | ⬜ |
| P-08 | 领号行锁持有到 COMMIT ⇒ 长事务串行化整条 run | 课件 C9"per-run 串行化" | 一个慢工具在事务内 ⇒ API 写 cancel_requested 被阻塞 | 事务只包本地写入，⛔ 外部调用/模型调用永不在事务内（课件 §13.4）；判据：事务时长 P99 < 50ms | 在事务内 sleep 2s 调工具 ⇒ cancel 请求延迟 > 2s | K2/K5 | ⬜ |
| P-09 | `FOR UPDATE SKIP LOCKED` 配 CTE+聚合的 claim SQL，索引不对就全表扫 | 现状 `claim_run` 用了 inflight 聚合子查询 | worker 多了以后队列轮询拖慢业务查询（课件 H24 同族） | Celery 化后 claim 只按 `run_id` 一行 CAS（课件 §8.2 形态），聚合公平分配的职责移交分队列；索引 `(status, lease_expires_at)` 已有 | EXPLAIN 出现 Seq Scan ⇒ 红 | K3 | ⬜ |
| P-10 | `pg_bootstrap.sh` 假定 root（useradd/su/chown） | 本机实测 09-02 | 干净的非 root 机器一条命令重建是假的（`clean-machine-only-gaps` 老病） | 已补用户态分支（`PG_USER_MODE`）；训练机 root 路径不变 | 在非 root 机上跑旧脚本 ⇒ useradd 失败 | K0 | ✅ 09-02 |
| P-11 | 用户态 PG 的 unix socket 目录与 `psql` 默认库 | 本机实测 09-02 | `/var/run/postgresql` 不可写；`psql` 默认连同名库 `samwang` 不存在 | `-k /tmp` + `-d postgres`；应用一律 127.0.0.1 | — | K0 | ✅ 09-02 |
| P-12 | `usage_records` 唯一约束的粒度：同一 run **合法地**执行多次（审批恢复 = attempts+1） | 09-02 实测：加 UNIQUE(org,run,call_index=0) 后审批恢复的 run 全部 failed（`usage_records_once` 撞） | 约束太粗 ⇒ 把合法的第二次执行当账单翻倍拒掉，run 失败而钱已花 | `call_index = agent_runs.attempts`（每次执行一行；同一次执行重放才被拒）；K9-3 改每模型调用一行时换成调用序号 | 把 call_index 写死 0 ⇒ `test_design_conformance::test_approved_case_lets_the_run_continue` 必红（已实证） | K2/K9 | ✅ 09-02 |

---

## 4 · 我们的接缝（现有代码 × 新中间件）

| # | 坑 | 出处 | 后果 | 防法 / 判据 | 归属 | 状态 |
|---|---|---|---|---|---|---|
| S-01 | 现有 `claim_run` **顺手接管 lease 过期的 running**（自动 requeue 语义写在 claim 里） | `db.py:197` | 与 K8 sweeper 成为同一件事的**两个写入者**（H102 同族）：sweeper 迁 queued 写事件，claim 又静默接管不写事件 ⇒ 恢复不留痕 | 一个所有者：claim 只认 `status='queued'`（课件 §8.2 原型），过期 lease 的回收**只**由 sweeper 走 `transition_run` + `run.requeued_by_sweeper` | K3/K8 | 🔶 09-02：Celery 定向 claim 只认 queued ✅（test_targeted_claim…）；轮询入口仍接管过期 lease，K8 sweeper 落地时撤 |
| S-02 | 幂等键**从 run_id 推** | `11 §3.9`（validation_errors 那段：client_request_id 从 run_id 推） | 课件 §11.4：同 run 内重试正常，**rerun 换 run_id 就双重扣款** | 写工具 key 绑业务实体+版本位（`campaign_id:动作:v`），只读工具才可带 run_id；K5-6 逐工具登记 key_fn；判据：rerun 同一业务意图 ⇒ 平台去重账本命中 | K5/K6 | ⬜ |
| S-03 | `finish_run` / `emit` 的 `MAX(seq)+1` | `db.py:283`、`worker.py:65` | 见 P-03 | 同 P-03 | K2 | 🔶 |
| S-04 | `system.wait` 睡到 lease 的一半 | 09 §4.5.9 | 引入心跳后 lease TTL 变短（60s）⇒ 可等时长跟着缩到 30s，模型被教过"先等够再查" | 心跳期间 wait 可跨多个 TTL（续租在跑就不是死），`truncated_by_lease` 语义改成"被 max_duration 截断"；训练侧描述不变（守则⑮：改 runtime 语义前先对照沙盒 spec） | K3/K5 | ⬜ |
| S-05 | 现有 worker 一进程 16 并发（asyncio）⇒ Celery prefork 变 N 进程各 1 条 | B-5/E33 压测形态 | goodput 192 那组数字作废；DB 连接数模型变 | K3 门槛⑨基线重测；E33 数字标"Celery 前" | K3 | ⬜ |
| S-06 | worker 按 `--org-id` 限定是**测试隔离**的结构保证（C-1） | `db.py claim_run` docstring | 换成共享 broker 后，测试和常驻 worker 又会互相抢活（08-20 中过两次） | 队列名带租户/环境前缀（`runs.org_demo` / `runs.test`），测试用独立 Redis db 或前缀；判据：常驻 worker 订阅的队列集合 ∩ 测试队列 = ∅ | K3 | ⬜ |
| S-07 | 存档密度：`save_transcript` 是否在"模型点名工具、结果未回"那一档也存 | `agent_loop.py:156/185/205` | 缺这一档 ⇒ 分支 C 无解（课件 CH5 §4.4） | K0 核实；不足则 K5-2 补 | K0/K5 | ⬜ |
| S-08 | `resume_after_approval` 只把状态改回 queued，恢复靠 worker 读 checkpoint | `db.py:297` + `agent_loop.py:140` | 若 claim 后没带 `resume=True` 就从头跑（重复模型调用、重复读工具配额） | K0 核实 worker 调用链；判据：审批恢复后模型调用次数 = 剩余轮数 | K0/K5 | ⬜ |
| S-09 | 取消：现有 cancelled 终态由 `finish_run` 写，但 running 中的**协作式取消**（cancel_requested_at）不存在 | 11 §0 / 27 K1-4 | running 的 run 无法取消或只能强杀 | K1-4 + K3-10 + K4-4 | K1 | ✅ 09-02（`db.request_cancel` + ActionGate 入口安全点；`test_cancel_request_is_honoured_by_worker_at_safety_point`） |
| S-13 | 测试把两个 404 的**整个响应体**做相等比较当"不可枚举"判据 | 09-02 K1-7 信封加 request_id 后两条老测试红 | 判据量的是"响应体逐字节相同"，而它真正要防的是 code/message 可探测——判据形状比它要防的事宽 | 比较时剔除 request_id（探测面 = code + message）；新写此类判据先问"哪些字段是探测面" | K1 | ✅ 09-02 |
| S-16 | `run.enqueued` 可能排在 `run.started` 之后 | 09-02 集成测试实测 | 前端时间线按 seq 展示会出现"先开跑后入队"；若有判据假设顺序会误红 | 这是"先 publish 后标记"顺序铁律的必然：worker 比标记事务快。判据只断言两者都在、run.created 在首位；时间线展示按事件语义不按 seq 排这两条 | K3/K7 | ✅ 登记（不修：修它 = 反转顺序铁律） |
| S-15 | `record_tool_call` 占坑是"先查再插"：并发同键两次调用都查到"没有"，第二个 INSERT 撞 `tool_calls_external_idem_uniq` 直接抛异常 | 09-02 K1 全套连跑暴露（老竞态，K2 的领号写入改变了时序才稳定复现；`test_concurrent_same_key_returns_the_original_result`） | 撞唯一键的那次调用带异常回到 worker ⇒ run failed，而另一路可能已经在执行副作用——"返回失败但钱在花"的形状 | 课件 H01 工具级形态：UniqueViolation = 命中，回到 `_await_settled_prior` 等原结果；救你的是约束不是查询 | ✅ 09-02（三遍 278/278 稳定） | K1/K6 | ✅ |
| S-14 | worker-驱动的 API 测试假设"队列里只有我这条" | 09-02 K1 测试首跑：p95 测试留下 100 条 queued，后续 claim 抢到别人的 | 与 C-1 同族：全局/同 org 队列 ⇒ 任何 run_once 都会抢别人的活，结论跟着错 | 测试内 `_drain(org, keep)`；Celery 化后按队列名隔离（S-06） | K1/K3 | ✅ 09-02（测试卫生） |
| S-10 | `tests/runtime/test_retrieval.py` 偶发红（09-02 本机：语料灌入后首跑 1 红、复跑全绿） | 本机实测 | 偶发红的测试 = 不可信的尺子（09 §4.5.5 原话）；K3 门槛⑨"零失败"会被它污染 | 复现三次定位（怀疑：生效期按 `now()` 算、与灌入时间戳同秒；或队列全局污染 C-1 同族）；定位前不许改判据 | K0/K2 | ⬜ |
| S-12 | clarify 收场后 run 停在 `running`（无人置 waiting_for_user）⇒ lease 过期被 `claim_run` 当崩溃重抢 | verl-22 09-02 通报；29 D25 | 追问一次 = 60s 后整条 run 重跑一遍（重复模型调用 + 读工具配额）；引入 sweeper 后会变成 `run.requeued_by_sweeper` 假恢复 | 与 S-01 同根：过期 lease 的处置只能有一个所有者且必须先分清"等人"和"死了"——waiting 必须清 lease（课件 CH4 ④） | K1/K4 | 🔶 已复现（R5 L4 记录） |
| S-11 | "哪些是写工具"有两份真相：`tool_registry` 的 `kind=="write"` 与 runtime 的 `WRITE_TOOLS` 抄本（`tools.py:41`） | 29 D12 | 登记≠实现的温床：一边加了写工具另一边不知道（08-19 已中过一次） | K6-3 删抄本，权限/幂等/五态全部从 registry 派生；判据 = `WRITE_TOOLS` 符号在 runtime 里不再存在 | K6 | ⬜ |

---

## 5 · 优化候选（做实验时从这里取题；每条先定读数再动手）

| # | 题 | 读数（在哪量） | 前提 | 归属 |
|---|---|---|---|---|
| O-01 | Outbox 延迟：created→enqueued 的间隔（dispatcher 轮询周期）；加 PG `NOTIFY` nudge 后降到多少。**nudge 是优化，扫表是正确性**（课件 CH3 待查#4） | `run.created`→`run.enqueued` 事件时间差 P50/P95 | K3 dispatcher 就位 | K3 |
| O-02 | dispatcher 批大小（LIMIT 100）与轮询间隔 vs PG 负载；`(status,next_attempt_at)` 索引命中 | EXPLAIN + PG `pg_stat_statements` | 同上 | K3 |
| O-03 | prefetch=1 + 进程数 N 的吞吐曲线；与 asyncio 单进程 16 并发对照（S-05） | goodput@SLO 阶梯（`runtime_loadtest.py`） | Celery 化完成 | K3 |
| O-04 | 心跳间隔 vs TTL：故障发现时间（≤2×TTL）与心跳写入负载的权衡 | sweeper 回收延迟 + PG 写 QPS | K3-7/K8-1 | K8 |
| O-05 | Redis AOF `everysec` vs `always` 的吞吐差与丢失窗口 | 投递 QPS + kill 测试丢失条数 | R-01 | K3 |
| O-06 | 分队列（interactive/batch）下交互式 P95 在批量灌入时的劣化 | 按队列的 `oldest_job_age` | C-10 | K3 |
| O-07 | 领号串行化（last_seq 行锁）在高事件率下的争用 | 事务等待时间 | P-03 修完 | K2 |
| O-08 | 存档密度的写放大：每 append 一条存一次 ⇒ 每 run 写多少字节；大 payload 拆 blob（附录 A §3.1） | checkpoints 表大小/run | K5-2 | K5/K9 |
| O-09 | outbox/run_events 归档策略（H108）：dispatched 超 N 天清理对查询的收益 | 表大小、扫表耗时 | K3 | K9 |
| O-10 | 连接预算：进程数 × 池大小的最优点（P-07） | `pg_stat_activity` 峰值 vs goodput | K3 | K3 |

---

## 6 · 排查纪律（怎么用这张表）

```
阶段开工前   把本表该阶段的行抄进该阶段的"课件已知缺口必修表"编号引用（不复述内容）
每条排查     ① 先写负向认证脚本/测试（人为制造）② 跑一次确认判据红 ③ 上防法 ④ 再跑确认绿
             ⑤ 状态改 ✅ 并在"防法/判据"格填 file:line 或测试名
不许         只改状态不留证据；把"配置写了"当成"防住了"（R-01/R-02/R-03 现在就是这个状态）
优化实验     §5 每题先注册读数与读数位置，再动手；结论进 27 对应阶段门槛表或 E 报告，本表只留指针
```
