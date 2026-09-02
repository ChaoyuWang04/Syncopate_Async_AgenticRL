# Syncopate · 29 · Serving 对照清单集（K0 总表 + 各阶段函数级细表）

> 立项：2026-09-02（K0 开工日）。`27 §0` 准则〇要求每阶段开工前打「已有 / 缺失 / 不同形」
> 三列对照清单并带 `file:line` 证据；这些表放进 27 会把施工文档撑成登记簿（Chaoyu 要求 27
> 保持纯洁），所以集中在本文。§1 是 K0 总表（能力级），§2 起是各阶段开工时补的函数级细表
> （K5-1 / K6-1 / K7-1 各占一节，开工时就地填）。状态只有四种：
> **✅ 已有 · ❌ 缺失 · 🔶 不同形（写清差异与去向）· ⛔ 裁剪（27 已标注理由）**。
>
> ⚠️ 课件笔记（`docs/reference/`）只按编号引用"49 条能力清单"（第 25/33/35/39/41 条），
> **从未逐条展开**。本表按十章各自的"你应该能实现 / 交付物"归纳出 **77 条**，
> 不冒充原 49 条（守则④：匹配不上宁可报没有）。

---

## 0 · K0 门槛核对（`27 §2`）

| # | 门槛 | 结果 | 证据 |
|---|---|---|---|
| ① 对照表零空格 | 77 条 × 状态/证据/差异 | ✅ §1 | 每条 ✅ 行带 file:line；❌/🔶 行带去向 |
| ② 重验"已落地"条目 | B-0～B-7 · F-1～F-6 逐条 | 🔶 **11/13 重验通过，2 条本机不可验** | §3；验法 = 本机跑测试 + 读实现，不读登记表 |
| ③ 选型呈报 | §16 四件 | ✅ 已裁（Chaoyu 09-02 先拍，K0 改为核实） | §4 核实结论 |
| ④ 差异定性 | 每条 🔶/❌ 标去向 | ✅ | §1 最后一列 |

**测试基线（本机，2026-09-02，PG 16.15 用户态 + 语料已灌）**：
`tests/runtime` **253 collected = 247 passed · 5 skipped（v15 契约专有）· 1 xfailed（降级信号缺生产者，老账）**。
⚠️ 首次跑 `test_retrieval` 3 红 = 语料未灌（08 §1 干净机器缺口，`scripts/ingest_corpus.py` 后消失）；
灌后又偶发 1 红、复跑全绿 ⇒ 登记 `28` S-10 偶发红待查。K3 门槛⑨的回归分母 = **247**。

---

## 1 · K0 总表：能力 × 现状（77 条，五组）

### 组 A · Run API（课件 CH1，K1 承接）

| # | 能力 | 状态 | 证据 | 差异 / 去向 |
|---|---|---|---|---|
| A1 | `POST /runs` 快速返回 run_id | ✅ | `api.py:259-280` | — |
| A2 | `GET /runs/{id}` | ✅ | `api.py:282-298` | — |
| A3 | `GET /runs/{id}/events`（拉历史 + 流） | 🔶 | `api.py:455-480` | 课件分两入口（`events?after=N` 拉历史 / `events/stream` 推）；我们一个入口只走 SSE，**无 `after` 查询参数**，只认 `Last-Event-ID` 头 → K7-1 |
| A4 | `GET /runs/{id}/trace`（八表聚合，独立权限） | ✅ 09-02 | 路由表 `api.py:235-483` 无 | K1-8 建入口，K8-4 聚合 |
| A5 | `POST /runs/{id}/cancel`（协作式） | ✅ 09-02（安全点：ActionGate 入口；其余 K5-5） | 无路由；`schema.sql:24-51` 无 `cancel_requested_at` | K1-4 + K2-1 |
| A6 | `POST /runs/{id}/resume`（resume_token + 新 input） | ✅ 09-02（input 落 run.resumed 事件，loop 消费归 K5-2） | 只有 `POST /approvals/{case_ref}` 间接恢复 `api.py:310-329` → `db.py:297` | 无 resume_token、无 input 带入、任何 pending 审批单裁决即恢复 → K1-5 收编 |
| A7 | 六状态三活三终 | ✅ | `schema.sql:30-32` CHECK | — |
| A8 | 幂等三态 201 / 200 / 409 | ✅ 09-02 | 命中返回 **201 + created=False** `api.py:266-280`；无 `input_hash` 列 | 同 key 不同 input 判不出 409（H11）→ K1-3 + K2-1 |
| A9 | 幂等靠约束不靠先查（H01） | ✅ | `db.py:159-189` ON CONFLICT DO NOTHING + 回查 | — |
| A10 | 隔离进 SQL，跨租户 404 | ✅ | `api.py:290-298`；11 §3 R3.2 | — |
| A11 | 双层错误码信封 `{error:{code,message,request_id}}` | ✅ 09-02 | `HTTPException(404,"run 不存在")` 纯字符串 `api.py:297` | K1-7 |
| A12 | response_model 出口白名单 | ✅ | 11 §3 R3.9；`api.py` 各路由 | — |
| A13 | input 按 run_type 分发子 schema 校验（H06） | ✅ 09-02（RUN_INPUT_MODELS，只有 chat） | `RunCreate` 只有 user_message/intent/automation_tier | 无 run_type 概念；一条消息=一个 run → K1-1 定 run_type 是否引入 |
| A14 | id 不可枚举 | ✅ | `api.py:272` `run_` + uuid4 hex12；`api.py:336` conv 同 | case_ref 见 §4-3 |
| A15 | `POST /runs` P95 < 300ms | ✅ 09-02 实测（test_post_runs_p95_under_300ms） | 11 §5 压测 I01 P95 1.3s 是整条 run；创建耗时未单独量 | K1 门槛⑥补量 |

### 组 B · 数据库（课件 CH2，K2 承接）

| # | 能力 | 状态 | 证据 | 差异 / 去向 |
|---|---|---|---|---|
| B1 | 两本账 agent_runs（余额）+ run_events（流水） | ✅ | `schema.sql:24,62` | — |
| B2 | 八张核心表 | ✅ | 9 张（+approval_cases）`schema.sql:24-216` | model_calls 表**无写入路径**（全仓无 INSERT）→ K5/K9 |
| B3 | 单调 sequence 同事务领号（last_seq） | ✅ 09-02 | `db.py:290-293` `MAX(seq)+1`；`worker.py:65-91` 有界重试掩盖撞号 | H13 → K2-2；`28` P-03 |
| B4 | 子表冗余 org_id | ✅ | run_events/agent_steps/model_calls/tool_calls/checkpoints/usage/audit 全有 | 可开 RLS 的前提已具备 |
| B5 | tool_calls `UNIQUE(org,tool,key)` + CHECK 写工具 key 非空 | ✅ 09-02（0002：side_effect 列 + CHECK；side_effect 由 K6 填） | `schema.sql:130-132` 部分唯一索引 `(org_id, key)` | 无 tool 维度（key 已含 tool 名，等价）；**无 CHECK**（H19）→ K2-1 |
| B6 | tool_calls 两阶段写入（意图日志 created/ended） | 🔶 | `db.py:439-462` 先占坑再执行，`ok` NULL=执行中 | 无 `ended_at`/`status` 列，"执行中"靠 NULL 三值表达 → K2-1/K6-2 五态 |
| B7 | usage_records UNIQUE 防账单翻倍 | ✅ 09-02（call_index=attempts，28 P-12） | `schema.sql:144-153` 零约束 | H15 → K2-1 |
| B8 | checkpoints UNIQUE(run, index) | ✅ | `schema.sql:141` `(org_id,run_id,step)` | — |
| B9 | `input_hash / cancel_requested_at / resume_token / last_seq / version` 五列 | ✅ 09-02（0002，另加 run_type/parent_run_id/rerun_reason） | `schema.sql:24-51` 一列都没有 | K2-1 一次建齐 |
| B10 | `updated_at` 触发器 | ✅ 09-02 | 应用层每处手写 `updated_at=now()` `db.py:245,280,325` | 漏写一处就停摆（H14）→ K2-4 触发器 |
| B11 | 索引 `(org_id,status,created_at)`；不建与 UNIQUE 重复的 | ✅ 09-02（加 agent_runs_list，删 run_events_replay） | `schema.sql:58-59` 有 `(status,lease_expires_at)`；`run_events_replay (org,run,seq)` 与 UNIQUE **完全重复** `schema.sql:75` | H16 同病 → K2-5 |
| B12 | 创建事务 = agent_runs + run.created 同事务 | ✅ 09-02（`db.create_run`；原子性测试在案） | `db.py:169-183` 只 INSERT agent_runs；**全仓无 `run.created` 事件**，事件流从 `run.started` 开始 | K2-6（K3-2 再加 outbox 行） |
| B13 | 版本化 migration（Alembic） | ✅ 09-02 | `schema.sql:427` 手工 `ADD COLUMN IF NOT EXISTS` 加列路径 | §16-2 已裁 → K2；`28` P-04/P-05 |
| B14 | `now()`=事务时间，排序只认 seq | ✅ | `api.py:426` `ORDER BY seq` | — |

### 组 C · 队列与 worker（课件 CH3，K3 承接）

| # | 能力 | 状态 | 证据 | 差异 / 去向 |
|---|---|---|---|---|
| C1 | Outbox 表 + dispatcher | ✅ 09-02 | 全仓 0 处（11 §3 R5.6） | K3-1～K3-3 |
| C2 | 队列本体（broker） | ✅ 09-02 Celery+Redis（轮询入口保留给测试/考场链） | 无 broker：worker 每 0.2s 轮询 `claim_run` `worker.py:144-145,535-541` | §16-1 已裁 Celery+Redis → K3-4 |
| C3 | claim = 条件 UPDATE + lease + COALESCE(started_at) | ✅ | `db.py:197-262`（SKIP LOCKED · org 公平分配 · attempt+1） | 🔶 **顺手接管过期 lease**（`db.py:230`）= 与 sweeper 双写入者 → `28` S-01，K3-6 收窄为只认 queued |
| C4 | lease 心跳续租 | ✅ 09-02（Celery 路径；轮询路径无心跳） | 全仓无 renew/heartbeat；`lease_expires_at` 只在 claim 写一次 `db.py:244` | H30 → K3-7 |
| C5 | 先写库再 ack | ✅ 09-02（acks_late；集成测试③） | 无 ack 概念（无 broker） | K3-8 |
| C6 | 错误分类 transient/permanent + 退避 + DLQ | ✅ 09-02（副作用感知归 K5/K6） | 工具级 retriable 重试 `tools.py:114-136`；run 级异常一律 `failed` `worker.py:318-322`；无 DLQ | K3-9 |
| C7 | worker 层不改 run 状态（一个写入者） | 🔶 | `run_once` 兜底 `finish_run(failed)` `worker.py:319-322` = worker 层写状态 | H102 → K3-5/K4-6 |
| C8 | 协作式取消消费端（四个安全点） | 🔶（1/4：工具调用前已接；模型调用前/step 后/下轮前 K5-5） | 无 cancel_requested_at 读点 | K3-10/K4-4/K5-5 |
| C9 | 积压指标 `oldest_job_age` + 告警 + 分队列 | ✅ 09-02（三队列建好；告警线 60s；面板归 K9） | `metrics.py:75` `queue_wait_seconds`（排队等待）；无告警、无分队列 | K3-11；分队列 = `28` C-10 |
| C10 | 三个 attempts 分账 | ✅ 09-02（outbox.attempts / Celery retries / agent_runs.attempts） | 只有 `agent_runs.attempt`（claim +1）`db.py:245` | K3-12；列名 `attempt` vs 课件 `attempts`，K2 迁移时统一 |
| C11 | job payload 只放 run_id | ✅ 09-02 | 无 job | K3-4；`28` H23 |
| C12 | 优雅关停（drain） | 🔶 | 信号置位不 exit，跑完当前动作 `worker.py:576-579` | 中途停的 run 等 lease 过期才可接管（无主动放 lease）→ K9-6 drain 演练 |
| C13 | 队列 SLO queue lag P95 < 10s | 🔶 | 11 §5 队列最老等待 1.9s（DB 轮询形态） | Celery 化后重测（`28` S-05） |

### 组 D · 状态机 · loop · Tool Runtime · 事件流（课件 CH4–CH7，K4–K7 承接）

| # | 能力 | 状态 | 证据 | 差异 / 去向 |
|---|---|---|---|---|
| D1 | 迁移白名单 + `transition_run` 唯一入口 + reason/actor 必填 | ❌ | 状态裸改四处：`db.py:241`（claim）`db.py:278`（finish）`db.py:324`（resume）`gateway.py:143`（open_approval_case） | K4-1/K4-6；K4 门槛② grep 判据 |
| D2 | 事件名映射（succeeded→`run.completed`） | 🔶 | `db.py:259` `_TERMINAL_EVENT` 用 **`run.succeeded`**；`api.py:406` TERMINAL 同名 | 课件口径 `run.completed`；前端 `sse.ts` 也认现名 ⇒ K4-2 改名要三处同步（含前端） |
| D3 | 非法迁移 → 409 `INVALID_RUN_TRANSITION` | ❌ | 无 | K4-1 |
| D4 | checkpoint 每 append 一条存一次，带 `last`/`completed_tool_calls` | 🔶 | `agent_loop.py:174-180` action+observation **一起 append 后存一次**；`agent_loop.py:93-105` 快照只有 `history` | **缺"模型已点名工具、结果未回"那一档** ⇒ 分支 C 无解（`28` S-07）→ K5-2 |
| D5 | 恢复 = 读最新快照重入 loop，不重跑已完成工具 | ✅ | `agent_loop.py:108-119,140` `load_transcript(resume=True)` | 🔶 resume 判定 = "有已裁决审批单" `worker.py:445-452`；崩溃恢复路径（sweeper 重投后 resume）不存在 → K5-2/K8 |
| D6 | 快照无"下一步"字段，下一步由模型定 | ✅ | `agent_loop.py:145` 每轮重新 decide | — |
| D7 | loop 内零横切，横切收口唯一出口 | ✅ | `action_gate.py:167-378`；`test_agent_loop` 源码判据 | — |
| D8 | 步数上限 = permanent | ✅ | `action_gate.py:187-193` → `worker.py:522-529` failed | — |
| D9 | timeout 三层（模型 / 工具 / run） | 🔶 | 模型 120s `decider.py:133,140`；工具 30s `tools.py:90` | **run 级无** → K5/K9-2 max_duration |
| D10 | 错误分层：transient 回队列 / permanent 终态 | 🔶 | 只有 permanent 路（failed）；动作失败观测回模型 `agent_loop.py:213-216` | K5-4 |
| D11 | Tool Runtime 四道闸（找定义→schema→权限→幂等） | ✅ | `action_gate.py:201`（存在）`:212`（必填）`tools.py:99`（权限）`db.py:405`（幂等） | 🔶 schema 校验只查**必填缺失**不校验类型（09 §4 ⑧ 字符串数字坑）→ K6-1 |
| D12 | 注册断言 side_effect ⇒ idempotency_required + key_fn + timeout + output_schema | 🔶 | `tool_registry.py:168` 断言 write⇒fact_key；**runtime 另抄一份 `WRITE_TOOLS`** `tools.py:41`（两份"哪些是写工具"的真相，`test_design_conformance` 靠测试对齐） | K6-3：以 registry `kind=="write"` 为唯一来源，删 WRITE_TOOLS 抄本；补 timeout/output_schema |
| D13 | tool_calls 五态含 `response_lost` | ❌ | `schema.sql:110` `ok BOOLEAN` 三值 | K6-2 |
| D14 | 幂等键绑业务实体、**不含 run_id** | 🔶 | `tools.py:71-84` `f"{org}:{run_id}:{tool}:{hash}"`；编排 `client_request_id=f"{run_id}:budget"` `worker.py:437` | rerun 即双重执行（课件 §11.4）→ `28` S-02，K5-6/K6-4 |
| D15 | 失败分诊 `error_json{code,message,retryable}` | 🔶 | `error TEXT` + `PlatformError.retriable` 内部字段 | K6-5 |
| D16 | output_schema 防反向污染 | ❌ | 无 | K6-6 |
| D17 | 拦下也落库（tool_calls 行） | 🔶 | 权限/步数/成本拦下只写 audit+event `action_gate.py:243-247,281-286` | 不写 tool_calls 行 → K6-1 |
| D18 | 幂等命中"执行中"返回处理中不冒充 | ✅ | `db.py:366-402` `_await_settled_prior` | 课件 `response_lost` 语义的雏形，K6-2 收编 |
| D19 | SSE 先落库再推 + heartbeat + terminal 关流 | ✅ | `api.py:409-452`；B-6a 终态事件并入 finish_run 事务 `db.py:265-293` | — |
| D20 | 双路续传 `after` + `Last-Event-ID` | 🔶 | 只有 header 路 `api.py:458-475` | K7-1 补 query 路（优先级 query > header） |
| D21 | 事件分层 public/internal/audit 过滤 | ❌ | payload 全量直推；`model.thinking` 6000 字直推 `agent_loop.py:148-151` | K7-2 |
| D22 | 通知唤醒替代扫库 | ✅ | PG NOTIFY 触发器 `schema.sql:432-440` + api sse_bell（E33） | — |
| D23 | SSE 鉴权 | 🔶 | Bearer dev-token 头（fetch 流）09 §0 | 同域 Cookie 方案未做 → K7-4 |
| D24 | 前端三步：拉状态→重放历史→接实时流 | ✅ | `frontend/src/lib/sse.ts:4,51` | 本机未构建，见 §3 F-2 |
| D25 | `waiting_for_user` 语义统一（等审批 / 等用户补充都映射到它，resume 回 queued） | 🔶 **真缺口（verl-22 09-02 通报，代码核实）** | `agent_loop.py:195-197` clarify 返回 halted 且 case_ref=None；`worker.py:519` halted 一律 return；**全库只有 `gateway.py:143` 写 waiting_for_user** ⇒ clarify 后 run 停在 running，60s 后被 `claim_run` 当崩溃重抢重跑（R5 考场 L4 8/25 status=running 即此） | K4 取舍表"waiting_for_user 改造"的实测依据；修法待裁在 `26 §2.5`/`25 §7㊱`（26 线只改 worker halted 分支，改前会通报）→ K1-5/K4-4 收编 |
| D26 | 会话历史回灌覆盖所有收场（含 reject/clarify 轮） | 🔶 | `db.py:143` `prior_turns` 只取 `status='succeeded'`；reject 收场归 cancelled 且 result=NULL | 拒绝/追问轮不进下一轮历史（守则⑮同形问题）；归 26 线 W2，K 线不动 `prior_turns` |

### 组 E · 恢复 · 硬化 · 回流 · 上线（课件 CH8–CH10 + 附录 A，K8–K11 承接）

| # | 能力 | 状态 | 证据 | 差异 / 去向 |
|---|---|---|---|---|
| E1 | sweeper（三分支顺序 + 四原则） | ❌ | 无 | K8-1 |
| E2 | reconciliation 对账（按幂等键查平台去重账本） | ❌ | 无；平台侧 `platform._seen_keys` 有去重但无查询接口 | K5-6 建账本查询 + K8-2 |
| E3 | 四动词 replay/retry/rerun/repair | 🔶 | 前端历史回放 = 只读 replay ✅；rerun/repair 无 | K8-3；`parent_run_id` 列 K4-5 |
| E4 | 七步排查 SOP + trace 聚合 | ❌ | 无 | K8-4/K11-2 |
| E5 | 端到端故障注入联测（分支 A/B/C） | 🔶 | `FaultPlan` `platform.py:60-72`（工具超时/限流/5xx/副作用已发生）+ `test_m97_stress` 12 条 + loadtest kill -9 恢复 3/3（11 §5） | 分支 C（写工具执行后记录前 kill）无自动化 → K8-5 |
| E6 | 九条 SLO 自动读数 | 🔶 | `metrics.py:58-137` 四项（按意图延迟/排队/工具延迟/读写分桶）+ loadtest §19 判定 | 九条对齐 → K9-1 |
| E7 | run 级预算四字段 + 超限转 waiting | 🔶 | 步数上限 ✅；org 日成本闸 `worker.py:286,374-379`（超限=failed/cancelled，非 waiting） | max_tokens/max_duration/max_model_calls 无 → K9-2 |
| E8 | usage 一轮一行 | 🔶 | 一 run 一行 `worker.py:495-500`（loop 结束后汇总写） | K9-3 |
| E9 | 结构化日志（11 字段） | ❌ | 判据行全是 `print` | K9-4 |
| E10 | `/metrics` + `duplicate_prevented_total` | 🔶 | `metrics.py` 函数无 endpoint；`tool_calls.replayed_from` 可算重复挡下次数但无指标 | K9-3 |
| E11 | 限流（API/org/写工具） | 🔶 | 平台侧 BUC 积分制 ✅（B-1a）；API 侧无 | K9 三维度 |
| E12 | schema_version / payload `v` | ❌ | 无 | K9-5 |
| E13 | 发布能力（开关/停队列/禁工具/回滚/drain）+ 演练 | 🔶 | `release.py:70-95` 开关 + fail-closed ✅；其余四项无、零演练 | K9-6 |
| E14 | 测试八类 | 🔶 | unit/integration/contract/故障注入/压测 ✅；migration/replay/idempotency(rerun) 缺 | K9-7 |
| E15 | `feedback_items` / `run_annotations` | ❌ | 无 | K10-1 |
| E16 | 版本号 run 级 + tool 级 | 🔶 | `conversations.model` 有模型标签（dev mode）；无 contract/prompt/registry 版本 | K10-5 |
| E17 | `training_exports` 留痕 + 出局/准入清单 | ❌ | 无（26 号管线有 DATA_VERSION，但 runtime→训练无通道） | K10-6 |
| E18 | 上线总清单 / Runbook / 复盘模板 | ❌ | 08/09 有"怎么起"，无 runbook 卡片 | K11 |
| E19 | 备份 RPO/RTO + 恢复演练 | ❌ | PG 是派生产物（08 §1.1）；业务数据无备份策略 | K11-4 |
| E20 | 多实例 SSE fanout / Model Gateway / K8s | ⛔ | 27 K7/K9 已裁剪，复活条件已登记 | — |

**汇总（09-02 K1+K2+K3 后）**：✅ 42 · 🔶 24 · ❌ 11 · ⛔ 0（合计 77）。缺失集中在四块：**Outbox/队列（C1–C5）· 状态机入口（D1–D3）· sweeper/对账（E1–E2）· 回流与运维（E15–E19）**；不同形集中在 **幂等键含 run_id（D14）· 存档密度（D4）· seq 分配（B3）· worker 写状态（C7）** 四条老病，都已在 `28` 有对应行。

---

## 2 · 各阶段函数级细表（开工时就地填，K5-1 / K6-1 / K7-1 各一节）

（K1 开工时从 §1 组 A 展开到函数级；本节现为空是正常的，不是漏填。）

---

## 3 · K0 门槛②：01 记载"已落地"条目的重验

验法 = 本机跑对应测试 + 读实现。⚠️ 两条本机不可验（需 vLLM 端点 / node 构建），**不当通过**。

| 条目 | 验法 | 结论 |
|---|---|---|
| B-0 loop 设计边界 | 读 `agent_loop.py:1-30` + `test_agent_loop` 8 条（含源码判据"横切不进循环"） | ✅ |
| B-3a ActionGate 收口 | `test_action_gate` 14 条（含签名判据：无 invoke= 形参） | ✅ |
| B-1a 平台 BUC/频次上限 | `test_platform_real_shapes` 22 条 | ✅ |
| B-1b 分页/异步任务/显式字段 | 同上 | ✅ |
| B-2 工具 30/30 | `test_tool_parity` 10 + `test_tool_impls` 10 + `test_write_tools` 9 + `test_creative_tools` 12 + `test_data_source_tools` 11 + `test_memory_and_safety` 9 | ✅ |
| B-3b 模型驱动循环 + transcript 恢复 | `test_agent_loop` 8 | ✅ 通过；🔶 存档密度不同形（D4） |
| B-4 vLLM 端点 + decider 接线 | `test_decider` 8 + `test_decider_v15` 11（假端点）；真端点**本机不可验** | ⚠️ 进程内通过，真端点未验 |
| B-5a/B-5b 对齐账本 / 对照台 | `test_tool_parity` + `test_tool_behaviour_parity` 6 + `test_tool_field_parity` 7 | ✅ |
| B-6 压测 24/25 | `test_m97_stress` 12（进程内故障注入）；loadtest 需真端点 | ⚠️ 进程内通过，压测未重跑 |
| B-7 灰测闸门 | `test_release_gate` 8（含 api.py 不许出现 SYNCOPATE_RELEASE） | ✅ |
| F-1 会话门面 | `test_conversations` 9 | ✅ |
| F-2/F-3 前端 + 部署接线 | 本机未装 node、dist 未构建 | ⚠️ 未验（只读 `sse.ts` 确认 Last-Event-ID 续传逻辑在） |
| F-5 多轮壳层 | `test_prior_turn_is_prose_v15` 2 条 **skipped（v15 专有）**；`test_decider` 覆盖 prior turns | 🔶 v14 契约下有效；v15 判据被 skip，随 26 线 |
| F-6 全量菜单 + 档位推导 | `test_tier_policy` 10 + `test_decider` 菜单条数 | ✅ |

---

## 4 · K0-3/K0-4：存储底座核实 + §16 裁定核实

**底座（本机 2026-09-02）**：PostgreSQL **16.15**（conda-forge，用户态，`~/.local/share/syncopate/pgdata/16`）· asyncpg 裸 SQL（`db.py:29-60`，JSONB/NUMERIC 显式编解码）· 事务 `db.tx()` · 行锁 `FOR UPDATE SKIP LOCKED` 在用 · 触发器在用（`notify_run_event`）· 22 张表（runtime 9 + RAG 2 + 参考数据 10 + conversations）· 行数 0（新库）· Redis **8.10.1**（用户态，`scripts/redis_bootstrap.sh`，requirepass/AOF/noeviction）。训练机：PG 16 deb（`/workspace/tools/postgres`），`max_connections` 曾调 300。

| §16 | 裁定 | K0 核实 |
|---|---|---|
| 1 队列 = Celery+Redis | 已裁 | 现状无 broker（C2）；接缝三条已登记 `28` C-09/S-05/S-06 |
| 2 PG 不换 + Alembic | 已裁 | 三能力全在用；`schema.sql` 有手工加列路径 ⇒ 两份真相风险（`28` P-04），K2 定 Alembic 为唯一真相 |
| 3 id 不迁 | 已裁 | run/conv 随机 ✅；**case_ref 待 K1 核**（`gateway.py:121` 生成处未读） |
| 4 快照式恢复 | 已裁 | `save_transcript` 即课件 checkpoint 的雏形；差一档存档 + 两个字段（D4/D5） |

---

## 5 · K0 出口：交给 K1 的三句话

1. **不缺表缺列**：K2 的 migration 第一版 = agent_runs 加 6 列（input_hash / cancel_requested_at / resume_token / last_seq / version / attempts 改名）+ usage UNIQUE + tool_calls CHECK/status/ended_at；K1 的 cancel/resume/trace 三入口全部压在这些列上，**K1 与 K2 要交错做**（27 §1 已说 K2 是 K1 地基）。
2. **四条老病先修再叠新机制**：seq 领号（B3）· 幂等键去 run_id（D14）· claim 不接管过期 lease（C3）· worker 不写状态（C7）。它们不修，Outbox/sweeper/五态叠上去只会把"两个写入者"从一处变成三处。
3. **本机能验的边界**：进程内 247 条 + 故障注入 + Celery/Redis 全部能验；真模型端点、前端构建、压测数字要回训练机。K3 门槛⑨的回归分母 = 247。
