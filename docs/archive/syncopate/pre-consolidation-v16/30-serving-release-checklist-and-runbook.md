# Syncopate · 30 · Serving 上线总清单 · Runbook · 复盘模板（K11 收口物）

> 📦 **历史发布清单与基线，不能直接用于当前机器放行。** 现行说明见
> [docs/syncopate/07-SERVING.md](../../../syncopate/07-SERVING.md)。

> 立项：2026-09-02（K11）。课件附录 A 的 A1（52 条清单）与 A2（七张 runbook）按准则五裁剪成我们版：
> 清单每条 = **可证伪的判断句 + 验法**；runbook 每张 = 十一段结构裁剪为八段，**止损栏（减法，不需根因）与
> 恢复栏（加法，必须有根因）分开**；每张卡的"第一步看什么"由 `scripts/serving/runbook_queries.py` 逐条真跑
> （K11 门槛②）。裁掉的条目显式登记复活条件。SLO 基线在 `_audit/serving_k11/slo_baseline_*.txt`。
> 六条总原则（A2 宪法）：先止损再恢复 · 先查证据再动作 · **回放不执行工具** · **重试必须检查幂等** ·
> 高风险工具先关自动化再人工确认 · 所有修复动作都要写事件和审计。

---

## 1 · 上线总清单（我们版 42 条 · 9 分区；✓ = 已验 · ✗ = 未达标 · 挂账 = 有归属与复活条件）

### 1.1 安全（6 条；裁 2）

| # | 判断句 | 验法 | 状态 |
|---|---|---|---|
| S1 | org_id 只从 token/Cookie 解析，请求体里没有这个字段 | `test_api::test_org_id_in_body_is_ignored` | ✓ |
| S2 | 每条 run/事件/tool_call 详情查询都带 org scope | `test_run_api_k1::test_every_agent_runs_query_in_api_carries_org_id` + trace 子表带 org | ✓ |
| S3 | 模型只能提 tool_call，碰外部世界唯一出口 = ActionGate | `test_action_gate` 签名判据（无 invoke= 形参）+ `test_agent_loop` 源码判据 | ✓ |
| S4 | 未注册工具直接拒绝（报"没有"不猜） | `test_tool_runtime_k6::test_every_gate_refusal_leaves_a_tool_calls_row…`（unknown_tool 行） | ✓ |
| S5 | secret 不进日志/事件/SSE；public 流零 internal 字段 | `test_hardening_k9::test_log_event_redacts_secrets…` + `test_events_k7::test_public_stream_strips_internal_fields…` | ✓ |
| S6 | SSE 已鉴权（Bearer 或同域 Cookie），跨租户 404 | `test_events_k7::test_same_origin_cookie_authenticates_and_cross_tenant_is_404` | ✓ |
| — | SSRF 防护 / sandbox 隔离 | **裁剪**：无外网请求类、无代码执行类工具。复活条件 = 新增任何外网/执行类工具时整节启用 | 挂账（无归属，条件触发） |

### 1.2 副作用正确性（5 条）

| # | 判断句 | 验法 | 状态 |
|---|---|---|---|
| W1 | 所有 side_effect 工具都有权限/超时/输出键登记，缺一个导入即炸 | `test_tool_runtime_k6::test_negative_registering_a_side_effect_tool…` + `assert_governance_complete` | ✓ |
| W2 | 幂等键不含 run_id；tool_calls UNIQUE(org, key) 只覆盖非 skipped_duplicate 行 | `test_worker` 键判据 + `test_same_key_second_call_skips_handler…`（UNIQUE 负向） | ✓ |
| W3 | 写动作执行前必过审批（tier C）或已裁决；D 档永不自动 | `test_design_conformance::test_tier_c_action_must_go_through_approval` · tier_d 落库行 | ✓ |
| W4 | 半成功写调用落为 response_lost 且禁止自动重试 | `test_loop_k5::test_response_lost_write_is_not_retried…` + 负向"抹掉意图日志必重发" | ✓ |
| W5 | 下游按 idempotency_key 可查（platform_ledger），不可查时转 manual_review | `test_recovery_k8::test_branch_c…` / `…ledger_unavailable` | ✓ |

### 1.3 恢复与对账（6 条）

| # | 判断句 | 验法 | 状态 |
|---|---|---|---|
| R1 | lease 心跳在跑，过期 lease 由 sweeper 回收（三分支顺序：取消→次数→重投） | `test_outbox_k3::test_heartbeat…` · `test_recovery_k8::test_cancel_request_with_dead_worker…` | ✓ |
| R2 | 恢复从最新快照续跑，只读工具不重调，模型只多一次 | `test_loop_k5::test_read_only_recovery…` · `test_recovery_k8::test_branch_a…` | ✓ |
| R3 | 一切状态变更走 transition_run（grep 判据全仓一处） | `test_state_machine_k4::test_no_status_write_outside_transition_run` | ✓ |
| R4 | 周期对账扫 response_lost，按键回填或 manual_review，三写入同事务 | `test_recovery_k8` 三条对账测试 | ✓ |
| R5 | retry 有 max_attempts/退避/上限，超限 failed 或死信 | `test_outbox_k3::test_backoff_is_capped…` · `test_transient_error…` | ✓ |
| R6 | replay 只回放事件不执行工具 | `test_events_k7::test_sse_endpoint_path_calls_nothing_but…`（AST） | ✓ |

### 1.4 版本治理（4 条；裁 1）

| # | 判断句 | 验法 | 状态 |
|---|---|---|---|
| V1 | run 创建时记 contract/prompt 版本，claim 后记 model 版本 | `test_flywheel_k10::test_metrics_slice_by_version` | ✓ |
| V2 | 每次 tool_call 记 registry 版本（描述哈希） | 0007 列 + `flywheel.registry_version` | ✓ |
| V3 | 新代码能读旧 checkpoint（无 v = v1）；未知版本拒绝 + manual_review 不崩 | `test_hardening_k9::test_old_checkpoint_without_version…` | ✓ |
| V4 | event 只加字段不改含义；outbox payload 加 v | **挂账 K9-5**：outbox/事件 payload 未加 v（读侧无消费者时无害）| 挂账（K 线；复活 = 第一次改 payload 含义前） |
| — | 工具破坏性变更新旧并存 | 裁剪：工具由沙盒 spec 单一真相 + 训练同形约束，破坏性变更 = 数据版本变更（26 号管线） | 挂账（26 线口径） |

### 1.5 数据生命周期（3 条；裁 2）

| # | 判断句 | 验法 | 状态 |
|---|---|---|---|
| D1 | outbox dispatched 超 30 天归档删除 | `sweeper.sweep_once` E 段（`test_recovery_k8` 未单测；计数在 counts） | ✓（机制在，归档量本机为 0） |
| D2 | 定义了 RPO/RTO 并演练过恢复 | §4 灾备演练：数据层干净重建 RTO 实测；模型权重那半未演练 | ✓/挂账 |
| D3 | 备份 = 迁移链 + 快照可重建；业务数据的备份策略 | **挂账**：dev 期数据库是派生产物（08 §1.1）；生产前定 pg_dump 周期 | 挂账（K 线；复活 = 灰测放真人） |
| — | 大表分区、trace payload 拆 blob | 裁剪：行数量级远未到（本机 <1e5）；复活 = run_events > 1e7 行 | 挂账 |

### 1.6 可观测与告警（5 条；裁 1）

| # | 判断句 | 验法 | 状态 |
|---|---|---|---|
| O1 | 错误路径结构化日志可按 run_id/org/error_code 搜；密钥打码 | `test_hardening_k9::test_log_event_redacts…` | ✓ |
| O2 | /metrics 暴露 run/队列/工具/成本/stuck 12 项 | `test_hardening_k9::test_metrics_endpoint_exposes_the_panel…` | ✓ |
| O3 | 告警落在业务健康上，每条绑 runbook 卡 | `metrics.alerts` + 本文 §2 六张卡 | ✓ |
| O4 | 判据行体系：每个机制生效都有一行（[dispatcher]/[worker-init]/[lease-heartbeat]/[sweeper]/[redis-config]…） | 集成测试断言判据行；`scripts/*_bootstrap.sh` | ✓ |
| O5 | 做过告警演练（告警真的会触发） | `scripts/serving/runbook_queries.py` 输出 [alert] 行（本机 dev 库上 4 条真触发） | ✓ |
| — | trace_id 贯穿 API/Worker/Tool | 裁剪：run_id 即贯穿键（单实例）；request_id 只在信封。复活 = 多实例或接 OpenTelemetry | 挂账 |

### 1.7 成本与 SLO（5 条；裁 1）

| # | 判断句 | 验法 | 状态 |
|---|---|---|---|
| C1 | 九条 SLO 一键打印，读数全部来自表 | `scripts/serving/slo_readout.py`；基线 `_audit/serving_k11/slo_baseline_2026-09-02.txt` | ✓（本机 dev 库上 2 条红，见 §5） |
| C2 | 每次模型调用一行 usage（token + usage_source） | `test_hardening_k9::test_non_converging_run…`（call_index 1..6） | ✓ |
| C3 | run 级预算四字段强制生效，超限转 waiting | 同上 + 负向"关闸烧到步数上限" | ✓ |
| C4 | org 日预算两档：warn 告警 / over 拒新建 | `test_org_over_budget_rejects_new_runs…` | ✓ |
| C5 | 限流在 API/Worker/Tool/Model 各层 | **挂账**：平台侧 BUC 限流 ✓、org 日预算 ✓；API 全局并发与写工具并发额度未做 | 挂账（K 线；复活 = 灰测对外） |

### 1.8 测试与演练（5 条；裁 1）

| # | 判断句 | 验法 | 状态 |
|---|---|---|---|
| T1 | 八类测试每类可点名 | 29 §2.4 | ✓（混沌测试裁剪，复活 = 多实例） |
| T2 | 故障注入覆盖 worker 崩溃/模型 429/工具超时/response_lost | `FaultPlan` 族 + K3/K5/K8 测试 + 真 Celery ack 前崩溃 | ✓ |
| T3 | 幂等测试：同键重复调用不重复执行 | K6 幂等闸 + K3 集成 | ✓ |
| T4 | migration/replay 测试 | `test_schema_migrations` · `test_old_checkpoint_without_version…` | ✓ |
| T5 | 压测在 Celery 化后重跑 | **挂账 S-05**：需训练机端点 | 挂账（K 线；复活 = 回训练机） |

### 1.9 发布流程（3 条）

| # | 判断句 | 验法 | 状态 |
|---|---|---|---|
| P1 | Alembic 迁移链是唯一真相，干净库 upgrade head == 快照 | `test_fresh_upgrade_head_matches_committed_snapshot` + pg_bootstrap 末尾 --check | ✓ |
| P2 | 五能力：开关/禁工具/drain 真演练；停队列/回滚只写命令 | `test_release_gate` · `test_disabled_tool_switch…` · `test_drain_warm_shutdown…`；命令在 09 §0 | ✓/挂账（停队列、回滚未真演练） |
| P3 | runbook 已写且与告警绑定 | 本文 §2 + `metrics.RUNBOOK` | ✓ |

**合计**：✓ 36 · ✓/挂账 3 · 挂账 3（V4/D3/C5/T5 归 K 线，条件触发型 3 条无归属但条件写死）。

---

## 2 · Runbook（六张卡；八段：症状 · 影响 · 常见原因 · 第一步看什么 · 关键判断 · **止损** · 恢复 · 验证）

### RUNBOOK 01 · queue lag 持续升高
- **症状**：`oldest_queued_run_age_s` > 60 持续；[alert] queue_lag。
- **影响**：用户看"排队中"变长；SLO queue lag P95 < 10s 破。
- **常见原因**：worker 数不够 / broker 断 / dispatcher 死 / 模型端点慢导致 worker 全在等。
- **第一步看什么**（`scripts/serving/runbook_queries.py` 卡 01）：outbox pending 与最老 pending 秒 · queued 数与最老秒 · "dispatched 但没 pending 也没 started"数 · running 数与活 lease 数。
- **关键判断**（现象 → 判断）：outbox pending 涨而 queued 不动 ⇒ **dispatcher 死/broker 断**（看 [dispatcher] 判据行、Redis ping）· queued 涨 pending=0 且 running 满、活 lease 满 ⇒ **worker 容量不够/模型慢**（看 [lease-heartbeat] 频率）· "dispatched 但未 started" >0 ⇒ **消息丢了**（Redis 重启/AOF 未 fsync）⇒ 人工 `requeue_outbox`。
- **⚠️ 止损（不需根因）**：`SYNCOPATE_RELEASE_HALTED=1`（写动作全停）· 暂停 batch 队列 `celery control cancel_consumer batch` · 降 worker 并发。⛔ 不要"重跑全部 running"。
- **恢复（需根因）**：起 dispatcher / 修 Redis / 加 worker（`celery … -c N`）/ 模型端点扩容。
- **验证**：oldest_queued 回落到 <10s；`stuck_runs` 未涨；无重复副作用（`duplicate_prevented_total` 增量 = 重投次数，不多）。

### RUNBOOK 02 · 卡死 run
- **症状**：`stuck_runs` ≥ 10；用户"一直转圈"。
- **影响**：单 run 卡死不影响别人（lease 隔离）；批量卡死 = 容量或依赖故障。
- **常见原因**：worker 死后 sweeper 没跑 / lease TTL 设短误判 / waiting_for_user 无人裁决 / 僵尸 queued。
- **第一步看什么**（卡 02）：running∧lease 过期数 · waiting 超 6h 数 · stuck_queued 告警未消数 · attempts 到上限的 running。
- **关键判断**：lease 过期数 >0 且 [sweeper] 判据行停了 ⇒ **sweeper 没起**；过期数 >0 且 sweeper 在跑 ⇒ 看 attempts（到上限 ⇒ 它在被反复救 ⇒ 看该 run 的 error_json）；waiting 超龄 ⇒ **人没裁决**（不是故障，去审批页）；stuck_queued ⇒ **投递层丢消息**，`requeue_outbox` 人工重投。
- **⚠️ 止损**：不动状态。先 `GET /runs/{id}/trace` 看括号法则（哪层括号没闭合）。
- **恢复**：起 sweeper；对确认死的 run 由 sweeper 自然回收（⛔ 不手改 status——走 repair/requeue 接口留痕）。
- **验证**：stuck_runs 回落；被救的 run 事件流有 `run.requeued_by_sweeper` + `run.restarted`。

### RUNBOOK 03 · 写工具报错 / 对账
- **症状**：`write_tool_error_rate` > 0.1%；`response_lost_open` > 0；死信 > 0。
- **影响**：真金白银那一侧——**先假设已经花了钱**。
- **常见原因**：平台 429/613 / 本地超时（结果未知）/ 工具实现崩 / 参数校验失败（模型病）。
- **第一步看什么**（卡 03）：24h 写调用失败/总 · response_lost 开放行 · duplicate_prevented 总数 · 按 error_json.code 分组 · 死信数。
- **关键判断**：code=client_timeout 且 side_effect ⇒ **结果未知，只能对账**（`reconcile_once` 按键查 platform_ledger）· code=429/rate_limited ⇒ 平台限流，带键重试安全 · code=validation_failed/unknown_tool 多 ⇒ **模型病不是平台病**，归 K10 回流 · duplicate_prevented 涨 ⇒ **前面有闸漏了**（兜底生效 ≠ 健康）。
- **⚠️ 止损**：`SYNCOPATE_DISABLED_TOOLS=campaign.update_budget,campaign.create`（禁写工具，读照常）；⛔ **禁止对 response_lost 手动重发**。
- **恢复**：对账回填（自动每 5 分钟；或 `POST /runs/{id}/tool_calls/{id}/repair` 人工四样留痕）；死信 `dead_letter_jobs` 逐条 reprocess。
- **验证**：response_lost_open = 0；每条修复有 `tool.repaired` 事件 + `tool.reconcile/tool.repair` 审计；平台账本与 tool_calls 逐键一致。

### RUNBOOK 04 · SSE 断线 / 时间线缺口
- **症状**：前端"连接中断"频繁；时间线跳格；流挂死不关。
- **影响**：只影响观感，不影响执行（事实在库）。
- **常见原因**：代理 idle timeout（keepalive 4s 应挡住）/ 门铃 listener 断（2s 轮询兜底）/ 终态没发终态事件（结构上已不可能）/ seq 撞号（分配器已消灭）。
- **第一步看什么**（卡 04）：seq 有洞的 run 数 · last_seq 与事件数不一致的 run · 终态但无终态事件的 run（⚠️ 本机 dev 库里的老 run 是 K2 之前直接 SQL 造的，属历史噪声）。
- **关键判断**：有洞 >0 ⇒ **有人绕过领号器写事件**（grep `INSERT INTO run_events`）· 终态无事件 >0 且是新 run ⇒ **有状态裸改**（K4 grep 判据应红）· 都为 0 而前端断 ⇒ 网络/代理层。
- **⚠️ 止损**：无（只读路径）。
- **恢复**：前端带 `after=N` 重连即补齐；listener 断了看 [sse-bell] 判据行。
- **验证**：`test_events_k7` 两路补发；`GET /runs/{id}/events?after=N` 手动补一次。

### RUNBOOK 05 · 版本迁移失败
- **症状**：`alembic upgrade head` 报错 / pg_bootstrap 末尾快照核对红 / run.manual_review（unsupported_checkpoint_version）。
- **影响**：进程起不来（迁移链断）或旧 run 恢复被拒。
- **常见原因**：revision 名 >32 字符（S-30）/ 有人手改库（漂移）/ DDL 需 CONCURRENTLY 却在事务内 / checkpoint 格式改了没升版本。
- **第一步看什么**（卡 05）：alembic 版本 · checkpoint 版本分布 · run 级版本分布 · manual_review 事件数。
- **关键判断**：`--check` 红 ⇒ 漂移，**先找谁改了库**（不许反向改快照）· manual_review 涨 ⇒ **读侧没转换器**，加版本转换后 resume。
- **⚠️ 止损**：不回滚 DDL（expand/contract：contract 不可逆）；新 worker 停投（`cancel_consumer`），旧 worker 继续吃在途。
- **恢复**：修迁移脚本 → `alembic upgrade head` → `schema_snapshot.py --write` 提交快照。
- **验证**：`test_schema_migrations` 全绿；bootstrap 末尾 ✅。

### RUNBOOK 06 · token 消耗异常
- **症状**：org 日用量突增；`budget_waiting_total` 涨；[budget] warn/over 判据行。
- **影响**：成本；被拦的 run 停在 waiting_for_user（不是失败）。
- **常见原因**：不收敛 run（模型来回调工具）/ 预算配置太低 / 某 org 刷量。
- **第一步看什么**（卡 06）：今日 org 用量 Top5 · 预算超限转 waiting 的 run 数 · 单 run token Top5 · org 预算配置。
- **关键判断**：单 run Top 远超中位 ⇒ **不收敛 run**（看它的 tool_calls 序列，多半 unknown_tool/validation 循环）⇒ K10 回流 · 全 org 均匀上涨 ⇒ 流量真涨，调 `org_budgets`。
- **⚠️ 止损**：`org_budgets.daily_tokens` 调低（新建即 429）；`max_model_calls` 调低。
- **恢复**：人看 waiting 的 run，`resume`（加预算）或 `cancel`。
- **验证**：/metrics `budget_waiting_total` 不再涨；SLO C9 org_budget_ratio < 1。

### 事故传导图（分诊：同时响的告警只有一个是因）

```
模型端点 429/慢 ──▶ worker 全在等 ──▶ 活 lease 满、queued 涨 ──▶ queue_lag 告警（01）
                                   └▶ 单 run 超 max_duration ──▶ budget 转 waiting（06）
Redis 重启/AOF 丢 ──▶ dispatched 未 started ──▶ stuck_queued 告警（02）──▶ 人工 requeue
平台 5xx/超时 ──▶ response_lost 涨（03）──▶ 对账 ──▶ 若账本不可用 ⇒ manual_review
sweeper 停 ──▶ running∧lease 过期堆积 ──▶ stuck_runs（02）
```

---

## 3 · 复盘模板（13 字段 + Agent 五问）与一次历史回填

**模板**：① 标题/时间窗 ② 影响面（org/run 数/钱） ③ 症状与首个告警 ④ 时间线（含止损与恢复分界） ⑤ 根因（不是"人为失误"）
⑥ 为什么检测没抓到 ⑦ 哪些检测有效 ⑧ 止损动作及其副作用 ⑨ 恢复动作及验证 ⑩ 重复副作用核查结果 ⑪ 对账完成情况
⑫ 新增检测/判据/runbook 改动 ⑬ 回流：导出了哪些 case（K10）。**Agent 五问（事故结束 = 五问全否）**：还有半成功吗？还有待对账吗？
有 checkpoint 读不了吗？有版本不兼容吗？有事件丢失吗？

**回填实例：2026-08-20「常驻 worker 抢走测试租户的 run」**（01 C-1 / 11 §3.9 任务一假阴性）
① 08-20，两次 ② 测试套件（org_acme/globex）与探针结论；无真金 ③ 探针报"C 档没走审批"，状态 queued、无审批单——全对得上
④ 时间线：起常驻 worker（无 --org-id）→ 跑符合性探针 → 假阴性 → 怀疑 claim_run → 发现队列全局 → 加 `--org-id`
⑤ 根因：队列全局 + 测试与常驻 worker 共享一个库；探针的"前提"（我建的 run 是被我跑的）从未被断言
⑥ 检测没抓到：没有任何判据问"这条 run 是谁跑的"；⑦ 有效的：探针的三个读数都对——**只是量错了对象**
⑧ 止损：停常驻 worker（无副作用）⑨ 恢复：`claim_run(org_id=…)` 结构限定 + 测试 `_drain`；验证 = 随机顺序连跑 3 次全绿
⑩ 无重复副作用（假平台）⑪ 无对账 ⑫ 新增：worker/dispatcher/sweeper 一律 `--org-id`；K3 起队列名带租户（S-06）；K8 sweeper org 过滤（S-25）
⑬ 回流：无（非模型问题）。五问：半成功否 · 待对账否 · checkpoint 否 · 版本否 · 事件丢失否 ⇒ 结束。

---

## 4 · 灾备演练（K11-4）

`bash scripts/serving/dr_drill.sh`：干净目录重建 **数据层**（PG initdb 备用端口 → Redis 备用端口 → alembic upgrade head →
快照核对 → 建 run/领取/收尾冒烟），记录在 `_audit/serving_k11/dr_drill_log.txt`。**实测 RTO = 1.5s**（09-02 16:22，
29 张表 · 事件 created/started/completed 三条齐；新暴露隐形前提 = 0，两条已知前提回填在下行）。
挂账：模型权重/candidate 那半（HF 仓库拉回 + 端点起）本机没有权重，未演练；隐形前提回填：conda-forge 的 PG/Redis
用户态可复现（08 §1.1 训练机是 deb 解包），`.venv` 需 `uv sync --inexact --extra runtime --extra dev`。

---

## 5 · SLO 基线（K11-5）

`_audit/serving_k11/slo_baseline_2026-09-02.txt`（本机 dev 库，含全天测试污染，**不是生产读数**）：九条里 6 绿、2 红
（run.created 成功率 0.977 与 run.failed 1.6% —— 红的来源是测试直接 SQL 造的无事件 run 与故意造的失败 run）、1 无读数
（POST /runs P95 需 `--api` 真打；K1 测试实测 P95 < 300ms）。生产 baseline 在灰测放真人后重打并替换本节。
