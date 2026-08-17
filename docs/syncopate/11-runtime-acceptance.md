# Syncopate · 11 — M9 Runtime：设计符合性审计

> 写于 **2026-08-17**（M9.1–M9.6 施工完成后的第一次独立审查）。
> 本文是 **M9 验收这条线的权威文档**：设计要求什么、代码兑现到哪、缺口归谁。
> Runtime 怎么起 / 施工时抓到的 bug → `09-runtime-handoff.md`
> 设计依据 → `../syncopate-project-design-v0.1.md` §3 §5 §11 §18 §19 §21 §36–40

---

## 0 · 三十秒读懂

**45 条既有测试全绿，但设计文档 §3 的核心机制一次都没被接上。**

那 45 条验的是「**写出来的东西自己对不对**」，没有一条问过
「**它是不是设计文档要求的那个东西**」。这两件事测试抓到的东西不一样。

```
40 条判据          ✅ 兑现 20   🟡 部分 7   ⛔ 未兑现 11   ⬜ 设计里本就没定 2
五个确认的缺口      F1–F5，全部有复现，已变成 5 条 xfail(strict) 测试
```

| 层 | 状态 |
|---|---|
| 数据模型（9 张表 / 列 / 约束作用域） | ✅ **全过**，是这次审计里唯一一组零缺口的 |
| 幂等三层（顺序投递） | ✅ 过 |
| 幂等三层（**并发**投递） | ⛔ **F3** |
| 越权 / 白名单 / org 注入 | ✅ 过 |
| **自动化四档（§3）** | ⛔ **F1 —— C 档动作不走审批，直接执行** |
| **审批闭环** | ⛔ **F2 —— 人点了同意，run 永久停住** |
| 六个降级触发器 | ⛔ **F4 —— 只有 2 个在真实路径上可达** |
| 沙盒 ⊆ runtime | ⛔ **F5 —— 纪律此前没有物理载体** |
| 编排 / 组件（Outbox / RAG / 模型服务） | ⛔ 已知的"最小可用"，逐条记账在 §4 |

⚠️ **别把这份读成"M9 做砸了"。** 数据模型和幂等这两块是真扎实的 ——
它们恰好是"错了就是真金白银"的那两块。缺的集中在**编排**这一层，
而 09 自己就写着「编排是最小可用」。这次审计的价值是把那句话
**从一句自述变成一张逐条的账**。

---

## 1 · 这次用的方法（照 M8 那套，见 `10-rag-retrieval.md`）

1. **先立判据，再看代码。** 40 条判据写死在读实现之前 —— 反过来做会照着代码写判据，是自证。
2. **每条缺口必须有复现**，不能只写"读代码看着不对"。三个主发现都跑了探针。
3. **缺口要变成机器可见的东西**，否则下次照样被忘掉 ⇒ `tests/runtime/test_design_conformance.py`。
4. **`xfail(strict=True)` 而不是注释**：现在不满足所以标 xfail，**修好那天会变成 XPASS = 失败**，
   逼下一个人回来翻标记。缺口自己会喊。

★ 审计当中被自己推翻过一次：先怀疑 `open_approval_case` 没把 run 状态和审批单
放在同一个事务里（09 §3 点名过这个坑）。**读代码后确认是对的**，同事务、还清了 lease。
真正断的是**回来**那半边（F2）。⇒ 记在这里，因为"我以为断在 A、其实断在 B"本身是信息。

---

## 2 · ★★★ 五个确认的缺口

### F1 · C 档动作不走审批，写动作直接执行了（§3）

```
automation_tier 落库 = 'C'        ← API 收下了，存进 agent_runs 了
开出的审批单        = 0           ← 期望 1
run 终态           = succeeded    ← 钱花出去了
```

**根因是三段都断的一条链**：`claim_run` 的 `RETURNING` 里**没有 automation_tier** ⇒
worker 拿不到 ⇒ worker 硬编码 `DecisionContext(automation_tier=None)` ⇒
`tier_c` 触发器在真实路径上**永远不可达**。

⚠️ **最刺的地方**：API 层认认真真校验了 `pattern="^[ABCD]$"`、schema 里认认真真写了
`CHECK (automation_tier IN ('A','B','C','D'))` —— **两道关卡都在，中间那段没接**。
一个字段被完整地收下、校验、落库，然后没有任何人读它。

**这是「机制在但没接上」的第六种形态：字段全程有效，只是没有消费者。**
前五种（忘了接 / 作用域 / 时间维度 / 另一条分支 / 钩子自己成了坑）都至少有个"接"的动作，
这一种连接的动作都不存在 —— 而它长得**特别像做完了**。

⇒ C 档覆盖的是「建 campaign · 大幅扩量 · 跨地域铺开 · 关停」，**全是不可逆的**。

### F2 · 人点了同意，run 永久停在 `waiting_for_user`

```
开单后 run 状态        = waiting_for_user   ✅ 对的
人点同意后 run 状态     = waiting_for_user   🔴 没动
worker 再抢，抢到      = None               🔴 claim_run 不认这个状态
```

开单那半边完全正确（审批单与 run 状态**同事务**翻转、lease 也清了）。
断的是回来：`POST /approvals/{case_ref}` 只 UPDATE `approval_cases`，
**没有任何代码把 run 放回 `queued`**，而 `claim_run` 只认
`queued` 或 `running 且 lease 过期`。

⇒ 后果：**飞轮回路 2 的燃料（`modified_params`）落库了，执行侧的闭环却是断的。**
人改完参数没有任何东西会去执行它。

⚠️ 恢复语义是个**真的设计决定**，不是补一行 UPDATE 就完：
从头重跑（幂等键会挡住第二次写，安全但浪费）还是从断点续（要 `checkpoints`，
而那张表**现在没有写入路径**，见 §3-R5.3）？**这一条不该由审计单方面拍。**

### F3 · 并发同键命中幂等，返回的是**空**，不是原结果

```
A: ok   replayed=False data={'new_budget': 120000}
B: ok   replayed=True  data=None            ← 🔴
execute() 真的跑了 1 次                      ← ✅ 没有双花，数据库挡住了
tool_calls: id=120 ok=True  result={...}
            id=121 ok=NULL  replayed_from=120 result=NULL
```

既有的 10 条幂等测试**都是顺序投两次** —— 第一次早就写完 ok/result 了。
并发时第二次命中的是一条**还在执行中的占坑行**（ok/result 都还是 NULL），
`record_tool_call` 不看这个，直接当"原结果"返回。

⚠️ **后果不是崩，是静默假失败**：worker 判 `not written.ok` ⇒
run 记成 `failed`（`error` 还是 `None`），**而钱已经花出去了**。
用户被告知失败，账单上却有这一笔。

★ **这正是 flash-attn 那条教训的同一形状：返回空比报错更毒。**
报错至少让上游报警；返回 `None` 没有异常、没有日志、状态机看着也正常。
⇒ 幸运的是最贵的那件事（重复扣款）**没有发生** —— 唯一索引兑现了它的职责。

⚠️ 修法同样是设计决定：撞上"执行中"是该等、该返回 409、还是该报"处理中"？
**设计文档 §38 只定义了"已完成"的命中，没有定义"执行中"的命中。**

### F4 · 六个降级触发器，只有 2 个在真实路径上有生产者

```
有生产者   tool_failed · write_amount
无生产者   validation_errors · data_maturity · cap_hits · retrieval_empty_tools · automation_tier
```

`evaluate_triggers` 的六个分支单元测试全绿 —— 但那验的是**机制本身对不对**，
不是**机制有没有被接上**。worker 全文只给 `ctx` 赋了两个字段。

逐条的后果：

| 信号 | 没有生产者意味着 |
|---|---|
| `data_maturity` | **归因延迟是第一性约束**（§0.3），而 `data_immature` 这个降级永远不会发生。worker 里 `data_maturity="mature"` 是**硬编码**的字符串，还只写进 `agent_steps`，没进 `ctx` |
| `cap_hits` | runtime 侧**根本没有 cap 这一层** —— 沙盒的 32→34 条护栏在线上一条都不生效 |
| `retrieval_empty_tools` | 09 §3 说它「直接接上 M8 的 no_match」—— **接口留了，线没拉**（runtime 侧没有任何检索） |
| `validation_errors` | 无参数校验层 |
| `automation_tier` | 见 F1 |

⚠️ **不可达这件事没有行为可测** —— 你测不出一个永远不会发生的事件。
所以这条判据用的是源码扫描（`DecisionContext` 的每个字段在 worker 里有没有被赋值），
不是行为测试。这是刻意的选择，记在这里免得下次被当成"偷懒的测试"。

### F5 · 沙盒 8 个写工具，runtime 只认识 4 个

```
✅ campaign.update_budget · campaign.create · campaign.scale_budget · approval.create_case
🔴 creative.upload · memory.write_proposal · memory.invalidate · memory.conflict_resolve
```

runtime 把不在 `WRITE_TOOLS` 里的工具**一律当读工具**：不校验权限，**也不生成外部幂等键**。
所以"漏登记"的代价不是报错，是**一个写动作悄悄没有幂等保护**。

★ 真正的问题不是这 4 个（它们目前还到不了 runtime），是
**「沙盒 ⊆ runtime」这条纪律此前没有任何物理载体** ——
谁在沙盒里加一个写工具，runtime 这边不会有任何东西响。现在有了（F5 那条测试）。

⚠️ 附带查出一处**设计与实现的三方不一致**：§11-① 说「**每个**写工具必须带外部幂等键」，
而沙盒 8 个写工具里只有 3 个 `idempotent=True`。沙盒和 runtime 是一致的，
**两边一起偏离了设计文档**。⇒ 要么改 §11-①（承认有些写动作天然不幂等），
要么给那 5 个补键。**这一条要业务侧定，不是工程决定。**

---

## 3 · 40 条判据逐条

判据全部写死在读实现之前。证据列是 `file:line` 或测试名。

### 组 1 · 数据模型（§37）—— ✅ 7/7，零缺口

| # | 要求 | 结论 | 证据 |
|---|---|---|---|
| R1.1 | 8+1 张表齐 | ✅ | `schema.sql` 九张全在 |
| R1.2 | 六个业务特有列齐 | ✅ | `intent`/`automation_tier`/`requires_approval`/`data_maturity_at_step`/`external_idempotency_key`/`param_source` |
| R1.3 | approval_cases 的飞轮三列 | ✅ | `modified_params`/`outcome_checked_at`/`outcome_result`（**列在，写入路径见 R5.5**） |
| R1.4 | 按 run_id 做键的表，唯一约束必须含 org_id | ✅ | 逐张核过：agent_runs / run_events / agent_steps / checkpoints 的 UNIQUE 都带 org_id；tool_calls 幂等索引按 `(org_id, key)`；model_calls / usage_records / audit_logs 是追加式无自然键，不需要。**09 §4-① 那个 bug 改干净了，不是只改了三张** |
| R1.5 | automation_tier 值域 A/B/C/D | ✅ | CHECK 约束在（⚠️ 但没人读，见 F1） |
| R1.6 | param_source 能区分 user vs tool_result | ✅ | CHECK 四值齐 |
| R1.7 | schema.sql 是唯一 DDL 真相来源 | ✅ | `pg_bootstrap.sh` 里没有独立 DDL |

🟡 附带两条（不算判据）：`tool_calls.latency_ms` 是 **TEXT** 而 `model_calls.latency_ms` 是 INTEGER（09 §5 已记）；
`schema.sql` 头部注释「PGDATA 放不进 /workspace」**已过期**（换机器后可以放，见 08 §1.1）。

### 组 2 · 幂等与副作用（§38 §34 §11）

| # | 要求 | 结论 | 证据 |
|---|---|---|---|
| R2.1 | 请求级幂等 | ✅ | `db.create_run` 用 `ON CONFLICT DO NOTHING` + 回查（**不是先查再插** —— 并发有竞态窗口）· 3 条测试 |
| R2.2 | 任务级幂等 | ✅ | `claim_run` 的 `FOR UPDATE SKIP LOCKED` + lease · 3 条测试 |
| R2.3 | 工具级幂等 | 🟡 | 顺序投两次 ✅（4 条测试）；**并发投两次 ⛔ = F3** |
| R2.4 | 外部幂等键真的传给平台 | ✅ | `tools.py` 把 key 塞进 `kwargs["idempotency_key"]`，`platform._seen_keys` 认它 |
| R2.5 | 先占坑再执行，占坑独立事务 | ✅ | `db.record_tool_call` 顺序如此。⚠️ **没有"执行中崩溃"的测试**，只有代码顺序 |
| R2.6 | 超时两形态不可从文本分辨 | ✅ | `TIMEOUT_MESSAGE` 只有一份；runtime 只读 `exc.retriable`，不读文本 |
| R2.7 | 沙盒 ⊆ runtime | ⛔ | **F5** |

★ **R2.1 和 R2.3 是同一个问题的两次作答，只有一次答对了。** `create_run` 的注释
把并发竞态讲得清清楚楚并用了正确写法，`record_tool_call` 就在同一个文件里，
用的却是"先查再插"。⇒ 同 M8 的「保底写在另一条分支里」：**正确写法在本文件内就有，只是没被复用。**

### 组 3 · 编排与契约（§5 §11 §36）

| # | 要求 | 结论 | 证据 |
|---|---|---|---|
| R3.1 | org_id 永不信前端 | ✅ | `RunCreate` 里**根本没有**这个字段；`Depends(current_org)` |
| R3.2 | 越权在 SQL 挡 | ✅ | 每条查询带 `WHERE org_id=$1`；**别人的 run 和不存在的 run 返回同一个 404** · 越权测试在 |
| R3.3 | behavior 五分类 | ⛔ | runtime 全仓 `behavior`/`clarify`/`defer` 各 **0** 处。`propose` 只以审批单的形式隐含存在 |
| R3.4 | 终答 schema 的可判定字段 | 🟡 | `evidence` ✅（approval_cases）· `requires_approval` ✅ · `proposed_action` 🟡（`proposed_params` 是等价物）· `data_maturity` 🟡（有列，值是硬编码的）· `confidence` ⛔ |
| R3.5 | 每个写工具必带外部幂等键 | 🟡 | 只对 `WRITE_TOOLS` 的 4 个生成。三方不一致见 F5 末段 |
| R3.6 | 工具描述写"我不做什么" | ⬜ | runtime 没有工具描述层（描述在沙盒 `tool_registry`）—— **设计里就没定 runtime 要不要有**，显式停放 |
| R3.7 | 工具名自解释 | ⬜ | 同上 |
| R3.8 | 按意图剪枝 allowed_tools | ⛔ | runtime 无 `allowed_tools` 概念。⚠️ 它同时是**研究实验的混淆变量**（§11-④），异步对比时要固定 |
| R3.9 | response_model 白名单 | ✅ | 除 `/healthz` 和 SSE（流式声明不了）外每个路由都有 |

### 组 4 · 网关与安全（§39 §18 §21 §3）

| # | 要求 | 结论 | 证据 |
|---|---|---|---|
| R4.1 | 六个降级触发器齐 | ✅ | `gateway.evaluate_triggers` 六个 + `tier_c` 共七个 |
| R4.2 | 六个都真的接得上 | ⛔ | **F4**（只有 2 个有生产者） |
| R4.3 | 没有一个读模型置信度 | ✅ | `DecisionContext` 七个字段**全是外部信号**，没有任何 logprob/confidence |
| R4.4 | 网关输出是带证据的审批单 | ✅ | `proposed_params` + `rationale` + `evidence` 三者都落库 |
| R4.5 | 审批单与 run 状态同事务翻转 | ✅ | `open_approval_case` 一个 `db.tx()` 里两条语句，并清了 lease（★ 我一开始怀疑这里，是错的） |
| R4.6 | C 档一律走审批 | ⛔ | **F1** |
| R4.7 | D 档永不自动 | ⛔ | runtime **没有账户/竞品作用域的概念**；且 `ToolRuntime` 默认持有**全部**写权限（`permissions=set(WRITE_TOOLS.values())`），worker 用的正是这个默认值 ⇒ 权限闸在真实路径上从不拒绝 |
| R4.8 | 读 ⊥ 写不可合并（§21） | ⛔ | `WRITE_TOOLS` 只在 `tools.py` 内部用于权限/幂等，**观测侧没有按读/写分桶** —— 而 eval 侧 2026-08-16 刚补过这把尺子，一量就发现写桶成功率只有 20% |

### 组 5 · 观测 · 成本 · 飞轮 · 组件（§19 §36 §40）

| # | 要求 | 结论 | 证据 |
|---|---|---|---|
| R5.1 | SSE 断线补发 | ✅ | `Last-Event-ID` → `seq` · 7 条测试 · 终态事件关流 · 心跳。⚠️ `worker.emit` 用 `max(seq)+1` 分配，**并发写同一条 run 会撞 UNIQUE**（当前单 worker 串行所以碰不到，压测场景①要留意） |
| R5.2 | 单 org 日预算触顶的降级行为 | 🟡 | `Worker._over_budget` 在**跑之前**查一次；① 单跑内不封顶（一条 run 可以冲过头）② 降级动作是 `status='failed'`，§19 要的是"明确降级路径"，failed 算不算要定 |
| R5.3 | 崩溃后恢复成功率 100% | 🟡 | lease 回收 ✅ 有测试；但 **`checkpoints` 表没有写入路径** ⇒ 恢复是"整条重跑"，不是"从断点续"。重跑靠幂等键兜底（安全但浪费）|
| R5.4 | 长任务不阻塞其他任务 | ⛔ | `Worker.run_once` 严格串行，一个 worker 一次一条。480s 工具会**堵死整条队列**。且 `claim_run` 是**全局 FIFO，不按 org 隔离** ⇒ 单 org 刷爆会饿死其他 org（压测场景⑤） |
| R5.5 | 飞轮接口有写入路径 | 🟡 | 回路 2 的 `modified_params` ✅（`POST /approvals/{case_ref}`）；回路 3 的 `outcome_checked_at`/`outcome_result` ⛔ **除 schema.sql 外无任何代码引用**（设计里归 M12，属预留，不算欠债） |
| R5.6 | Outbox + 队列（§36） | ⛔ | 全仓 0 处。当前是"直接写库 + worker 轮询" |
| R5.7 | RAG 服务（§36） | ⛔ | runtime 侧无任何检索。⇒ `retrieval_empty` 触发器（F4）和 §19 的「RAG 检索 P95 < 200ms」**都没有承载物** |
| R5.8 | 模型服务 + LoRA 热加载（§36） | ⛔ | 无。**`model_calls` 表建了但没有写入路径** ⇒ §19 的 TTFT/TPOT 和「单任务 token 成本」量不了（`usage_records` 里的 token 数是 worker **硬编码**的 800/120） |
| R5.9 | 按意图的延迟可观测 | 🟡 | `agent_runs` 有 `intent` + `created_at`/`updated_at`，端到端 P50/P95/P99 **算得出来**；但没有查询/落库，且分不出排队时间与执行时间 |

---

## 4 · 缺口归谁（★ 空格子当场开条目，不许留白）

| 缺口 | 性质 | 谁在打 |
|---|---|---|
| **F1** C 档不走审批 | 🔴 真 bug，链路三段断在中间一段 | **未定** —— 修法机械（`claim_run` 多 RETURNING 一列 + worker 传进 ctx），但会改变行为，**要 Chaoyu 点头** |
| **F2** 审批通过后 run 停住 | 🔴 闭环缺失 + **恢复语义是设计决定** | **未定** —— 从头重跑 vs 从断点续，取决于要不要先给 `checkpoints` 补写入路径（R5.3） |
| **F3** 并发同键返回空 | 🔴 真 bug + **§38 没定义"执行中命中"** | **未定** —— 等/409/"处理中" 三选一，要先补设计 |
| **F4** 五个降级信号没有生产者 | 🟡 编排最小可用的必然结果 | 归 **M9.4 下半场**（真 Agent Loop）。⚠️ 但 `data_maturity` 那条**不该等** —— 归因延迟是第一性约束，它现在是硬编码的 `"mature"` |
| **F5** 沙盒写工具 runtime 只认 4/8 | 🟡 纪律缺物理载体（已补） | 载体已就位；**那 4 个的登记**归 M9.4 下半场 |
| §11-① 与沙盒/runtime 三方不一致 | ⬜ 设计问题不是工程问题 | **要业务侧定**：改 §11-① 还是给 5 个写工具补键 |
| R3.3 behavior 五分类 / R3.8 意图剪枝 | ⬜ 刻意的最小可用 | **M9.4 下半场**（09 已自述"编排是最小可用"，本表把它拆成了逐条） |
| R4.7 D 档 / 权限闸从不拒绝 | 🔴 安全相关 | **未定** —— 至少 `ToolRuntime` 的默认权限不该是"全给" |
| R4.8 观测无读/写分桶 | 🟡 尺子缺一把 | **未定**（eval 侧已有 `_report_read_write` 可参照） |
| R5.4 串行 + 全局 FIFO | 🟡 | **压测前必须定**：场景④⑤直接考它 |
| R5.6 Outbox / R5.7 RAG / R5.8 模型服务 | ⬜ 组件未建 | **M9.4 下半场 / M10** |
| R5.5 回路 3 的 outcome_* | ⬜ 预留 | **M12**，设计里本就如此，不算欠债 |

---

## 5 · 压测（M9.7）之前必须先做的两件

**① §19 的【待定】门槛要填。** 现在挂着：按意图的 P50/P95/P99、模型 TTFT/TPOT、
并发 run 数、SSE 断线补发成功率、单任务 token 成本。
⚠️ **门槛空着就跑，跑完还是"无法判定"** —— M7 那六条毕业条件就是这么废掉的。
（填法照 05 §2.9：能实测反填的反填，填不出来的**从毕业条件里显式划掉**，不许挂着。）

**② 五个压测场景里有三个现在没有承载物**：

```
① 突发 10× 流量    🟡 能跑，但 R5.4（串行 + 全局 FIFO）决定了结果几乎注定难看
② 模型服务挂掉      ⛔ 没有模型服务（R5.8）—— 这个场景现在没有被测对象
③ 工具超时         ✅ 有（FaultPlan 能注入，且两种超时都建模了）
④ RAG 不可用       ⛔ 没有 RAG（R5.7）—— 同②
⑤ 单 org 刷爆预算   🟡 成本闸在，但 R5.2 只在跑前查一次、R5.4 无 org 隔离
```

⇒ **场景②④ 不是"没测"，是"没有被测对象"。** 压测之前要么把组件建起来，
要么把这两个场景从 M9.7 的毕业条件里**显式划掉并写明理由** ——
按 05 §2.9 的口径，**留着不做等于又造一条空门槛**。

---

## 6 · 已就位的判据

| 判据 | 位置 | 守什么 |
|---|---|---|
| 设计符合性 🆕 | `tests/runtime/test_design_conformance.py` | F1–F5，全部 `xfail(strict=True)`，**修好会变成 XPASS 报错** |
| 三层幂等 | `tests/runtime/test_idempotency.py` | 10 条，全是真的投两次（顺序） |
| API + 越权 | `tests/runtime/test_api.py` | 12 条 |
| Worker 编排 | `tests/runtime/test_worker.py` | 14 条 |
| SSE 断线补发 | `tests/runtime/test_sse.py` | 7 条 |

```
全量：365 passed, 5 xfailed, 0 skipped
```

⚠️ **0 skipped 是验收口径**（要 PG 起着）。⚠️ **5 xfailed 不是 0 缺口，是 5 个缺口正在被盯着。**

---

## 7 · 别再重新讨论的

- 数据模型这一层是对的（R1.1–R1.7 全过），别再翻。
- `open_approval_case` 的同事务翻转是对的（审计怀疑过一次，错的是审计）。
- 幂等的**顺序**路径是对的、超时两形态的建模是对的。缺口只在**并发**（F3）。
- 假平台不接真 Meta（2026-08-14 已定），真接入留到 M10。
