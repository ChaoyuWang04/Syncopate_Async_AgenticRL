# Syncopate · 27 · Serving Harness 生产化施工计划书

> 来源：`docs/reference/Serving-Harness-*.md`（老师课件 takeaway，10 章 + 附录 A，共 18472 行）。
> 目标：把我们的 agent runtime/serving 从 toy 级抬到生产级——课件里每个关键点都要变成
> 我们系统的一部分。**本文是施工计划书，不是开工令**；当前主线队首仍是 `26`（v15 维修）。
>
> **按章分轮撰写**：本轮 = 总纲 + K0 盘点 + K1（Run API·课件 CH1）+ K2（数据库·CH2）
> + K3（队列与 worker·CH3）。K4–K11 只留占位（§6），下轮读完对应章再补，补写时
> 沿用本文结构：步骤 → 课件已知缺口必修表 → 量化门槛表。
> 纪律与 26 同款：**每步门槛跑前注册，数字不达标不进下一步。**

---

## 0 · 核心准则（写在一切步骤之前）

**准则〇（Chaoyu 09-01 立，本计划总纲）：任何一个阶段开工之前，先检查该阶段涉及的
现有实现，⛔ 禁止纯粹的增量更新。**

我们的 runtime 已经实现了课件的一部分（Run API / 队列 / worker / SSE / 审批 / 会话
都有雏形，`01 §B/§F` 有登记）。所以每个阶段的开工动作永远是三步：

```
① 先盘点：打出「已有 / 缺失 / 不同形」三列对照清单，每格带 file:line 证据
   ⚠️ 登记≠实现（守则）：01-TASKS 写着"已落地"的也要验证，不许默认成立
② 再整合：已有实现与计划冲突 ⇒ 就地改写整合进一份实现
   ⛔ 不许在旁边再搭一份——同一件事两份实现，最后两份都不准（21 号登记过的老病）
③ 清单过了该阶段的「盘点门槛」才许动第一行代码
```

**准则一：** 每步的量化门槛跑前注册；注册前过三查（可测/可达/阶段归属，26 §W0 学费）。

**准则二：** 判据优先写成「两个东西应当相同」的断言（守则①），且每条新判据要做
**负向认证**——人为制造该防的事故，判据必须红；不红的判据不算数（空门槛不等于通过）。

**准则三：** 课件是教学配置，不是生产配置（单实例、`run_123` 可枚举 id、无心跳续租），
且课件自带坑表 **H01–H108**，其中一部分课件自己都没修（如 lease 心跳跨六章未给）。
⇒ **照抄课件代码 = 把课件的已知坑一起抄进来。** 每阶段附「课件已知缺口必修表」，
修完才算该阶段过。

**准则四（课件总纲，整个计划的尺子）：进程可以死，表不能丢。**
每个机制都要能回答"哪个进程死掉之后，怎么靠表活过来"。每阶段验收里都有一条
kill 测试，就是在量这句话。

**准则五（Chaoyu 09-01 立）：不照单全收，按项目背景取舍。**
课件面向的是"多租户对客、真金退款"的通用客服后端；我们是单机四卡、自有平台、
dev mode 的业务 agent。每条课件机制在计划里标三种处置之一并写明理由：
**采纳**（照做）· **改造**（保留意图、换成贴合我们现状的实现——尤其已有同构件时
收编而不重写）· **裁剪**（当前规模不做，登记复活条件）。⛔ 无标注的照抄不算过计划评审。

---

## 1 · 全景：阶段表 K0–K11

| 阶段 | 对应课件 | 一句话 | 本文状态 |
|---|---|---|---|
| **K0** | — | 现状盘点：49 条能力清单 × 现有实现对照 | ✅ 本轮已写（§2） |
| **K1** | CH1 | Run API：六入口 · 幂等 · 协作式取消 · 双层错误码 | ✅ 本轮已写（§3） |
| **K2** | CH2 | 数据库落库：两本账 + 8 表 + 补课件自认的缺口 | ✅ 本轮已写（§4） |
| **K3** | CH3 | 队列与 worker：Outbox · lease · 五层幂等防线 | ✅ 本轮已写（§5） |
| **K4** | CH4 | 状态机：白名单 · 唯一迁移入口 · 一事务写全 | ✅ 本轮已写（§6） |
| **K5** | CH5 | 执行 loop：对照收编 · 意图日志 · 错误分层 | ✅ 本轮已写（§7） |
| **K6** | CH6 | Tool Runtime：四道闸收编 · 五态 tool_calls · 注册断言 | ✅ 本轮已写（§8） |
| **K7** | CH7 | SSE 事件流：对照收编 · 事件分层 · 通知唤醒 | ✅ 本轮已写（§9） |
| **K8** | CH8 | 恢复与重放：sweeper · 对账闭环 · 四动词纪律 | ✅ 本轮已写（§10） |
| **K9** | CH9 | 生产硬化：SLO · 预算 · 观测三件套 · 变更可逆 | ✅ 本轮已写（§11） |
| **K10** | CH10 | 回流飞轮：结果信号回流 · 与训练线的接口 | ✅ 已写（§12） |
| **K11** | 附录 A | 上线总清单 · Runbook · 复盘模板 | ✅ 已写（§13） |

**依赖关系**：K0 先于一切；K1–K3 按序（K2 的表是 K1 的地基，K3 的 outbox 改写 K1 的
创建事务）；K4 起的顺序在补写时定。**每阶段末尾必须全量测试零失败才进下一阶段。**

---

## 2 · K0 · 现状盘点（0 GPU，~1 天；一切阶段的前置）

> ✅ **K0 已完成（2026-09-02）**：产物 = `29-serving-k0-inventory.md`（77 条对照 + 门槛②重验表
> + 底座核实 + 交给 K1 的三句话）；测试基线 **247 passed**；§16 四件已裁（本文 §16）。

### 步骤

```
K0-1 抽出课件的「能力清单 49 条」（CH1 §0.3 提及，分 5 组；逐条列成表——
     若某条在已读章节里没有展开，标注"定义在 CH<n>，占位"）
K0-2 对我们的 runtime 做全面对照：每条能力标「已有 / 缺失 / 不同形」+ file:line 证据。
     重点核对（01-TASKS 记载"已落地"、按准则〇必须重验的）：
       · run 的资源化程度：现有 runs 表 / conversations 表的字段 vs 课件 agent_runs
       · 六入口对应物：创建/查询/取消/恢复/事件流/trace 各自有没有、形状差多少
       · 幂等：F-1 记载"幂等测试"——是哪一层的幂等？有没有 input_hash 第二把锁？
       · 队列：现有 claim_run（org_id 限定，C-1）是什么机制？有没有 outbox？
       · worker：崩溃后 run 会怎样？有没有 lease/attempts 字段？
       · SSE：Last-Event-ID 重连（F-2 记载）对应课件 after=N 补发到什么程度？
       · 取消：cancelled 终态事件并入 finish_run 事务（B-6a）——协作式取消有没有？
       · 事件表：有没有单调 sequence？谁分配？
       · 多租户：org_id 在哪几张表上？哪些查询带了？
       · 成本记账：model/tool 调用有没有落库？usage 有没有？
K0-3 存储底座盘点：现在用的什么数据库、什么驱动、事务能力如何；
     现有 run id 的形态（可枚举吗）；表的行数规模
K0-4 产出《选型呈报》交 Chaoyu 裁定（见 §8）：队列底座 · 数据库引擎 · id 形态。
     ⚠️ 这三件不盘点完不许拍——盘点前问就是盲拍
```

### 门槛（不过不进 K1）

| # | 门槛 | 数字 |
|---|---|---|
| ① | 对照表零空格 | 49 条能力 × 3 列（状态/证据/差异说明）无空格；「已有」的每条有 file:line |
| ② | 重验完成 | 01-TASKS 记载"已落地"的 runtime 条目（B-0~B-7 · F-1~F-6）逐条重验，验法=跑其现有测试 + 读实现，不是读登记表 |
| ③ | 选型呈报 | §16 四件的呈报材料齐（现状 + 迁移代价估算），已提交 Chaoyu |
| ④ | 差异定性 | 每条「不同形」标注去向：K几阶段改写 / 显式保留（保留要写理由，= 登记欠账） |

---

## 3 · K1 · Run API（课件 CH1；~2 天）

> ✅ **K1 已落地（2026-09-02）**：六入口齐（cancel/resume/trace 新增；events 的 `after` 查询路归 K7-1）·
> 幂等三态 201/200/409（input_hash 第二把锁）· 协作式取消（API 写意图，ActionGate 入口 = 第一个
> 安全点兑现，其余安全点 K5-5）· resume 四道检查（一次性 resume_token，回 queued 不回 running）·
> 错误信封 `{error:{code,message,request_id}}` + 13 个注册码 · run_type 分发校验（当前只有 chat）·
> /trace 独立角色位（`dev-token-acme-trace`）。门槛①–⑥ = `tests/runtime/test_run_api_k1.py` 22 条
> （并发双击、6×2 终态矩阵、五入口 404、SQL org_id 扫描、P95 实测）。K1-2 的投递仍是"worker 轮询"，
> K3-2 换 Outbox（欠账已在 K3）。⚠️ 前端 `api.ts` 改读信封（本机无 node，未构建验证）。

### 目标形状（课件的骨架，一次列全）

```
六入口     POST /runs · GET /runs/{id} · GET /runs/{id}/events · GET /runs/{id}/trace
           · POST /runs/{id}/cancel · POST /runs/{id}/resume（没有 DELETE、没有 PUT——
           run 是已发生的历史，只能推进不能改删）
六状态     queued · running · waiting_for_user（活）/ succeeded · failed · cancelled（终）
           对终态 cancel/resume ⇒ 409（不是 400，不是静默成功）
八硬要求   快速返回（=解耦本身）· 可查询/可观察/可取消/可恢复（四抓手）·
           幂等/多租户隔离/错误稳定（三承诺）
```

### 步骤

```
K1-1 数据契约层：RunStatus 枚举 + CreateRunRequest（run_type / idempotency_key
     min_length=8 / input）+ RunResponse（出口白名单，response_model 是白名单不是
     格式化）+ CancelRunRequest（reason 落库）+ ResumeRunRequest（resume_token +
     input——resume 是"带着新信息继续"，不只是继续）
K1-2 create_run 五步（从便宜到贵）：算 input_hash（json.dumps sort_keys=True
     ensure_ascii=False 规范化）→ 查幂等键 → 建 run → 记 run.created → 投递
     （K3 前允许裸投递，但要在代码注释登记"K3 换 Outbox"——欠账显式化）
K1-3 幂等三态：新 key→201 · 同 key 同 hash→200（不是错误！是分布式正常现象）·
     同 key 不同 hash→409 IDEMPOTENCY_CONFLICT。
     ★ 必须 catch unique 约束的 IntegrityError → 重查 → 按 200 返回（课件坑 H01：
     "先查再插"有竞态，救你的是数据库约束，不是查询）
K1-4 取消（协作式）：queued/waiting_for_user → API 直接迁 cancelled；
     running → API 只写 cancel_requested_at 字段 + run.cancel_requested 事件，
     worker 在安全点自己迁。⛔ cancel_requested 永不进状态枚举（意图≠状态）
K1-5 resume 四道检查：404 租户 → 409 状态（仅 waiting_for_user 可 resume）→
     校验 resume_token → 回到 queued（⛔ 不是 running——必须重新走队列被领取，
     保住"同一时刻只有一个 worker 持有"）
K1-6 多租户隔离进 SQL：一切详情查询 WHERE id=:run_id AND org_id=:org_id，
     org_id 只来自鉴权注入、永不来自请求体；跨租户一律 404（不是 403，防枚举）；
     403 只用于同组织内角色不足
K1-7 双层错误码：统一信封 {error:{code,message,request_id}}；
     code 给前端代码判断（稳定），message 给人看（可改）；409 至少三个业务码
     （IDEMPOTENCY_CONFLICT / RUN_ALREADY_TERMINAL / RUN_NOT_WAITING_FOR_USER）
K1-8 /trace 单独权限：trace 含完整 prompt/参数/推理，org_id 校验挡不住
     "组织内普通成员看别人全文"（课件坑 H08）⇒ 独立角色位 + 默认关
```

### 课件已知缺口必修表（K1 部分）

| 课件坑 | 必修动作 |
|---|---|
| H01 先查再插竞态 | K1-3 的 IntegrityError 处理 + 并发双击测试 |
| H04 同 key 同 input 返 400 | K1-3 三态里明确 200 |
| H05 幂等 key 粒度 | key 形态定为「实体:动作:v版本」；key 由**客户端**在点击那一刻生成（服务端生成毫无意义） |
| H06 input 不做二次校验 | input 按 run_type 分发到子 schema 校验，不许"格式对但字段缺"漏到 worker |
| H08 /trace 同权限 | K1-8 |
| H09 resume 直接置 running | K1-5 |

### 门槛（不过不进 K2）

| # | 门槛 | 数字 |
|---|---|---|
| ① 幂等三态 | 三个契约测试各绿：201/200/409；**并发测试**：同 key 两请求同时发，恰好一个 201 一个 200、库里恰好一行（负向认证：注释掉 IntegrityError 处理必红） |
| ② 终态守卫 | 对六状态 × cancel/resume 全组合表驱动测试：终态一律 409 + 正确业务码 |
| ③ 隔离 | 跨租户访问测试：GET/cancel/resume/events/trace 五入口全部 404；grep 判据：runtime 详情查询 SQL 无一条缺 org_id（或 K2 的 RLS 顶上） |
| ④ resume 不变量 | resume 后断言 status=queued 且 lease 为空；随后被 worker 正常领取跑完 |
| ⑤ 错误信封 | 全部非 2xx 响应过 schema 校验（code∈注册表 · request_id 非空）；前端代码 grep 不到任何对 message 文本的判断 |
| ⑥ 响应时延 | POST /runs 本机 P95 < 300ms（课件 SLO；100 次采样） |

---

## 4 · K2 · 数据库落库（课件 CH2 + 课件自认缺口的修补；~2 天）

> ✅ **K2 已落地（2026-09-02，与 K1 交错先做）**：迁移链 `syncopate/runtime/migrations/`
> （0001 baseline 内嵌旧 schema.sql，0002 = K2-1/K2-2/K2-4/K2-5 一次建齐），`schema.sql` 退役为
> 生成快照 `schema.snapshot.txt`；领号器 `db.next_seq/append_event` 收编全部四处事件写入；
> 创建事务写 `run.created`（K2-6）。门槛 ①②③④⑤⑦ = `tests/runtime/test_schema_migrations.py`
> 7 条（含两条负向认证）；⑥ org_id 全表冗余已核（29 B4），trace 子表查询随 K1-8/K8-4；
> K2-7 归 K8-4。`usage_records` 粒度暂定"每次执行一行"（call_index=attempts，28 P-12）。

### 目标形状

```
两本账     agent_runs（余额：覆盖写，一行）+ run_events（流水：只追加，单调 sequence）
           ——流水能算出余额，余额推不出流水
六个展开   agent_steps（过程）· model_calls/tool_calls（调用与代价）·
           checkpoints（恢复）· usage_records（成本）· audit_logs（审计，run_id 可空）
命名口径   成功事件叫 run.completed（状态叫 succeeded）——两边都不改
```

### 步骤

⚠️ **K2 不是照抄课件 DDL**——课件 CH2 的 schema 与 CH1 的承诺对不上（课件自己承认
"地基缺三根钢筋"），下表把课件版和必修版并排列出，我们直接建必修版：

```
K2-1 建 8 张表，其中在课件 DDL 基础上必改的：
     agent_runs   + input_hash（幂等第二把锁，H11）+ cancel_requested_at（H12）
                  + resume_token（H12）+ lease_owner/lease_expires_at/attempts（K3 用，
                  一次建齐免得三次 migration）+ last_seq（序号分配器，见 K2-2）
                  + version（CAS 守卫，K4 用）
     run_events / agent_steps / model_calls / tool_calls
                  全部冗余 org_id（课件坑 H17：图例的锁是假的；冗余是将来能开
                  Row Level Security 的前提，生产答案=全部冗余）
     tool_calls   UNIQUE(org_id, tool_name, idempotency_key)（课件版少 org_id ⇒
                  跨租户副作用丢失，H19）+ CHECK（写工具 idempotency_key 非空）
                  + created_at/ended_at 两阶段写入（intent log：先记"我要做"再做）
     usage_records 加 UNIQUE（课件版零约束 ⇒ worker 重跑账单翻倍，H15）；
                  粒度定为「每 run 每模型调用一行」，聚合查询按 org+日出账
     checkpoints  UNIQUE(run_id, checkpoint_index)
K2-2 序号分配器（课件坑 H13，三处同病）：run_events.sequence /
     agent_steps.step_index / checkpoints.checkpoint_index 一律走
     UPDATE agent_runs SET last_seq=last_seq+1 WHERE id=:run_id RETURNING last_seq
     （同事务领号，行锁到 COMMIT ⇒ 分配顺序=可见顺序，SSE 补发无空洞）
     ⛔ 全库禁止 SELECT MAX(sequence)+1（两个写者必撞）
K2-3 id 形态：一律不可预测 id（ULID/UUID）——run_123 可枚举会抵消"404 防枚举"
     整套防护（H18）。存量 id 迁移方案按 K0 呈报的裁定执行
K2-4 updated_at 触发器（H14：Postgres 的 DEFAULT now() 只在 INSERT 生效一次，
     不加触发器它永远停在创建时间）；若底座非 PG，用应用层统一写入点
K2-5 索引：建 (org_id, status, created_at)（列表页真正需要的那条）；
     ⛔ 不建与 UNIQUE 重复的索引（UNIQUE 本身就是 B-tree，重复=白维护一棵树）
K2-6 创建事务：INSERT agent_runs + INSERT run_events(run.created) 同一事务
     （只写这两样；outbox 行 K3 加进来）——"让系统失败得干净"
K2-7 trace 聚合查询：先验 org 再查子表的写法废除，子表查询直接带 org_id
     （子表已冗余）；model_calls.request_json 的归档/脱敏标记为 K9 欠账，先登记
```

### 门槛（不过不进 K3）

| # | 门槛 | 数字 |
|---|---|---|
| ① 约束就位 | DDL 断言测试：全部 UNIQUE/CHECK/触发器存在且生效（逐条 INSERT 违例数据必被拒） |
| ② 序号无洞无撞 | 并发测试：两个写者（模拟 API 写 cancel_requested 事件 + worker 写 tool 事件）各写 100 条，断言 sequence 连续无重复无空洞；负向认证：换回 MAX+1 实现必现撞号 |
| ③ 事务原子性 | 创建事务中途 kill：重启后 run 与 run.created **同生共死**（要么都在要么都不在，不许"状态在、事件缺"） |
| ④ 账单防重 | 同一 run 重复执行注入：usage_records 唯一约束拒绝第二条；对账查询（usage 汇总 vs model_calls 明细求和）差值 = 0 |
| ⑤ updated_at | UPDATE 后断言 updated_at 变化（负向：去掉触发器必红） |
| ⑥ 隔离完备 | 每张含业务数据的表都有 org_id 列；trace 全部子表查询带 org_id（grep 判据零漏网） |
| ⑦ id 不可枚举 | 新建 run 的 id 通过随机性检查（连续创建 100 个，任意两 id 无可预测递增关系） |

---

## 5 · K3 · 队列与 worker（课件 CH3；~3 天）

### 目标形状（三道防线）

```
队列唯一的承诺 = at-least-once（至少一次 = 可能多次），三道防线接住三个后果：
  不丢   Outbox：写库和"要投递"同一事务；dispatcher 扫表补投（消息丢失变成不可能）
  不争   lease：队列只做投递不做授权；执行权靠一条条件 UPDATE 在数据库里抢
  不重   幂等五层：终态检查 → claim → 工具幂等键(K6) → checkpoint(K5) → 唯一约束(K2)
心法：从「消灭重复」（不可能）转向「让重复无害」（可行）；宁可多做，不可少做
```

### 步骤

```
K3-1 outbox_jobs 表（课件 DDL 可用）+ 索引 (status, next_attempt_at)（这条不是可选
     优化——表只增不减且绝大多数行是 dispatched，没索引=每几秒全表扫）
     + 归档策略（课件坑 H108 全书未给）：dispatched 超过 N 天定期清理，N 首标 30
K3-2 创建事务升级：agent_runs + run_events(run.created) + outbox_jobs 三 INSERT
     同事务；⛔ 事务里没有任何 publish。K1-2 登记的欠账在此销账
K3-3 dispatcher 循环：扫 pending 且 next_attempt_at 到期，LIMIT 100；
     publish 成功后**同一事务**做两件事：标 dispatched + 写 run.enqueued 事件
     （只成一半=事件流悬空缺口，H107）；失败退避 next_attempt_at=now+min(cap,
     2^(attempts-1))，cap 首标 300s（不设 cap 重试 20 次=12 天后，H103）
     顺序铁律：先 publish 后标记（"投了没记"下轮重投无害；"记了没投"任务永久消失）
     ——把不可恢复的一步放前面
K3-4 队列本体：**Celery + Redis（§16-1 已裁，Chaoyu 09-02）**。Celery 只承担
     "投递/消费/ack/重试/分队列"，⛔ 不承担 run 状态、事务、幂等、lease（课件 §4
     "装个 Celery ≠ 有了 Agent 后端"）。enqueue_job 仍抽象成单一函数（内部=
     `execute_run.apply_async`），task 体只放 run_id（H23）。Celery/Redis 的配置项
     必修（acks_late · prefetch=1 · visibility_timeout 与 lease 对齐 · reject_on_worker_lost
     · JSON 序列化 · ignore_result · Redis AOF+noeviction+requirepass）全部登记在 **28**，
     每条带负向认证，K3 门槛⑩=28 §1/§2 的 K3 归属项全部关闭
K3-5 worker 分层（课件 §14.3，全章最值钱）：
     worker 层只碰基础设施（claim job / ack / requeue），⛔ 不改 run 状态；
     业务错误在 execute_run 内部被消化并走状态迁移。
     一个可变状态只有一个写入者（H102：两个写入者的协调在跨进程+崩溃环境下无解）
K3-6 claim run（整章技术核心）：
     UPDATE agent_runs SET status='running', lease_owner=:worker_id,
       lease_expires_at=:expires, attempts=attempts+1,
       started_at=COALESCE(started_at, now())
     WHERE id=:run_id AND status='queued'
     返回 1 行=抢到；0 行=退出并 ack。⛔ 禁止先 SELECT 判断再 UPDATE（H27）。
     started_at 用 COALESCE：恢复重跑不覆盖第一次开始时间（事实不可修改）。
     ★ 课件坑 H28 在此定死：run.attempts 在 claim 时 +1 = 「被执行次数」；
     sweeper 重投不加这个数（它写 run.requeued_by_sweeper 事件，从事件流数）
K3-7 lease 心跳（课件坑 H30，跨六章未解，我们必须自己定）：
     worker 每 20s 续租（UPDATE lease_expires_at=now()+60s WHERE id=:run_id AND
     lease_owner=:worker_id）；TTL=60s=3×心跳间隔。
     判据：TTL 必须大于「两次续租的最大间隔」，而不是大于「run 总时长」——
     有心跳 TTL 才能短，故障发现才快（等人工 10 分钟的 run 照样只要 60s TTL）。
     续租失败（0 行）=lease 已被收走 ⇒ worker 立即停止执行不再写库。
     回收进程（sweeper）属 K8；本阶段先让「续租在跑」有判据行可见
K3-8 成功收尾顺序：写 status=succeeded + output + run.completed → 最后才 ack
     （反了=队列认为完成而库里没完成，无人再救，H29）
K3-9 错误分类（急诊分诊表）：transient（503/429/超时/断连）→ 退避重试；
     permanent（不存在/无权限/schema 非法）→ 立即 failed；取消 → cancelled；
     超 max_attempts → dead_letter_jobs（病历不是垃圾桶：original_job_id/payload/
     attempts/error 全留，可人工 reprocess）
     ⚠️ 超时后能否安全重试还要看有没有做过副作用（查 tool_calls 意图日志），
     不能光看错误类型——此判定在 K5/K6 完善，本阶段先保守（有未闭合的写工具
     意图记录 ⇒ 不自动重试，进 DLQ）
K3-10 协作式取消消费端：worker 在四个安全点（模型调用前/工具调用前/step 完成后/
     下轮 loop 前）检查 cancel_requested_at，发现即由 worker 自己迁 cancelled+ack
K3-11 积压监控：八指标落地，告警挂 oldest_job_age_seconds（用户视角）而不是
     queue_length（系统视角）；三队列（interactive 高优 / batch 低 / maintenance 低）
     ——分队列的价值还有故障隔离：写类工具可以单独暂停
K3-12 三个 attempts 分账（H22/H25）：outbox.attempts=投递次数（max 10）·
     job_queue.attempts=消费次数（max 3）· agent_runs.attempts=执行次数（max 3）。
     监控指标分别出，⛔ 永不混算（混算的失败率会导出错误的扩容决策）
```

### 门槛（不过不进 K4）

| # | 门槛 | 数字 |
|---|---|---|
| ① 不丢 | 故障注入：publish 前 kill dispatcher → 重启后 run 最终 enqueued（outbox pending 还在）；负向认证：绕过 outbox 直接裸 publish 并注入失败 ⇒ 丢 run 判据必红 |
| ② 不争 | 同一 job 投递两次给两个 worker：恰一个 claim 成功（1 行/0 行各一）；副作用计数=1；`attempts` 只 +1 |
| ③ 不重 | ack 前 kill worker → 重投后新 worker 读到终态直接跳过并 ack，run 不重跑不卡死 |
| ④ 顺序铁律 | 两条负向认证：（a）dispatcher 改成先标记后 publish ⇒ 「任务消失」测试必红（b）worker 改成先 ack 后写库 ⇒ 「run 永久卡死」测试必红 |
| ⑤ 错误分类 | 表驱动测试：transient 注入 → 按 1/5/15min 档退避重投最终成功；permanent → 立即 failed 零重试；连败超限 → DLQ 且 payload/error 完整可 reprocess |
| ⑥ 心跳 | 正常慢 run（单步 > TTL 但持续续租）不被误收；停掉续租线程 ⇒ lease 过期可被接管（K8 前先断言 lease_expires_at 停止推进）；判据行 `[lease-heartbeat]` 每次续租必打 |
| ⑦ 取消 | running 中发 cancel：worker 在下一个安全点内停下、状态=cancelled、终态事件在库；强杀路径不存在（grep 无 kill 逻辑） |
| ⑧ 积压 | oldest_job_age 指标可读且有告警阈值；构造 100 条积压，指标读数与实际等待吻合（±10%） |
| ⑨ 全量回归 | 既有 runtime 测试套件零失败（对照 K0 盘点时的基线数） |

---

## 6 · K4 · 状态机（课件 CH4；~1.5 天）

### 取舍总表（准则五）

| 课件机制 | 处置 | 理由 |
|---|---|---|
| 六状态 + 九条边 + 三个终态空集白名单 | **采纳** | 核心中的核心；空集是"不需要额外代码的禁止"，防的正是我们最贵的失效形状（旧 job 复活终态） |
| `transition_run` 唯一入口 + 一事务写全 + reason/actor 必填 | **采纳** | "签名即制度"；我们 B-6a 已把终态事件并进 finish_run 同一事务——**收编进 transition_run，不另起炉灶**（准则〇） |
| 事件名映射函数（吃 (from,to) 二元组） | **采纳+改造** | 课件自己有两处口径冲突（sweeper 事件名 / run.restarted，CH4 待查#1#2）——我们直接定死：映射函数**加 actor_type 参数**，sweeper 路径产 `run.requeued_by_sweeper`；`attempts>1` 的 `queued→running` 产 `run.restarted` |
| `waiting_for_user` 状态 | **改造** | 我们已有审批中断/恢复（F-2）——把"等审批/等用户补充"统一映射成 `running→waiting_for_user`、批准/补充 = resume→queued；不发明第二套暂停语义 |
| 触发者矩阵（403 层完整 RBAC） | **裁剪→轻量断言** | 课件自己没给代码（CH4 待查#3）；我们单租户 dev mode，API 调用方可信。留其意图：transition_run 内一张 `(边, actor_type)` 允许表，违例即 raise——防的是**我们自己的代码走错路**，不是恶意用户。复活条件：对外多租户时升级 RBAC |
| retry vs rerun 分界（终态后只能新建 run） | **采纳** | 前端"再来一条"本来就新建 run；补 `parent_run_id` + `rerun_reason` 两列把两次执行串起来 |
| `cancelling` 不进状态枚举（意图≠状态） | **采纳** | `cancel_requested_at` 只是字段；其生命周期定死（课件待查#6）：**终态后保留不清**（它是历史事实），判定只在活态读 |
| version 列 CAS 守卫 | **采纳（K2 已建列）** | 行锁 + 条件更新为主，version 防 sweeper 与活 worker 互相盖写（K8 消费） |

### 步骤

```
K4-1 白名单 + 唯一入口：ALLOWED_RUN_TRANSITIONS（9 边 3 空集）· transition_run 五节
     流水线（行锁→查白名单→apply_status_fields→写事件→写审计，③④⑤同一事务）·
     InvalidRunTransition 携带 from/to → API 层 409 INVALID_RUN_TRANSITION
     ⚠️ 非法迁移不写事件不写审计（它没发生；写了会污染 K8 重放）——撞 409 的计数
     归 metrics，另一本账
K4-2 事件名映射 event_type_for_transition(from, to, actor_type)：
     succeeded→run.completed（⛔ 禁 run.succeeded）· running→queued 按 actor 分
     run.retry_scheduled / run.requeued_by_sweeper · resume 产 run.resumed
K4-3 派生字段 apply_status_fields：started_at=COALESCE 首次不覆盖 · 终态设 ended_at
     +清 lease · waiting 清 lease 不设 ended_at；attempts+1 定在 claim（承 K3-6）；
     MAX_ATTEMPTS 判断定死在调用方 transient 分支（课件待查#5）：超限改请求 failed
K4-4 协作式取消消费端收编（K3-10 的状态机侧）：running 的取消由 worker 在安全点
     transition；钱已动才收到取消 ⇒ cancelled + audit 两条事实并存，不删真相
K4-5 rerun 通道：parent_run_id / rerun_reason 落列；终态永不回队
K4-6 现有实现收编（准则〇）：finish_run 的终态事务、cancelled 终态事件（B-6a）、
     claim_run 全部改走/对齐 transition_run；改完 grep 全仓裸改点
K4-7 grep 判据进 CI/pre-commit：`\.status\s*=` 命中白名单只留 transition_run 内部
     与创建事务两处
```

### 门槛（不过不进 K5）

| # | 门槛 | 数字 |
|---|---|---|
| ① 白名单全枚举 | 表驱动测试 6×6=36 组合：9 条合法通过、27 条非法全部 409（含五条明确禁止）；三个终态出边断言为空集 |
| ② 无第二条路 | CI grep：全仓 `.status` 赋值命中数 = 白名单登记数（transition_run + 创建事务），新增命中即红 |
| ③ 事务原子 | 迁移中途注入失败：状态/事件/审计三者同生共死（部分提交 = 必红） |
| ④ 事件名 | 映射函数单测覆盖全部命名 + actor 分支（sweeper/retry 两名字各归各）；负向：手拼 `run.succeeded` 出现即红 |
| ⑤ 取消语义 | running 中 cancel → 状态不变仅写标记；worker 安全点后 → cancelled；"钱已动"场景断言 cancelled 与 audit 并存 |
| ⑥ 全量回归 | runtime 测试套件零失败 |

---

## 7 · K5 · 执行 loop（课件 CH5；~2 天）

> ⚠️ **本阶段的性质与其他阶段不同：我们已经有一个在生产跑的 agent_loop + ActionGate +
> transcript 恢复 + 假模型驱动测试（B-0/B-3a/B-3b，01 §1）**，且横切（权限/幂等/审批/
> 审计/事件/步数上限）已收口在 ActionGate——比课件"散点检查"更结构化。
> 所以 K5 是**对照收编**，不是按课件重写。红线：loop 的渲染/解析与训练侧共用一份
> （N5/守则⑮），任何改动必须过同形断言。

### 取舍总表（准则五）

| 课件机制 | 处置 | 理由 |
|---|---|---|
| 每 append 一条存一次 checkpoint（快照哲学） | **采纳（§16-4 已裁，Chaoyu 09-02）** | 课件 CH5 §4.4/CH8 §4.3：**存档密度决定恢复分辨率**——只有"模型已点名工具、结果未回"那一档存在，恢复才能走第二路（查 tool_calls 而不是重问模型/重跑工具），否则分支 C（写工具半成功）无解。我们现有 `save_transcript` 本质就是课件的 context 快照，收编而非新建：K0 核存档密度（工具调用前那一档有没有）+ 补 `last`/`completed_tool_calls` 两个恢复判断字段；`run_events` 只做回放展示，⛔ 不做恢复来源 |
| 意图日志（写工具"先记后做"两阶段写入） | **采纳（本阶段最重要的增量）** | "崩在副作用发生与记录之间"是重放和快照都救不了的唯一黑洞；执行前 INSERT tool_calls(幂等键, status=running) → 执行后 UPDATE 结果。恢复时读它判断"钱动没动" |
| 恢复两条路（last=无结果的写调用 ⇒ 不重试、查证对账） | **采纳** | response_lost 语义 + "结果未知禁止自动重试"铁律 |
| 错误分层（业务错误 Loop 内消化不外抛；worker 层薄） | **采纳** | "一个状态一个负责人"；与我们 K3-5 同一条。取消异常按课件待查#7 定死：ensure_not_cancelled 内部完成 transition 后直接 return，不设第三类异常 |
| 安全点检查（尤其副作用工具前） | **改造** | 不做散点检查——**挂在 ActionGate 出口**（一切副作用的唯一出口，天然的结构化安全点），取消/审批/超限在同一处判 |
| MAX_LOOP_STEPS=permanent 熔断 | **采纳（已有件对齐）** | 我们已有步数上限（ActionGate）；改造点＝超限语义定为 permanent（failed + 现场），不是 transient 重试 |
| 幂等键规范（动作:业务实体:版本，写工具禁带 run_id） | **采纳** | 下游平台也是我们的（B-1）⇒ 平台侧同步建去重账本（refund_records 同款：UNIQUE(org, key)），两侧一起认 key——课件里"下游可能不认"的最大不可控项在我们这不存在，必须用满这个优势 |
| FakeModel 剧本测试 + 只断言数据库状态 | **采纳（已有件对齐）** | B-3b 就是假模型驱动；对齐课件断言法：只断言表状态（行数/序号连续/终态），不断言内部调用 |
| prompt 逐轮变长的成本账 / provider 圈断器 | **裁剪** | 本地 vLLM 单一"provider"；成本记账归 K9 轻量做，圈断器复活条件=接入外部 API |
| 五层 timeout | **改造→三层** | 模型调用 / 工具调用 / 整 run 三层（我们已有工具 timeout 于 ActionGate）；等审批超时按业务定（归 K4 waiting 语义）；lease 超时归 K3/K8 |

### 步骤

```
K5-1 对照清单先行（准则〇）：课件"九件事"逐条 × 我们 agent_loop/ActionGate/worker 的
     现状，打「已有/缺/不同形」表（K0 已做粗表，此处细化到函数级）
K5-2 恢复哲学落地（§16-4 已裁=课件快照式）：`save_transcript` 收编为课件 checkpoint：
     每 append 一条消息存一次（一轮两档：模型输出后 / 工具结果回灌后），快照带
     `last` + `completed_tool_calls`；不变量 checkpoint #k 恰 k 条消息；恢复 = 读最新
     快照重入 loop，⛔ 快照里没有"下一步"字段（下一步永远由模型决定）；
     渲染仍与 decider 同一条代码路径（守则⑮）
K5-3 意图日志：写类工具经 ActionGate 执行前先落 tool_calls(幂等键,status=running)，
     执行后 UPDATE(status, result, ended_at)；恢复逻辑分两路——last 为无结果写调用
     ⇒ 查意图日志：succeeded 则补结果续跑 / running·response_lost 则停下转对账（K8）
K5-4 错误分层落地：TransientRunError/PermanentRunError 两类；transient=transition
     +enqueue 两步；permanent=failed+失败现场；execute_run 永远正常 return
K5-5 安全点收口：取消/审批/步数/预算判定统一挂 ActionGate 出口；副作用工具执行前
     必过（结构保证，不靠散点自觉）
K5-6 幂等键两层落库：API 层 UNIQUE(org,key)（K1）+ 工具层 UNIQUE(org,tool,key)
     （K2）+ 平台侧去重账本；写工具 key_fn 逐个登记（业务实体，禁 run_id），
     只读工具 key 可带 run_id（作用域=本次执行内）
K5-7 测试收编：剧本假模型跑全链，只断言库状态；恢复测试=杀进程后换 worker 续跑
```

### 门槛（不过不进 K6）

| # | 门槛 | 数字 |
|---|---|---|
| ① 对照清单 | 九件事 × 现状三列零空格（函数级 file:line） |
| ② 副作用黑洞 | kill 注入：写工具执行后、记录结果前杀进程 → 恢复后**平台侧副作用计数 = 1**（不重复执行）且 run 不卡死、转对账路径有事件可见；负向认证：去掉意图日志此测试必红 |
| ③ 只读恢复 | 中途杀进程恢复：只读工具调用计数不变（结果复用），模型调用 +1，终态正常 |
| ④ 错误分层 | 业务错误注入 → worker 层零状态写入；步数超限 → failed 且不再入队；transient → 退避后最终成功 |
| ⑤ 安全点 | cancel 后：写类工具执行数 = 0（ActionGate 出口拦截断言） |
| ⑥ 同形红线 | 渲染/解析共用断言全绿（tests/train 一致性测试 + 26-W2 同形家族）——loop 改动不许造成训练/线上分叉 |
| ⑦ 全量回归 | 测试套件零失败 |

---

## 8 · K6 · Tool Runtime（课件 CH6；~2 天）

> 我们已有 ActionGate（agent 碰外部世界的**唯一出口**，横切收口：权限/幂等/重试/成本闸/
> 审批/审计/事件/步数上限）+ tool_registry（spec 唯一真相来源）+ tool_parity（30/30 账本）。
> K6 = 拿课件的"四道闸 + 八步生命周期 + 十三条职责"当**对照清单**验收 ActionGate，
> 补真缺口，不另建一层。⚠️ 老病警示：WRITE_TOOLS 曾"登记 8 实现 2"（登记≠实现）——
> 本阶段的注册断言就是治它的。

### 取舍总表（准则五）

| 课件机制 | 处置 | 理由 |
|---|---|---|
| 四道闸（找定义→schema→权限→幂等）+ 被拦下也落库 | **改造=对照验收 ActionGate** | 结构已有；逐闸核对实现与判据，缺的补进 ActionGate 而不是旁边新建 |
| `tool_calls` 五态（running/succeeded/failed/skipped_duplicate/**response_lost**） | **采纳（本阶段最重要的增量）** | "系统必须能表达『我不知道』"；承接 K5-3 意图日志；response_lost **禁止自动重试**、只能对账回填（K8）。⚠️ 课件待查#4（skipped_duplicate 与 UNIQUE 冲突）我们定死：**UNIQUE 部分索引只覆盖非 skipped_duplicate 行**，重复行写 `duplicate_of` 引用列 |
| 注册断言（side_effect ⇒ idempotency_required + key_fn 存在） | **采纳** | 硬规则写进注册函数不是 checklist；直接治"登记≠实现" |
| `output_schema`（防脏数据反向灌回 context） | **采纳** | 我们的观测直接进训练同形渲染——脏观测既是线上幻觉燃料也是训练污染源，双重理由 |
| 失败分类 + error_json 三字段（code/message/retryable） | **采纳** | 现有 tool_crashed 兜底升级成结构化分诊；"防线生效不是故障"（权限拒绝记 failed 不告警） |
| "不给字段 > 加校验"（金额由后端算） | **改造** | 我们的业务本来就是"模型提案数值 + 审批/灰测闸把关"（tier_policy C 档拦停），等价保护已在；对确实不该由模型定的字段（如 org/账户标识）逐一核对删除 |
| 权限 = 工具×用户×**资源归属**三判 | **改造轻量** | 单租户 dev mode；资源归属校验以断言形式进 ActionGate（防自己代码走错），完整多租户判定登记复活条件 |
| MCP 接入治理 | **裁剪** | 无 MCP 工具；复活条件=接入任何外部工具时，一律过 Registry 四道闸不豁免 |
| 治理字段六项 | **改造** | risk_level≈已有 automation_tier（能升不能降）+ 审批策略 + release 灰测阶梯，映射而非新建；采纳 timeout_seconds/audit_required 进注册表；network_policy 裁剪（工具全是本地平台），复活条件=接外部 API |

### 步骤

```
K6-1 对照验收：课件十三条职责 × ActionGate 现状，逐条打「已有/缺/不同形」（函数级）
K6-2 tool_calls 表补全五态 + 状态机（谁在什么条件下改哪行——课件全书没定义，我们
     定死：Runtime 写 running→succeeded/failed/skipped_duplicate；worker 死后
     的 running 行由 K8 sweeper 按超龄判据标 response_lost；对账任务回填终值）
     超龄判据按工具分（注册表加 expected_max_ms，禁全局常量——课件自己给了反例：
     读工具 ~480ms vs 写工具 ~1260ms）
K6-3 注册断言进 tool_registry：side_effect⇒idempotency_required + key_fn 非空 +
     timeout_seconds 必填 + 写工具 output_schema 必填；断言失败=注册不进来
K6-4 幂等闸补全：同键已 succeeded → handler 零执行、新写 skipped_duplicate 行
     （duplicate_of 引用）、返回第一次结果；UNIQUE 部分索引兜底（K2 的约束落位）
K6-5 失败分诊器：异常 → error_json{code,message,retryable}，分类映射由注册表
     retryable_errors 驱动；写工具 retryable_errors=空集（显式决策入注册表）
K6-6 output_schema 逐工具登记并在回灌前校验；不符 → 不进 context、按上游错误分类
     （副作用已发生的场景标记待对账，不当普通失败）
```

### 门槛（不过不进 K7）

| # | 门槛 | 数字 |
|---|---|---|
| ① 对照清单 | 十三条职责 × 现状零空格（file:line） |
| ② 注册断言 | 负向认证：注册"有副作用但无 key_fn/无 output_schema/无 timeout"的工具必被拒；现有 30 工具全量重注册通过 |
| ③ 拦下也落库 | 四道闸各构造一次违例 → tool_calls(failed) 行 + error_json 三字段齐 + 事件在；无一例"只报错不留痕" |
| ④ 幂等闸 | 同键二次调用：handler 执行数=0、skipped_duplicate 新行、返回首次结果；负向：绕过代码判重直接 INSERT 第二条 succeeded 被 UNIQUE 拒 |
| ⑤ 反向污染 | 脏返回注入 → 不进 context（同形断言联动：观测渲染两侧一致仍成立） |
| ⑥ 分诊 | 表驱动：七类错误各归其类（含"权限拒绝记 failed 不触发告警"断言） |
| ⑦ 回归 | tool_parity 账本仍 30/30；全量测试零失败 |

---

## 9 · K7 · SSE 事件流（课件 CH7；~1.5 天）

> 我们已有：SSE + `Last-Event-ID` 重连（F-2，Node 冒烟过）· 多路 SSE 归约 · 终态事件
> 并入 finish_run 同一事务（B-6a）· emit seq 竞态修过（B-6）· `model.thinking` 独立
> 事件种类（"模型在想也是事实"课件要点已实现）· 前端历史回放+审批中断恢复。
> K7 = 对照收编 + 补三样真缺口：**事件分层过滤 · 通知唤醒 · 鉴权硬化**。

### 取舍总表（准则五）

| 课件机制 | 处置 | 理由 |
|---|---|---|
| 先落库再推送（事实层/唤醒层/传输层三分） | **采纳（对照验收）** | B-6a 已把终态事件并入事务，全链顺序逐处核对；负向认证补上 |
| sequence 游标 + after/Last-Event-ID 双路续传 | **采纳（已有件对齐）** | 取号收编进 K2-2 的 last_seq 分配器 + K4 行锁（transition_run 持锁写事件天然串行，课件方案 A）；冲突重试兜底保留 |
| 事件分层（public/internal/audit）+ payload 两视图 | **采纳（真缺口）** | "库里存全的，推出去只留人话"：prompt/token/purpose 等 internal 字段推前端前剥掉——现状是否漏推敏感字段，盘点必查 |
| 通知唤醒替代每连接扫库 | **改造** | 单机单进程 ⇒ 进程内 channel 即可（写事件后 notify），不引 Redis/LISTEN-NOTIFY；收到通知仍回库取数（通知只是唤醒不是事实） |
| heartbeat 注释行 + terminal 双侧关闭 | **采纳** | 防代理掐连接；两边都关防僵尸连接 |
| SSE 鉴权三方案 | **采纳（首选方案天然成立）** | 前端 dist 由 API 同源挂载 /app（F-3）⇒ 同域 Cookie 方案零成本；铁律入 CI：长效 key 禁止出现在 URL 构造 |
| WebSocket / 多实例 fanout / Kafka | **裁剪** | 单向下行够用、单实例部署；复活条件=需要双向打断交互 / 多实例部署 |
| Polling 兜底 | **采纳（已有件）** | GET events?after=N 拉历史即是；确认保留 |

### 步骤

```
K7-1 对照验收：课件"七样交付物"× 现状（SSE 格式含 id 行与空行 · 双路续传优先级
     query>header · heartbeat · terminal 关闭 · 取号 · 分层 · 鉴权）
K7-2 事件分层落地：事件注册表标 public/internal/audit；public payload 只含
     display 类字段，推送前过滤器剥 internal 字段；internal 走 trace 接口（独立权限，
     承接 K1-8）
K7-3 通知唤醒：写事件 COMMIT 后 notify 进程内 channel；SSE 循环改"等通知→回库查
     增量→推"；无事件只发 heartbeat
K7-4 鉴权硬化：同域 Cookie 判定用户→org→run 三查；CI 判据：前端代码无长效 key
     进 URL
K7-5 SSE endpoint 只读纪律固化：endpoint 代码路径零业务调用（读、推、游标、心跳、
     终态关闭五个动作之外无其他）
```

### 门槛（不过不进 K8）

| # | 门槛 | 数字 |
|---|---|---|
| ① 顺序断言 | 推送通道故障注入 → 库中事件一条不少、重连全补齐；负向：改成先推后写 ⇒ 丢事实测试必红 |
| ② 断线补发 | 断开期间写 M 条 → 重连（after 与 Last-Event-ID 各测一路）收到恰 M 条、sequence 连续无重复 |
| ③ 分层过滤 | public 流正则扫描：prompt/token/purpose 等 internal 字段零命中；internal trace 接口全量可见 |
| ④ 唤醒效率 | 静默期查询计数=0（只有 heartbeat）；事件写入到推出延迟 < 1s |
| ⑤ 鉴权 | 跨租户订阅 404；CI grep：URL 构造中无长效凭证 |
| ⑥ 只读纪律 | endpoint 路径结构断言：无模型/工具/状态迁移调用 |
| ⑦ 回归 | 前端时间线全链实测（含审批中断恢复重放）+ 全量测试零失败 |

---

## 10 · K8 · 恢复与重放（课件 CH8；~2 天）

> 课件这章的核心认知：**恢复不是一层，是前面各层做对之后"涌现"的能力**——K8 自己
> 不建新表，只把 K2 的证据表、K3 的 lease/心跳、K5 的意图日志、K6 的五态串起来。
> 所以 K8 的施工一半是两个新角色（sweeper/对账），另一半是**端到端故障注入联测**。

### 取舍总表（准则五）

| 课件机制 | 处置 | 理由 |
|---|---|---|
| Sweeper（三分支顺序 cancel→attempts→requeue + 四原则） | **采纳** | 顺序不可换（先判取消，否则"已取消+次数耗尽"被标 failed，责任归属错）；四原则第一条"异常条件必须能写成 SQL"照抄；单机实现=常驻定时线程/systemd timer，不需要 CronJob |
| Reconciliation 对账闭环 | **采纳+改造（我们有独占优势）** | 课件最难的"去问下游"在我们这退化成**查自己平台的去重账本**（下游也是我们的，K5-6 已建）——对账可做到强可靠。节奏定死（课件没给）：response_lost 产生即触发一次 + 每 5 分钟周期兜底；回填+事件+审计同一事务（修复动作自身也要被保护） |
| 四动词纪律（Replay/Retry/Rerun/Repair） | **采纳** | 接口名/按钮/日志用准四个词；Replay 零副作用（"每次复盘真金白银再退一次款"是课件定性的最致命错误）；Repair 必留四样（audit/reason/operator/before-after） |
| 恢复必须留痕 | **采纳** | requeued_by_sweeper + 第二条 run.started 是特性不是 bug——事后要能区分"正常执行"和"救援" |
| 七步排查 SOP + trace 聚合 | **采纳** | 按 run_id 一键拉齐八表证据；第 2 步"最后一个事件"是分诊台（对应我们已有的括号式事件流） |
| checkpoint 读档重入 | **采纳** | §16-4 已裁=课件快照式：读最新 checkpoint 还原 context 重入 loop，`last` 两路判断在 K5-2/K5-3 已定；`run_events` 只用于回放展示与排查 |
| 僵尸 queued 清理（课件 H37 全书未解） | **采纳（我们定死）** | sweeper 加扫描类：queued 超龄且 outbox 无 pending ⇒ 告警 + 允许人工 requeue outbox（不自动，避免掩盖投递层病根） |

### 步骤

```
K8-1 sweeper：扫 running∧lease 过期（K3-7 心跳判据）→ 三分支走 transition_run +
     事件 + 审计；另扫 waiting 超龄（提醒）、queued 超龄（告警）、写类 tool_calls
     超龄 running（按注册表 expected_max_ms 判，标 response_lost，绝不自动重发）
K8-2 对账任务：扫 response_lost / 超龄 running 的写调用 → 按幂等键查平台去重账本 →
     命中回填 succeeded+tool.repaired / 未命中回填 failed / 账本不可用转 manual_review
     事件（⛔ manual_review 是事件不是状态）；三写入同一事务
K8-3 四动词落地：replay 接口（只读 run_events 渲染）；rerun=新 run+parent_run_id
     （K4-5 已建列）；repair 管理通道带四样留痕
K8-4 排查 SOP 写进 runbook（K11 汇总）+ trace 聚合接口（八表按 run_id 聚合，
     internal 权限，承接 K7-2 分层）
K8-5 端到端故障注入联测（本阶段的主体工作量）：分支 A/B/C 三场景 + 取消兑现 +
     慢不当死，全部自动化进测试套件
```

### 门槛（不过不进 K9）

| # | 门槛 | 数字 |
|---|---|---|
| ① 分支 A | 模型调用前 kill worker → sweeper 在 ≤2×TTL 内回收 → 新 worker 续跑至终态；事件流含 requeued_by_sweeper + 第二条 run.started（恢复留痕断言）；只读工具调用计数不变 |
| ② 分支 B | 同 job 双投递 → 副作用计数=1（K3② 在全链场景复测） |
| ③ 分支 C | 写工具执行后、记录前 kill → response_lost → 对账回填 → run 跑完；平台侧副作用计数=1；负向：跳过对账直接重试的实现必红 |
| ④ 取消兑现 | cancel_requested + worker 死 → sweeper 兑现为 cancelled（非 failed）；分支顺序负向认证（换序必红） |
| ⑤ 慢不当死 | 单步超 TTL 但持续续租的 run 零误回收（连续 3×TTL 观察） |
| ⑥ Replay 零副作用 | replay 全量调用前后，平台写操作计数差=0（结构断言） |
| ⑦ 证据链 | trace 接口对任一 run 返回八表聚合零缺项；SOP 文档七步与接口字段一一对应 |
| ⑧ 回归 | 全量测试零失败 |

---

## 11 · K9 · 生产硬化（课件 CH9；~2 天；准则五裁剪面最大的一章）

> 课件三圈壳：入口先拦 · 过程看得见 · 变更可逆。我们的现状：release.py 灰测闸门
> （automation_tier 阶梯 + fail-closed）≈ feature flag/canary 的等价物已有；
> B-1a 平台限流已有；测试租户与真人租户结构隔离（C-1）≈ "staging 不连 prod"已有。
> 单机自有平台 ⇒ K8s/Helm/多租户七维限流/多 provider 圈断器/美元账单全裁。

### 取舍总表（准则五）

| 课件机制 | 处置 | 理由 |
|---|---|---|
| SLO 先行（九条=前八层不变量翻译成数字） | **采纳** | 已列 §15；K9 给每条配上读数来源；"没有对应 SLO 的层，其正确性在线上不可验证" |
| run 级预算四字段 + 超限转 waiting_for_user | **采纳** | 步数上限已有（ActionGate），补 max_tokens/max_duration；**还有救就转 waiting 不判死**（挡晋级不挡起跑的同族思想）；`max_model_calls` 是 agent 后端独有的预算维度 |
| usage 一轮一行记账 | **采纳+改造** | "记录的粒度决定控制的粒度"；本地 vLLM 无美元账 ⇒ 口径=token+时延；org 日 token 预算两档（近限告警/超限拒新建） |
| 结构化日志 + 检索四维度 | **采纳（裁剪字段）** | run_id/org/step/tool_call_id/error_code/latency + request_id；"少记一个字段=少一条排查路径且不可补救"；密钥/prompt 全文不进日志（CI 正则） |
| 指标 15 项 + 告警绑定行动 | **采纳（裁剪至 ~10）** | 必含 `duplicate_prevented_total`（**兜底生效次数是一等公民指标——它量的是"离出事还有多远"**；口径定死：本地幂等闸命中 skipped_duplicate 计一次）、stuck_runs、oldest_job_age；告警输出必带 runbook 节引用 |
| 限流七维度 | **改造→三维度** | 全局并发 / org 并发 / 写工具并发（refund 式更严额度）；api_key/IP/provider 维度裁（单机自用），复活条件=对外开放 |
| expand/contract 迁移 + 四种"事实上的 schema" + schema_version | **采纳（重要）** | checkpoint/事件 payload/工具 args/job payload 都是"跨时间存活的数据"；课件待查#2 我们定死：payload 顶层 `v` 整数版本，读到不认识的版本 ⇒ 拒绝处理+manual_review 事件（不崩不猜）；"改字段含义比删字段更毒（不报错）"与我们守则同源 |
| 发布七能力 + 回滚动作链（按生效速度排序） | **改造→五能力** | 开关（release.py 收编）/停队列（K3 分队列）/禁工具（registry 开关）/回滚版本（git+进程重启，无镜像）/drain（worker 优雅退场：跑完当前轮放 lease）。⛔ 全部要演练——"没演练过的回滚等于没有回滚" |
| 测试八类对照 | **采纳** | 我们已有前五类（unit/integration/contract/故障注入/压测），补 migration/replay/idempotency 三类；"有几类测试=真正在守护几层承诺" |
| K8s/Helm/Compose · 多 provider 圈断器 · PII 脱敏 · prompt injection 治理 | **裁剪** | 单机 systemd/脚本部署；单一本地模型；自用系统。各登记复活条件（多实例部署/接外部 API/对外开放） |

### 步骤

```
K9-1 SLO 落数：§15 九条逐条配读数脚本/接口（可自动打印，不许人工查库口算）
K9-2 预算闸：run 级四字段进 agent_runs+ActionGate 出口判定；org 日 token 预算两档；
     超限"还有救"分支转 waiting_for_user
K9-3 usage 一轮一行 + 指标面板（~10 项）+ 告警判据行（每条绑 runbook 节）
K9-4 日志结构化收口：错误路径统一走结构化 emit；CI 正则挡裸 print/敏感字段
K9-5 schema_version：checkpoint/事件 payload/job payload 顶层加 v；读侧版本网关
     （认识→转换器，不认识→拒绝+manual_review）；迁移纪律文档化（expand/contract
     四步 + 四禁忌）
K9-6 发布五能力演练：逐项实际操作一次并记录（演练记录进 K11 runbook）
K9-7 测试八类对照表补齐 migration/replay/idempotency 三类
```

### 门槛（不过不进 K10）

| # | 门槛 | 数字 |
|---|---|---|
| ① SLO 可读 | 九条 SLO 每条有自动读数（跑一条命令全表打印），零"无法判定" |
| ② 预算 | 注入不收敛 run → max_model_calls 拦停且转 waiting（负向：关闸必红）；org 超限 → 新建被拒且已跑的不受影响 |
| ③ 指标 | duplicate_prevented 埋点生效（构造重复场景计数+1）；stuck_runs/oldest_age 读数与实际吻合 |
| ④ 日志 | 全部错误路径结构化（CI grep 零裸 print 落盘路径）；敏感字段正则扫描零命中 |
| ⑤ 版本网关 | 旧格式样本被新代码正确读取（replay test）；未知版本 → 拒绝+manual_review 事件，进程不崩 |
| ⑥ 演练 | 五能力各一次实际演练记录（时间/操作/恢复时长）；"理论上支持"记 0 分 |
| ⑦ 测试对照 | 八类测试每类 ≥1 例在套件中可点名；缺类显式登记 |
| ⑧ 回归 | 全量测试零失败 |

---

## 12 · K10 · 回流飞轮（课件 CH10；~2 天；与训练线的交界章）

> 课件的两把尺子（"执行完成"≠"问题解决"）正是我们项目的核心张力：runtime 说跑通了、
> 真人一试全是环境的错（00 §3 那四课）。⚠️ 关键判断：**我们已有一套成熟的评测/训练
> 管线**（冻结考卷 + 判分器负向认证 + DATA_VERSION + 出厂体检 + 影子重建），课件的
> eval_datasets/eval_cases/eval_runs 三张表**不重建**（准则〇：那就是我们的考卷体系）。
> K10 只补真缺口：**结果信号的结构化落库 + trace→训练线的安全通道 + 版本归因**。

### 取舍总表（准则五）

| 课件机制 | 处置 | 理由 |
|---|---|---|
| `feedback_items` + `run_annotations` 两张回流表 | **采纳（真缺口）** | dev mode 真人对话的反馈现在靠口头/文档，没有绑 run_id 的结构化落库；两把尺子有时间差 ⇒ 同 run 多行 append、后来的可推翻先前的；**label（症状）/reason_code（病因）必须分列**——同一症状不同病因走向完全不同的修法 |
| 归因分类含 `external_system_error` | **采纳（与我们最贵的教训同源）** | "模型没错，是环境错"= 我们 08-20 四课 + 行为异常先查输入的结构化版本。**没归因的负样本符号可能是反的——喂进 RL 是在惩罚做对了的模型，且 loss 曲线上看不出来** |
| 抽候选覆盖面（不只 failed/点踩，要有业务结果通道） | **采纳** | "你抽不到的 case 永远不会被优化"；只回流失败样本 = 优化尾部丢掉主体（我们 R5 已经吃过分布病的亏） |
| trace→case 四步（抽取→拼 trace→脱敏→归因）+ expected 人工签字 | **采纳（脱敏轻量化）** | 单租户自有数据，PII 面小 ⇒ 脱敏收敛为"密钥/token 删除 + 深拷贝不污染原始 trace"两条；**expected 只能人签**= 我们 gold 纪律同源（26-W2 那条"gold 终答自带数字"） |
| 三条硬门禁（unsafe=0 第一 · pass≥线上 · 人工 review） | **采纳（收编进 candidate_gate）** | 与我们"挡晋级不挡起跑"完全同构；安全类判据零容忍不取平均 = 我们 cap 红线文化；**"指标没接到 return False 上就只是看板装饰"** |
| 版本号记录（run 级 + tool_call 级） | **采纳** | run 行记 contract/model/prompt 版本、tool_call 记 registry 版本——没有版本切片，"错误率涨了"永远拆不成可排除的假设（我们 21 号作废登记的病根之一就是版本口径） |
| 三句限制 + `training_exports` 导出留痕 | **采纳** | runtime 数据不直接训练 / 手动导出 / 不过门禁不上线——与守则⑮/26 号管线天然咬合；导出登记进 _audit（manifest 带 dataset_version + 条数 + 去向） |
| 六表 proposal 平台 + Optimization Skill | **改造→轻量** | 我们的"优化提案"就是文档裁定流（22 号决策记录 + Chaoyu 批准），已是人工闸门；只加一条纪律：**提案必带 evidence（source_run_ids + 数量 + reason_code），"我觉得这样更好"不是证据**；`/auto-apply` 这类接口**永远不实现**（最强的权限控制是能力根本不存在） |
| 标注 UI / 自动聚类 / A/B 分流平台 | **裁剪** | 课件自己也说别先建平台——"先建漂亮工具没有信号流进来 = 工具空转"；四步顺序：先有信号→再有归因→再有题目→最后才有优化 |

### 步骤

```
K10-1 两张回流表落库 + 前端反馈入口（POST feedback/annotation；chatbox 加 👍👎 与
      问题标签——第一步就有价值：上线当天消掉"不知道用户满不满意"的盲区）
K10-2 归因词表定稿：label 集 + reason_code 集（必含 external/env 类）；与 26 号
      триage/难例口径对齐，不另造一套
K10-3 抽候选任务：failed / 负反馈 / 业务结果异常 / 高价值成功四路 SQL（带时间窗
      +LIMIT，常驻周期任务不是一次性脚本）
K10-4 trace→case 通道：拉齐八表 → 深拷贝 → 密钥类删除 → 人工定 expected →
      产物进 26 号管线的考卷/数据体系（带 source_run_id 可追回）
K10-5 版本归因：agent_runs 记 contract/model/prompt 版本，tool_calls 记 registry
      版本；指标按版本切片
K10-6 导出通道：training_exports 登记 + 出局清单（OR 逻辑：未归因失败/幻觉/
      密钥残留任一命中即出局）+ 准入清单（AND 逻辑）——默认拒绝
```

### 门槛（不过不进 K11）

| # | 门槛 | 数字 |
|---|---|---|
| ① 反馈闭环 | 前端反馈落库端到端通（真浏览器实测）；同 run 多条反馈、后条推翻前条的场景有测试 |
| ② 归因强制 | 负样本进导出通道时 reason_code 非空是硬校验（负向：无归因样本必被拒）；external 类样本零进入负样本池（断言） |
| ③ 抽取覆盖 | 四路抽取各有实测产出；构造"跑对了但业务结果坏"的样本，仅业务结果通道能抽到（负向认证覆盖面） |
| ④ 通道纪律 | 原始 trace 逐字节不被加工过程修改（前后哈希一致）；case 无 expected 不能入库；导出 manifest 与实际条数一致 |
| ⑤ 版本切片 | 任一指标可按 contract/prompt/model 版本切出分组读数 |
| ⑥ 回归 | 全量测试零失败 |

---

## 13 · K11 · 上线总清单与 Runbook（附录 A；~1.5 天；全计划的收口）

> 附录 A 的定位：把前十层的承诺压成"可勾选的判断句"（A1）和"半夜照着做的动作卡"
> （A2）。对我们同理：**K11 不新增机制，是把 K1–K10 的门槛沉淀成两份常驻文档 +
> 一轮演练**。裁剪面：SSRF/sandbox/多实例/多 provider/双人审批（复活条件各登记）。

### 取舍总表（准则五）

| 课件机制 | 处置 | 理由 |
|---|---|---|
| 上线总清单（52 条 9 分区，每条可证伪的判断句） | **采纳（裁剪成我们版 ~40 条）** | "清单质量取决于每条能不能被证伪"= 我们守则①；9 分区保留，多租户/多实例条目裁剪并登记 |
| P0/P1/P2 优先级判据 | **采纳（判据本身最值钱）** | **"出错之后能不能靠重跑或加资源救回来？不能 ⇒ P0"**——可判定，直接进我们的排序原则 |
| Runbook 十一段结构 + 止损/恢复分离 + 关键判断两列表 | **采纳** | 止损是减法不需根因、恢复是加法必须有根因——防"边查边改时间线乱掉"；**单一指标永远不判根因，判据是指标组合**（queue 积压 × worker 活跃度那张表直接抄） |
| 六条总原则 + 事故传导图（分诊） | **采纳** | "同时响的告警只有一个是因"；给我们的 runbook 画自己的传导图（模型 429→队列积压→…）；**"禁止盲目重跑全部 running"写进止损栏**——副作用系统里"什么都不做"经常比"做点什么"安全 |
| 复盘模板 + Agent 五问 | **采纳** | "事故结束的标志 = 五问全否"（半成功？待对账？checkpoint 读不了？版本不兼容？事件丢失？）——处置过程自己会造新债；"哪些检测有效/缺失"两栏喂 K10 飞轮 |
| 备份恢复演练 + RPO/RTO | **采纳（我们有真实痛点）** | "没演练过的备份等于没有备份"（第三次出现的原理）；我们的不可再生资产（bases/ 合并底座、生产 candidate）已有 HF 异地仓库惯例——K11 把"从 HF 恢复全链"做一次实测演练（08 §换机器清单就是脚本底稿） |
| 风险分级四级 + 数据分类五级 | **改造→映射与简化** | 风险分级映射到已有 automation_tier（工具注册表补 risk_level 与 tier 对齐）；数据分类简化三级（public/internal/secret），secret 三条"永远不进"照抄（prompt/日志/事件） |
| Prompt injection/SSRF/sandbox/网络策略 | **裁剪（结构已豁免大半）** | 模型只能调 registry 注册工具（allowlist 姿态已有）、无外网请求类工具、无代码执行工具；复活条件=新增任何外网/执行类工具时，网络策略九字段 + sandbox 七条整节启用 |
| Model Gateway / 多实例 SSE / maker-checker | **裁剪/改造** | 单 provider 单实例；"非法输出不静默吞"已有（parse_err 计数文化）；双人审批降级为"高危动作 preview + Chaoyu 确认"，审批记录留 requested_by/approved_by 两列（结构在，人少先同人） |

### 步骤

```
K11-1 上线总清单我们版：K1–K10 全部门槛 + 附录 A 九分区裁剪合并，每条 = 可证伪
      判断句 + 验法（命令/查询）；裁掉的条目显式登记复活条件
K11-2 Runbook 手册：按我们的告警面写 5–6 张卡（队列积压/卡死 run/写工具报错/
      SSE 断线/版本迁移失败/token 消耗异常），十一段结构 + 止损栏黄框 + 判断两列表
      + 传导图；告警输出带卡片引用（K9-3 已埋）
K11-3 复盘模板入库（13 字段 + Agent 五问），并回填一次历史实例（用 ㉞ 陈旧 worker
      抢队列那次事故试写，验证模板可用性）
K11-4 灾备演练：从 HF 仓库 + git 在干净环境重建 serving 全链一次，记录 RTO 实测；
      发现的隐形前提回填 08 §换机器清单（干净机器才暴露的缺口——我们的老教训）
K11-5 全计划收口：九条 SLO 全表实测打印一轮；52→我们版清单逐条勾一遍，
      未达标项显式挂账（守则⑦：空着的门槛读作无法判定）
```

### 门槛（全计划完成判据）

| # | 门槛 | 数字 |
|---|---|---|
| ① 清单 | 我们版总清单零空格：每条勾过（✓/✗/挂账），✗ 与挂账项有归属与复活条件 |
| ② Runbook | 每张卡的"第一批查询"逐条真跑过一遍（命令有效性验证）；每条告警绑定到卡 |
| ③ 复盘 | 模板用历史实例回填一次成功（可读、五问可答） |
| ④ 灾备 | 干净环境重建演练完成，RTO 实测记录在案；新暴露的隐形前提 = 0 或已回填文档 |
| ⑤ SLO | 九条全表一键打印，实测值入档作为 baseline |
| ⑥ 回归 | 全量测试零失败；tool_parity 30/30；26 号线判据不受影响（交界确认） |

---

## 14 · 全计划收口

**K0–K11 十二个阶段全部成稿（2026-09-01）。** 执行顺序：K0 盘点（可与 26 号 W 步
并行）→ §16 四件裁定 → K1 起按序施工，每阶段门槛不过不进下一阶段。全程本机可做，
无 GPU 依赖；总工期粗估 **~18 个工作日**（K0 1 + K1 2 + K2 2 + K3 3 + K4 1.5 +
K5 2 + K6 2 + K7 1.5 + K8 2 + K9 2 + K10 2 + K11 1.5，含各阶段验收）。
⚠️ 这是纯施工估计，不含裁定等待与 26 号维修穿插；排期由 Chaoyu 定。

---

## 15 · 全局验收：九条 SLO（课件 CH1 §8，收尾时逐条核）

| SLO | 归属阶段 |
|---|---|
| POST /runs P95 < 300ms | K1 |
| run.created 成功率 > 99.9% | K2/K3（落库+outbox 事务） |
| 普通任务 P95 完成 < 60s | 整链（K11 实测） |
| run.failed 比例 < 1% | K5 |
| 写类工具错误率 < 0.1% | K6 |
| queue lag P95 < 10s | K3 |
| SSE 断线后可通过 after 补齐 | K7 |
| stuck run 数量 < 10 | K8 |
| 单 org 每日成本不超预算 | K9 |

---

## 16 · 已裁定（Chaoyu 2026-09-02；原"盘点后再拍"改为读完课件相关章后先拍，K0 只做核实）

| # | 事 | 裁定 | 理由（课件出处） | K0 仍要核实的 |
|---|---|---|---|---|
| 1 | 队列底座 | **Celery + Redis**（Dramatiq 为备选，RabbitMQ 登记复活条件=需要更强路由语义） | CH3 §4：框架"使用"中间件；Redis 全书默认且身兼四职（broker/限流/信号量/缓存，CH9 §13）；Chaoyu 要求对齐工业惯例并把中间件的坑逐条学到手 | 现有 asyncio worker 与 Celery prefork 的接法（每任务一个事件循环、连接池在 worker_process_init 建）；B-5 goodput 数字作废需重测 |
| 2 | 数据库引擎 | **PostgreSQL 不换 + 引入 Alembic 做版本化 migration；不引 SQLAlchemy ORM** | CH1 §8 技术栈 = PostgreSQL + Alembic；CH2 C8"schema 一定会变，光补 CH1 承诺的列就三次 migration"；现有裸 SQL + asyncpg 是刻意选择（09 §4 Decimal 等坑在此层修） | Alembic 与 schema.sql 谁是唯一真相（准则〇：只能一份）；DDL 不能进事务的项（CONCURRENTLY） |
| 3 | 存量 id 形态 | **不迁移**：run_id/conversation_id 已是随机十六进制 | 实查 api.py:272/336；H18 前提不成立 | case_ref 及其他表编号是否同样随机 |
| 4 | 恢复哲学 | **课件快照式**：每 append 一条存一次 checkpoint，快照带 last/completed_tool_calls；run_events 只做回放展示 | CH5 §4.4 / CH8 §4.3"存档密度决定恢复分辨率"；分支 C 在稀疏存档下无解 | 现有 save_transcript 的存档密度与字段；resume_after_approval 是否读最新快照 |

⚠️ 四件的坑与后续排查/优化项**不写在本文**，全部登记在 `28-serving-middleware-hazards.md`（独立文档集，保持本文只有施工步骤与门槛）。

---

## 17 · 与现有主线的关系

- 优先级：`26`（v15 数据/尺子维修）仍是唯一队首；本计划的 K0 可以与 26 的 W 步并行
  （都是本机 0 GPU 的活），K1 起的施工排期等 Chaoyu 定。
- 训练侧不受影响：本计划只动 runtime/serving；`交界` 的东西（tool_registry、
  rollout 渲染共用件）动之前按铁律先写 MAINLINE-INFRA。
- 课件 CH10（回流飞轮）与我们训练线的对接，在补写 K10 时单独出接口设计。
