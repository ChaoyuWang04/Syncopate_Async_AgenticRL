# Syncopate · 09 — M9 Runtime 上线态

> 更新于 **2026-08-17**。M9.1–M9.6 施工完成，**M9.7 压测未做**（那是最终的考试）。
> ★★ **验收 / 设计符合性 → `11-runtime-acceptance.md`**（2026-08-17 独立审计，40 条判据
> 逐条核对；**结论：45 条测试全绿，但设计 §3 的 C 档审批一次都没被接上**）——
> 这一份只写「怎么起、施工时抓到了什么」，**别在这里找验收结论**。
> 环境怎么起 → `08-machine-and-environment.md`
> 设计依据 → `../syncopate-project-design-v0.1.md` §36–39

---

## 0 · 三十秒读懂

M0–M8 造的是**训练用的东西**（数据、沙盒、判据、模型）。M9 是第一次造**真服务**：
接真实请求、调广告平台 API、把钱花出去。几乎不复用前面的代码。

```
bash scripts/pg_bootstrap.sh                    # 起库（幂等，一条命令重建）
python -m pytest tests/runtime/ -q              # 45 条
uvicorn syncopate.runtime.api:app --port 8000   # 起服务
```

| 步 | 内容 | 状态 |
|---|---|---|
| M9.1 | 8+1 张表 | ✅ |
| M9.2 | 三层幂等 | ✅ 10 条测试**全是真的投两次** |
| M9.3 | FastAPI + org_id 注入 | ✅ 12 条，主验收是越权打不穿 |
| M9.4 | Tool Runtime + Worker | ✅ 含假平台与故障注入 |
| M9.5 | 审批网关 + 六个降级触发器 | ✅ |
| M9.6 | SSE + 观测 + 成本控制 | ✅ 9 条，主验收是断线补发 |
| M9.8 🆕 | **Runtime 检索服务**（三态契约 + PG 语料） | ✅ 2026-08-17，入口 **`12-rag-runtime-design.md`** |
| **M9.7** | **压测五场景** | ⬜ **最终的考试**。⚠️ 场景②（模型服务挂掉）等训练跑完才有被测对象；场景④已就绪，见 `11 §5` |

⚠️ **上表是"施工"状态，不是"验收"状态。** 2026-08-17 的符合性审计结论：
40 条判据 **✅20 / 🟡7 / ⛔11 / ⬜2**，五个确认缺口 F1–F5 已变成
`tests/runtime/test_design_conformance.py` 里 5 条 `xfail(strict=True)`。
**全表见 `11-runtime-acceptance.md`。**

---

## 1 · 三条贯穿全局的纪律

**① 永不信前端。** `org_id` 一律从鉴权注入（`Depends(current_org)`），
请求模型里**根本没有这个字段**。越权在 **SQL 的 `WHERE org_id=$1`** 里挡，
不是在应用层判断 —— 后者一次 typo 就穿了。

**② 幂等是唯一一个"错了就是真金白银"的东西。** 其余组件出问题最多是服务不可用，
重复扣款是**不可逆损失**。⇒ 判据必须是**实测重复投递**，不能是代码 review。

**③ ★ 沙盒是 runtime 的子集，且契约由 runtime 定义。**
沙盒（`syncopate/core`）可以简化实现，但**不能有 runtime 没有的行为**
（比如"重试一定成功"）。同一个工具两边行为不一致，训出来的策略在线上就不成立。
⇒ 新增工具行为时，**先在这边定契约，再让沙盒去满足它**。

---

## 2 · 三层幂等（§38）实现在哪

| 层 | 谁重复 | 物理保证 |
|---|---|---|
| 请求级 | 用户点两次 | `agent_runs` 上 `UNIQUE(org_id, idempotency_key)`（PARTIAL：只对非空 key 生效） |
| 任务级 | 队列重投 | `claim_run` 的原子 `UPDATE ... FOR UPDATE SKIP LOCKED` + lease |
| **工具级** | 同一次预算变更被调两次 | `tool_calls` 上 `UNIQUE(org_id, external_idempotency_key)` |

**只有第三层是外部系统认的。** 实查过：**Meta Marketing API 本身没有幂等机制** ——
所以这层保证由我们兑现。

**两个刻意的实现选择**：

- **先占坑，再执行。** 反过来的话，进程在"执行完但还没记账"的窗口里崩掉，
  重试就会**真的再扣一次钱**。
- **占坑用独立事务提交。** 执行是外部副作用（HTTP 调用），事务回滚**撤销不了它**。

### ★★★ 超时的两种形态

```
请求没发出去    重试是安全的
到了但回包丢了  重试 = 重复扣款
```

⚠️ 现象**一模一样**，而且 `platform.TIMEOUT_MESSAGE` **只有一份**（错误文本逐字相同）。
分得开的话 runtime 就会去读文本做决策，而那个信号在真平台上不存在 —— 上线即失效。
⇒ **只能靠幂等键分辨**：重试带同一个键，平台/我们的库告诉你"这个键见过"。
这就是第三层存在的全部理由。沙盒里 `EnvSnapshot.failures.side_effect_applied` 是同一条。

---

## 3 · 六个降级触发器（§39）

**全部是外部信号，没有一个读模型置信度** —— 设计文档原话：
「LLM 的 token 概率和"答案对不对"关系很弱（**编造时往往最自信**）」。

```
① tool_failed            工具重试用尽仍失败
② validation_failed      参数校验不通过
③ data_immature          数据未收敛（归因延迟是第一性约束）
④ cap_hit                命中任一护栏
⑤ amount_over_threshold  写动作金额超阈值
⑥ retrieval_empty        RAG 检索为空   ← ★ 直接接上 M8 的 no_match
```

★ ⑥ 值得单说：**M8 之前"检索为空"在系统里没有表示**，所以它不可能成为降级条件。
M8 把 `no_match` 做成明确的信号位之后，runtime 才拿得到它。
**机制先要存在，才谈得上被接上。**

★ 网关的输出**不是"拒绝"，是一张带证据的审批单**（`proposed_params` + `rationale`
+ `evidence`）。人看的是证据不是结论。而 `modified_params`（人改了什么）
是**飞轮回路 2 的燃料**。

★ 审批单和 run 状态**在同一个事务里**翻转。分开会留下
「单开了但 run 还在跑」（重复执行）或「run 停了但没有单」（永久卡死）。

---

## 4 · 施工中抓到的六个真 bug（都有测试守着）

**① 所有按 `run_id` 做键的表作用域都错了。** `run_id` 只在 org 内唯一
（`agent_runs` 的 UNIQUE 是 `(org_id, run_id)`），而我给 `run_events` /
`agent_steps` / `checkpoints` 的唯一约束只写了 `run_id`。
后果：两个 org 用同名 run_id 时 seq 被对方顶高 ⇒ **SSE 断线补发定位到错的位置**。

**② 模块常量被当默认参数值。** `DEFAULT_AMOUNT_THRESHOLD` 写成
`def f(..., amount_threshold=DEFAULT_AMOUNT_THRESHOLD)`，绑定在**函数定义时** ⇒
之后改模块属性（配置热更、测试调阈值）**一律不生效，而且悄无声息**。
⇒ 阈值在函数体里解析，配置走 `WorkerConfig`。

**③ `_shutdown` 关了池却没清引用。** 同一个 app 再次 startup 时判成"已有 db"，
攥着一个**已关闭的池**继续跑。`uvicorn --reload` 也会踩。

**④ `DB = Annotated[...]` 定义在函数内部会失效。** `from __future__ import annotations`
让注解变成字符串，FastAPI 在**模块作用域**查名字 —— 查不到就退化成"这是查询参数"，
表现是所有接口一律 422 `missing query param: db`，和依赖注入八竿子打不着。

**⑤ asyncpg 的连接池绑定在创建它的事件循环上。** 报错是
`InterfaceError: another operation is in progress`，**完全没提循环**，
很容易误诊成并发 bug。⇒ 测试里一个 case = 一个循环 + 一份 lifespan。

**⑥ `fastapi.testclient.TestClient` 在本环境挂死**（连最小 app 都卡在它的 portal 线程上）。
⇒ 测试用 `httpx.ASGITransport` 直连，附带好处是 startup 和请求同循环。

---

## 5 · 已知缺口与下一步

| 缺口 | 说明 |
|---|---|
| **M9.7 压测五场景** | ①10× 流量 ②模型服务挂 ③工具超时 ④RAG 不可用 ⑤单 org 刷爆预算。**每个都要有明确降级路径** |
| 编排是**最小可用** | 一次 metrics 读 + 一次预算写。真正的多轮 Agent Loop 是下一步；先把幂等/审批/事件/计费四条横切接通并测住，再变复杂 |
| 鉴权是占位 | Bearer token → org 映射表。真上线换 OIDC/JWT —— **但注入的形状不变**，只改 `current_org` |
| 【待定】指标 | 延迟 P50/P95/P99（按意图）、并发 run 数、QPS、SSE 断线补发成功率、单任务 token 成本。**M9.7 压测时用实测值反填**，别提前拍 |
| 假平台 vs 真 API | 按 2026-08-14 的决定不接真 Meta（会真烧钱），真接入留到 M10 影子模式 |
| `latency_ms` 列类型写成了 TEXT | `tool_calls.latency_ms` 应该是 INTEGER。现在没往里写值，改的时候一起修 |

🔴 **2026-08-17 换机器后更正：PGDATA 现在就放在 `/workspace`。**
`/workspace` 是本地 XFS，`chmod 700` 生效 ⇒ `PGDATA=/workspace/pgdata/16/syncopate`
（已写进 `/workspace/.env`）。~~旧机器的 mfs 不支持权限位才放不进~~。
⚠️ 但**数据库仍然是派生产物**，这条没变：工具在 `/workspace/tools/postgres`、
schema 在仓库（`syncopate/runtime/schema.sql` 是真相来源）、
`bash scripts/pg_bootstrap.sh` 一条命令重建。详见 `08-machine-and-environment.md` §1.1。
⚠️ 干净机器上重建会撞两个坑（都已修进脚本）：`dpkg -x` **不跑 maintainer 脚本**
⇒ 不建 `postgres` 用户；`libpq.so.5` 不进 ldconfig ⇒ 要 `LD_LIBRARY_PATH`。
