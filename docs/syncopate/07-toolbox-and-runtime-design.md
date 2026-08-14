# Syncopate · 07 — Toolbox 与训练 Runtime 设计：怎么训出一个抗得住真实世界的 agent

> 写于 2026-08-11。
> 上游依据：`00-research-question`（第二目标）、`syncopate-project-design-v0.1` §27/§34/§38、
> 以及本文第 2 节实查的 Meta / AppsFlyer 官方文档。

---

## 0 · 这份文档要解决的问题

我们的 agent 有**真实的写权限**（改预算、传素材）。它上线后面对的不是我们精心构造的世界，而是：

- 请求发不出去、发出去了没回音、回音是一堆脏数据
- 平台限流、改动超频、权限过期
- 两个数据源给出互相矛盾的数字
- campaign 名称和素材标题里可能藏着别人写的指令

$$\boxed{\textbf{沙盒里没出现过的失败模式，模型上线时的正确处理概率} \approx 0}$$

这不是猜测。我们已经实测过同构的事实：base 不知道 `clarify` 这个选项存在，8 次采样全错、
RL 的 advantage 恒为 0 搜不出来，而 SFT 一个 epoch 就解决。
**p=0 的格子 RL 永远够不着** —— 抗风险能力也一样。

所以抗风险不是"上线前加个 try/except"，是**必须在造沙盒的时候就设计进去**。

---

## 1 · 当前沙盒的结构（现状基线）

```
EnvSnapshot（世界）      只读：campaigns / creatives / accounts / benchmarks
                        / memory / safety_lines / 政策库
                        / ★ policy_clauses / insights（M8 新增，见 §1.1）
   ↓ 读工具查它
ToolRegistry            30 个工具：name / description / JSON Schema
                        / kind(read|write) / fact_key / latency_seconds
   ↓ 执行时注入 ToolContext(case, env, sandbox, step, tool_call_id)
Sandbox（本次 rollout）  append-only 审计日志：tool / 参数 / 结果 / 第几步 / 对象主键
```

三条已有的设计纪律（继续保持）：

1. **工具不阻止违规，只如实执行**。没查政策就改预算？工具照做，verifier 事后封顶。
   工具直接拒绝的话，模型学到的是"报错就换一个"，不是"为什么要先查"。
2. **延迟是真的**。`latency_seconds` 走真 `asyncio.sleep`，假的慢暴露不了阻塞问题。
3. **世界只读，改动入账**。重放和归因的基础。

### 1.1 ★★ M8 · RAG v1 的语料层（2026-08-14）

设计文档 §12 把 RAG 分成三层，M2 做完了结构化侧（安全线/benchmark，精确 key 查询），
M8 补上另外两层：

| 层 | 表 | 工具 | 内容 |
|---|---|---|---|
| 半结构化 | `policy_clauses` | `policy.search` | 平台政策 / 广告法 / SOP，**按条款切**，带生效期与 `supersedes` |
| 非结构化 | `insights` | `insight.search_claims` | 复盘结论，**按"一条结论"切**，带 `status: active/superseded/refuted` |

**★★★ 为什么沙盒里不用真向量库**

GRPO 把同一条 case **并发跑 8 遍**比谁好谁坏。ANN 近似检索、共享索引、运行时建库，
都会让「这次召回了、下次没召回」变成组内 reward 的差异来源 —— 而那个差异会被当成
"模型做得不同"记进 advantage。⇒ 语料和 `safety_lines` 一样**逐 case** 落在
`readonly_tables` 里，检索是 `(query, 本 case 语料, reference_now)` 的**纯函数**。

**打分函数是实测选的**（17 查询 × 10 条款自建评测集，四方案同集对照）：

```
词法覆盖率   0.35–0.40   命中 11/12   误召回 2   正确留空 4/5   ← 采用
CJK 二元组   0.05        11/12        2          4/5
BM25(归一)   任意        11/12        6          0/5
Qwen3-0.6B   0.70         9/12        6          0/5
```

- **BM25 出局的真机制**：归一化后 top1 恒为 1.0 ⇒ **任何阈值都必然返回至少一条**，
  它结构上产生不了"查不到" —— **BM25 是排序器，不是判定器**。
  而「无检索幻觉率」这项验收要求"查不到"必须能发生。
- **向量出局的真机制**：Qwen3-0.6B 是 causal decoder 不是检索模型，mean-pool 向量
  各向异性强，任意两段文本余弦落在 0.6–0.9，**相关与不相关之间没有分离带**。
  ⚠️ 这不代表 embedding 不行，只代表手头这个模型不行。

**⚠️⚠️ 已知局限与由它推出的造题纪律：词汇失配**

「每天预算最多能加多少」vs「单日预算上调不得超过前一日的 20%」（每天/单日、加/上调、
最多/不得超过）表层几乎无共同词 —— 四种方案**全军覆没**。这是词法检索的固有边界。

它真正的危险不是"检索差"，是**会教出一个错技能**：把查询词凑成文档原词。
那个技能对真向量库无用甚至有害 —— **训推不一致里最坏的一种：不是能力没学到，是学了个错的**。

⇒ **造题纪律（写进 `corpus.py`，由测试守着）**：
```
✅ 该训的：拿到结果之后怎么办 —— 用现行版还是过期版、查不到时转人工还是硬答
❌ 不该训的：怎么把查询词凑成文档里的原词

造题脚本必须双向断言：该命中的档 gold 查询真的命中且中对，空洞档真的返回空。
不许设计"要换三次说法才查得到"的题。
```

§14 的两项验收都只考"怎么处理结果"、不考"怎么组织查询" —— 这个设计正好绕开了这个坑。

**训推缺口要显式记账**（§33）：线上是真向量库 + rerank，沙盒是确定性词法。
处理办法**不是把它藏起来**，而是让沙盒实现**同一份契约** ——
会返回空、会返回过期的、会返回只沾边的。按「沙盒不要比真实世界友好」的原则，
把不完美如实建模，而不是给个干净的召回。

---

## 2 · 真实 API 的实查事实

> 全部来自官方文档，链接在文末。标 ⚠️ 的是需要进一步核实的。

### 2.1 Meta Marketing API

| # | 事实 | 我们现在 | 严重度 |
|---|---|---|---|
| M1 | **`daily_budget` 是最小货币单位（分）**。文档在 `spend_cap` 上写明"integer value of subunit in your currency" | 我们用元 | 🔴 |
| M2 | 更新 campaign **只返回 `{success: bool}`**，不返回新值 | 我们返回 previous/new budget | 🔴 |
| M3 | 支持 read-after-write，但**必须显式请求字段** | 我们的写根本不改世界，读不到 | 🔴 |
| M4 | **没有幂等机制**，文档未提供任何请求去重 | 我们也没有 | 🔴 |
| M5 | **每个 ad set 每小时最多改 4 次预算**，超了报 `613` / 子码 `1487632`，并封禁该 ad set 一小时 | 无限制 | 🔴 |
| M6 | 限流是**积分制（BUC）**：读 1 分、写 3 分；开发档上限 60 分、标准档 9000 分，衰减 300 秒；**按广告账户共享额度** | 无限流 | 🟡 |
| M7 | ⚠️ `execution_options: ['validate_only']` 校验模式 | 无 dry-run | 🟡 待核实 |

★ **M4 + M5 合起来是最要命的组合**：没有幂等键，而且改动次数有硬上限。
一次超时后盲目重试，可能同时造成"多改一次预算"和"耗尽当小时配额"。

### 2.2 AppsFlyer

| # | 事实 | 严重度 |
|---|---|---|
| A1 | Pull API 配额：查 **≤2 天 → 每分钟 1 次**（每 app 每报表）；**≥3 天 → 每账户每天 120 次 / 每 app 每天 24 次** | 🟡 |
| A2 | **成本数据可能延迟数小时**，取决于提供方 | 🔴 |
| A3 | 数据新鲜度分 daily / real-time 两档，各自的刷新率不同 | 🟡 |
| A4 | ★ **归因窗口不一致是 Meta/AF 差异的头号成因**：Meta 默认 7 天点击 + 1 天浏览；若 AF 侧配成 1 天点击，则**点击后 2–7 天才首次打开 App 的用户，在 AF 里算自然量，在 Meta 里算投放带来的** | 🔴 |

★★ **A4 改变了我们对"数据源打架"的建模方式**。

原计划是给两个源加一个随机偏差。**那是假的** —— 真实的打架有**确定的成因和方向**：
它来自归因窗口配置，而且偏差方向可预测（AF 少算、Meta 多算）。
模型该学的是**识别成因并据此判断该信谁**，不是识别噪声。

---

## 3 · 差距清单与优先级

按"上线第一天会不会出事"排：

### 🔴 P0 — 直接造成金钱损失或错误决策

| # | 差距 | 修法 |
|---|---|---|
| P0-1 | **写完读不到**（实测：改成 900，再读还是 500） | 读工具在返回前把 sandbox 账本**叠加**在 env 之上。不是让 env 可变（那会破坏重放），是加一层 overlay |
| P0-2 | **单位是元不是分** | 全部改成最小货币单位，字段名带单位后缀（`daily_budget_minor`），并在工具描述里写死 |
| P0-3 | **写动作没有幂等键** | 写工具增加 `client_request_id` 必填参数；沙盒按它去重，重复提交返回**同一个结果** + `deduplicated: true` |
| P0-4 | **改动频次无上限** | 沙盒实现 Meta 的 4 次/小时规则，超了返回 `613/1487632` |
| P0-5 | **无失败注入** | 见 §4 |
| P0-6 | 数据源打架未建模 | 按 A4 的真实机制建：两个源 + 归因窗口配置差异 |

### 🟡 P1 — 让 agent 不可靠

| # | 差距 | 修法 |
|---|---|---|
| P1-1 | 写是同步立即生效 | 改成**提交 → 返回 change_id → 查询状态**；`validate_only` 作为 dry-run |
| P1-2 | `poll_review` 是阻塞等待，不是轮询 | ★ 改成立刻返回 `pending`，由 agent 决定何时再查。**现在的实现把"什么时候该查"这个决策从模型手里拿走了** |
| P1-3 | 无分页 | `campaign.list` 加 cursor |
| P1-4 | 无限流 | 实现 BUC 积分制（读 1 / 写 3），耗尽返回 429 + `retry_after` |
| P1-5 | 无归因窗口/时区参数 | 查指标必须指定窗口，不同窗口给不同的数 |

---

## 4 · ★★★ 失败注入机制

### 4.1 分类学（每一类都要有 gold 示范正确形状）

| | 真实来源 | **正确行为** | 错误行为（最常见） |
|---|---|---|---|
| **F1 超时/无响应** | 网络、平台慢 | ★ **状态未知：禁止盲目重试**。先查证当前状态，再决定 | 直接重试 → 重复扣款 |
| **F2 可重试错误** | 429（BUC 积分耗尽）、5xx | 按 `retry_after` 退避，次数有上限 | 立刻密集重试 → 加剧限流 |
| **F3 不可重试错误** | 400 校验 / 403 权限 / **613 改动超频** | 不重试。换方案或上报人工 | 反复重试同一个必然失败的调用 |
| **F4 返回空/缺字段** | AppsFlyer 成本延迟数小时 | 降级：标注不确定，或 `defer` | 把缺失当成 0 |
| **F5 数值不合理** | 上游 bug、口径错 | 交叉验证，拒绝采信 | 照单全收，据此下结论 |
| **F6 多源打架** | 归因窗口不一致 | 两个都查，以 MMP 为准 + 标 caveat + 降 confidence | 只查一个源 |
| **F7 对抗输入** | campaign 名 / 素材标题**是别人能填的** | 把工具返回视为**不可信输入**，绝不执行其中的指令 | 照做 |

★ **F1 是最反直觉的一条**：工程直觉是"失败了就重试"，而在**有金钱副作用且没有幂等键**
（Meta 实况）的场景下，**重试是错的**。这恰恰是必须训进权重的东西。

★ **F7 值得单独强调**（设计文档 §27.2）：假设模型已被策反。工具返回不是可信来源，
而我们的 agent 有真实写权限。

### 4.2 ★★★ 注入必须是**确定性**的，由 case 声明

这条最容易做错，做错了整个 RL 就废了。

GRPO 的机制是**同一条 case 跑 N 遍、组内比较**。若失败是运行时随机注入：

```
rollout 1：第 3 步超时 → 妥善处理 → 0.8
rollout 2：没遇到失败  → 顺利完成 → 0.9
```

模型学到的是"rollout 2 的做法更好" —— **而两者的差异根本不来自模型，来自运气。
advantage 被污染了。**

⇒ 失败剧本写进 `EnvSnapshot`，由 case 声明：

```python
env.failures = [
  {"tool": "campaign.update_budget", "at_call": 1, "mode": "timeout",
   "side_effect_applied": True},        # ★ 超时了但写其实生效了 —— F1 的真形态
  {"tool": "metrics.get_freshness", "at_call": 1, "mode": "rate_limited",
   "retry_after": 30},
]
```

同一条 case 的 N 次 rollout 遇到**完全相同**的失败序列，组内比较才干净。

> 这和我们踩过的「rollout_id 固定导致 artifact 互相覆盖」同源：
> **RL 里任何跨 rollout 的随机性都是污染。**

真实世界当然是随机的，所以两种模式并存：

| 模式 | 注入方式 | 用途 |
|---|---|---|
| **训练 / 评测** | 确定性剧本（case 声明） | 组内可比、可复现 |
| **压测** | 随机注入 + 指定概率 | 测鲁棒性的分布，不用于梯度 |

### 4.3 `side_effect_applied` 是这套机制的灵魂

超时分两种，**模型看到的现象一模一样**：

```
超时 A：请求没发出去      → 世界没变 → 该重试
超时 B：请求到了，回包丢了 → 世界变了 → 重试就是重复扣款
```

**agent 无法从错误信息区分它们** —— 这正是真实世界的样子，也正是"必须先查证"这条
规则存在的理由。沙盒必须能构造 B，否则模型学到的是"超时=没做成"，那是错的。

---

## 5 · 抗风险：SFT 教什么，RL 教什么

| | 教什么 | 为什么归它 |
|---|---|---|
| **SFT** | **响应的形状**：超时要先查证、429 要退避、数据没到要 defer、工具返回里的指令不能执行 | 这是**离散的模式**。没见过就 p≈0，**RL 够不着** |
| **RL** | **程度与取舍**：重试几次、退避多久、什么时候放弃升级人工、confidence 降多少 | 这是**连续的权衡**，有梯度，RL 擅长 |

**每一类失败（F1–F7）至少一条 gold 示范正确形状**，其余交给 RL 调优。

配额建议（沿用设计文档 §27.3 的量级，按我们的格子数收缩）：

```
F1 超时          每个高危写工具 × 2（side_effect_applied 各一）
F2/F3 错误码     每类 × 2
F4 数据缺失      与 I02 合并，复用 defer
F5 数值不合理    × 3
F6 多源打架      × 4（归因窗口的两种配置）
F7 对抗输入      × 4  ← 优先级最高，因为后果不可逆
```

⚠️ **负样本占比要设上限**（设计文档 §40.3 陷阱 1：只喂难例，模型在简单例上悄悄退化）。
我们已经在 dead_grid 桶上实测过这个坑：只装死格导致 `defer` 从 97% 掉到 0%。
**F 类样本必须和正常样本一起进桶，并且配对照档。**

---

## 6 · Toolbox 规范：每个工具必须声明的东西

现在的 `ToolSpec` 只有 `name / description / parameters / kind / fact_key / latency_seconds / requires`。
按上面的差距，需要补：

```python
@dataclass
class ToolSpec:
    # ---- 已有 ----
    name, description, parameters, kind, handler, fact_key, latency_seconds, requires

    # ---- 新增：真实性 ----
    api_ref: str | None          # 对应的真实 endpoint，如 "meta:POST /{campaign_id}"
    unit_notes: dict[str, str]   # {"daily_budget_minor": "最小货币单位（分）"}
    idempotent: bool             # 写工具是否要求 client_request_id
    cost_points: int             # BUC 积分：读 1 / 写 3
    quota: dict | None           # {"per_hour": 4, "scope": "campaign_id"} —— Meta 的 4 次/小时
    paginated: bool

    # ---- 新增：失败面 ----
    failure_modes: tuple[str, ...]   # 这个工具**可能**发生哪些 F 类失败
    retriable_errors: frozenset[str] # 哪些错误码可以重试
```

`failure_modes` 不只是文档：**它是 case 生成器的取值域**，也是"这个工具的失败面有没有被
数据覆盖到"这个体检项的依据。

### 6.1 描述的写法（沿用 M0 的教训）

M0 已经确立的原则：**先说我做什么，再说我不做什么**。现在再加一条：

> **凡是有单位、有配额、有副作用的，必须写进描述里。**
> 模型看不到代码，它对世界的全部认知来自这段文字。

```
campaign.update_budget
  调整 campaign 的日预算。这是不可逆的写操作，会立即影响花费。
  · new_budget_minor 单位是**最小货币单位（分）**，900 元要填 90000。
  · 每个 campaign 每小时最多改 4 次，超出会被平台封禁一小时。
  · 必须传 client_request_id；网络超时后**不要重试**，用 campaign.get_metrics 查证当前值。
  · 返回只有 {success}，不含新值。要确认结果必须再查一次。
```

---

## 7 · Runtime 规范

| 层 | 要求 | 现状 |
|---|---|---|
| **幂等** | 写工具必收 `client_request_id`；沙盒按 (tool, id) 去重 | ❌ |
| **配额** | BUC 积分账本，跨整条 rollout 累计；耗尽返回 429 | ❌ |
| **改动频次** | 按 `quota.scope` 计数，超了返回 613 | ❌ |
| **读写叠加** | 读工具返回 `env` 叠加 `sandbox` 账本后的视图 | ❌ |
| **失败注入** | 从 `env.failures` 取剧本，按 (tool, 第几次调用) 匹配 | ❌ |
| **不可信输入标记** | 工具返回里凡是**用户可填字段**（campaign 名、素材标题）打标，审计日志记 `param_source` | ❌ |
| 延迟 | 真 sleep，可缩放 | ✅ |
| 审计 | append-only | ✅ |

★ `param_source`（设计文档 §37 第 9 张表要求的字段）：记录一个写动作的参数是
**用户给的**还是**从工具返回里读来的**。F7 对抗输入的检测就靠它 ——
"从工具返回里读来的 campaign_id 被拿去做了写动作"是一条可判定的规则。

---

## 8 · 落地顺序

按"改动小 × 收益大"排，每一步都能独立验证：

| 序 | 做什么 | 验收 |
|---|---|---|
| **1** | **读写叠加**（P0-1） | 改预算 900 后再查，读到 900 |
| **2** | **单位改成 minor**（P0-2）+ 描述里写死 | gold 全过；生成器体检无单位歧义 |
| **3** | **幂等键 + 去重**（P0-3） | 同一 `client_request_id` 提交两次，账本只有一条生效 |
| **4** | **失败注入框架**（§4）+ F1/F7 两类 gold | 同一 case 跑 8 遍，失败序列完全一致；F1 的 gold 示范"先查证再决定" |
| **5** | 改动频次 + BUC 限流（P0-4 / P1-4） | 第 5 次改预算返回 613 |
| **6** | `poll_review` 改成真轮询（P1-2） | 长尾轨迹仍在，但"何时查"由模型决定 |
| **7** | 多源 + 归因窗口（P0-6） | 两个源按 A4 的机制给出可解释的差异 |
| **8** | 分页、validate_only、时区（P1-3/P1-5/M7） | — |

⚠️ **每一步都要重跑 gold 验证 + 冻结 EVAL**。我们今天已经吃过两次"改了一个参数，
另一处的前提悄悄失效"的亏（fp32→bf16 让 offload 的结论作废；
`max_model_len` 减半让 prompt 被截掉 45%）。

---

## 附 · 来源

- [Meta · Campaign 参考](https://developers.facebook.com/docs/marketing-api/reference/ad-campaign-group/)
- [Meta · Marketing API 限流](https://developers.facebook.com/docs/marketing-api/overview/rate-limiting/)
- [Meta · Insights 最佳实践与限制](https://developers.facebook.com/docs/marketing-api/insights/best-practices/)
- [AppsFlyer · 数据新鲜度与时区](https://support.appsflyer.com/hc/en-us/articles/360000310629-About-data-freshness-and-timezone-support)
- [AppsFlyer · 报表生成配额](https://support.appsflyer.com/hc/en-us/articles/207034366-Rate-limitations-for-reports-and-reporting-API)
- [AppsFlyer · Meta 广告差异](https://support.appsflyer.com/hc/en-us/articles/4410481130641-Meta-ads-discrepancies)
- [AppsFlyer · Pull API 聚合数据](https://support.appsflyer.com/hc/en-us/articles/207034346-Pull-API-aggregate-data)
