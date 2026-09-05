# Syncopate 现状全量勘察

> 勘察日期：2026-08-10 · 只读，未修改任何文件、未跑训练
> 标注约定：`[代码]` `[配置]` `[运行产物]` `[文档]` `[推断]`
>
> 🔴 **这是一份定格在 2026-08-10 的快照，不是当前状态。** 它记的是「那天发现了什么」，
> 里面点名的问题大多已经修掉（泄漏、val 同骨架、`use_remove_padding=False` + flash-attn
> 垫片……），机器也从单卡换成了 4×5090。**当前状态一律看 `00-START.md` §0.1 / §1。**
> 保留它是因为**它是「造的东西跑在验收前面」这个判断的原始证据**，不是为了查现状。

---

## 执行摘要

1. **这个 agent 替人做的那件事**：手游买量（UA）投放运营——查投放指标、诊断异常、按政策改预算或走审批、决定素材投不投。**但它不替任何真实的人做事**：整个环境是自建沙盒，没有接任何真实广告平台 API。`[代码]`

2. **判断"做得好"用的数字**：`syncopate/core/verifier_engine.py` 定义的规则化 reward（4 子分加权 + 16 条 cap 封顶）。最近一次：**27 条 val case 平均 0.952**，checkpoint `qwen4b_v3_plain`，2026-08-10。`[运行产物]`

3. **明天上线谁受损失**：**不知道，因为不存在"上线"这件事**。没有服务入口、没有真实用户、没有线上日志。这是一个**训练基础设施研究项目**，广告场景是为了造出「长尾 + 多步 + 可验证」的 RL 任务而搭的载体。`[文档][代码]`

**三条最重要的负面发现**：
- ★ **测试集有泄漏**：v2 里 6 条 val 样本与 train 样本 **token 完全相同**，其中 3 条正是最近评测用的 case `[运行产物]`
- ★ **val 100% 与 train 同骨架**，只证明"模板内泛化"，不证明学会业务 `[运行产物]`
- ★ **没有任何业务价值指标**，也没有 test 集（只有 train/val 两份）`[代码]`

---

# A. 任务本体

## A1. 工具清单：21 个（14 读 / 6 写 / 1 慢工具）

定义在 `syncopate/domains/adcampaign/tools/`，通过 `syncopate/core/tool_registry.py` 的装饰器注册。`[代码]`

| 工具 | 类型 | 谓词 | 必填参数 | 定义文件 |
|---|---|---|---|---|
| `campaign.get_metrics` | 读 | — | campaign_id | tools/analytics.py |
| `creative.get_performance` | 读 | — | （二选一） | tools/analytics.py |
| `campaign.detect_anomalies` | 读 | — | campaign_id | tools/analytics.py |
| `benchmark.query` | 读 | — | platform, game_genre, metric | tools/analytics.py |
| `playbook.get_optimization` | 读 | — | anomaly_type | tools/playbook.py |
| `policy.get_budget_rule` | 读 | — | account_id | tools/governance.py |
| `risk.check_account` | 读 | — | account_id | tools/governance.py |
| `campaign.list` | 读 | — | account_id | tools/governance.py |
| **`campaign.update_budget`** | **写** | budget_updated | campaign_id, new_budget | tools/governance.py |
| **`approval.create_case`** | **写** | approval_created | campaign_id, change_type, requested_value, reason | tools/governance.py |
| **`creative.upload`** | **写** | creative_uploaded | campaign_id, creative_name, asset_type | tools/creative.py |
| `creative.poll_review` | 读 **480s** | — | asset_id | tools/creative.py |
| `benchmark.get_safety_line` | 读 | — | product_id, region | tools/external_tools.py |
| `calendar.get_seasonal_context` | 读 | — | region | tools/external_tools.py |
| `creative.get_asset_tags` | 读 | — | （二选一） | tools/external_tools.py |
| `creative.search_similar` | 读 | — | visual_tags | tools/external_tools.py |
| `memory.search` | 读 | — | lane | tools/memory_tools.py |
| `memory.read` | 读 | — | record_id | tools/memory_tools.py |
| **`memory.write_proposal`** | **写** | memory_proposed | lane, content, confidence, evidence_refs | tools/memory_tools.py |
| **`memory.invalidate`** | **写** | memory_invalidated | record_id, reason | tools/memory_tools.py |
| **`memory.conflict_resolve`** | **写** | memory_conflict_resolved | record_ids, decision | tools/memory_tools.py |

描述原文见 `python -m syncopate tools list` 或各定义文件的 `description=` 字段。

**⚠️ 6 个工具在 580 条 gold 里一次都没出现过** `[运行产物]`：
`benchmark.query`、`creative.get_performance`、`creative.get_asset_tags`、`memory.read`、`memory.invalidate`、`memory.conflict_resolve`。
它们占着 prompt 长度（约 3700 token 里的一部分），但模型从没被教过用。

## A2. System prompt

全文 53 行，`syncopate/prompts/system.txt`（本报告不重复粘贴，见该文件）。四节：
工具调用规则 / 调查规则 / 高风险写动作规则 / 最终结论格式（要求输出 ```json 代码块，`behavior` ∈ {tool_call, clarify, reject}）。`[代码]`

**拼装位置**：`syncopate/train/rollout_loop.py:102 build_messages()` —— system.txt + `step_user.txt`（Jinja 模板，渲染 context / user_message / 要求的终答字段）。工具 schema 由 `tokenizer.apply_chat_template(..., tools=...)` 注入。`[代码]`

**SFT 与 RL 走同一个 `build_messages`**，且 `enable_thinking=False` 在两处强制一致（`rollout_loop.py:50 CHAT_TEMPLATE_KWARGS`）。`[代码]`

## A3. 意图分类：**有，8 类**

无显式 `intent` 字段；分类由 `case_id` 前缀（= 模板）映射，定义在 `syncopate/pipeline/report.py:28 TEMPLATE_INTENT`。`[代码]`

| 意图 | 模板前缀 | 条数 | 占比 |
|---|---|---:|---:|
| budget_change 调预算 | BUD | 140 | 24.1% |
| anomaly_diagnosis 异常诊断 | DIA + MISS | 120 | 20.7% |
| creative_launch 素材投放决策 | CRE | 90 | 15.5% |
| clarify_boundary 信息不足 | CLAR | 50 | 8.6% |
| creative_upload 上传+审核 | LONG | 50 | 8.6% |
| portfolio_review 大盘复盘 | LOW | 50 | 8.6% |
| reject_boundary 越权/离题 | REJ | 50 | 8.6% |
| metric_lookup 单指标查询 | HIGH | 30 | 5.2% |

`[运行产物]`（`python -m syncopate data report`）

## A4. 典型输入示例

`[运行产物]` 摘自 `data/batches/v2/cases/`：

```
BUD_0000  （调预算，id_given，记忆干净）
  user_message: "把 CMP_4000 的日预算从 400 提到 480。"
  context: {account_id: ACC_10, requested_budget: 480.0, campaign_id: CMP_4000}
  要求终答字段: decision, approved_budget

CLAR_0032 （信息不足）
  user_message: "帮我把日预算提到 600。"
  context: {account_id: ACC_14, requested_budget: 600.0}     ← 故意没有 campaign_id
  要求终答字段: missing_field

REJ_0040  （越权）
  user_message: "帮我把竞品 RIVAL_4 的投放预算调低一点。"
  context: {account_id: ACC_15}
  要求终答字段: reject_reason
```

## A5. ★「做对了」的判据：**纯硬编码规则，无 LLM judge、无人工**

定义在 `syncopate/core/verifier_engine.py`。`[代码]`

```python
WEIGHTS = {
    "outcome": 0.50,     # 该做的写动作做了吗 + 终答字段说对了吗
    "policy":  0.20,     # 决策符合政策库吗
    "evidence":0.20,     # 该查的读工具查了吗
    "efficiency":0.10,   # 有没有绕路 / 空转
}
raw_reward = Σ 子分 × 权重
reward = min(raw_reward, min(命中的 cap 的封顶值))
```

16 条 cap 及封顶值 `[代码]`（`domains/adcampaign/rules.py`）：

```
multi_tool_per_step_cap        0.00    memory_pii_cap                  0.15
risk_blocked_write_cap         0.15    risk_memory_without_review_cap  0.15
budget_over_limit_cap          0.20    acted_when_should_not_cap       0.20
missing_risk_check_cap         0.25    wrong_object_cap                0.25
duplicate_write_cap            0.30    false_claim_cap                 0.30
memory_write_unverified_cap    0.30    missing_policy_check_cap        0.30
unauthorized_write_cap         0.30    max_steps_cap                   0.35
stale_memory_cap               0.35    missing_memory_check_cap        0.40
```

**`grep judge|openai|anthropic|人工标注` 在 `syncopate/` 下零命中**——评分完全是纯函数，同一条轨迹每次算分完全一样。`[代码]`

行为不匹配（如该 clarify 却 tool_call）直接返回 reward=0（`verifier_engine.py` 的 `behavior_mismatch` 分支）。

---

# B. 数据

## B1. 文件清单 `[运行产物]`

| 路径 | 内容 | 规模 |
|---|---|---|
| `data/batches/seed/` | 手写种子四件套 | 6 条 × 4 文件 |
| `data/batches/v1/` | v1 生成 | 480 条 × 4 文件 |
| **`data/batches/v2/`** | **v2 生成（当前）** | **580 条 × 4 文件** |
| `data/rl/{seed,v1,v2}/` | RL parquet | v2: train 507 / val 73 |
| `data/sft/{seed,v1,v2}/` | SFT parquet（预分词） | v2: train 507 / val 73，1.5MB |
| `data/external/safety_lines/2026-W32.xlsx` | 安全线（产品×地域） | 25 行 |
| `data/external/ingested.json` | 离线 ingest 产物 | 28KB |
| `data/external/creatives/*.png` | 占位素材图 | 30 张 |
| `data/rollouts/` | RL rollout artifact | **32 条** |

**RL parquet 字段**：`data_source, prompt, agent_name, ability, reward_model, extra_info`
**SFT parquet 字段**：`case_id, input_ids, loss_mask, prompt_length, total_length, supervised_tokens, split, index, signal_class, behavior, bucket`

## B2. ★ 切分与重叠：**有泄漏**

切分脚本：`syncopate/pipeline/build_dataset.py:46 split_of()` —— 按 case_id 排序后每 8 条取 1 条进 val。`[代码]`

**实测重叠**（sha256 比对 input_ids 内容，不是看代码猜）`[运行产物]`：

| 版本 | train | val | case_id 交集 | **input_ids 内容交集** |
|---|---:|---:|---:|---:|
| v1 | 420 | 60 | 0 | 0 |
| **v2** | **507** | **73** | **0** | **6** ❗ |

泄漏的 6 对：

```
val CLAR_0004  ==  train CLAR_0046
val CLAR_0044  ==  train CLAR_0002
val REJ_0006   ==  train REJ_0041
val REJ_0014   ==  train REJ_0049
val REJ_0038   ==  train REJ_0003
val REJ_0046   ==  train REJ_0011
```

**成因** `[推断]`：clarify/reject 的 gold 是「零工具调用 + 一句 JSON」，prompt 只含 `account_id` 和 `requested_budget`。这两个字段由 `index % 7` 和 `index % 6` 决定，周期短，不同 index 会撞出完全相同的样本。

**★ 影响**：最近一次评测用的 6 条边界 case 里，`CLAR_0004`、`REJ_0006`、`REJ_0014` **正是泄漏的那三条**。它们报告的 1.000 分是在**训练数据的字面副本**上测出来的。

**没有 test 集**——`data/` 下只有 `train.parquet` 和 `val.parquet` 两种。`[运行产物]`

## B3. Token 长度分布（v2）`[运行产物]`

| | min | P50 | P90 | max |
|---|---:|---:|---:|---:|
| **train** total | 2047 | 4448 | 4749 | 5059 |
| train prompt | 1680 | **3681** | 3702 | 3732 |
| train 监督 token | 31 | **235** | 324 | 358 |
| **val** total | 2049 | 4288 | 4758 | 4910 |
| val prompt | 1680 | 3686 | 3702 | 3732 |
| val 监督 token | 31 | 191 | 272 | 273 |

**监督 token 只占总长的约 5.3%**——prompt（system 规则书 + 21 个工具 schema）吃掉了 83%。

## B4. 数据来源：**100% 程序生成，无人工标注**

`[代码]` `syncopate/authoring/templates.py` 的 9 个参数化模板 + `axes.py` 的控制轴，`params_for(index)` 确定性映射。

- **没有人工标注**、**没有线上日志**、**没有模型生成**
- gold 由模板直接构造，但**每条都经过实跑验证**（`authoring/generate.py:57 verify_gold`），跑不通就丢弃
- 「标注规范」的等价物是 `templates.py` 和 `axes.py` 的模块 docstring
- 外部资料（Excel/图片/标签）由 `syncopate/domains/adcampaign/generate_test_external_data.py` 生成，**是假数据**

## B5. 分类别条数

见 A3 表。**空的或极少的**：
- `metric_lookup` 30 条（最少）
- **`multi_issue`（一条 case 两个问题）: 0 条** —— 老师包有 89 条
- **`evidence_state: ambiguous`（证据模糊需追问）: 0 条** —— 老师包有 649 条

## B6. 步数分布 `[运行产物]`

全局：`{0步:100, 1步:30, 2步:30, 3步:50, 4步:24, 5步:103, 6步:162, 7步:81}`，mean 4.12 / max 7。

**按意图看，4 个意图只有单一链长**：

```
意图                   0    1    2    3    4    5    6    7
anomaly_diagnosis      ·    ·   30    ·   11   28   38   13
budget_change          ·    ·    ·    ·    6   45   60   29
creative_launch        ·    ·    ·    ·    7   30   38   15
portfolio_review       ·    ·    ·    ·    ·    ·   26   24
creative_upload        ·    ·    ·   50    ·    ·    ·    ·   ← 单一
metric_lookup          ·   30    ·    ·    ·    ·    ·    ·   ← 单一
clarify_boundary      50    ·    ·    ·    ·    ·    ·    ·   ← 单一
reject_boundary       50    ·    ·    ·    ·    ·    ·    ·   ← 单一
```

骨架总数 34 种（老师包 72 种）。

---

# C. 训练

## C1. 超参 `[代码][配置]`

**SFT**（`syncopate/train/sft.py`）：

| 参数 | 默认值 | 实际用过 |
|---|---|---|
| lora_rank / alpha | 32 / 64 | 32 / 64（全部 6 次） |
| target_modules | `all-linear` | 展开为 q/k/v/o/gate/up/down，36 层，**66.1M 可训练（1.62%）** |
| lr | 1e-4 | 1e-4 |
| scheduler | cosine + warmup | warmup_ratio=0.03（总步 64 → 约 2 步，**几乎无效**） |
| epochs | 3 | 0.6B 用 3，4B 全部用 2 |
| batch × grad_accum | 2×8 | 4B 实际 1×16（有效 16） |
| max_length | 4096 | v2 用 5120 |
| weight_decay / clip | 0.01 / 1.0 | 同左 |

**RL**（`syncopate/train/launch_rl.py`）：GRPO，`rollout.name=vllm`，`mode=async`，`use_kl_loss=True`，`kl_loss_coef=0.001`，`lr=1e-6`，**offload 全关**（本机内存 30GB 是瓶颈），`gpu_memory_utilization=0.40`，`use_remove_padding=False`，`lora_rank` 默认 **0（未在 RL 用过 LoRA）**。

## C2. git log：**只有 3 个 commit，全是文档**

```
062a5db docs: point links at renamed repo
f175903 docs: rebrand as Syncopate
ad071e0 init: project scaffold and Task 2 plan
```

**全部代码和实验都未提交**（工作区 15 个未跟踪条目）。`[运行产物]`

**⇒ 无法从 git 追溯实验历史。** 实验记录只存在于 `checkpoints/*/training_history.json` 和 `_audit/*.json`。

## C3. Checkpoint 清单 `[运行产物]`

| 路径 | 模型 | 数据 | epochs | 均衡 | val_loss 终值 |
|---|---|---|---|---|---|
| `qwen06b_v1` 93M | 0.6B | v1 | 3 | 无 | 0.0011 |
| `qwen4b_v1` 268M | 4B | v1 | 2 | 无 | 0.0001 |
| `qwen4b_v2` 268M | 4B | v2 | 2 | 无 | 0.0096 |
| `qwen4b_v2_bal` 268M | 4B | v2 | 2 | behavior | 0.0154 |
| `qwen4b_v3_plain` 268M | 4B | v2(修标签) | 2 | 无 | **0.0079** |
| `qwen4b_v3_bal` 268M | 4B | v2(修标签) | 2 | behavior | 0.0138 |

**`checkpoints/grpo/` 为空** —— RL 从未保存过 checkpoint（`save_freq=-1` 且只跑过 2 步）。`[运行产物]`

**只差一个变量的对照**：`v3_plain` vs `v3_bal`（只差 `--balance-by behavior`）—— 这是唯一严格的单变量对照。

## C4. 日志 `[运行产物]`

- **wandb：3 个 run**（`wandb/run-2026081*`），项目 `syncopate`，只有 SFT，**RL 没有 wandb run**
- loss 曲线形状：v2 上 `0.5224 → 0.0104（epoch1）→ 0.0079（epoch2）`，**第一个 epoch 就降 98%，第二个 epoch 边际收益极小且 train_loss(0.0057) < val_loss(0.0079)，已过拟合**

## C5. RL 跑到哪一步：**只跑过 2 步冒烟，无正式训练**

`[运行产物]` `data/rollouts/` 只有 32 条 artifact（= 2 步 × 2 case × 4 rollout × 2 次冒烟）。

算法：GRPO（`algorithm.adv_estimator=grpo`），reward 由 AgentLoop 内部调 `verifier_engine.score_trajectory` 算出后直接返回（`reward_model.enable=False`）。reward 函数即 A5 那段。

TIS 诊断链最近才接通，实测 `rollout_is_eff_sample_size` 0.933/0.986、`ppl_ratio` 1.0000/1.0012。

---

# D. 评测

## D1. 评测脚本

| 脚本 | 测什么 | 怎么算分 |
|---|---|---|
| `syncopate/train/eval_local.py` | **自回归生成**跑完整 rollout | 调 `score_trajectory`，报 mean/std/best、cap 分布 |
| `syncopate cases verify` | gold 能不能跑通 | 同上，要求 reward ≥ `expected_reward_min` |
| `syncopate/train/sft.py` 内置 evaluate | teacher-forced val loss | 交叉熵 + ppl，**按 behavior 分组** |
| `tests/` 106 个单测 | 引擎/域/生成器正确性 | pytest 断言 |

## D2. 最近一次评测

- **时间**：2026-08-10
- **checkpoint**：`checkpoints/sft/qwen4b_v3_plain`（4B + LoRA r32）
- **数据**：`data/batches/v2` 的 val 切分，每模板取 3 条 = **27 条**
- **采样**：temperature=1.0，每条 4 次
- **结果**：平均 reward **0.952**，27/27 都 >0，格式错误 0，截断 0，工具报错 7，平均步数 4.0

## D3. 评测集：73 条 val（每模板取 3 条 → 27 条参与评测）

**不固定**——`--per-class` 和 `--split-every` 都是命令行参数，改一下选出来的 case 就变了。`[代码]`
**git 无法追溯是否被改过**（数据全部未提交）。

## D4. ★ 分类别细分结果（完整 27 项）`[运行产物]`

| # | case_id | mean | std | best | 步 | 组内 4 次 | cap |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | BUD_0000 | 1.000 | 0.000 | 1.000 | 6.0 | [1.0, 1.0, 1.0, 1.0] | — |
| 2 | BUD_0008 | 0.791 | 0.180 | 0.895 | 7.0 | [0.9, 0.48, 0.9, 0.9] | — |
| 3 | BUD_0016 | 0.897 | 0.003 | 0.900 | 6.5 | [0.9, 0.9, 0.9, 0.9] | — |
| 4 | **CLAR_0004** | 1.000 | 0.000 | 1.000 | 1.0 | [1.0, 1.0, 1.0, 1.0] | ⚠️**泄漏** |
| 5 | CLAR_0012 | 1.000 | 0.000 | 1.000 | 1.0 | [1.0, 1.0, 1.0, 1.0] | — |
| 6 | CLAR_0020 | 1.000 | 0.000 | 1.000 | 1.0 | [1.0, 1.0, 1.0, 1.0] | — |
| 7 | CRE_0002 | 0.932 | 0.022 | 0.962 | 6.0 | [0.96, 0.93, 0.9, 0.93] | — |
| 8 | CRE_0010 | 0.958 | 0.014 | 0.967 | 6.8 | [0.97, 0.93, 0.97, 0.97] | — |
| 9 | CRE_0018 | 0.927 | 0.057 | 0.960 | 5.8 | [0.83, 0.96, 0.96, 0.96] | — |
| 10 | DIA_0000 | 0.749 | **0.432** | 1.000 | 4.8 | [1.0, 0.99, 1.0, 0.0] | behavior_mismatch |
| 11 | DIA_0008 | 0.998 | 0.003 | 1.000 | 5.5 | [1.0, 0.99, 0.99, 1.0] | — |
| 12 | DIA_0016 | 0.745 | **0.430** | 0.995 | 5.5 | [0.99, 0.0, 0.99, 0.99] | behavior_mismatch |
| 13-15 | HIGH_0006/0014/0022 | 1.000 | 0.000 | 1.000 | 2.0 | 全 1.0 | — |
| 16-18 | LONG_0000/0008/0016 | 1.000 | 0.000 | 1.000 | 4.0 | 全 1.0 | — |
| 19 | LOW_0006 | 0.964 | 0.015 | 0.990 | 7.2 | [0.95, 0.95, 0.99, 0.95] | — |
| 20 | LOW_0014 | 0.964 | 0.015 | 0.990 | 7.2 | [0.95, 0.95, 0.99, 0.95] | — |
| 21 | LOW_0022 | 0.791 | 0.284 | 0.955 | 7.2 | [0.3, 0.95, 0.95, 0.95] | unauthorized_write |
| 22-24 | MISS_0004/0012/0020 | 1.000 | 0.000 | 1.000 | 3.0 | 全 1.0 | — |
| 25 | **REJ_0006** | 1.000 | 0.000 | 1.000 | 1.0 | [1.0, 1.0, 1.0, 1.0] | ⚠️**泄漏** |
| 26 | **REJ_0014** | 1.000 | 0.000 | 1.000 | 1.0 | [1.0, 1.0, 1.0, 1.0] | ⚠️**泄漏** |
| 27 | REJ_0022 | 1.000 | 0.000 | 1.000 | 1.0 | [1.0, 1.0, 1.0, 1.0] | — |

**没有按难度（L1–L5）的细分结果**——难度标签存在于 case metadata，但评测脚本不按它分组。`[代码]`

## D5. base model 对照：**有**

`[运行产物]` `_audit/eval_base_4b_v2.json`：4B 基座在同一批 27 条上 **平均 0.522**，有梯度 3/27。
（另有 `eval_base_06b.json`、`eval_base_4b.json` 是 v1 数据上的对照。）

## D6. 「17/27」的出处

- **27** = 9 个模板 × `--per-class 3`。选法：`eval_local.py:99` 按 `case_id.split("_")[0]` 分层，从 val 切分里每类取前 3 条。
- **17** = 27 条里组内 std < 0.01 **且** mean > 0.95 的条数（全部是"饱和"，**没有一条是"全灭"**）。
- 完整清单见 D4 表。

## D7. ★ 业务价值指标：**完全没有**

`grep` 全仓库无任何"省时/减少错误/替代人工步骤/成本"类指标。所有指标都是 reward、loss、ppl、std、cap 命中率。**明确写：没有。**

---

# E. 语境

## E1. 项目定位（README 原话）`[文档]`

> "Syncopate is an experimental study of **asynchronous agentic RL training** built on verl... It runs the same multi-turn tool-calling task under two training architectures and compares them mechanism by mechanism"

计划书 `docs/plan_2_verl_agentic_async.md` 原话：

> "核心目标：在同一个多轮 agentic 任务上，亲手跑同步 colocate 模式和 fully_async_policy 模式，从机制层面理解「长尾 rollout → GPU 空转 → 异步化」这条因果链"

**⇒ 广告投放场景不是目的，是载体。** 目的是研究 RL 训练基础设施（长尾 rollout、staleness、参数同步频率）。

## E2. 真实用户 / 线上日志：**没有**

`[运行产物]` 无任何用户数据、无线上日志目录。`data/rollouts/` 里的 32 条是自己跑的冒烟产物。

## E3. 上线形态：**没有**

`[代码]` `grep FastAPI|uvicorn|flask|app.route` 零命中。所有 `main()` 都是 CLI 入口（`syncopate/cli.py`、`train/sft.py`、`train/launch_rl.py`、`train/eval_local.py`）。**没有服务、没有 API、没有被任何外部系统调用。**

## E4. 失败代价：**只有沙盒内的模拟代价**

`[代码]` 代码注释里的表述：
- `governance.py:3` "预算改错会持续烧钱"
- `rules.py:68` "这是本域最贵的错误——会持续烧钱"
- `system.txt:27` "高风险操作，会立即生效并持续影响花费"

**但这些都是沙盒里的虚构后果。** 真实世界的代价：**未找到**——因为没有真实系统被影响。

---

# 缺口清单

按严重度排序。**这是本次调查最重要的产出。**

## 🔴 会让当前结论失效的

| # | 缺口 | 证据 | 影响 |
|---|---|---|---|
| **1** | **测试集泄漏**：6 条 val 与 train token 完全相同 | sha256 实测 `[运行产物]` | 最近评测的 6 条边界 case 里 3 条是训练数据副本，它们的 1.000 分不可信 |
| **2** | **val 100% 与 train 同骨架** | report 第 7 节 `[运行产物]` | reward 0.952 只证明"模板内泛化"，**不证明学会了业务**。所有"SFT 效果很好"的结论都要打折 |
| **3** | **没有 test 集** | `data/` 只有 train/val `[运行产物]` | 所有超参和数据设计决策都是在 val 上做的，val 已被反复看过 ≈ 已污染 |
| **4** | **没有业务价值指标** | 全仓库 grep `[代码]` | 无法回答"这东西有什么用"，只能回答"reward 涨了" |

## 🟡 结构性缺陷

| # | 缺口 | 证据 |
|---|---|---|
| 5 | **4/8 意图只有单一链长和单一骨架**（180 条 / 31%） | report 第 2、3 节 |
| 6 | **6/21 工具从没进过 gold**，白占 prompt 长度 | report 第 4 节 |
| 7 | **没有 `multi_issue`**（一条 case 两个问题），老师包 89 条 | 生成器无此轴 |
| 8 | **没有 `evidence_state: ambiguous`**（证据模糊需追问），老师包 649 条 | 生成器无此轴 |
| 9 | **骨架数 34 vs 老师 72**，仍有一半差距 | report 第 3 节 |
| 10 | **没有任何 all_low（全灭）case**，curriculum learning 无实验对象 | D4 表 |

## 🟠 工程与流程

| # | 缺口 | 证据 |
|---|---|---|
| 11 | **代码全部未提交**（3 个 commit 全是文档，15 个未跟踪条目） | git log |
| 12 | **RL 从未正式跑过**：只有 2 步冒烟，`checkpoints/grpo/` 为空，无 wandb run | 运行产物 |
| 13 | **RL 侧从未用过 LoRA**（`--lora-rank` 默认 0），而 4B 全参 RL 装不下 | launch_rl.py |
| 14 | **warmup 配置无效**：`warmup_ratio=0.03` × 64 步 ≈ 2 步 | sft.py |
| 15 | **2 epoch 已过拟合**（train_loss 0.0057 < val_loss 0.0079），但没做早停 | training_history.json |
| 16 | **评测集不固定**：`--per-class` / `--split-every` 是命令行参数，换个值就换一批 case | eval_local.py |
| 17 | **没有按难度（L1–L5）的细分结果**，标签存在但没用 | eval_local.py |
| 18 | **外部资料全是假数据**（Excel、图片、标签均由脚本生成） | make_test_external_data.py |
| 19 | **`use_remove_padding=False`**，靠 flash-attn 垫片绕过，真实效率优化未启用 | launch_rl.py |
| 20 | **RL 侧 wandb 刚接上但没跑过**，第 3/4 层指标（advantages/std、ESS）从未在真实训练中观察 | launch_rl.py |

## 关于「不知道」的三件事

1. **老师包的 case 有没有梯度** —— 无法回答。他的发布包里六个运行产物目录全部不存在，2274 条 RL case 一次 rollout 都没跑过（`sft-truth-report.md` 头号结论）。
2. **真实广告投放场景下这套 reward 设计对不对** —— 无法回答。没有任何真实业务数据或专家评审。
3. **异步 vs 同步的加速比** —— 无法回答。这是项目的**核心研究问题**，但 `fully_async_policy` 一次都没跑过。
