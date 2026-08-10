# Syncopate 交接文档

> 写于 2026-08-10（M0），2026-08-10 更新（SFT 桶重切 + M1）。给下一个上下文窗口。
> **先读这份，再读 `docs/syncopate-project-design-v0.1.md`（权威设计），最后按需查代码。**

---

## 0 · 三十秒读懂这个项目

**第一目标（真实业务）**：手游买量（UA）投放的**全链路闭环** agent——
`L1 idea 收集 → L2 feature 化+批量生成素材 → L3 投放 → L4 跨平台收数 → L5 分析+feature 归因 → L6 决策与扩量 → L7 治理`。
业务价值指标是 **span of control**（一个优化师能管住的 平台×产品×地域×素材 组合数）。

**第二目标（并行，不阻塞）**：异步 agentic RL 研究（sync colocate vs `fully_async_policy`）。

⚠️ **这个定位曾被搞反过**，导致给出过错误建议（例如建议删掉「6 个从没进过 gold 的工具」，而其中三个正是 L5 归因→L6 扩量的全部构件）。**别再犯。**

**三条钉死的前提**（设计文档 §0）：
1. **会变的进 RAG，不变的进权重，绝不能错的进代码**
2. **沙盒里只有过程奖励**——结果正确性（7 天后真赚钱了吗）沙盒不可验证 ⇒ 灰度上线不是验收，是训练的第二阶段
3. **归因延迟是第一性约束**——D7 意味着今天的决策 7 天后才知道对错，而 D1 数据今天就有且极易被误当结论

---

## 1 · 当前状态

### 已完成：M0（commit `f588f6e`）

| 项 | 结果 |
|---|---|
| 工具改名 | `creative.get_performance`→`get_metrics_by_asset`、`benchmark.query`→`get_industry_baseline`；描述改成"先说我做什么，**再说我不做什么**" |
| 泄漏修复 | 6 条内容级泄漏 → **0**。加了生成器的**内容去重守卫**（SHA-256 of prompt+答案+behavior），参数设计再怎么改都撞不了 |
| 三桶切分 | `data/splits/v2/`：EVAL **52**（冻结）/ SFT 105 / RL 423，SHA-256 实测两两零重叠 |
| base 基线 | 4B 基座在冻结 EVAL 上：52 条 × 8 采样，**平均 reward 0.524** |
| 其它 | EOS 实测正常（截断率 13.7%）；labels 实测正确；129 文件入库 |

### 代码结构

```
syncopate/
├── core/        场景无关引擎：schemas(四件套) sandbox tool_registry
│                trajectory runner parsing verifier_engine
├── domains/adcampaign/   21 工具 · 16 cap · 5-lane memory · policies · world
├── authoring/   axes(控制轴) templates(9 模板) generate(含去重守卫) seed_cases
├── pipeline/    build_dataset sft_replay split(三桶) report(分布体检)
├── train/       rollout_loop(框架无关) verl_agent_loop(薄适配)
│                sft eval_local launch_rl
└── prompts/     system.txt step_user.txt prompt_hash
```

**106 个测试全过。** 环境：独立 `.venv`（py3.12 + torch2.9+cu128 + vllm0.12 + verl0.8.0，**numpy 必须 2.2.6**）。

### 常用命令

```bash
python -m syncopate cases generate --spec configs/buckets/v2.yaml --out data/batches/v2
python -m syncopate cases verify
python -m syncopate data split  --batch data/batches/v2 --out data/splits/v2
python -m syncopate data build  --pool rl|sft --batch data/batches/v2 --out data/{rl,sft}/v2
python -m syncopate data report --batch data/batches/v2
python -m syncopate tools list

python -m syncopate.train.sft        --model models/Qwen3-4B --train-file ... --wandb-project syncopate
python -m syncopate.train.eval_local --model models/Qwen3-4B --split-dir data/splits/v2 --samples-per-case 8
python -m syncopate.train.launch_rl  --dry-run
```

---

## 2 · ★★★ 最重要的一个发现：`p=0` 有两种成因

base 的零梯度构成：

```
有梯度 σ>0.01      8/52
饱和   σ=0,r>0.9   4/52
全灭   σ=0,r<0.15  5/52
卡死   σ=0,中间分 35/52   ← 最大的一块
```

**「全灭」不等于「难」。** 我们那 5 条全灭（CLAR/REJ）8 次采样全部 `behavior_mismatch`、截断率 88–100%——base 只是**不知道 `behavior: clarify/reject` 这个输出约定**，就一路调工具撞到 `max_steps=4`。证据：v3_plain SFT **一轮就把它们从 0.000 拉到 1.000**。

| 成因 | 特征 | 代价 |
|---|---|---|
| **约定未知** | base 没见过我们的输出约定 | **廉价**，SFT 一个 epoch 解决 |
| **能力不足** | 真的不会做 | 昂贵，这才是"死格"的实质 |

**真正的死格藏在「卡死」的 35 条里，分两类**：

- **A 类 · 系统性跳过前置（16 条）** ← ★ SFT 冷启动的真正目标
  分数卡在 0.3（12 条 BUD/CRE/LOW，`false_claim` 68 次 + `missing_memory_check` 40 + `unauthorized_write` 40）和 0.4（4 条 BUD，纯 `missing_memory_check`）。
  **不是撞步数上限，是走捷径**：跳过 `memory.search`、跳过安全线核查就下结论。
- **B 类 · 流程对但子分丢分（19 条）**：卡在 0.63–0.9，**零 cap 命中**。这是 RL 该管的。

⇒ **SFT 桶应该是「5 条约定型全灭 + 16 条 A 类」，不是现在按难度标签选的 105 条。**

---

## 2.5 · ✅ SFT 桶已按 dead_grid 重切（2026-08-10）

`split.py` 原来只按 case_id 精确匹配死格 —— 而死格是在**冻结 EVAL** 上测出来的，
精确匹配的结果是空桶。现在走一次外推：EVAL 里的死 case → 它所在的格子 → 池子里同格的 case，
**每格按死因配额取**（见 `pipeline/dead_grid.py`）。

| | 旧（difficulty_proxy） | 新（dead_grid） |
|---|---|---|
| SFT 桶 | 105 条 | 108 条 |
| 模板构成 | BUD 25 / CRE 34 / LOW 46 | BUD 40 / CRE 30 / LOW 20 / **CLAR 12 / REJ 6** |
| 与旧桶重合 | — | 仅 33 条 |

配额 `{convention: 6/格, shortcut: 10/格}`。总量刻意卡在 ~105：
**只换成分不换规模，新旧两次 SFT 才是单变量对比。**

⚠️ **整格全取是错的**：12 个死格对应池子里 288/528 条，SFT 桶会吃掉一多半 RL 池。
而且 12 个死格里有 3 个的 EVAL 证据是 1/2（同格里有活的）——
`(模板, behavior, 结局, entry)` 这个粒度**分辨不出同格内的死活**。
想彻底消掉这个外推，只能直接在池子上跑 base 评测拿 per-case 的 p。

### ★★★ 顺带挖出的泄漏：三桶从来没被数据构造消费过

`build_dataset.build()` 按整个 manifest 建数据集，**从没读过 split 目录**。
所以 `data/sft/v2` 一直是全部 580 条，**52 条冻结 EVAL 从 M0 起就在训练数据里**，
而 `split_report.json` 还写着「三桶零重叠 ✅」。

⇒ **切分文件和训练数据是两件事。切得再干净，构造那步不读它照样泄漏。**
现在 `build()` 必须二选一（`--split-dir` 或显式 `--full-batch`），忘了声明当场报错。

---

## 2.6 · ✅ M1 · 数据成熟度机制（2026-08-10）

| 层 | 产出 |
|---|---|
| **成熟曲线** | `domains/adcampaign/maturity.py`：ROAS 7 天 / CPI 3 天 / CTR 1 天收敛；观测值 = 终值 × 形变（ROAS 早期偏低、CPI 早期偏高）；**未收敛返回区间**，区间宽度随进度收窄到 0 |
| **工具** | `metrics.get_freshness(campaign_id, metric)` → 成熟度 / 已跑天数 / 还差几天 / 样本量 / 预期区间。工具菜单 21 → 22 |
| **行为** | `defer` 进 `VALID_BEHAVIORS` + system.txt，要求给出 `recheck_after_days` |
| **cap** | `premature_decision_cap` 0.15 / `insufficient_sample_cap` 0.20 / `missing_safety_line_cap` 0.20 |
| **数据轴** | `DATA_MATURITIES = [mature, partial, immature]`，对应开投 14/3/1 天 |
| **意图** | I02 `data_freshness_check` 模板（`FRESH_*`），90 条，三档齐全 |
| **批次** | `data/batches/v3` = v2 的 580 条**逐字节不变** + 90 条 FRESH；`data/splits/v3` EVAL 64 / SFT 108 / RL 498 |

### 这一段踩到/绕开的三个坑

1. **新 cap 会追溯判旧数据**。cap 一注册，`active_caps=None` 的存量 case 立刻被打中，
   gold 跌分、基线不可比。⇒ 三条新 cap 的判据都写成**自动闭合**：存量 campaign 默认
   `started_days_ago=30`、安装量充足，永远算 mature，规则对它们恒为 None。
   实测 580 条 gold 重放，新 cap 命中 **0**。
   （`missing_safety_line_cap` 无法自动闭合，改为**只在 case 声明了安全线是必查项时**生效。）
2. **`literal:false` 被判成「期望 True」**。`values_equal` 里 `bool("false")` 是 True，
   于是所有期望 false 的字段悄悄失分，而 `literal:true` 一直是**碰巧**对的。
   实测抓到点：I02 partial 档 outcome 卡在 0.67。已修 + 加测试。
3. **`premature` 和 `insufficient` 必须分开**。样本量不足也会让 maturity 变成 IMMATURE，
   最初的判据 `maturity == IMMATURE` 让两条 cap 同时命中。
   改成按天数判。⇒ 「等就好了」和「等也没用」是两种病，合成一条会让模型学到错误的解法。

### ⚠️ 代价：base 基线必须重测

system.txt 加了 `defer` 和「数据成熟度规则」两节 ⇒ **每条 case 的 prompt 都变了**
⇒ M0 那个 **0.524 不再可比**。新基线在 `_audit/M1_base_4b.json`（EVAL 64 条 × 8 采样）。
冻结 EVAL 的 52 个原 case_id 全部保留，只是又加了 12 条 FRESH。

`eval_local` 现在会直接算 **`defer` 双向准确率**（该 defer 的 defer 率 / 不该 defer 的误判率），
这是 M1 的验收指标——只测单向会训出一个什么都不敢做的 agent。

---

## 3 · 下一步（按优先级）

### 尚未做完的 M1 收尾

- 用新 SFT 桶（`data/sft/v3`）重训一版，确认 A 类死格（跳过前置）真的被抬起来
- I02 的骨架只有 2 种（`get_metrics → get_freshness` ± `list`）。这是刻意的：
  **三档成熟度走同一条链，区别全在读到 freshness 之后怎么判** —— 这比换条链更难，
  但要注意别让它退化成"认模板"

### 然后：M2 RAG v0 → M3 L5 归因 → M4 L6 扩量 → M5 负面数据 → M6 SFT 毕业

见设计文档 §十 的 M0–M12 表。

---

## 4 · 已确定的决策（别再重新讨论）

| 决策 | 结论 | 依据 |
|---|---|---|
| **SFT 要不要混通用数据** | **第一版不混** | 手册 §3.1：大厂混数据是因为用 FFT，**LoRA 已经交了参数空间的保险费**；§18.1：混数据只治「知识/能力遗忘」，而 function calling 的主要杀手是「格式坍缩」和「过拟合」，**混数据治不了** |
| **该做什么替代** | epoch 2→**1**、**监控输出熵**、加两个专项测试（不需要工具的问题测格式坍缩；没见过的 schema 测过拟合） | 手册 §18.5：遗忘量 ∝ lr × 步数 × 数据窄度，**80% 的"灾难性遗忘"是 LR 太大 + 训太多轮** |
| **RL 用不用 LoRA** | **必须用** | 4B 全参 AdamW 优化器状态 48GB > 系统内存 30GB；LoRA r32 只要 0.79GB |
| **reference model 指向谁** | **SFT ckpt，不是 base** | 手册 §24③：否则 KL 一直把模型往"没学过业务"拉 |
| **ckpt 怎么选** | **不选 val loss 最低那个**，选「格式学会了但行为没定型」的 | 手册 §20：SFT 训得越狠 → 熵越低 → GRPO 探索不动。我们已经踩过（选了 val loss 最低，零梯度格子 63%） |
| **安全线维度** | 产品 × 地域，**不加平台** | 平台差异已由 `benchmark.get_industry_baseline` 覆盖；加平台格子 25→125 填不满 |
| **数据源打架听谁的** | **以 MMP 为准，但两个都必须查**；差异 >15% 降 confidence | 平台后台有自归因偏向，差异本身是信号 |

---

## 5 · 踩过的坑（都已有测试守卫，别重蹈）

1. **SFT 标签 bug**：`gold_script(behavior="tool_call")` 默认值从没被覆盖 ⇒ clarify/reject 的监督目标是错的。症状极具迷惑性——**分组 val_loss 降到 0.0000**（完美学会了错误标签），生成时 behavior 恒为 tool_call。当时误判成"token 失衡"做了加权采样，那只是让它把错的学得更牢。
   ⇒ **loss 降到 0 只说明学到了标签，不说明标签是对的。**
2. **伪多样性**：`index*k % n` 在模数相同时只是重排，两个轴会同步变化——25 种组合实际只有 5 种。要让轴同时依赖 index 的**高位和低位**。
3. **Qwen3 模板不对称**：只给**最后一个** assistant 轮加空 `<think>` 块 ⇒ 整段渲染和增量拼接**天生逐 token 不相等**，无论 `enable_thinking` 设什么。解法是只保留一条代码路径（SFT 数据由同一个 rollout 循环回放 gold 产出）。
4. **`active_caps=[]` 曾被当成"跑全部 cap"** ⇒ 现在 `None`=全部、`[]`=全关。
5. **rollout_id 固定** ⇒ GRPO 的 n 份复制品写到同一 artifact 路径互相覆盖，且**不报错**。
6. **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`**（从老师脚本抄来）和 vLLM 内存池冲突，engine 启动就挂。
7. **verl 0.8.0 的 `numpy<2.0.0` 是陈旧钉**：照做会连锁打断 scipy → transformers → verl 自己。装完必须推翻回 numpy 2.2.6。
8. **flash-attn 在 verl 0.8.0 里是硬依赖**（`use_remove_padding=False` 关不掉），但被 import 的四个函数全是纯 PyTorch 工具 ⇒ 用 `scripts/install_flash_attn_shim.py` 的垫片即可，不必在 sm_120 上编译。

---

## 6 · 未验证 / 未做的事（别当成已完成）

| 项 | 状态 |
|---|---|
| **RL 从未正式跑过** | 只有 2 步冒烟，`checkpoints/grpo/` 为空，无 wandb run |
| **`fully_async_policy` 一次没跑** | 第二目标（异步研究）的核心，完全空白 |
| **业务价值指标** | **完全没有**。所有指标都是 reward/loss/ppl/std |
| ~~**数据成熟度轴 / `defer` 行为**~~ | ✅ M1 已做：90 条 I02 gold，三档齐全 |
| **负面数据 N1–N6** | 工具失败/返回空/数据打架/**对抗输入** 均 0 条。N6 尤其要紧——广告平台的 campaign 名称、素材标题都是别人能填的，一次 prompt injection，而这个 agent 有真实写权限 |
| **L1/L2 意图、L5 归因、L6 扩量** | 全空（设计文档 §7 的 I02/I07/I09/I11 是技术制高点） |
| **tools schema 两侧 hash 比对** | 未测（结构上共用 `build_messages`，但没做 SHA-256 比对） |
| **`multi_issue` / `evidence_ambiguous`** | 各 0 条（老师包分别有 89 / 649 条） |
| **按难度（L1–L5）的细分评测** | 标签存在但评测不按它分组 |
| **外部资料全是假数据** | Excel/图片/标签均由 `scripts/make_test_external_data.py` 生成 |

---

## 7 · 关键文件索引

| 文件 | 内容 |
|---|---|
| **`docs/syncopate-project-design-v0.1.md`** | ★ **权威设计**（1246 行）：18 意图体系、六维评估、数据飞轮三回路、M0–M12、附录 A 待定问题 |
| `docs/syncopate/04-status-audit.md` | M0 前的全量勘察报告（含 20 条缺口清单） |
| `docs/syncopate/03-data-distribution.md` | 数据分布体检（意图 × 链长 × 骨架 × 工具频次） |
| `docs/syncopate/00-research-question.md` | 第二目标：sequence-level TIS 的 T 依赖与 staleness |
| `docs/syncopate/02-credit-assignment.md` | 步级信用分配探底（结论：该用 caps 归因，不是谓词翻转） |
| `sft-truth-report.md` | 老师包的真相核查（**头号结论：包里零运行产物**） |
| `_audit/M0_base_4b.json` | ★ base 基线明细，SFT 桶重切的数据来源 |
| `data/splits/v2/split_report.json` | 三桶互斥性实测报告 |
| `/home/samwang/code/projects/核心手册/AgenticRL/sft-finetune-takeaways.md` | ★ **方法论权威**（1648 行），讨论 SFT/RL 方法论时以它为准 |

---

## 8 · 给下一个窗口的第一句话

> 「读 `docs/syncopate/05-handoff.md`。SFT 桶已按 dead_grid 重切、M1 数据成熟度机制已落地，
> 数据在 `data/batches/v3` + `data/splits/v3`。下一步是用新 SFT 桶重训并验 `defer` 双向准确率，然后进 M2。」

如果 Chaoyu 问的是方法论问题（数据量、LoRA、遗忘、评估指标），**先查 `核心手册/`，别凭通用经验答**。
