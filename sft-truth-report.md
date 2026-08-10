# SFT 数据/训练/评估 真实情况核查报告

> 调查日期：2026-08-05
> 调查对象：`reference/industrial_posttrain_training_release/`（下简称 `REL/`），verl 快照简称 `UP/`
> 纪律：只读 + 分析脚本（`_audit/`）；未训练、未修改任何原文件
> 定位方式：全部 grep 符号名，未使用课件行号
> 分析用 Python：`/home/samwang/Downloads/ENTER/envs/verl-omni/bin/python`（pandas 3.0.3 / pyarrow 24.0.0 / transformers 5.8.1）
> tokenizer：`/home/samwang/code/projects/models/Qwen3-0.6B`（与 Qwen3-8B 同族同词表）

---

## ★ 头号结论（先读这一条）

**这个发布包里没有任何运行产物。** `REL/README.md:12-17` 明确写了不包含 checkpoint / W&B 历史 / rollout 结果；实测 `data/rollouts_verl/`、`data/evals/`、`runs/`、`checkpoints/sft/`、`checkpoints/grpo/`、`data/metrics_verl/` **六个目录全部不存在**，全包内除 `data/{sft,rl}/**` 的 9 个数据 jsonl 外**没有任何 `scores.jsonl` / summary / 日志**。

因此 **P1 全部三项（T11 隔离评测集成绩、T12 死格复算、T13c max_step_hit 升降）在本机无法复算**，`8.1 / 9.5 / 92.3` 和 `275→66` 这些数字**在本包内没有任何可核对的来源**。它们只可能来自课件或老师自己的 W&B。这是简历上风险最高的一块。

---

## 1 · 速查表

| 编号 | 问题 | 一句话结论 | 级别 | 简历是否需改 |
|---|---|---|---|---|
| **T1** | gold 轨迹怎么来的 | **不是拒绝采样**。gold 是脚本编排的确定性轨迹，2737 条 `gold_score.reward` **全部恰好 = 1.0**、五个子分全 1.0、零 cap 命中；生成器不在包内 | **C**（拒绝采样确认不存在）/ **A**（"编排+确定性打分"有实测） | ✅ 必改：不能说"教师模型采样 K 条取最优" |
| **T2** | 185 / 166 / 135 | 135 = SFT 池；`val_every=10` 排序切分 → **train 121 / val 14**；`total_epochs=1` 硬锁一个 epoch，bs=1 ⇒ **步数 ≡ 训练样本数**。**185 池 → 19 val + 166 train = step166**，算术完全吻合 | **A**（135/121/14 实测）/ **B**（185→166 为高置信推断，该池不在包内） | ✅ 必改：包内是 121 条，不是 185/166 |
| **T3** | "最难 ~35% + 持久 cap 优先" | **代码里不存在**。`routing/route_case.py` 的决策树**零调用方**，`routing/sampling_policy.py` 是 6 行空壳；shipped 池的 `source=authored_stage5`，与 `pool_writer` 的输出 schema 不同 ⇒ 选池发生在包外。实测难度分布近似均匀（L1 27/L2 29/L3 31/L4 28/L5 6），**不是"最难 35%"** | **C** | ✅ 必改：这是最容易被问穿的一句 |
| **T4** | 配比体检 | 121 条：17 个 intent（`damaged_item_refund` 32 条 = 26%），难度近均匀，**工具列 31 个 schema 全表恒定（1 个 hash）不随 intent 变**，`enable_thinking` 全 False，**86% 样本 ≥3 轮工具调用**；**终答段仅占有 loss token 的 16.6%** | **A** | 可新增（真实数字比模糊描述强） |
| **T5** | 去重 | 管线内**确认不存在**任何 dedup/hash/simhash/minhash/embedding 相似度逻辑。实测：135 条只有 **56 个不同(intent+工具序列)骨架**，110/135（81%）与他人共享骨架，最大重复 16 条；逐字相同的 messages = 0 | **C** | ✅ 必改为"识别了缺口" |
| **T6** | 评测集污染 | **无任何交集断言**（全包 grep 无 assert/disjoint/overlap，且 `tests/` 目录根本不存在）。case_id 层面干净（EVAL∩训练 = 0）；但**实体层面严重重叠**：**283/305（92.8%）EVAL case 的"案例家族"在训练侧有兄弟**，64 个共享 order_id 的订单记录**逐字节完全相同** | **C**（断言）/ **A**（污染实测） | ✅ 必改：不能无限定地说"隔离评测集" |
| **T7** | 质量门 | 入库前**只有一道**：gold 动作缺对应 observation 则 `raise`（`sft_builder.py:186-188`）。**没有**分数阈值、cap 检查、步数上限、parse 合法性检查。**"终答宣称 vs sandbox 对账"确认不存在**于 SFT 侧（数据字段 `sandbox_final_state` 存在但构造期不读） | **A**（那一道）/ **C**（其余） | ✅ 必改 |
| **T8** | 训练方式 | **全参 SFT**（全包无 LoRA 字样），FSDP + bf16 + param/optimizer offload，AdamW lr=1e-5、wd=0.01、clip 1.0、**constant LR 无 warmup**，`total_epochs=1`，**无早停**，`save_freq=-1` ⇒ **只存最后一个 ckpt，不是最优** | **A**（代码）/ **B**（无产物证明真跑过） | 可写"设计并实现" |
| **T9** | loss mask 与截断预检 | loss mask 由 verl `MultiTurnSFTDataset` 负责（本项目未自写）。preflight **逐条算 loss token，但只对整个 split 的总和断言 `>0`**——单条全 0 不会失败，只会体现在打印的 min 上。**实测 45/121（37.2%）样本总长超 `max_length=12288`**，`truncation=left` ⇒ 被切的是**左侧 system prompt + 31 个工具 schema**，监督 token 侥幸幸存 | **A** | ✅ 可作为"发现的真问题"加分项 |
| **T10** | SFT↔RL 接口 | prompt **确实同源**（三处调同一 `render_prompt`+`_case_context`）；`tool_schema_hash` **算了、落盘了、但全仓库没有任何一处做跨阶段比对**；`enable_thinking`：SFT 侧硬编码 False，**RL 侧从不传** ⇒ 走 Qwen3 模板默认 = **允许真 thinking**，两阶段不一致 | **A** | ✅ 可作为"发现的真问题" |
| **T11** | 隔离评测集 305 条成绩 | **无法计算**：零产物。305 个 EVAL case id 可从 manifest 取到，但没有任何 scores.jsonl 可过滤 | **C**（数字无来源） | 🔴 **最高风险**：92.3% 无法自证 |
| **T12** | 死格数（275→66） | **无法计算**：需要按 case 分组的 rollout reward，包内不存在 | **C** | 🔴 同上 |
| **T13** | 三个疑点 | ① communication 机制上**可变**（`1-0.5×forbidden_hits-0.1×not clear`），但 gold 上恒为 1.0（judgement 是预烤进 gold 的）；运行时是否恒 1.0 **无产物可判**。② **`high_risk_no_check_cap` 明确在 `DEFERRED_CAPS` 里"留表不激活"** ⇒ 触发数必然为 0，且**根本没有 risk 子分**（只有 5 个子分）。③ max_step_hit 升降**无法判定**，且发现**代码里两套口径不一致** | ①**C** ②**A** ③**C** | ✅ ② 可直接写；①③ 删数字 |

---

## 2 · 逐项详情

### T0 · 版本与地图

**本仓库**（`verl-async-agentic-rl`，即 Syncopate）
```
commit 062a5dbd8e6cb98decf170c70c652d1537fbdfb6  (main)
docs: point links at renamed repo
未提交改动：?? docs/learning-notes/  ?? docs/syncopate/
```

**verl 接入方式：vendored 源码快照**，不是 pip 也不是 fork——`REL/verl/upstream/` 随包固定（`REL/.gitignore:19` 把它排除出 git）。版本 `0.8.0.dev`。`REL/.venv-verl/` 存在但只是半成品（无 pandas/torch，安装被带宽掐断）。

**关键路径地图**

| 角色 | 路径 |
|---|---|
| SFT 数据构造 | `scripts/build_sft.py` → `train/sft_builder.py::build_sft_dataset` |
| GRPO 数据构造 | `scripts/build_grpo.py` → `train/grpo_builder.py` |
| 一键构造 | `scripts/build_training_data.sh` |
| route_split（同功能） | `routing/route_case.py` + `routing/pool_writer.py`——**均无调用方** |
| gold 轨迹生成脚本 | **不在包内**（`gold_trajectory` 全仓库只有一处读：`sft_builder.py:153`） |
| SFT 入口 | `scripts/run_sft_stage.sh` → `scripts/train_sft.py` → `UP/verl/trainer/sft_trainer.py` |
| GRPO 入口 | `scripts/run_agenticrl_stage.sh` → `scripts/train_grpo_verl.py` → `UP/verl/trainer/main_ppo.py` |
| parquet | `data/sft/stage5/{train,val}.parquet`、`data/rl/stage5/*`、`data/rl/eval/*` |
| 评估产物 | **不存在**（`train/reports.py` + `scripts/summarize_runs.py` 是消费方，无输入） |
| checkpoint | `checkpoints/` 只有 `.gitkeep` |

数据源目录 `data/batches/{sft,stage5_full,eval,rl}/` 四份，每份 `cases/ verifier_specs/ gold_paths/ env_snapshots/` 各 **2737 个文件**（四份是同一批 case 的四个视图，`manifest_id` 都是 `batch_full_v2`），差别只在 `manifest.json` 的 `pool` / `count` / `entries`。

---

### T1 · gold 轨迹来源：**不是拒绝采样**

**结论**：gold 是**脚本编排出来的确定性轨迹**，然后由 verifier 打分打到满分；不存在"每 case 采 K 条取最优"。

**证据 1 —— 生成脚本不在包内。** grep `gold_trajectory` 全仓库（排除 `verl/`、`.venv-verl`）只有一处，且是**读**：
```
train/sft_builder.py:153:    trajectory = gold["gold_trajectory"]
```

**证据 2 —— gold 文件里没有任何采样痕迹，反而有预烤的判定。** 扫描全部 2737 份 `data/batches/stage5_full/gold_paths/*.gold.json`，顶层字段恒定为 6 个：
```
{'actions': 2737, 'final_text': 2737, 'llm_judgement': 2737,
 'expected_reward_min': 2737, 'gold_trajectory': 2737, 'gold_score': 2737}
```
**没有 model / temperature / K / candidates / 采样轮次任何字段**；反而带了一个 `llm_judgement`——即 verifier 三层回退里的第 1 层"注入判定"（`agent/verifier.py:365-368`），意味着 gold 的打分**完全确定、不调 LLM**。

**证据 3 —— 2737 条 gold 全部满分，零 cap。**（`_audit` 内联脚本输出）
```
gold_score.reward 分布: {1.0: 2737}
active_caps 分布   : {'(无cap)': 2737}
confidence 分布    : {1.0: 2737}
子分 outcome/policy/evidence/efficiency/communication：全部 min=max=1.000
```
如果是拒绝采样，分数应有分布；**恒等于 1.0 只可能来自"照着 verifier 的判分规则反向编排动作"**。这与 `agent/policy_eval.py:8-11` 的自述一致：
```python
"""...供：
  - verifier 算"正确答案"对照 agent 实际写动作；
  - case 模板算 gold 的派生动作（保证 gold 与 verifier 同源）。
```

**置信度：高。**

**对简历/面试的影响**：任何"拒绝采样 / best-of-K / 教师模型蒸馏"的表述必须删掉。可换成真实且同样有含金量的说法：*"gold 轨迹由 policy-KB 确定性求值器派生，与 verifier 同源，保证监督信号与奖励口径严格一致"*——这其实是个更工整的设计，不必虚构。

---

### T2 · 条数之谜：185 / 166 / 135

**实测（`_audit/t0_counts.py`）**
```
data/sft/stage5/train.parquet   rows=  121
data/sft/stage5/val.parquet     rows=   14
data/rl/stage5/train.parquet    rows= 2273
data/rl/eval/train.parquet      rows=  304
manifest: sft count=135 pool=sft | eval count=305 | rl count=2274 | stage5_full count=2737
```
与 `REL/README.md:113-118` 的"期望输出"完全一致（121 / 14 / 2273 / 304）。

**135 → 121+14 的机制**（`train/sft_builder.py:63-75`）：
```python
entries = sorted(manifest["entries"], key=lambda item: item["id"])
for index, entry in enumerate(entries):
    split = "val" if index % val_every == 0 else "train"
```
`val_every=10`（`scripts/build_training_data.sh:22`）⇒ val = ⌈135/10⌉ = 14，train = 121。✅

**166 步的语义。** `scripts/train_sft.py:123` 固定 `trainer.total_epochs=1`；`UP/verl/trainer/sft_trainer.py:356` 的外层循环是
```python
for epoch in range(start_epoch, self.config.trainer.total_epochs):
```
一个 epoch 走完 `len(train_dataloader)` 个 batch 就退出，`total_training_steps` 只是提前终止的上限（`:409 is_last_step = global_step >= self.total_training_steps`）。`TRAIN_BATCH_SIZE=1`（`run_sft_stage.sh:79`）⇒ **步数 ≡ 训练样本数，且不可能超过它**。所以 **121 条数据跑不出 166 步**。

**185 的解释（推断）**：若 SFT 池 = 185 条，同样 `val_every=10` ⇒ val = ⌈185/10⌉ = 19，train = 185−19 = **166** = step166。三个数字一次对齐。run 名 `sft_stage5_185_qwen3_4b_step166` 即 `<池大小>_<模型>_step<训练样本数>`。

⚠️ 该 185 池**不在本包内**（包内是 135），且模型是 **Qwen3-4B** 而非包内默认 Qwen3-8B ⇒ 那次 run 用的是**另一版数据 + 另一个模型**。

**置信度**：121/14/135 = 高（实测）；185→166 的算术链 = 高，但"185 就是池大小"是推断，标 **B**。

**一句话钉死**：**SFT 实际训练样本 = 121 条（本包）；来源 = `data/batches/sft/manifest.json` 的 135 条池，按 id 排序每 10 条抽 1 条作 val。课件里的 166 步对应一个包外的 185 条池。**

---

### T3 · "最难 ~35% + 持久 cap 优先"：**代码里不存在**

**确认不存在的搜索范围与关键词**：全包（排除 `verl/`、`.venv-verl`）grep `route_case` / `write_pools` / `pool_manifest` / `pools_dir` / `routing.` / `authored_stage5` / `batch_full_v2` / `\b35\b`。

**证据 1 —— routing 模块零调用方**：
```
=== who imports routing ===
routing/pool_writer.py:16:from routing.route_case import ROUTE_TO_POOL
（除此之外无任何 import）
```
即 `routing/` 只有内部自引用，`scripts/`、`train/`、`agent/` 无一处使用。

**证据 2 —— `routing/sampling_policy.py` 是空壳**（全文 6 行，仅版权头 + 一句 docstring `"""Training sampling policy helpers."""`）。

**证据 3 —— shipped 池不是 `pool_writer` 写的。** `pool_writer.write_pools` 产出的 entry schema 是（`:36-42`）`{case_id, route, reasons, metrics, rollouts_ref}`、`source="routing_v1"`、目录名 `pool_sft_curriculum` 之类；而实际 `data/batches/sft/manifest.json` 的 entry 是
```
['case_id','control_axis','difficulty','files','gold_reward','hashes','id','metadata','primary_intent','source']
source = "authored_stage5"    pool = "sft"
```
**两套 schema 完全不同** ⇒ 选池发生在这个发布包之外。

**证据 4 —— 难度分布不支持"最难 35%"。**（`_audit/t4_mix.py`）
| | L1 | L2 | L3 | L4 | L5 |
|---|---|---|---|---|---|
| SFT train(121) | 27 | 29 | 31 | 28 | 6 |
| 全池(2737) | 811 | 719 | 490 | 512 | 205 |

SFT 里 L3+L4+L5 = 65/121 = **53.7%**，全池 43.7%——**只是轻微偏难，不是"最难 35%"**，且 L5 反而被稀释（4.9% vs 7.5%）。

**证据 5 —— "持久 cap 优先"没有对应字段。** SFT parquet 的 17 列里没有任何 cap 相关列；`gold_reward` 121 条全 = 1.0（std=0），**没有可用来排序"持久 cap"的信号**。route_case 里唯一沾边的是 `gold_reward < 0.80 → quarantine`（`route_case.py:82-83`），但它从不执行，且所有 gold 都是 1.0。

**附带发现（真实缺陷）**：SFT parquet 的 `routing_bucket` 列 **121 条全为 None**。原因在 `sft_builder.py:228-238`——它从 `batch_dir/classification.json` 取值，而 `data/batches/sft/` 下**没有** `classification.json`（只有 `stage5_full/` 有）。这是一处静默降级，不报错。

**置信度：高（确认不存在）。**

**面试口径**：改成"路由决策树（quarantine → tool_gap → parse/tool 失败归因 → spread 分流）在代码里读过并理解，但选池在发布包外完成；我据此识别出'池的构造过程不可复现'这个缺口"。

---

### T4 · 配比体检（真实表，`_audit/t4_mix.py` + `_audit/t4_tokens.py`）

**intent 分布（train 121）**——17 个 intent，头部集中：
```
damaged_item_refund 32 | wrong_item_received 13 | cancel_unshipped_order 12
payment_dispute_or_chargeback 9 | duplicate_charge 8 | reshipment_or_replacement 8
delivered_not_received 7 | partial_shipment 6 | invoice_vat_change 6 | missing_accessory 5
return_label_or_pickup_issue 4 | warranty_or_repair 3 | 其余 5 个 intent 各 1–2 条
```
最大类占 26.4%；全池 2737 里 `damaged_item_refund` 占 28.7%——**SFT 大致等比缩放，未做类别均衡**。

**难度**：L3 31 / L2 29 / L4 28 / L1 27 / L5 6（近均匀，L5 明显欠采）

**assistant 轮数分布（= 工具调用轮数）**
| 工具轮数 | 1 | 2 | 3+ |
|---|---|---|---|
| 样本数 | 8 | 9 | **104（86.0%）** |
最长 9 轮（10 条 assistant 消息）。

**列完整性**：`messages` / `tools` / `enable_thinking` 三列齐全。
- `enable_thinking`：135 条**全为 False**
- `tools`：**只有 1 个 distinct hash**（`b551f79d7634cd83`），每条都是同一份 **31 个工具 schema** ⇒ **不随 intent 变化**。原因在 `sft_builder.py:65`，`tool_schemas = ToolFactory().tool_schemas()` 在循环**外**只算一次。

**token 构成（Qwen3 tokenizer，`enable_thinking=False`，121 条）**
```
        total    assistant   final(终答段)
mean   12279.7      229.1       38.0
P50    12224        222         36
P90    12664        381         47
max    13059        534         80
合计 total=1,485,847 | assistant=27,722 | final=4,595
```
- **assistant（≈有 loss 的 token）只占全序列 1.9%** —— 12k token 里 98% 是 system prompt + 31 个工具 schema + 工具返回
- **★ 终答段 / 全部有 loss token = 16.6%**（逐条中位数 19.4%，P10 9.0%，P90 35.4%）⇒ **83% 的监督信号落在中间的 tool_call 上，只有 1/6 在最终答复上**
- `assistant` 的 P50=222 与 `docs/learning-notes/05` 用同一 tokenizer 得到的 T(P50)=222 **完全吻合**，互为交叉验证

**置信度：高。**

---

### T5 · 去重：**管线里确认不存在**

**搜索范围**：全包 `*.py` `*.sh` `*.yaml`，排除 `verl/upstream/`、`.venv-verl/`
**关键词**：`dedup` `duplicate` `hash` `md5` `sha` `sha1` `sha256` `simhash` `minhash` `jaccard` `similarity` `near_dup` `embedding`

命中的 sha256 全部**与去重无关**：
```
train/sft_builder.py:270      _stable_hash → prompt_trace_hash（审计用，注释明说"不作为安全哈希"）
agent/runtime.py:140          ticket_id = TKT_ + sha256(case_id)[:12]
agent/prompts/templates.py:65 prompt_hash / tool_schema_hash
agent/verifier.py:899         duplicate_write_tools —— 这是 reward 里的"重复写副作用"检测，不是数据去重
```
⇒ **确认不存在**任何样本级去重/近重检测。

**补充事实检查（实体遮蔽后的结构签名，`_audit/t5_t6.py`）**
签名 = `(primary_intent, 工具调用名序列)`，即抹掉全部实体值只看骨架：
```
样本数 135  |  distinct 签名 56
重复组(>1) 31 组，涉及 110 条样本（81.5%）
最大重复条数 16
逐字相同的 messages 重复组：0
Top:
  x16  damaged_item_refund | oms.list_orders->oms.get_order->attachment.list->attachment.inspect
  x8   damaged_item_refund | oms.get_order->attachment.list->attachment.inspect
  x6   cancel_unshipped_order | oms.get_order
仅看工具序列（跨 intent）：distinct 47，最大重复 23
```

**如何诚实解读**：工具调用任务里同一 intent 走同一条规范路径是**设计意图**，骨架重复本身不等于"脏数据"。真正可说的是：**135 条样本只覆盖了 56 种行为骨架，且最大一类占 16 条（11.9%）——有效多样性远低于名义样本量，而管线里没有任何机制去度量或控制这一点。**

**置信度：高。**

**面试口径**：*"管线里没有去重，我做了实体遮蔽后的骨架签名统计，发现 135 条只有 56 种骨架、最大一类 16 条；我的方案是在入库时按骨架签名做配额上限。"* —— 这是标准的"识别缺口 + 给方案"，比编一个不存在的 minhash 强得多。

---

### T6 · 评测集污染：隔离是**承诺**，不是**断言**

#### (a) 有没有交集断言 → **确认不存在**

搜索范围：全包 `*.py`（排除 `verl/`、`.venv-verl`），关键词 `assert` `disjoint` `intersect` `overlap` `leak` `contamin` `holdout` `held_out` `no_overlap`。全部命中只有 2 条，且都无关：
```
agent/providers/local_hf_provider.py:56-57:  assert self._tokenizer is not None / assert self._model is not None
```
另外：`run_tests.py` 存在，但 **`tests/` 目录在发布包里根本不存在** ⇒ 那个"唯一在跑的真红线"测试 runner 无测试可跑。

#### (b) case_id 层面 → **干净**（`_audit/t6_overlap.py`）
```
eval ∩ sft = 0
eval ∩ rl  = 0
rl  ∩ sft  = 135     ← SFT 池是 RL 池的完全子集
|full|=2737  |sft∪eval∪rl|=2579  （158 条未进任何池）
```
⚠️ 顺带一个必须知道的事实：**SFT 用的 135 条 case 100% 也在 GRPO 训练池里**。不是错，但"SFT 集和 RL 集"不是两批数据。

#### (c) 实体层面 → **严重重叠**（`_audit/t6_contamination.py`）

case 确实带来源字段（`cases/*.json` 顶层）：`ticket_id` `customer_id` `order_id` `market` `expected_policy_id`。
（注意 `ticket_id = TKT_+sha256(case_id)[:12]`，`agent/runtime.py:140`，是 case_id 的派生量，不构成独立来源。）

| 字段 | train distinct | eval distinct | 共享值 | **受影响 EVAL case** |
|---|---|---|---|---|
| `order_id` | 137 | 71 | 64 | **149 / 305 (48.9%)** |
| `customer_id` | 265 | 145 | 129 | **289 / 305 (94.8%)** |
| `expected_policy_id` | 87 | 80 | 78 | 303 / 305（政策是有限公共集，属正常） |
| `customer_message` | 2270 | 305 | **1** | **1 / 305（prompt 文本几乎不重复）** |

**决定性证据（`_audit/t6_env.py`）**：64 个共享 order_id，逐个比对两侧 `env_snapshots/*.env.json` 里该订单的记录——
```
同 order_id 的订单记录：完全相同=64  不同=0  取不到=0
  O_2CE707: EVAL ADR_DE_clarify_ask_order_given_b01  ==  TRAIN ADR_DE_clarify_ask_order_given_b02
  O_E59585: EVAL ADR_FR_deny_explain_order_given_d08 ==  TRAIN ADR_FR_deny_explain_order_given_d01
```
**100% 逐字节相同。** 而且 case_id 是**兄弟编号**（`_b01` vs `_b02`）。

**家族级统计（`_audit/t6_family.py`，家族 = case_id 去掉尾部 `_x##`）**
```
EVAL 家族 148 | 训练侧家族 272 | 共享家族 126
★ 落在共享家族里的 EVAL case = 283/305 (92.8%)
   EVAL 独占家族里的 case      =  22
```

**诚实解读**：这是**同一 case 模板批量生成的兄弟样本被随机切到了两侧**。prompt 文本本身不重复（只有 1 条逐字重合），所以不是最粗暴的泄漏；但"同市场 + 同 intent + 同 outcome 类型 + 同一张订单记录"的近邻样本在训练侧存在，**评测集测的更接近"对已见模板的泛化"，而不是"对未见任务的泛化"**。真正干净的 held-out 只有 **22/305（7.2%）**。

**置信度：高（全部实测）。**

**面试口径**：*"case_id 层面是隔离的，我验证过是 0 交集；但我进一步按实体和 case 家族查，发现 92.8% 的评测 case 在训练侧有同模板兄弟，共享的 64 个订单记录逐字节相同。所以那个分数应该被理解为模板内泛化。我会加两道：入库时的 id 交集断言，和按家族而不是按 case 切分。"*

---

### T7 · 质量门：只有一道

**SFT 侧入库过滤的全部条件**（`train/sft_builder.py` 内全部 3 处 `raise`）：

| 位置 | 条件 | 性质 |
|---|---|---|
| `:58-59` | `val_every < 2` | 参数校验，与数据质量无关 |
| **`:186-188`** | **gold 有 action 但缺对应 tool_observation → `raise ValueError`** | **唯一的真质量门** |
| `:246-247` | pandas 未安装 | 环境校验 |

```python
# train/sft_builder.py:185-188
        observation = observations_by_id.get(tool_call_id)
        if observation is None:
            # gold 不完整时必须失败，不能生成"有动作无 observation"的半截监督样本。
            raise ValueError(f"gold trajectory missing observation for {entry['id']} {tool_call_id}")
```

**确认不存在的检查**（搜索范围：`train/sft_builder.py` + `scripts/build_sft.py`）：
- 分数阈值：无（`gold_reward` 只是被原样抄进 `:215` 的列，不做判断）
- cap 检查：无
- 步数上限：无（`case.max_steps` 字段存在但构造期不读）
- parse 合法性：无
- **★ 终答宣称 vs `sandbox_final_state` 对账：确认不存在**。grep `sandbox_final_state|final_state|reconcile|claimed.*execut|false_promise` 于这两个文件 → **0 命中**。

**值得强调的反差**：对账所需的数据**全都在**——每份 gold 都有 `gold_trajectory.sandbox_final_state`（2737/2737），verifier 里也有现成的 `false_promise_cap`（"声称的写 − 实际执行的写 ≠ ∅"，`schemas/reward_schema.py:35`）。**RL 侧用 reward 惩罚空头承诺，SFT 侧却没有对应的入库校验**——这是一个具体、廉价、可实现的缺口（复用 verifier 的集合差逻辑即可）。

`grpo_builder.py` 侧同样只有 `val_every` 和 pandas 两处 raise，无数据质量门。

**置信度：高。**

---

### T8 · 训练方式

| 项 | 值 | 证据 |
|---|---|---|
| 全参 / LoRA | **全参**。全包 grep `lora`（排除 verl/venv）**0 命中** | — |
| 引擎 | FSDP，bf16，`param_offload=True`，`optimizer_offload=True`，`enable_activation_offload=True` | `train_sft.py:108-117` |
| optimizer | **AdamW**（verl 默认），betas [0.9, 0.999]，weight_decay **0.01**，clip_grad **1.0** | `UP/verl/trainer/config/optim/fsdp.yaml` |
| lr | **1e-5**（`run_sft_stage.sh:83` → `train_sft.py:107`），覆盖 verl 默认 1e-3 | — |
| **warmup** | **无**。`lr_warmup_steps_ratio: 0.0`、`lr_warmup_steps: -1`，项目未覆盖 | `optim/fsdp.yaml` |
| **LR schedule** | **constant**（`lr_scheduler_type: constant`），无 cosine 衰减 | 同上 |
| epoch/step | `total_epochs=1`（`train_sft.py:123`）+ `TOTAL_STEPS` 默认 **121**（`run_sft_stage.sh:66`）。因 `total_epochs=1` 硬锁外层循环，**step 上限 ≡ 样本数** | `sft_trainer.py:356,409` |
| **早停** | **无**。`total_training_steps` 是硬性上限，代码里无 patience / best-metric 逻辑 | `sft_trainer.py:409-450` |
| **checkpoint 选择** | **最后一个，不是最优**。`save_freq=-1` ⇒ `sft_trainer.py:438` 的 `(self.save_freq > 0 and is_save_step)` 恒 False，只有 `is_last_step` 触发保存；shell 侧读 `latest_checkpointed_iteration.txt` 取最终步 | `run_sft_stage.sh:234-240` |
| 长度 | `max_length=12288`，`truncation=left`，`pad_mode=no_padding`，`use_dynamic_bsz=False` | `train_sft.py:99-105` |
| batch | `train_batch_size=1`，`micro_batch_size_per_gpu=1` | `run_sft_stage.sh:79-80` |
| 规模 | `NNODES=1`，`NPROC_PER_NODE=64` | `run_sft_stage.sh:37-38` |
| resume | `trainer.resume_mode=disable`，不从 ckpt 续训 | `train_sft.py:129` |

⚠️ `configs/train_sft.yaml` 写的是 `max_length: 4096` / `total_training_steps: 1` / `train_max_samples: 1`，但 **grep 显示无任何代码读取该文件**——真实参数全部来自 `run_sft_stage.sh` 的环境变量。**照 YAML 理解配置会得到错误结论。**

**级别**：代码事实 **A**；"这套配置真的跑过并产出了模型" **B**（无 checkpoint/日志佐证）。

---

### T9 · 损失掩码与截断预检

**loss_mask 构造位置**：**不在本项目**。项目只负责把 `messages`（含 `role: "tool"` 的消息）写进 parquet，掩码由 verl 的 `MultiTurnSFTDataset` 依 chat template 生成。接线在 `train_sft.py:92-95`：
```python
"data.messages_key=messages",
"data.tools_key=tools",
"data.enable_thinking_key=enable_thinking",
"data.enable_thinking_default=False",
```
`sft_builder.py:203-204` 的注释确认了预期语义："gold final_text 是最后一条 assistant 监督目标，训练时 loss 会落在这类 assistant tokens 上"。

**`SKIP_LOSS_MASK_CHECK` 的 preflight 到底做什么**（`scripts/train_sft.py:144-187`）——这是本项目自写的：

1. 用**真实 tokenizer** + **verl 自己的 `MultiTurnSFTDataset`** 读同一份 parquet（`:176`），config 与训练侧同口径（`:157-169`）；
2. **逐条**算 loss token 数：`loss_counts = [_loss_token_count(dataset[index]["loss_mask"]) for index in range(len(dataset))]`（`:177`）——**不是抽样，是全量**；
3. **但断言只对整个 split 的总和**：
```python
# scripts/train_sft.py:178-183
        loss_sum = sum(loss_counts)
        if loss_sum <= 0:
            raise RuntimeError(
                f"{split} SFT loss_mask is empty after max_length={args.max_length}; ..."
```
4. 逐条的 min/max 只被**打印**，不参与判定（`:184-187`）。

⇒ **精确表述：逐条计算、全量覆盖，但只对 split 总和断言 `>0`。单条 `loss_sum=0` 不会让训练失败，只会体现在日志的 `min=0` 上。**

**★ 实测发现的真问题**：`_audit/t4_tokens.py` 显示
```
121 条样本 total token: mean 12280 / P50 12224 / P90 12664 / max 13059
超过 max_length=12288 的样本：45/121 (37.2%)，超出量中位数 183 token
```
`truncation=left` ⇒ 被切掉的是**序列最左端 = system prompt + 31 个工具 schema 的开头部分**。监督 token 在最右端，全部幸存 ⇒ preflight 必然通过。`run_sft_stage.sh:74` 的注释"12288 是已验证不会截掉监督 token 的安全长度"**字面正确，但掩盖了 37% 的样本在训练时看不到完整工具定义**——模型被要求调用它当时看不见 schema 的工具。

（另：若真按 `configs/train_sft.yaml` 的 `max_length: 4096`，121/121 条全部超长，中位数超出 8128 token。所幸该 YAML 无人读取。）

**置信度：高。**

**面试口径**：这是一个高质量的"我自己查出来的坑"，建议写进简历的"发现的问题"一栏。

---

### T10 · SFT ↔ RL 接口

#### (a) 是否共用同一拼 prompt 代码路径 → **是**（A 级）

三处调用点，同一对函数：
```
train/sft_builder.py:157-158            render_prompt("system.txt", {}) / render_prompt("step_user.txt", {"case": _case_context(case)})
train/grpo_builder.py:156-157           同上
agent/runtime.py:243-244                同上（standalone runtime）
```
`_case_context` 定义在 `agent/runtime.py:159`，被 SFT/GRPO builder 各自 import（`sft_builder.py:22`、`grpo_builder.py:22`）。
`sft_builder.py:155` 的注释也明写："首轮 prompt 必须和 runtime/GRPO 同源：同一 system.txt、同一 step_user.txt、同一 _case_context 投影。"

同样地，工具 schema 两侧都来自 `ToolFactory().tool_schemas()`（`sft_builder.py:65` / `verl_agent_loop_adapter.py:49-50` / `runtime.py:238`），是同一份 31 个 schema。

#### (b) `tool_schema_hash` → 算了、存了、**从不比对**

全部出现位置（全包 grep，排除 verl/venv）：
```
agent/prompts/templates.py:9,11,13   定义 stable_hash + docstring 声称用于 consistency audit
agent/runtime.py:240,264,279         runtime 侧计算并写进 trajectory
train/verl_agent_loop_adapter.py:52,84,134   RL rollout 侧计算并写进 trajectory
agent/trajectory.py:57,90            schema 字段
agent/rollout_store.py:137           落盘
```
**没有任何一处做 `==` 比较、断言或告警。** 且 **SFT parquet 的 17 列里根本没有 `tool_schema_hash` 列**（列名见 T0 输出）——SFT 侧连记都没记。

`agent/prompts/templates.py:11-13` 的 docstring 写着"GRPO group 要求同一组内 prompt 版本、tool_schema_hash、env/verifier/model 版本全部一致，consistency audit 靠这些哈希" ⇒ **这是设计意图，不是已实现的校验**。

**顺手算的一致性**：两侧调用的是同一行代码 `ToolFactory().tool_schemas()`，SFT parquet 的 tools 列 121 条只有 1 个 hash（`b551f79d7634cd83`，31 个工具）⇒ **在同一份代码下必然一致**；但一致性来自"共用同一函数"，而**不是来自任何校验**。任何一侧改了 ToolFactory 而另一侧数据没重建，系统不会告警。

#### (c) `enable_thinking` → **两阶段不一致，笔记的判断被证实**

- **SFT 侧**：`sft_builder.py:214` 硬编码 `"enable_thinking": False`；`train_sft.py:94-95` 传 `enable_thinking_key=enable_thinking` + `enable_thinking_default=False`。实测 135 条全 False。
- **RL 侧**：`verl_agent_loop_adapter.py:94-101` 调 `self.apply_chat_template(messages, tools=..., images=..., videos=..., audios=..., mm_processor_kwargs=...)` —— **签名里根本没有 enable_thinking 参数**（`UP/verl/experimental/agent_loop/agent_loop.py:321-330`）。
- verl 内部走 `**self.apply_chat_template_kwargs`（`agent_loop.py:356, 380`），该值取自 `data.apply_chat_template_kwargs`，**默认 `{}`**（`UP/verl/trainer/config/_generated_ppo_trainer.yaml:471`），且**本项目全包 grep `apply_chat_template_kwargs` 为 0 命中** ⇒ **从不设置**。
- Qwen3 chat template 的逻辑是：
```jinja
{%- if enable_thinking is defined and enable_thinking is false %}
    {{- '<think>\n\n</think>\n\n' }}
{%- endif %}
```
**不传 ⇒ 不注入空 think 块 ⇒ 模型可以自由输出真 thinking，且这些 token 的 `response_mask=1`（算梯度）。**

⇒ **证实**：SFT 学的是"无 thinking 直接出 tool_call"，RL rollout 却允许并训练真 thinking。这是一个真实的两阶段分布不一致。

**置信度：高。**

---

### T11 · 隔离评测集 305 条单独成绩 → **无法计算**

**能拿到的**：305 个 EVAL case id（`data/batches/eval/manifest.json`，`count=305`，其中 304 进 `data/rl/eval/train.parquet`、1 条进 val）。

**拿不到的**：任何 `scores.jsonl`。

`scores.jsonl` 的唯一产出点是 GRPO 训练期间的 reward adapter：
```
train/verl_reward_adapter.py:119:    scores_path = run_dir / "scores.jsonl"
train/verl_reward_adapter.py:11:  "...追加 run 级 scores.jsonl / summary.json，方便训练后做 before/after 对比。"
run_dir = data/rollouts_verl/<VERL_RUN_ID>/   (verl_agent_loop_adapter.py:58)
```
消费方是 `train/reports.py:19-21` 和 `scripts/summarize_runs.py`。

**实测目录存在性**：
```
不存在: data/rollouts_verl   不存在: data/evals       不存在: runs
不存在: checkpoints/sft      不存在: checkpoints/grpo  不存在: data/metrics_verl
全包内 .jsonl 文件（排除 verl/venv/batches）：只有 data/{sft,rl}/** 的 9 个数据文件
```
与 `REL/README.md:12-17` 自述一致（明确声明不含 checkpoint / W&B / rollout 结果）。

另注：`scripts/train_grpo_verl.py:134` 设 `trainer.val_before_train=False` ⇒ **连 base 模型的评测都不会自动跑**，"base / SFT 后 / GRPO 后"三份对照在这套默认配置下本来就不会自动产生。

⇒ **`8.1 / 9.5 / 92.3` 三个数字在本包内没有任何可核对的来源，无法验证也无法证伪。**

**级别：C（对本包而言）。**

**建议口径**：这三个数字如果要留在简历上，必须能指出它们来自老师的 W&B run（并且你看过）。否则应删除，或改写为**你能自证的东西**：*"评测池 305 条与训练池 case_id 零交集；我进一步核查发现 92.8% 的评测 case 在训练侧有同模板兄弟。"* —— 后者你有脚本、有输出、当场能复现，面试价值反而更高。

---

### T12 · 死格数（275 → 66）→ **无法计算**

复算需要"每个 case 的 K 条 rollout reward"以算组内方差。该数据只存在于 `data/rollouts_verl/<run_id>/scores.jsonl`（不存在）。

包内唯一的 reward 数据是 gold 的 `gold_score.reward`，**2737 条全 = 1.0**（方差恒为 0）——是"标准答案"，不是策略采样，**不能用来算死格**。

计算逻辑本身在包内可读（`routing/metrics.py:28,41,55` 聚合 `reward_spread` / `max_step_hit_rate` 等），但无输入。

**级别：C。**

---

### T13 · 三个疑点

#### ① communication 是否恒 1.0 / judge 是否每条都真调

**机制上可变**（`agent/verifier.py:812-821`）：
```python
def calculate_communication_score(judgement: dict[str, Any]) -> float:
    hits = len(judgement.get("forbidden_hits", []))
    clear = bool(judgement.get("clear", True))
    return clamp(1.0 - 0.50 * hits - (0.10 if not clear else 0.0))
```
权重 0.05（`verifier.py:58`），是五个子分里最轻的。

**在 gold 上恒 = 1.0**（实测 2737/2737，min=max=1.000）——但这是**构造使然**：gold 自带 `llm_judgement`（`forbidden_hits: []`, `clear: true`），走的是 verifier 三层回退里的**第 1 层"注入"**（`verifier.py:365-368`），根本不调 LLM。

**"每条 rollout 是否都真调 judge" —— 代码上可判，运行时不可判**：
- 调用点：`run_merged_verifier_llm`（`verifier.py:346`），三层回退 = 注入 > provider（真调 LLM，temperature=0.0，`:384-397`）> heuristic 兜底（`:405`）；
- **每次调用都会写 `judge_meta["source"]` ∈ {injected, provider, heuristic}**（`:368, 388, 405`），落进 `diagnostics`；
- ⇒ **只要有 scores.jsonl，这个问题一查即知**；但产物不存在。
- `.env` 里 `VERIFIER_PROVIDER=none` 会走第 3 层 heuristic（`REL/README.md:81-87` 明说"只想先本地检查接线"用），那种情况下就不是正式判分口径。

**级别：机制 A / 运行时 C。**

#### ② risk 子分 / `high_risk_no_check_cap` 是否死代码 → **是，且是主动声明的**

**没有 risk 子分。** `schemas/reward_schema.py:52-59` 的 `Subscores` 只有 5 个字段：`outcome / policy / evidence / efficiency / communication`。

`high_risk_no_check_cap` 明确在**未激活**列表里：
```python
# schemas/reward_schema.py:40-49
# 当前版本「留表不激活」的 5 个 cap：schema/标签保留以稳定数据结构，但当前版本不触发。
DEFERRED_CAPS = [
    "high_risk_no_check_cap",  # high_risk 动作未做核查
    "approval_bypass_cap", "privacy_violation_cap", "stale_commit_cap", "tool_gap_cap",
]
```
`calculate_caps`（`verifier.py:824`）只处理 `ACTIVE_CAP_VALUES` 里的 9 个（注释说 8 个，实际 9 条——`wrong_object_cap` 是后加的，注释没跟上，`reward_schema.py:28-38`）。

**实测佐证**：2737 份 gold 的 `active_caps` 全为空、`confidence` 全 = 1.0。

⇒ **触发记录必然为 0，但这不是 bug，是显式的分期设计。**

**级别：A。** 这条可以直接写进简历/面试，措辞用"分期激活"而非"死代码"。

#### ③ max_step_hit 是升是降 → **无法判定，且发现口径不一致**

无日志/产物 ⇒ 升降不可判。

但顺手查出**代码里至少两套口径**：
```
train/verl_reward_adapter.py:89:  "max_step_hit": not bool((trajectory.get("final_text") or "").strip())
                                   ← 只看"有没有终答"，完全不看步数
agent/verifier.py:800:            hit_max_steps = actual >= spec.max_steps and not trajectory.get("final_text")
                                   ← 步数达上限 且 无终答
```
（`docs/learning-notes/07` §1.2 已记录第三处 `spec.max_steps` 与 case/runtime 的 max_steps 也各自独立。）
⇒ **同名指标在 reward 侧和 verifier 侧统计口径不同，两条曲线不可直接对比。**

**级别：升降 C / 口径不一致 A。**

---

## 3 · P2 顺手带走

| 项 | 结论 | 证据 |
|---|---|---|
| `AgentLoopOutput` 完整字段 | `prompt_ids, response_ids, response_mask, response_logprobs, routed_experts, multi_modal_data, reward_score, num_turns, metrics(AgentLoopMetrics), extra_fields(dict={}), mm_processor_kwargs`。**`metrics` 和 `extra_fields` 都是现成的透传口** | `UP/verl/experimental/agent_loop/agent_loop.py:121-147` |
| `fraction_high` 是否被记录 | 指标名 `rollout_corr/rollout_is_ratio_fraction_high`，由 `UP/verl/trainer/ppo/rollout_corr_helper.py` 计算并进 W&B。**代码里存在；末期值无产物可查（C 级）** | 见 `docs/learning-notes/02` §5 |
| `norm_adv_by_std_in_grpo` | **存在，默认 `True`**；老师的 `train_grpo_verl.py` 未覆盖 ⇒ 生效为 True | `UP/verl/trainer/config/ppo_trainer.yaml:74`；`UP/verl/trainer/ppo/core_algos.py:273` |

---

## 4 · 未完成项 / 卡在哪

| 项 | 卡点 |
|---|---|
| T11 隔离评测集 305 条成绩 | **零运行产物**。除非能拿到老师的 W&B run 或 `data/rollouts_verl/`，否则本机无解 |
| T12 死格复算（275→66） | 同上 |
| T13① communication 运行时是否恒 1.0 | 同上（有 scores.jsonl 即可，`diagnostics.judge_meta.source` 现成） |
| T13③ max_step_hit 升降 | 同上 |
| T2 中 185 池的直接证据 | 185 条的 SFT 池不在本包内；算术链（185→19 val+166 train）自洽，但无法出示那份 manifest |
| T1 gold 生成器 | `data/authoring/`（`policy_eval.py:16` 提到的 `data/authoring/policy_kb.py`）**未随包发布**，无法看到 case/gold 模板的编排逻辑 |
| loss_mask 的逐 token 直接验证 | 需要在装好 verl 的环境里实例化 `MultiTurnSFTDataset`；本机 `.venv-verl` 是半成品（无 torch），未做。T4 的 token 数用增量法近似，与 `docs/learning-notes/05` 的独立测量吻合（P50=222），但不是从 `loss_mask` 直读 |

---

## 5 · 附录：`_audit/` 脚本清单

| 脚本 | 作用 | 对应任务 |
|---|---|---|
| `t0_counts.py` | parquet/jsonl 行数、manifest 元信息 | T0, T2 |
| `t4_mix.py` | intent/difficulty/routing_bucket/gold_reward 分布、tools 列 hash、轮数分布、intent×difficulty 交叉表 | T3, T4 |
| `t4_tokens.py` → `t4_tokens.csv` | 真实 tokenizer 下的 total/assistant/final token 统计 | T4, T9 |
| `t5_t6.py` | 实体遮蔽后的结构签名去重统计 + case 字段探查 | T5, T6 |
| `t6_overlap.py` | 三池 case_id 交并集 | T6 |
| `t6_contamination.py` | order_id/customer_id/customer_message 跨池共享统计 | T6 |
| `t6_env.py` | 共享 order_id 的 env_snapshot 订单记录逐字节比对 | T6 |
| `t6_family.py` | case 家族级重叠（92.8% 的来源） | T6 |

运行方式：`cd <repo root> && /home/samwang/Downloads/ENTER/envs/verl-omni/bin/python _audit/<script>.py`
全部只读，未写入 `reference/` 下任何文件。

---

## 6 · 给简历的三条落地建议

1. **删掉三类无法自证的表述**：拒绝采样（T1）、"最难 35% + 持久 cap 优先"（T3）、`8.1/9.5/92.3` 与 `275→66`（T11/T12）。前两条代码里确认不存在，后两条在本包内无来源。
2. **把"发现的问题"换上去，这些你能当场复现**：37.2% 样本超 `max_length` 被左截断（T9）、92.8% 评测 case 有训练侧同模板兄弟（T6）、SFT 无 thinking / RL 默认开 thinking（T10）、`tool_schema_hash` 算而不比（T10）、SFT 侧缺"宣称 vs sandbox"对账而 RL 侧有 `false_promise_cap`（T7）。**"我审计出这五个坑"比"我做了一个 92.3% 的系统"更抗追问。**
3. **数据规模按真实写**：SFT 121 条训练样本（135 池，14 val），GRPO 2273 条，评测 304/305 条，全参 Qwen3-8B，单机 64 卡，1 epoch、constant LR、无 warmup、无早停、取最后一个 ckpt。
