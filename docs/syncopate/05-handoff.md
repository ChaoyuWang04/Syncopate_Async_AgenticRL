# Syncopate 交接文档

> 更新于 2026-08-11（沙盒保真度大改 + v8 基线之后）。给下一个上下文窗口。
> **先读这份 → `07-toolbox-and-runtime-design.md`（沙盒设计）→ `syncopate-project-design-v0.1.md`（权威设计）→ 按需查代码。**

---

## 0 · 三十秒读懂

**第一目标（真实业务）**：手游买量投放的**全链路闭环** agent。
业务价值指标是 **span of control**（一个优化师能管住的 平台×产品×地域×素材 组合数）。

**第二目标（并行，不阻塞）**：异步 agentic RL 研究（sync colocate vs fully_async）。

⚠️ **这个定位曾被搞反过**，导致给出过错误建议。别再犯。

**三条钉死的前提**（设计文档 §0）：
1. 会变的进 RAG，不变的进权重，绝不能错的进代码
2. 沙盒里只有过程奖励 ⇒ 灰度上线不是验收，是训练的第二阶段
3. 归因延迟是第一性约束（D7 才知对错，D1 数据极易被误当结论）

---

## 1 · 当前状态（**先看这张表**）

| 项 | 状态 |
|---|---|
| M0 地基 | ✅ 三桶切分 / 泄漏修复 / base 基线 |
| M1 数据成熟度 | ✅ 机制 + I02 + `defer`，验收达标（100% / 0.2%） |
| **沙盒保真度改造** | ✅ **本轮主体**，见 §3 |
| **v8 base 基线** | ✅ 已测，见 §4 —— **六条预测错了四条** |
| SFT（v8 数据上） | ⬜ **下一步**，桶已切好待确认 |
| RL 正式训练 | ⬜ 管线打通过（50 步跑完），但那一轮**结果作废**（prompt 被截断） |
| `fully_async` | ⬜ **单卡跑不了**，见 §6 |

**数据版本：v8**（`data/batches/v8` 820 条 / `data/splits/v8` / `data/sft/v8` / `data/rl/v8`）
**188 个测试全过**；`.venv` 独立环境（py3.12 + torch2.9+cu128 + vllm0.12 + verl0.8.0，numpy 必须 2.2.6）

---

## 2 · ★ 下一步该做什么（按顺序）

### 2.1 立刻做：确认 SFT 桶 → 训 SFT → 评

```bash
# 实际用的是默认配额 {convention 6, shortcut 10, control 4} → SFT 279 / RL 437
# （2026-08-12 补齐四个空模板后的数，见 §2.2）
python -m syncopate data split --batch data/batches/v8 --out data/splits/v8 \
    --dead-from _audit/v8_base.json
python -m syncopate data build --pool sft --batch data/batches/v8 \
    --out data/sft/v8 --split-dir data/splits/v8 --val-every 6 --model models/Qwen3-4B
python -m syncopate.train.sft --model models/Qwen3-4B \
    --train-file data/sft/v8/train.parquet --val-file data/sft/v8/val.parquet \
    --out checkpoints/sft/v8 --epochs 2 --batch-size 1 --grad-accum 4 \
    --lr 1e-4 --warmup-ratio 0.1 --lora-rank 32 --max-length 6144
python -m syncopate.train.eval_local --model models/Qwen3-4B \
    --adapter checkpoints/sft/v8/epoch1 --batch data/batches/v8 \
    --split-dir data/splits/v8 --samples-per-case 8 --out _audit/v8_sft_e1.json
python -m syncopate.train.compare _audit/v8_base.json _audit/v8_sft_e1.json
```

⚠️ **`--batch-size 1`**：batch 2 会 OOM（logits 张量 = batch × 5600 token × 15 万词表）。

### 2.2 ★ SFT 桶的成分：F 类占 35%（原 45%），四个空模板已补齐

> **2026-08-12 更新**：查这个问题时发现了一个更严重的洞，已修。
>
> **DIA / HIGH / LONG / MISS 四个模板在 SFT 桶里各 0 条**，而它们在 EVAL 里有 20 条、
> RL 池里有 180 条。原因是 `add_controls()` 有两道闸各挡掉两个模板：
> 闸①「只在有死格的模板下找对照」挡掉 HIGH/LONG（它们没有死格）；
> 闸②「对照必须是 gradient/saturated」挡掉 DIA/MISS（它们是 `subscore`，
> 分 0.745–0.900、**一条 cap 都没打中**，只是零梯度被 0.9 的边界划到了另一边）。
>
> ⇒ 上一次退化的教训**只修了一半**：修了「同一意图内的其它档」，没修「整个模板没进桶」。
> 铁证：M1fix 那一轮（**已经带对照档**）里 `MISS 0.745 → 0.649（-0.096）、截断率 0%→28%`，
> 而 MISS 正是被闸②挡住的模板之一。
>
> **修法**：闸① 去掉（每个模板都要有对照）；闸② 放宽成 `control_eligible()`
> —— 新增 `CONTROL_SUBSCORE_FLOOR = 0.7`，收「零梯度但 ≥0.7 且无 cap」的那批。
> 判据是「格子里**有**够格的样本」而不是「全部够格」：试过后者，它会连带丢掉
> `FRESH|defer|immature`（0.598/0.683）——**那正是本机制当初为之而生的那一格**。
>
> **结果**：对照格 17 → 25（纯增量，丢 0）；SFT 247 → 279；四个模板各 0 → 8；
> F 占比 39% → **35%**；RL 池 469 → 437。197 个测试全过。

**F 占比的机制**（已查清，不要重新推导）：

```
进桶格子 42 个 → F 类 20 个（48%），非 F 22 个
F 类每格池子   4–7 条   ← 配额 7 用不满，能拿多少拿多少
非 F 每格池子  10–41 条  ← 配额才是瓶颈
```

⇒ **占比由「格子数」决定，不由池子大小或配额决定。**
**多造非 F 数据没用**（非 F 每格已有 10–41 条，取的还是 7 条）。
**降配额反而让 F 占比上升**（39% → 49%），因为只砍到了池子大的非 F。

真正能降的杠杆只有两个：
1. **减少 F 的格子数**（F 类不按 `entry_mode` 细分 → 10 格 → F 占比 31%）
2. 给 F 类单独降配额（配额从「按 kind」改成「按 kind × 模板」）

**但当前判断是：先不降，直接训，然后用指标测。** 理由：

- 我们的 F 类**任务就是改预算**，和 140 条正常 BUD 同构。每条 F 轨迹的前三步
  （查现状 → 查政策 → 过风控）和正常路径一模一样 ⇒
  **「学 F 类」不会挤掉「学正常路径」，正常路径嵌在每一条 F 样本里。**
  经典的「只喂难例导致退化」，难例往往是另一类任务；我们不是。
- 上次真正翻车（`defer` 97%→0%）是**桶里一条 defer 都没有**，不是占比问题。
  现在 `FRESH` 三档齐、`BUD` 三结局齐、behavior 四种齐、**25 个 control 格子、11 个模板全覆盖**。

**判据已经就位**：`eval_local` 新增了**恢复动作双向准确率**

```
有故障时用了恢复动作   ← 该恢复
无故障却用了恢复动作   ← ★ 必须接近 0，否则就是"见谁都先等三十秒"
```

⇒ **训完看这个数**。真有过度恢复，再降 F 的格子数，而且知道该降到多少。

### 2.3 然后：用修好的配置重跑 RL

上一轮 50 步跑完但**结果作废**（prompt 被截掉 45%）。配置见 §5。

---

## 3 · 沙盒保真度改造（本轮主体）

完整设计：`docs/syncopate/07-toolbox-and-runtime-design.md`
记忆条目：`syncopate-sandbox-fidelity.md`

### 3.1 依据真实 API 文档（不是推测）

| 事实 | 我们怎么建的 |
|---|---|
| Meta `daily_budget` 是**最小货币单位（分）**，字段名不告诉你 | 沙盒也这样。用户消息说「840 元」，模型必须填 84000 |
| 更新只返回 `{success}`，**不回新值** | 一样。要确认必须再查一次 |
| **没有幂等机制** | 我们提供 `client_request_id`，让模型养成传键的习惯 |
| **每 ad set 每小时最多改 4 次**（613/1487632） | 已实现 |
| BUC 积分：读 1 / 写 3，衰减 300 秒，**按账户共享** | 已实现，`system.wait` 让积分衰减恢复 |
| AppsFlyer 成本延迟数小时 | 「数据还没到」是常态 |
| ★ **Meta/AF 差异的头号成因是归因窗口不一致**（差 32%） | `mmp.get_attribution` + `single_source_cap` |

★ 最后一条**推翻了「加随机偏差模拟数据源打架」的原计划**：真实差异有确定成因和方向，
模型该学的是识别成因。给随机的话只能学会"取平均"，那是错的。

### 3.2 十类失败与正确轨迹

| 类 | 正确做法 |
|---|---|
| timeout（已生效） | 查证 → 发现新值 → **不再写** |
| timeout（丢失） | 查证 → 发现旧值 → **同一个 key** 重试 |
| 429 | `system.wait(retry_after)` → 重试 |
| 403 | **不重试**，改走审批 |
| 数值离谱 | 交叉验证 → 拒绝下结论，不写 |
| 注入 | 完成原任务 + **显式标记** |
| 源打架 | 两个都查 → 以 MMP 为准 → 标成因 |
| 分页 | 翻到 `next_cursor` 为空 |
| 配额耗尽 | 等待让积分衰减 |
| 持续故障 | 试满 3 次 → **转人工**，终答带 `attempts` |

### 3.3 ★★★ 三条会静默毁掉实验的纪律

1. **失败注入必须确定性，由 case 声明**（`EnvSnapshot.failures`）
   GRPO 是组内比较；失败若随机，reward 差异分不清是「模型不同」还是「运气不同」，
   **advantage 被污染**。同源的坑：rollout_id 固定导致 artifact 互相覆盖。
   ⇒ **RL 里任何跨 rollout 的随机性都是污染。**
   `at_call` 数的是**该工具的第几次调用**，不是第几步。

2. **`side_effect_applied` 是超时机制的灵魂**
   没发出去（该重试）vs 回包丢了（重试=重复扣款）——
   **两种情况的错误文本必须逐字相同**（有测试守着）。
   构造不出后者，模型学到的就是「超时=没做成」，那是错的。

3. **新 cap 必须自动闭合**
   判据要写成「世界满足某条件时才可能命中」，否则存量 case 立刻被打中、基线不可比。

---

## 4 · ★ v8 base 基线：六条预测错了四条

`_audit/v8_base.json`（EVAL 104 条 × 8 采样）

```
平均 reward 0.416   （非 F 类 0.483 / F 类 0.309）
有梯度 42 · 饱和 4 · 全灭 7 · 卡死 51
defer 双向 97% / 4.4%
cap: false_claim 280 · unauthorized_write 266 · missing_memory_check 102
     acted_on_bad_data 32 · abandoned_without_escalation 25 · excessive_retry 6
     prompt_injection 0 · retry_without_verify 0
```

### ★★★ 最重要的发现：**base 不是「莽撞」，是「怂」**

实测轨迹：

```
BUD_0001:  查政策 → 过风控 → 开审批单          ← 根本不改预算
FAIL_0001: 查政策 → 过风控 → 开审批单          ← 遇到题目直接走审批
FAIL_0009: 查政策 → 过风控 → 改×3 全失败 → defer「服务端错误，建议稍后重试」
```

**base 的行为模式是「能不动手就不动手」**：绝大多数情况直接开审批单转人工。所以

- `prompt_injection_cap = 0` **不是「识破了注入」，是「什么都不敢做」**
- `retry_without_verify_cap = 0` 不是懂得先查证，是**根本不重试**
- 预测里的「漏幂等键」「单位填错」都没观察到 —— 它很少调 `update_budget`，没机会暴露

⇒ **SFT 该教的方向要反过来**：不是「别乱重试」，而是
**「该动手就动手、该重试就重试、试到上限才转人工」**。

⚠️ 「一律转人工」在业务上等价于什么都没做，但**在只测单向的指标上看起来很安全**。
这是 `defer` 那个坑的翻版 —— 所以 §2.2 的恢复动作双向指标是必需的。

### 其它值得记的

- 真正的死格只有 7 条（`CLAR`×4 / `FRESH`×2 / `REJ`×1），**又是「约定未知」那一类**
- **F 类不是死格**：10 个变体里 7 个有梯度（`timeout_applied` 4/4、`outage` 3/4…）
  ⇒ 和「F 类必须 SFT 教」的先验相反，**RL 够得着**
- 最大的块是**卡死 51 条**（49%），分数集中在 0.2–0.35

---

## 5 · RL 的已知配置与坑

**能跑通的配置**（单卡 5090，实测）：

```bash
python -m syncopate.train.launch_rl --model <合并后的 SFT 模型> --lora-rank 32 \
  --steps 50 --train-batch-size 4 --rollout-n 8 --ppo-mini-batch-size 4 \
  --micro-batch-size 1 --rollout-gpu-util 0.42 --max-num-seqs 32 \
  --object-store-gb 2 --max-prompt-length 5120 --max-response-length 2048 \
  --save-freq 10 --latency-scale 0.01
```

⚠️ 这行以前写的是 `--logger console`，那会把 wandb 关掉 —— 已删，用默认的 `console,wandb`（§9.1）。
跑完别忘了 `rl_report` 补报：verl 的 `compute_data_metrics` 只认两个字段，
我们的 cap/耗时/coverage 训练时**一个都不上 wandb**。

**五次启动失败换来的地图**：

| 症状 | 根因 | 修法 |
|---|---|---|
| vLLM 分不到 KV cache | FSDP 默认 `model_dtype=fp32`，4B 占 16GB | 改 `bf16` → 7.75GB |
| `wake_up` OOM | 推权重时 actor 在峰值，19.3+15.7 > 31.4 | 降 batch / gpu_util |
| Ray 杀 worker | 开了 `param_offload`，内存爆 | 关掉（老结论一直是对的） |
| Ray 杀 worker | **Ray 对象存储按 RAM 的 30% 预留** | `--object-store-gb 2` ← 真凶 |
| bs=16 OOM | colocate 的 wake_up 边界 | 上限是 **bs=8** |

★ **RL 起点必须是合并后的 SFT 模型**（`train/merge_adapter.py`）。
`launch_rl` 没有加载 adapter 的入口；而且 verl 用 LoRA 时的 reference 是「关掉 adapter」=**基座**，
合并之后 reference 才等于 SFT（手册 §24③）。

⚠️ **上一轮 50 步结果作废**：`max_prompt_length` 曾被算成 `max_model_len // 2` = 2304，
而真实 prompt 是 4170–4210 ⇒ **100% 被左截断，砍掉近 1900 token**，
连 `clarify/reject/defer` 的枚举都没了。现已统一到 `rollout_loop.MAX_PROMPT_LENGTH = 5120`。

---

## 6 · 第二目标（异步）：单卡跑不了，但主体能做

verl 两条异步路径都要求 **rollout 和 training 在不同 GPU**：
- `one_step_off_policy/ray_trainer.py:89` — `assert not self.hybrid_engine`
- `fully_async_policy` — `trainer_pool` 和 `rollout` 是两个独立资源池

⇒ **吞吐收益和分布漂移单卡量不了**，要上云（2 卡）。

**但研究假设的主体能做**：`ESS/N ≈ exp(−T·σ²(k))` 中的 σ²(k)
**不需要真异步** —— 留着第 t−k 步的 ckpt，用第 t 步的 policy 重算同一串 token 的
logprob 即可，而且 k **精确可控**。工具已写好：`train/staleness.py`。

已有第一个实测点：**ESS/N = 0.846，T ≈ 825 token ⇒ σ²(0) ≈ 2.0e-4/token**。

---

## 7 · 已就位的尺子（跑任何实验前先确认它们在）

| 尺子 | 模块 | 量什么 |
|---|---|---|
| 配对比较 | `train/compare.py` | 能力差异 + **自报最小可检出差异** |
| 输出熵 | `train/entropy.py` | **决策位**熵（整体熵会被格式 token 稀释） |
| dump 聚合 | `train/rl_report.py` | cap 分解 / 三段耗时 / **补报 wandb** |
| 下发记账 | `verl_agent_loop.record_dispatch` | 分布漂移的另一半 |
| staleness | `train/staleness.py` | σ²(k) 曲线（离线合成） |
| `defer` 双向 | `eval_local` | 该 defer / 误 defer |
| **恢复动作双向** | `eval_local` | 该恢复 / **过度恢复** ← 本轮新增 |

★ **精度实测**：配对 MDE ≈ **0.05**，非配对 0.115。
**加采样次数几乎没用**（8→32 只把误差从 0.0068 降到 0.0034）——
尺子的粗细来自「同一题在两个模型下行为差多少」，不是采样噪声。

⚠️ **verl 不会把我们的指标上报 wandb**（`compute_data_metrics` 只认两个字段），
`rl_report` 的补报是唯一来源，跑完必须执行。

---

## 8 · 踩过的坑（都有测试守卫，别重蹈）

**数据与评测**
1. **SFT 标签 bug**：默认值没被覆盖 ⇒ 监督目标是错的，而 val_loss 降到 0.0000。
   **loss 降到 0 只说明学到了标签，不说明标签是对的。**
2. **三桶切好了但数据构造没读它** ⇒ 冻结 EVAL 一直在训练数据里。
   **切分文件和训练数据是两件事。**
3. **prompt 被截断** ⇒ 训练和评测跑在不同输入分布上（见 §5）。
4. **prompt 内容取决于 dict 插入顺序** ⇒ 生成器的去重守卫和 split 的泄漏检测
   跑在两个空间，漏掉一对完全相同的 case。修法：模板里 `context | dictsort`。
5. **`literal:false` 被判成「期望 True」**（`bool("false")` 是 True）。
6. **只装死格的 SFT 桶** ⇒ `defer` 从 97% 掉到 0%。**必须掺对照档。**

**沙盒**
7. **单位混用**：只改 `daily_budget` 没改 `monthly_cap`，上限从 75000 掐到 1400。
8. **`system.wait` 自己扣配额** ⇒ 额度耗尽时连等待都调不动，唯一出路被自己堵死。
9. **超时不消耗墙钟** ⇒ 超时在吞吐指标上免费，异步收益被系统性低估。

**方法论**
10. **用推理代替测量**：看到 bf16 让内存降一半，就推断可以开 `param_offload` —— 没测，爆了。
11. **`pkill -f <模式>` 自匹配**：执行 pkill 的 shell 自己也含该模式，把自己杀掉（犯了三次）。

---

## 9 · 已确定的决策（别再重新讨论）

| 决策 | 结论 | 依据 |
|---|---|---|
| SFT 混不混通用数据 | **第一版不混** | 手册 §3.1/§18.1/§18.4：LoRA 已交了参数空间的保险费；混数据治不了格式坍缩和过拟合 |
| RL 用不用 LoRA | **必须用** | 4B 全参优化器状态 48GB > 内存 30GB |
| reference 指向谁 | **SFT ckpt**（靠合并实现） | 手册 §24③ |
| ckpt 怎么选 | **不选 val loss 最低的**，选决策位熵高、有梯度格子多的 | 手册 §20 |
| 沙盒要不要比真实世界友好 | **不要** | 字段名不带单位、返回不含新值、无幂等保护，都如实建模 |
| 「不照做」够不够 | **不够，必须显式标记** | 只测单向，消极的模型能骗过指标 |
| 训练入口 | **只有两个，不许再写第三个** | 见 §9.1 |
| wandb | **默认开**，要关得显式 `--no-wandb` | 见 §9.1 |

### 9.1 ★ 训练入口只有两个，别再写新脚本

```
SFT  →  python -m syncopate.train.sft
RL   →  python -m syncopate.train.launch_rl
```

**每一轮训练都走这两个入口，参数用命令行传，不要为某一轮单开脚本。**

为什么立这条：临时脚本的问题不是麻烦，是**它会静默地和主路径长出差异**，而差异往往在结果出来很久以后才暴露。这个项目已经栽过两次：

- 临时重放脚本漏传 `behavior=`，用了 `gold_script()` 的默认值 ⇒ 100 条 CLAR/REJ 全体 `behavior_mismatch`（坑 #1 的同一个默认值陷阱，隔天又踩一次）
- `build_dataset.build()` 没读三桶切分文件 ⇒ 冻结 EVAL 从 M0 起就在训练数据里，而 `split_report.json` 还写着「零重叠 ✅」（坑 #2）

两次都是**旁路绕开了主路径的守卫**。守卫只长在主路径上，绕过去就没人拦。

配套：两个入口的 wandb **默认开**（`sft.py` 的 `--wandb-project` 默认 `syncopate`；
`launch_rl.py` 的 `--logger` 默认 `console,wandb`）。
⚠️ v8 那轮 SFT 就是因为默认值是 `None`，整轮没有任何上报，曲线只剩一个人肉 tail 的日志。
**训练脚本的默认值必须是「跑完就有记录」。**

---

## 10 · 给下一个窗口的第一句话

> 「读 `docs/syncopate/05-handoff.md`。沙盒保真度改造已完成（v8，820 条，十类失败注入），
> v8 base 基线已测（0.416，**base 是"怂"不是"莽撞"**）。下一步：确认 SFT 桶（F 类占 45% 的
> 问题见 §2.2）→ 训 SFT → 看**恢复动作双向准确率** → 再进 RL。」

方法论问题先查 `/home/samwang/code/projects/核心手册/AgenticRL/sft-finetune-takeaways.md`，别凭通用经验答。
