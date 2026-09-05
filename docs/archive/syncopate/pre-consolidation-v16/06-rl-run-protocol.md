# Syncopate · 06 — 训练协议：自查清单 · 停机与完成 · 看哪些指标

> 📦 **历史快照，不代表当前训练协议。** 现行说明见
> [docs/syncopate/04-TRAINING.md](../../../syncopate/04-TRAINING.md)。

> **纪律：预期写在跑之前。** 跑完再解释曲线，人一定会为看到的任何形状编一个理由。
> 结构按生命周期排：**§1 起跑前后的自查清单 → §2 跑中的停机与完成判据 → H 部分 指标红线**。
> 2026-08-11 首跑预期与 M7 兑现在 `../archive/06-first-rl-expectations-2026-08-11.md`；
> 异步/离线合成的测量设计在 `23 §9.9`。原 14 号（健康度）= 本文 H 部分，14 空号不复用。

## 1 · ★★★ 训练前自查清单（每次起跑必过，Chaoyu 2026-08-19 立）

> **每一项都是「动作 + 证据」**：证据没出现就是没过，不许靠记得自己做过。
> 每条对应一次真实付过的学费（括号里是出处）。
> ⚠️ 登记型内容：加新项要带证据行；失效的项删掉，不许攒。

### 1.0 环境（30 秒）

```
□ set -a; . /workspace/.env; set +a         证据：echo $RAY_TMPDIR 非空（16G overlay 会爆）
□ bash scripts/infra/gpu_gate.sh                  证据：退出码 0（抢卡纪律；⛔ 别用 `;` 放行）
□ python scripts/tools/disk_report.py             证据：可用 ≥ 40 GB（收尾一次落 27 GB，M7 丢过 ckpt）
```

### 1.1 静态检查器（2 分钟，⚠️ 必须用项目 `.venv`）

```
□ python -m syncopate.pipeline.invariants
  证据：0 条违反（或只剩登记在案的遗留红）；⛔ 新出现的红一条都不许带着起跑
  覆盖：contract（脚本不许传契约参数）· data（RL prompt 同源+零截断 / SFT 终答收尾）·
        rollout（组重复零容忍 / 失败注入指纹唯一）· budget · rank · merge · audit
□ 若换过 flash-attn 轮子/机器：python scripts/infra/check_flash_attn_backward.py
  证据：反向数值通过（有轮子前向全对、反向全错，指标好看什么都没学）
```

### 1.2 数据（只在动过数据时）

```
□ 重建前 D 族 / 重切后 L 族：python -m syncopate.pipeline.data_gates --batch <批> [--split-dir <切>]
□ SFT↔RL 同构抽样：python scripts/v16/probe_sft_rl_consistency.py     证据：Q5/Q6 双绿
□ 增量重建冻结验证：case 集合逐一致 + 未涉样本逐 token 不变（13 §L2 / 22 §G-1 的做法）
□ 若加了 cap / 改了 schema：自动闭合验证（存量恒不命中）+ 21 号登记是否要作废旧基线
```

### 1.3 起 SFT

```
□ 数据是 2026-08-19 重建后的（旧 v13 缺终答 131 条）   证据：检查器 data 组绿
□ 启动日志出现「[契约] SFT 长度上限 = 7168」           证据：没有这行 = 没走契约
  （⚠️ 训练器没有采样均衡开关——Chaoyu 08-19 拆除；要调分布去调数据/分桶）
□ wandb 在报（默认开；关掉要显式 --no-wandb 且说得出为什么）
□ 跑完三件套：entropy + eval_local + compare（都带 --out，不落文件选点表恒显示"—"）
□ 选点定式（22 §G-7）：**e1 默认起点**；⛔ 不看 val_loss（模板家族重合，尺子失效）；
  <1 epoch 打点已废除（sft.py 硬过滤）；要再选就评 1–2 epoch 间三点 + e2，判据仍是熵+格子
```

### 1.4 起 RL

```
□ 起点目录里没有 lora_adapter/（launch_rl 会拦；有 = 拿了没合并的模型当起点）
□ 配对基线 = 起点模型自己的冻结 EVAL 审计（merged 形态；合并损失 −0.025 是实测的）
  ⛔ 别照抄历史审计路径 —— v3/M7 时代的审计对应的模型已不在本机，是历史记录不是对照
□ --purpose 声明了（candidate 受 ≥400 步 + 完成判据约束；probe 不受限）
□ 三条纪律：不用 lr 1e-4 出候选 · 不用训练分选 ckpt · 守卫必挂：
  bash scripts/tools/rl_guard.sh <日志> <ckpt目录> --kill
□ 长跑（save-freq × 步数 × 27GB > 剩余磁盘时）加挂滚动瘦身：
  bash scripts/tools/rl_ckpt_rolling_prune.sh <ckpt目录> &   # 只留最新全量，其余提 253MB adapter 后删
□ 契约参数一个都没传（检查器 contract 组守着；要 A/B 就写 CONTRACT-OVERRIDE 留痕）
```

### 1.5 起跑后 10 分钟内（判据行必须逐条出现，缺一行 = 那个机制没接上）

```
□ [pool] 动态分池启用：N 条 case（fully_async 下看 rollouter 进程那行，driver 的不算）
□ [agent-loop] 下发记账 ✓ 三类事件
□ [lora-probe] list_loras() 非空（step≥1）
□ [sync-payload] 第 2 次同步起 lora_ > 0（首次推基座是设计行为）
□ rollout_corr/kl 每次同步后回落 ~3.4e-4（不回落 = 权重没推过去，E22 的形状）
  ★ KL-off 也照常有效 —— 该指标来自 rollout-IS 诊断（rollout logprob vs trainer 重算），
  不吃 ref；E17 B 臂实证（15 次、地板 3.7e-4）
□ prompt_length/clip_ratio = 0.0000（非零守卫会停机；100% 截断翻过一整条归因链）
□ grep -c UserWarning <日志> —— 新出现的必须有人看过（E21 的告警躺了两个月）
□ W&B 两个 view 在收数（面板由 make_wandb_panels.py 建，不手拖）
```

### 1.6 跑完 · 晋级前（❗做完才算跑完，不是训练进程退出就算）

```
□ python -m syncopate.train.rl_report <run_dir> --wandb-run <id>
  （cap 分解/零梯度/池覆盖率的唯一来源；⚠️ run id 要去 W&B 查——这步最容易被忘）
□ python -m syncopate.train.pool_readout <run_dir> --export-triage _audit/triage/<run>/
□ python -m syncopate.train.candidate_gate <run_dir> --strict（晋级闸；probe 跑可跳过）
□ 冻结 EVAL 评测 + compare —— MDE 读它自己打印的数（配对与不配对差近四倍），别引用记忆
□ 检查器再过一遍（budget 组会对 overrides 与契约；这一跑会让"最近一跑"判据转绿或变红）
□ 27 GB 的 global_step_* 处理掉（dispatched.jsonl / rollout_dumps / 要过尺子的 ckpt 留）
```

---

## 2 · 跑中：停机（坏了）与完成（做完了），两族别混

> ⛔ 旧版停止条件只有四条（ESS/熵/reward-cap/grad_norm），**全在问「模型学得怎么样」**
> —— 于是两个基石级 bug（E21 梯度不同步 · E22 权重没推过去）静默跑完两轮训练，
> 一条线都没响。**缺的不是更好的阈值，是一整族「训练系统还活着吗」的判据。**

### 2.A · 系统还活着吗（每条都对应一次真实发生过、且完全静默的失效）

**四条全是「两个东西应当相同」型**：非黑即白、不需要阈值、不随基线漂移失效。

| # | 判据 | 从哪读 | 不满足说明 |
|---|---|---|---|
| **A1** | 每次权重同步后 `rollout_corr/kl` **回落到首步数量级** | 训练日志 | 权重没推给 rollout（E22 实测：第 3 个版本起不再回落，末尾 36×） |
| **A2** | 第 1 步各 rank 的 `lora_B` **梯度逐位相同** | `SYNCOPATE_DDP_PROBE=1` | 梯度没跨 rank 归约（E21）。前提：B 零初始化 ⇒ 起点必然一致 |
| **A3** | 训练/评测**契约五元组**相等 `(max_prompt, max_response, max_turns, top_p, top_k)` | `rollout_budget.py` + `overrides.yaml` | 评测测的不是训练那个策略（检查器 budget 组静态守住） |
| **A4** | **绝对有效条数** `N × ESS/N ≥ 24` | `rollout_is_eff_sample_size × batch` | 梯度已经是噪声 |

⚠️ A1/A2 只能运行时抓（产物上看不出来），A3 是静态判据 ⇒ 两套工具都要有（§2.D）。

### 2.B · 估计量还有效吗

```
🟡 ESS/N < 0.3                        ⇒ 不停机，逃生口：换 --rollout-is token（默认是 sequence）
🔴 fraction_low + fraction_high > 40%  ⇒ 停机（IS 修正退化成常数缩放 = 研究问题 H3 的正面症状）
```

**ESS 为什么从「停机线」降级成「换配置」——三条**：
① 它是比例，隐含大 batch 假设（上游 ~1024 条时 0.3=有效 307；我们 48 条时 0.3=**14 条**，
含义差 20 倍）⇒ 该看绝对条数（A4）；
② 上游原话是「consider switching to geometric aggregation」，我们曾误抄成「立即停」——
「何时该换口径」正是研究问题 H3，**别把研究问题的答案当训练的刹车**；
③ verl 报的 ESS 是 `clamp(0,2.0)` 后算的 ⇒ 系统性偏高，`fraction_high` 越大越乐观
⇒ B 族两条要一起看。

### 2.B.0 ⚠️⚠️ 引用本节任何 ESS 结论之前，先读这一段（2026-08-19）

`[实测 seqis_long120]` `partial_ratio` 30 个点全是 0.0 ⇒ **没有任何轨迹跨越过权重版本边界**
（trainer 一步远慢于 rollout，rollout 每次早早做完在等）⇒ π_rollout ≈ π_train ⇒ IS 近乎恒等。

```
① 「token 级与序列级任务分打平（+0.000）」≠ 两者一样好 —— 是这个条件下 IS 几乎没参与
② 「ESS 的作用没被观测到」≠ ESS 没用 —— 它从没面对过它该检测的那个条件
```

⇒ **我们至今没真正跑出过 fully_async 的陈旧度条件。** 陈旧度真起来之后
（长尾延迟 / rollout 变快 / sync_every 更大），本节结论**全部要重审**——那时序列级
可能真的塌，而那正是它会告诉我们的（token 级 ESS 恒 ≈0.999，是永远不响的警报器）。
★「没观测到 X 有用」和「X 没用」是两回事（空门槛同族）。

### 2.C · 模型学得怎么样

```
⚠️ 熵（actor/entropy）**只报不停**        [08-19 同日两次误杀后除名] 健康跑 e17a/b 下探到
                                       起点 31–42% 再回升；★真塌陷跑 e20f（defer→0%）
                                       全程 ≥0.0265 从没报警 ⇒ 对我们的失效模式
                                       **既误报也不真报**。塌陷硬停 = D 族（正是抓到 e20f 的）
🔴 grad_norm 突跳两个数量级             查坏样本或 lr（nan/inf 立刻停 = flash-attn 反向坏的症状）
🔴 reward 涨但 cap 不降（连续 3 步）     reward hacking 的指纹
🔴 defer 连零 ≥25 步（D 族）            拒绝能力塌陷，不可逆（组内 std=0 后再无梯度）
🔴 prompt clip_ratio > 0（P 族）        输入分布被污染，跑得越久废得越多
```

⚠️ **「reward 涨 cap 不降」没有自动停机者**（cap 数据只在 dump 里，守卫读不到）——
靠面板③人盯 + `rl_report` 事后复核。不要因为 reward 曲线好看就继续（设计 §31.3 回退优先）。

### 2.D · 判据由谁执行

```
check_pipeline_invariants   静态/离线    跑前跑后，查文档/产物/源码          A3
scripts/tools/rl_guard.sh         运行时       盯指标触发就停                     A1 A4 + 2.B + C/D/P 族
SYNCOPATE_DDP_PROBE=1       首步一次     比各 rank 梯度                     A2
```

⚠️ 此前守卫是跑时临时写的 shell、跑完就没，且只盯 2.C 那族 ——
**这是两个基石 bug 能静默跑完两轮的另一半原因**。守卫必须进仓库。

### 2.E 两个阈值：已用干净跑复核（2026-08-19）

`A4 的 24 条` 与 `2.B 的 40%` 原是按 batch=48 反填的工程值。复核（5120 干净跑
e17a/e17b）：ESS/N≈0.999 ⇒ 有效 ≈48（阈值的 2×）；`fraction` ~1e-4（阈值的千分之一下）。
⇒ **两条都是宽保险丝，维持不动**——职责是接住「估计量整个崩掉」，不是调参旋钮。
⚠️ 换 batch 构成（mini_batch × rollout_n ≠ 48）时 A4 按条数重算。

## 2.0 · ★★★ 完成判据：不是跑到步数，是跑到没东西可学（2026-08-19 立）

> **停机** = 坏了，继续跑在浪费或把错的训进权重；**完成** = 做完了，继续跑只是收益递减。
> 混在一起，「跑完了」和「崩了」会长得一样。

**两种用途，只有一种受约束**：`--purpose probe`（默认，不受任何约束）/
`--purpose candidate`（上线候选，受 ≥400 步 + 完成判据约束）。
★ 约束加在**晋级**不加在起跑——infra 的短实验一点不该被挡；忘了声明的后果是
**晋级时被 `run_purpose.json` 拦下**，不是靠记性。

**完成判据**：`零梯度率连续 3 个窗口没有创新高` ⇒ 没有新东西被学会，可以收工。
★ 零梯度率上升 = 越来越多的题被学会/饱和，**到顶才说明学不动了**；
判据是「不创新高」不是「低于某值」——那个值取决于数据里有多少死格，是数据的性质。
`[实测 e17a 60 步]` 轨迹 15%→52% 还在创新高、池只覆盖 22.7% ⇒ 60 步远没到头。
配套曲线：`syncopate/zero_grad_group_ratio` 与 `syncopate/pool_coverage`（面板④）。

**400 步是下限不是目标**：分池要几十轮才转起来（`WEIGHT_FLOOR=0.05` ≈ 每 20 轮体检一次）。
位移有余量（`22 §F`：位移 ∝ lr×√N，7 个 epoch 才漂到 SFT 一遍的量级）。

**跑完的三类格子出口完全不同**（清单 1.6 的 `pool_readout` 产出）：

```
饱和  分高、无方差    已经会了      ⇒ 降权，保留地板做回归体检
卡死  中间分、无方差  在里面打转    ⇒ 查缺工具还是缺信息；curriculum 只适用这一类
死格  分低、无方差    从没探索到    ⇒ RL 结构上救不了（只能强化出现过的行为），该由 SFT 覆盖
```

---

## 3 · 异步的测量设计（指针）

分卡是 verl 两条异步路径的硬前提（已打通，4×5090）。
⚠️ M7 实测 `partial_ratio = 0`、漂移 `total_variation = 0` —— **不是没测到漂移，  ⛔(21)
是那一跑里没有漂移可测**（轨迹太短，碰不到版本边界）。
⇒ σ²(k) 曲线用**离线合成**扫（k 精确可控）、吞吐与漂移用真异步补 ——
分工与做法见 **`23 §9.9`**，本文不再展开。

---

# H 部分 · 训练健康度：看哪些指标、红线在哪（原 14 号）

> 面板程序化建：`python -m syncopate.train.wandb_panels`（进仓库不手拖——
> "该看哪几条线"要跟着代码走）。W&B 项目 `spaemtuerl-northwestern-university/syncopate`。

## H0 · 一句话：先看「训没训动」，再看 loss

M7 跑完 150 步、零错误、曲线好看，结论却是「什么都没测出」——根因是一个不在任何
常规面板上的数：`‖ΔW‖/‖W‖ = 0.0093%`（正常 LoRA 0.5%–5%）。  ⛔(21)
**loss 降了、指标好看、而权重几乎没动。** 所以第一屏放位移，不放 loss。

## H1 · SFT 要盯的四组

### ① 训没训动（第一屏）

| 指标 | 期望 | 🔴 红线 |
|---|---|---|
| `health/delta_w_ratio` ‖ΔW‖/‖W‖ | 训完 **0.5%–5%** | <0.1% 白训 · >10% 训过头（熵塌，GRPO 探索不动） |
| `train/grad_norm` | 平稳同数量级 | 跳两个数量级 ⇒ 停；nan/inf 立刻停（flash-attn 反向坏的症状） |
| `train/lr` | warmup→cosine | 形状不对 = scheduler 没接上 |
| `health/skipped_micro_steps` | **恒 0** | >0 ⇒ 有样本被静默跳过（本项目最贵的失效形状） |

> ⚠️⚠️ **位移口径踩过一次坑**：第一版算 `‖Δθ_trainable‖/‖θ_trainable(初始)‖`，而 LoRA 的 B
> 零初始化 ⇒ epoch1 报 15.99% 看着像训过头 —— **同一份权重按正确口径是 0.485%，尺子错 33 倍**，
> 且那个数不能和 M7 的 0.0093% 比。  ⛔(21)
> 正确口径：`ΔW_eff = scaling·B@A`，`ratio = ‖ΔW_eff‖_F/‖W_base‖_F`，只对被适配层算。
> 复算：`python -m syncopate.train.weight_shift --base models/Qwen3-4B --adapter <ckpt>`
> ★ 教训：**一条指标报红有两种可能 —— 数据不行，或者尺子不行。**

### ② 学得对不对

| 指标 | 期望 | 🔴 红线 |
|---|---|---|
| `val/loss` | 降后走平 | **降到 ≈0 可疑，多半是标签 bug**；先降后升 = 过拟合 |
| `val/loss_<分组>` | 各组都降 | 某组不降/上升 ⇒ 那一类被牺牲了（POL/CONF 0 条就是这么发现的） |

⚠️⚠️ **选 ckpt 不看 val loss**（v13 起额外失效：val 模板家族 100% 在 train 里，`18 §7`）。
判据 = **决策位熵高 + 有梯度格子多**，选「格式学会了、行为还没定型」那版。
踩过一次：选了 val loss 最低那版 ⇒ 零梯度格子 63%。

★ **零梯度格子分三类别混看**（`select_sft_ckpt.py` 四类全打、和 = 总数可对账）：

```
饱和  分高 >0.9   已经会了，不是问题        死格  分低 <0.15  RL 够不着，该由 SFT 覆盖解
卡死  中间分      在里面打转（GEO 一类）
[实测 v13]  e1 有梯度 222·饱和 96·卡死 22·死格 3（零梯度 35.3%） · e2 零梯度 55.1%
```

⚠️ M6 毕业条件问「零梯度 <30%」——这个数必须直接打在屏幕上（判据不在屏幕上=没接上）。

### ③ 吞吐与资源

`perf/peak_memory_gb` **>28 GB** 要警惕（32 GB 卡上限）；`perf/steps_per_sec` 对比历史
（v11 单卡基线 412 s/epoch）。

### ④ 跑完必须补的三件（不在 W&B 自动曲线里）

见清单 **§1.3 末三项 / §1.6** —— 决策位熵（不是整体熵，会被格式 token 稀释）·
有梯度/饱和格子数 · 读/写分桶。

## H2 · RL 的面板键名对照（判据权威在 §2，这里只放"看哪条线"）

| 族 | 判据 | W&B / 来源 |
|---|---|---|
| A | 同步后 kl 回落地板 | `rollout_corr/kl` vs `syncopate/kl_floor`（⚠️ `rollout_corr/*` 只在 `bypass_mode=False` 下产出） |
| A | 跨 rank 梯度同 | `SYNCOPATE_DDP_PROBE=1` 看日志 |
| A | 绝对有效条数 ≥24 | `syncopate/effective_seqs`（`rl_report` 补报） |
| B | ESS/N（⚠️ 键名不叫 ess） | `rollout_corr/rollout_is_eff_sample_size` |
| B | 截到界比例 | `rollout_corr/rollout_is_ratio_fraction_{low,high}`（ESS 是 clamp 后算的，两条一起看） |
| C | 决策位熵 | 控制台看不到，聚合 `rollout_dumps/*.jsonl` |
| D | defer/reject 连零 | `syncopate/defer_count` `syncopate/reject_count`（`rl_report` 补报；守卫用 defer_watch） |
| P | prompt 零截断 | `prompt_length/clip_ratio` 必须 0.0000 |
| 完成 | 零梯度率 / 池覆盖率 | `syncopate/zero_grad_group_ratio` `syncopate/pool_coverage`（`rl_report` 补报） |

⚠️⚠️ **verl 不会把我们的指标上报 W&B**（`compute_data_metrics` 只认两个字段）
⇒ `rl_report` 补报是 `syncopate/*` 全系的唯一来源，跑完必须执行。

## H3 · SFT 的实际参数与为什么

```bash
python -m syncopate.train.sft --model models/Qwen3-4B \
  --train-file data/sft/v13/train.parquet --val-file data/sft/v13/val.parquet \
  --out checkpoints/sft/v13 --epochs 2 --batch-size 1 --grad-accum 4 \
  --lr 1e-4 --warmup-ratio 0.1 --lora-rank 32 --wandb-run sft-v13
```

⚠️ **2026-08-19 起没有 `--max-length` 了**：上限从 `rollout_budget` 推（7168），超长硬报错
（`08 §4.1`；"对值的副本"正是下一次漂移的来源，守则⑨）。
⚠️ 别改 `--batch-size`：实测 bs=1 最快，改了要同步改 `--grad-accum`。

## H4 · 面板脚本

`python -m syncopate.train.wandb_panels`（`--only sft` 可选）。
⚠️ 垫片：`wandb-workspaces` 0.4.5 还在调 `wandb.util.generate_id`（wandb 0.28 挪走了），
换 wandb 大版本报错先看那里。
