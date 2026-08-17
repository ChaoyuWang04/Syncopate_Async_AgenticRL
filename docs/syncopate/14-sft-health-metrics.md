# Syncopate · 14 — 训练健康度：看哪些指标、红线在哪

> 写于 **2026-08-17**，v13 第一次 SFT 之前。
> **面板已经程序化建好了，不用手拖**：`python scripts/make_wandb_panels.py`
> W&B 项目 `spaemtuerl-northwestern-university/syncopate`。
>
> 分工：**这份说红线在哪、为什么**；`scripts/make_wandb_panels.py` 负责把该看的摆到屏幕上。
> 面板不判定 —— 红线数值会随基线变，而改文档比改面板便宜。

---

## 0 · 一句话：先看「训没训动」，再看 loss

M7 那次跑完 150 步、零错误、曲线好看，结论却是「**什么都没测出**」。
根因是一个不在任何常规面板上的数：

```
‖ΔW‖/‖W‖ = 0.0093%      正常 LoRA 微调是 0.5%–5%，小了两三个数量级
```

**loss 降了、指标好看、而权重几乎没动。** 所以面板第一屏放的是位移，不是 loss。

---

## 1 · SFT 要盯的四组

### ① 训没训动（第一屏）

| 指标 | 期望 | 🔴 红线 |
|---|---|---|
| `health/delta_w_ratio` **‖ΔW‖/‖W‖** | 训完在 **0.5%–5%** | **< 0.1%** ⇒ 白训（lr 太小或没接上）<br>**> 10%** ⇒ 训过头，熵会塌、GRPO 探索不动 |

> ⚠️⚠️ **这条指标的口径踩过一次坑，2026-08-17 当场修的**：
> 第一版算的是 `‖Δθ_trainable‖/‖θ_trainable(初始)‖`，而 **LoRA 的 B 矩阵初始化为零**
> ⇒ 分母里只有 A、B 从 0 长起来 ⇒ epoch 1 报 **15.99%**，看着像"训过头"要停车。
> **同一份权重按正确口径量出来是 0.485%** —— 尺子错了 33 倍，而且那个数
> **根本不能和 M7 的 0.0093% 比**。
> ⇒ 正确口径：**LoRA 实际叠加到基座上的增量比基座本身**，只对被适配的层算：
> `ΔW_eff = scaling · B @ A`，`ratio = ‖ΔW_eff‖_F / ‖W_base‖_F`。
> 命令行复算：`python -m syncopate.train.weight_shift --base models/Qwen3-4B --adapter <ckpt>`
> ★ 教训还是那条：**一条指标报红有两种可能 —— 数据不行，或者尺子不行。**
| `train/grad_norm` | 平稳，同一数量级内波动 | **突然跳两个数量级** ⇒ 停，查坏样本或 lr<br>**nan / inf** ⇒ 立刻停（flash-attn 反向坏掉就是这个症状） |
| `train/lr` | 按 warmup→cosine 走 | 形状不对说明 scheduler 没接上 |
| `health/skipped_micro_steps` | **恒为 0** | **> 0** ⇒ 有样本没有监督 token 被静默跳过。**这是本项目最贵的失效形状**（SFT 标签 bug 那次 val_loss 降到 0.0000 却什么都没学对） |

### ② 学得对不对

| 指标 | 期望 | 🔴 红线 |
|---|---|---|
| `val/loss` | 单调下降后走平 | **降到 ≈0** ⇒ 可疑，多半是标签 bug 不是学得好<br>**先降后升** ⇒ 过拟合，用更早的 epoch |
| `train/loss` vs `val/loss` | 差距小 | train 远低于 val ⇒ 过拟合 |
| `val/ppl` | 跟着 loss 走 | — |
| `val/loss_<分组>` | 各组都降 | **某一组不降甚至上升** ⇒ 那一类被牺牲了（M8 的 POL/CONF 就是这么发现桶里 0 条的） |

⚠️⚠️ **选哪个 epoch 的 ckpt，不看 val loss。**
手册 §20：训得越狠输出熵越低，接上 GRPO 就探索不动。
我们踩过一次 —— 选了 val loss 最低的那版，结果**零梯度格子 63%**。
⇒ 要选的是「格式学会了、行为还没定型」的那一版，判据是
**决策位熵高 + 有梯度格子多**，跑完用 `train/entropy.py` 和 `eval_local` 量。
**每个 epoch 都存了 adapter 就是为了这个。**

### ③ 吞吐与资源

| 指标 | v11 基线（单卡） | 用途 |
|---|---|---|
| `perf/supervised_tokens_per_sec` | — | 掉一半说明有东西退化了 |
| `perf/steps_per_sec` | 412 s/epoch | 对比历史 |
| `perf/peak_memory_gb` | **12.0 GB** | **> 28 GB** ⇒ 逼近 32 GB 卡上限，要降 max-length |
| `health/epoch_seconds` | ~412 s | — |

### ④ 跑完必须补的三件（不在 W&B 自动曲线里）

```bash
python -m syncopate.train.entropy    --adapter checkpoints/sft/v13/epoch1 --limit 24
python -m syncopate.train.eval_local --adapter checkpoints/sft/v13/epoch1 ...
python -m syncopate.train.compare    <基线审计> <新审计>
```

- **决策位熵**（不是整体熵 —— 整体熵会被格式 token 稀释）
- **有梯度格子数 / 饱和格子数** ⇒ 决定选哪个 epoch
- **读/写分桶**（§21：混在一起，大量读操作会稀释掉写操作的风险）

---

## 2 · RL 要盯的（面板第二个 view）

**停止条件优先级高于分数曲线** —— 这四条任一触发就停，不看 reward 好不好看：

| 判据 | 🔴 立即停 | W&B 键名 |
|---|---|---|
| ESS/N | **< 0.3** | `rollout_corr/rollout_is_eff_sample_size` ⚠️ **键名不叫 ess** |
| 决策位熵 | **< 0.05** | 控制台看不到，按步段聚合 `rollout_dumps/*.jsonl` |
| reward 涨但 cap 不降 | **连续 3 步** | 需要 `rl_report` 补报 |
| grad_norm | **跳两个数量级** | `actor/grad_norm` |

⚠️⚠️ **verl 不会把我们的指标上报 W&B**（它的 `compute_data_metrics` 只认两个字段）
⇒ **`rl_report` 的补报是 cap 分解的唯一来源，跑完必须执行**：

```bash
python -m syncopate.train.rl_report checkpoints/grpo/<run>
```

---

## 3 · 这次 SFT 的实际参数与为什么

```bash
python -m syncopate.train.sft --model models/Qwen3-4B \
  --train-file data/sft/v13/train.parquet --val-file data/sft/v13/val.parquet \
  --out checkpoints/sft/v13 --epochs 2 --batch-size 1 --grad-accum 4 \
  --lr 1e-4 --warmup-ratio 0.1 --lora-rank 32 --max-length 6656 \
  --wandb-project syncopate --wandb-run sft-v13
```

★ **`--max-length` 从 6144 提到 6656，是量出来的**：
v13 最长序列 **6300** token（v11 是 5806）。不提就会截断 ——
而「prompt 被截断 ⇒ 训练和评测跑在不同输入分布上」是记录在案的坑 #3。
变长的原因是这一版**有意加的三样**：终答字段说明、system prompt 的检索规则、
更长的 `context`/`multi` 风格题面。
⇒ **换数据版本必须重量最长序列**，文档里那句警告是对的。

⚠️ **别改 `--batch-size`**：实测 bs=1 反而最快，改了要同步改 `--grad-accum`。

---

## 4 · 面板怎么来的

```bash
python scripts/make_wandb_panels.py            # SFT + RL 两个 view
python scripts/make_wandb_panels.py --only sft
```

★ **面板是判据的一部分，所以进仓库而不是在网页上拖。**
手拖的面板换个人、换台机器就没了，而"该看哪几条线"要跟着代码走。

⚠️ 脚本里有一行垫片：`wandb-workspaces` 0.4.5 还在调 `wandb.util.generate_id`，
而 wandb 0.28 把它挪走了。换 wandb 大版本后报错先看那里。
