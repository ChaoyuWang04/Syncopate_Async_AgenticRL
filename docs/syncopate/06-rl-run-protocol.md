# Syncopate · 06 — RL 跑之前的预期与解读方案

> 写于 2026-08-11，第一次正式 RL 之前。
> **纪律：预期写在跑之前。** 跑完再解释曲线，人一定会为看到的任何形状编一个理由。

---

## 0 · 起点与设定

| 项 | 值 | 依据 |
|---|---|---|
| 起点 ckpt | `checkpoints/sft/v3_ctrl/**epoch1**` | 决策位熵 0.484（e2 只有 0.336）、有梯度 44 格（e2 只有 36）。手册 §20：训得越狠熵越低，GRPO 探索不动 |
| 训练池 | `data/rl/v3` 467 train / 67 val | 与冻结 EVAL 零重叠（SHA-256 实测） |
| 评测集 | 冻结 EVAL 64 条 | case_id 从 M0 起未变 |
| 长尾 | `latency_scale=0.01` | 480 秒压成 4.8 秒。**这一档跑不出异步的收益**，是刻意的：先验算法，再验调度 |
| TIS | `sequence`，阈值 2.0 | verl 默认；研究主线要对比的对象 |

---

## 1 · ★ 可证伪的预测（跑之前写死）

### 1.1 能力

| # | 预测 | 判定 |
|---|---|---|
| P1 | 涨的主要是 **CRE(0.30) / MISS(0.55) / BUD(0.73)**——低分且有梯度的那批 | 配对比较按模板分组 |
| P2 | **饱和的 22 格基本不动**（HIGH/LONG/FRESH/CLAR/REJ） | 组内 std ≈ 0 的那批 |
| P3 | 总体提升 **小于 SFT 那次的 +0.352**，量级在 0.05–0.15 | RL 是在 SFT 的地板上抬天花板，不是再抬一次地板 |

⚠️ **P3 小于最小可检出差异（配对 MDE ≈ 0.05）时，结论是「没测出」不是「没提升」。**

### 1.2 过程（比分数更早报警）

| # | 预测 | 崩了的样子 |
|---|---|---|
| P4 | `cap/*` 命中率**同步下降**，尤其 `unauthorized_write`、`missing_memory_check` | **reward 涨但 cap 不降 = reward hacking**，立即停 |
| P5 | 决策位熵从 0.484 缓慢下降 | **跌破 base 的 0.09** = 探索死了，停 |
| P6 | ESS/N 维持在 0.8 以上 | **跌破 0.3** = sequence-level TIS 失效，该换 token-level |
| P7 | `response_length` 不单调增长 | 暴涨 = 典型的长度 hacking |

### 1.3 效率

| # | 预测 |
|---|---|
| P8 | `head_of_line_ratio`（最慢/平均）≈ 1.3–1.6，和 bs=2 冒烟一致 |
| P9 | 分布漂移 `total_variation` **恒等于 0**——sync 有 barrier，构造上不可能漂移 |

**P9 是对照组**：只有先证明 sync 下是 0，async 下测到的非 0 才有意义。

---

## 2 · 停止条件（优先级高于分数曲线）

```
立即停：ESS/N < 0.3
       决策位熵 < 0.05
       reward 涨但 cap 不降（连续 3 步）
       grad_norm 突然跳两个数量级
正常停：跑满预定步数
```

⚠️ **不要因为 reward 曲线好看就继续。** 设计文档 §31.3 的回退条件优先。

---

## 3 · 跑完怎么查

```bash
python -m syncopate.train.rl_report   checkpoints/grpo/<run>          # cap/耗时/漂移，补报 wandb
python -m syncopate.train.eval_local  --adapter <ckpt> ...            # 冻结 EVAL 重评
python -m syncopate.train.compare     _audit/M1_ctrl_epoch1.json _audit/RL_<run>.json
python -m syncopate.train.entropy     --adapter <ckpt> --limit 24     # 决策位熵
```

`compare` 会自己打印最小可检出差异，**不要绕过它读均值**。

---

## 4 · ★★★ 一个必须写下来的限制：单卡跑不了真异步

verl 的两条异步路径都要求 **rollout 和 training 在不同的 GPU 上**：

- `one_step_off_policy/ray_trainer.py:89` — `assert not self.hybrid_engine`（显式禁止 colocate）
- `fully_async_policy` — `trainer_pool` 和 `config.rollout.n_gpus_per_node` 是两个独立资源池

本机只有一张 5090。**所以「真异步的吞吐收益」和「分布漂移」这两件事，单卡量不了**，要么上云（2 卡），要么换办法。

### 4.1 换的办法：staleness 可以离线合成，而且更干净

研究假设的核心量是

$$\mathrm{ESS}/N \approx \exp(-T\,\sigma^2(k))$$

要量 $\sigma^2(k)$，**并不需要真的异步**——异步只是产生 staleness 的一种方式。
直接做法：把第 $t-k$ 步 policy 生成的轨迹留着，用第 $t$ 步的 policy 重算 logprob，
得到 ratio 分布 → ESS。$k$ 由我们精确控制，而不是由调度随机决定。

| | 真异步 | 离线合成 |
|---|---|---|
| $k$ 的取值 | 随机、由调度决定 | **精确可控**，能扫 k=0,1,2,4,8 |
| 硬件 | ≥2 GPU | **单卡** |
| 能测 σ²(k) 曲线 | 能，但含混杂 | **能，且干净** |
| 能测吞吐/空转 | ✅ | ❌ |
| 能测分布漂移 | ✅ | ❌ |

⇒ **单卡先把 H1/H2 的核心曲线做出来**（这是论文的主体），
   吞吐和漂移等上云再补（那是工程侧的佐证，不是假设本身）。

我们已经有 σ²(k=0) 的第一个实测点：ESS/N = 0.846，T ≈ 825 token
⇒ $\sigma^2(0) \approx 2.0\times10^{-4}$/token。

---

## 5 · 已就位的尺子

| 尺子 | 模块 | 量什么 |
|---|---|---|
| 配对比较 | `train/compare.py` | 能力差异 + **自报最小可检出差异** |
| 输出熵 | `train/entropy.py` | 决策位熵（整体熵会被格式 token 稀释） |
| dump 聚合 | `train/rl_report.py` | cap 分解 / 三段耗时 / 补报 wandb |
| 下发记账 | `verl_agent_loop.record_dispatch` | 分布漂移的另一半 |
| TIS | verl `rollout_corr/*` | ESS、IS ratio 分布、超界比例 |

⚠️ **verl 不会把我们的指标上报 wandb**（`compute_data_metrics` 只认两个字段），
所以 `rl_report` 的补报是唯一来源，跑完必须执行，否则曲线上只有 verl 自带的那些。
