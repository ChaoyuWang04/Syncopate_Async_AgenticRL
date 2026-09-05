---
name: silent-degradation-fsdp-nosync
description: "三个 trainer rank 两个月没同步过梯度；上游只用一行 UserWarning 通知，四处信号我们一处都没接住"
metadata:
  node_type: memory
  type: feedback
---

**2026-08-18 的旧栈事故（完整记录：`docs/archive/infra_exp/legacy-4x5090/E21-ddp-not-syncing.md`）。**

## 事故本身

`fsdp_size=1`（本意"不分片、纯数据并行"）⇒ verl 造出 `(3,1)` 网格 ⇒ `HYBRID_SHARD`
⇒ **PyTorch 见分片维=1，自动降级成 `NO_SHARD`，却把梯度归约留在那个大小为 1 的组上**
⇒ **空操作** ⇒ 三个 rank 各训各的 LoRA，**每次更新只用 1/3 的数据**。

**而它不报错、不崩、loss 会降、所有指标正常。** 唯一的提示是一行 `UserWarning`，
内容还是「我换了个策略」——**没说"你的梯度不再同步了"**。

**判据为什么干净**：LoRA 的 `B` 是**零初始化**的 ⇒ step1 三个 rank 权重都是 `0.000000`
（起点相同），而梯度范数不同 ⇒ **只可能是没 all-reduce**。

## Why：为什么我们两个月没发现（四处信号，一处都没接住）

1. **那行 warning 每跑打两次，就在我们自己的日志里** —— 我们只盯"自己打的判据行"，
   **从没反过来读框架已经打出来的告警**。
2. **我们读了同一段源码，三句描述全对，第四句是推断 —— 错就错在那一句**
   （"取 1 ⇒ 只在 ddp 维 all-reduce 梯度 = DDP"）。
   ★ 推断夹在一串正确的事实中间、排版完全一样 ⇒ 没人会去质疑它。
3. **我们把那个推断变成了"看起来像实测"的数字**（"DDP 每步同步 260 MB"），
   在 5 份文档 6 处引用，还用它推出了别的结论（跨 socket 代价 0.004%、B11 的理由）。
   **而 260 MB 是算的（66M×4 字节），不是量的 —— 那段流量当时根本不存在。**
4. **E00 量了 all-reduce 带宽，我们当成了"训练确实在做 all-reduce"** ——
   **能力 ≠ 发生。**

## How to apply

1. **框架的 `UserWarning` 要当判据读**：训练起来后 grep 一遍，**新出现的必须有人看过**。
2. **注释里把"读码得出的事实"和"据此的推断"分开**，推断句标 `[推断，未验证]`。
3. **"每步同步 X MB"这类数字要有一次实测**，算出来的不许直接进结论表。
4. ★★ **写工具时把"我假设 X 成立"写成断言** —— 成本几乎为零。
   **这次就是被 `rl_ckpt_to_adapter.py` 里随手一句断言抓到的，而四处显式信号都没抓住。**
5. **重测时省力的判据**：同一批里两臂都受同样影响的 A/B，**比值仍可信；绝对值一律作废**。

相关：[[feedback-measure-dont-infer]] [[project-mechanism-not-wired]]
[[blank-thresholds-are-not-passes]] [[collective-alignment-cliff]]
