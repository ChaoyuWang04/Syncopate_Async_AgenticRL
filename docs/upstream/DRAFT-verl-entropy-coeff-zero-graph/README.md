# DRAFT · verl：entropy_coeff=0 时 entropy 仍被连进损失图

> 状态：DRAFT（infra 已交发现；待考据成稿）。发现于 2026-08-27 E31 第 1 步冒烟首跑。

## 现象与代码位置

`verl/workers/utils/losses.py`（0.8.0 快照 L122-128）：

```python
if entropy is not None:
    entropy_loss = agg_loss(...)
    entropy_coeff = config.entropy_coeff
    policy_loss -= entropy_coeff * entropy_loss   # ← coeff==0 也无条件连图
```

只要模型输出里带 entropy（calculate_entropy 路径常开，因为要报 actor/entropy_loss 指标），
entropy 就被乘 0 连进 policy_loss —— **每步 update_actor 都会对 entropy 分支做一遍
数学上恒为零的反向传播**。

## 为什么值得修（三层）

1. **算力浪费**：非融合路径下 entropy = logsumexp − Σp·logits，对它反向要物化
   全词表尺寸（[T, V] fp32）的中间梯度缓冲——entropy_coeff=0 是 GRPO 系的常见配置，
   人人白付。（融合路径浪费较小；具体省多少未测，[推断，未验证]）
2. **autograd 契约意外**：自定义 autograd.Function 按惯例假设"未参与损失的输出收到
   None 梯度"（set_materialize_grads 语义）；这里收到的是**全零张量**。我们的
   MXFP8 投影 Function 因此在守卫上炸掉一次真实训练（E31 冒烟首跑，已修为容忍零梯度）。
3. **修法一行且不伤指标**：`if entropy is not None and config.entropy_coeff != 0:`
   连损失；metric 行单独保留（现在 metric 也在这个 if 里，要拆开）。

## 证据

- 崩溃现场：logs/e31s1_smoke_0827.log（首跑版，git 历史）——backward 收到非 None 全零 dentropy
- 我们侧回归测试：tests/train/test_e31_step1.py::test_u3_zero_entropy_grad_is_legal
- 配置：actor_rollout_ref.actor.entropy_coeff=0（launch_rl 默认，GRPO 常规）

## 提交前待办（Claude 考据）

- [ ] 上游 main 分支该文件现状（我们看的是 0.8.0 快照）
- [ ] 搜 verl issue 区有没有人报过 entropy 反向浪费
- [ ] 量一次非融合路径下的省时/省显存数字（有数才好卖）
