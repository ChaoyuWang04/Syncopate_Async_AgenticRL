---
name: rl-step-size-is-lr-times-steps
description: GRPO+AdamW 下 reward 只决定方向不决定距离，位移≈lr×步数；M7 实测模型只动了 0.0093%
metadata: 
  node_type: memory
  type: project
  originSessionId: 957ae9f2-2820-4a54-ab6d-75be32051e25
  modified: 2026-08-14T10:07:26.567Z
---

**2026-08-14 M7 实测：跑完 100 次更新后，`||ΔW||/||W|| = 0.0093%`**（万分之九）。
正常 LoRA 微调是 0.5%–5%，**小了两三个数量级**。

⇒ 冻结 EVAL 上配对差值 +0.011，而配对 MDE 是 **0.013** ⇒ 结论是「没测出」。
（⚠️ MDE **0.013 是配对的**；0.048 是不配对的。别引用错，差近四倍。）

**为什么 reward 再强也推不动——两层叠加：**

1. **GRPO 的 advantage 组内归一化**（减均值除标准差）⇒「答得特别好」和「稍微好一点」
   归一化后梯度**方向一样、幅度也一样**。
2. **AdamW 每步位移 ≈ lr**，与梯度大小无关（除以自身二阶矩）。

⇒ **位移 ≈ lr × 步数 × 方向一致性，reward 完全不参与。**
100 × 1e-6 = 1e-4，实测 9.3e-5 —— 几乎正好贴着这个上界。

★ **但方向是对的，而且模型极其敏感**：在 0.0093% 的位移下，`false_claim_cap`
就掉了 19%（130→105），同时 `unauthorized_write` +3、`unconfirmed_irreversible` +2、
两类基线上不存在的新违规出现。⇒ 杠杆在，只是几乎没用力；
**放大 lr 会同时放大好的和坏的方向**，下一跑必须带着刹车。

**下一步（未跑）**：只改一个变量 `lr 1e-6 → 1e-5`。理由：步数是钱、lr 免费；
grad_norm 整跑 0.011–0.06 稳得发闷，稳定性有余量。不稳就退到 5e-6。

相关：[[syncopate-project-framing]] [[feedback-measure-dont-infer]]
