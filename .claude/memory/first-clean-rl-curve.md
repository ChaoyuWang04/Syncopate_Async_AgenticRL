---
name: first-clean-rl-curve
description: 正确系统第一条完整 RL 曲线(08-19):峰值在 200 步、过训回落;cap 单调恶化+总分涨 = reward 盲区显影法;选点按 cap 干净度
metadata: 
  node_type: memory
  type: project
  originSessionId: 140d7814-8829-5438-9195-f1451b4a03a1
  modified: 2026-08-19T19:53:57.544Z
---

`cand_v13r2_e1`(400 步,PG+KL-off)的六点冻结考场曲线:
**base 0.356 → SFT 0.711 → RL-100 0.897 → RL-200 0.902(峰) → RL-300 0.898 → RL-400 0.856**

**三条可迁移的教训**:

1. **池子饱和后继续训 = 冻结考场净伤害**。池 ~80 步吃透(覆盖 100%、零梯度贴顶),
   之后 200 步在残余梯度口袋里打转,考场分从峰值掉 0.046。
   「完成判据达成」之后的步数不是免费的。
2. ★ **reward 盲区显影法**:总分还在涨、而某条 cap 随步数**单调恶化**
   (`missing_safety_line` 4→60、`abandoned_without_escalation` 2→12→20→20)
   ⇒ 那正是 reward 没罚到的地方——**RL 会搬进去住**。这两条已登记为 v14 reward 的输入。
3. **统计并列的点按 cap 干净度定胜负**:RL-100 与 RL-200 差 0.005 < MDE 0.023,
   选了 abandoned_esc 更干净的 RL-100(且更早的点熵余量更大,利于后续轮次)。

**配套曲线**(都指向同一个故事):
- 决策位熵:SFT 0.196 → RL-100 0.028 → 200 **0.007** → 300 0.033 → 400 **0.075**
  ——先坍缩后**回升**:400 点的熵回升+质量下跌 = 旧路线与新捷径打架(变形不是收敛)
- 解法多样性(工具序列种数):390 → 83 → 83 → 60 → 59
- 卡死格子:12 → 54 → 61 → 64 → **76**(RL 收敛时自造;FAIL/BUD/SCALE 为主)
- 有梯度格子:277 → 61 → 41 → 44 → 30(RL 的钱 100 步内花完)

另:GEO 六点全程 0.245±0.005 纹丝不动 = 死格实锤(RL 结构上救不了,SFT 覆盖是唯一出路)。
⚠️ 工程附账:RL adapter 的 target_modules 曾被提取脚本写成容器名(mlp/self_attn)——
vLLM 按张量名装载从没暴露,PEFT 按模块名注入才炸;「两条路径共享一份产物,
只有一条真正校验它」又一例(已修)。
候选 = `checkpoints/grpo/cand_v13r2_e1/adapter_global_step_25`,配对 +0.186(t≈16)。

相关:[[budget-truncation-family]] [[gate-the-promotion-not-the-run]] [[rl-step-size-is-lr-times-steps]]
