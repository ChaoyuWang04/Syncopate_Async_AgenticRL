---
name: budget-truncation-family
description: 截断家族一天三例：预算/上限没从契约派生 + 截断不报错不计数；★v13 SFT 26% 样本没终答（轮数上限默认 8）
metadata: 
  node_type: memory
  type: project
  originSessionId: 140d7814-8829-5438-9195-f1451b4a03a1
  modified: 2026-08-19T12:57:31.316Z
---

**同一个失效形状,2026-08-19 一天数出三例**(第三例是当天新抓的):

```
① RL prompt 左截断     3584 预算 ⇒ 100% 截掉「可以拒绝」⇒ defer 崩塌错误归因链（已翻案）
② SFT --max-length     默认 4096 + 静默 [:max_length] 切片；v13 92.6% 超限，
                       实跑手传 6656 侥幸躲过（潜伏未发作）
③ SFT 轮数上限         build_dataset 没跟 case.max_steps，落在默认 8
                       ⇒ ★ v13 有 131/503 条（26%）gold 回放被掐断，
                         **最终结论从没进过训练数据** —— 模型学的是「调满 8 次工具然后停」
                       ⚠️ 与 A-5「GEO 类卡死（打转不下结论）」行为形状一致，重训后要对一下
```

**共同解剖**:预算/上限存在**第二份副本**(默认值),且截断发生时**不报错、不计数、不留痕**。
③ 的发现路径值得记:是给 Q5/Q6 写探针时**首跑就炸**,按「判据异常先怀疑探针」解剖了一条
才确认是数据的病——**探针的第一次失败往往就是它最有价值的一次**。

**Why**: 截断砍掉的永远是「模型最该看见/最该学」的部分(选项枚举、终答),
而它对指标的伤害是**间接且自洽的**(loss 正常降、评测能跑),不会自己喊。

**How to apply**:
- 任何长度/轮数/预算类参数,唯一来源是契约(`rollout_budget.py` / `case.max_steps`);
  脚本或函数里出现同类默认值 = 事故预备役。守则⑨:该一致的值,这里根本不该有。
- 截断的出口只有两种合法写法:**硬报错**(SFT 构造/训练)或**计数+判据钉 0**
  (RL 的 `clip_ratio`,已进 `rl_guard` P 族,非零停机)。
- 已落的守卫:`check_pipeline_invariants --only contract data`(源码+数据两头)、
  `sft_replay` 回放 truncated 直接 raise、`probe_sft_rl_consistency.py`(Q5/Q6)。
- ⚠️ v13 RL prompt 实测最长 4654/5120,**余量只剩 466 token**——数据再长一点就撞①。

相关:[[behavior-collapse-check-input-first]] [[check-the-input-before-blaming-the-model]]
[[project-mechanism-not-wired]] [[blank-thresholds-are-not-passes]] [[incremental-rebuild-freeze]]
