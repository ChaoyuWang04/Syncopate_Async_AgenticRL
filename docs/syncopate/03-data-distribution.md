# 训练数据分布报告

> 数据源：`data/batches/v2`

```
共 580 条 case

==============================================================================
1 · 按业务意图
==============================================================================
意图                     条数     占比  链长 min/中位/max        骨架数  分布
budget_change         140  24.1%    4 /   6 /   7          12  ██████····················
anomaly_diagnosis     120  20.7%    2 /   5 /   7           9  █████·····················
creative_launch        90  15.5%    4 /   6 /   7           8  ████······················
clarify_boundary       50   8.6%    0 /   0 /   0           1  ██························
creative_upload        50   8.6%    3 /   3 /   3           1  ██························
portfolio_review       50   8.6%    6 /   6 /   7           2  ██························
reject_boundary        50   8.6%    0 /   0 /   0           1  ██························
metric_lookup          30   5.2%    1 /   1 /   1           1  █·························

==============================================================================
2 · 链长分布（每个意图内部是不是从短到长都有）
==============================================================================
意图                      0    1    2    3    4    5    6    7    8   ← gold 步数
anomaly_diagnosis       ·    ·   30    ·   11   28   38   13    ·
budget_change           ·    ·    ·    ·    6   45   60   29    ·
clarify_boundary       50    ·    ·    ·    ·    ·    ·    ·    ·
creative_launch         ·    ·    ·    ·    7   30   38   15    ·
creative_upload         ·    ·    ·   50    ·    ·    ·    ·    ·
metric_lookup           ·   30    ·    ·    ·    ·    ·    ·    ·
portfolio_review        ·    ·    ·    ·    ·    ·   26   24    ·
reject_boundary        50    ·    ·    ·    ·    ·    ·    ·    ·

  全局：{0: 100, 1: 30, 2: 30, 3: 50, 4: 24, 5: 103, 6: 162, 7: 81}
  ⚠️ 若某个意图只集中在 1-2 个长度上，说明它内部没有「由简到繁」的梯度，
     模型认出意图 = 知道要走几步，curriculum 也无从下手。

==============================================================================
3 · 骨架集中度（同一意图有几种走法）
==============================================================================
  budget_change        12 种骨架，最常见的占 17.1%
        24 条  list → get_metrics → search → get_budget_rule → check_account → create_case → write_prop
        22 条  get_metrics → search → get_budget_rule → check_account → create_case
        18 条  get_metrics → search → get_budget_rule → check_account → create_case → write_proposal
  anomaly_diagnosis     9 种骨架，最常见的占 25.0%
        30 条  get_metrics → get_safety_line
        15 条  get_metrics → detect_anomalies → search → get_optimization → get_optimization → write_pr
        14 条  list → get_metrics → detect_anomalies → search → get_optimization → get_optimization
  creative_launch       8 种骨架，最常见的占 17.8%
        16 条  get_metrics → search → get_safety_line → get_seasonal_context → search_similar
        16 条  get_metrics → search → get_safety_line → get_seasonal_context → search_similar → write_p
        15 条  list → get_metrics → search → get_safety_line → get_seasonal_context → search_similar
  clarify_boundary      1 种骨架，最常见的占 100.0%  ⚠️ 只有一种走法
        50 条  (不调工具)
  creative_upload       1 种骨架，最常见的占 100.0%  ⚠️ 只有一种走法
        50 条  get_metrics → upload → poll_review
  portfolio_review      2 种骨架，最常见的占 52.0%
        26 条  get_metrics → get_metrics → get_metrics → get_safety_line → detect_anomalies → get_optim
        24 条  list → get_metrics → get_metrics → get_metrics → get_safety_line → detect_anomalies → ge
  reject_boundary       1 种骨架，最常见的占 100.0%  ⚠️ 只有一种走法
        50 条  (不调工具)
  metric_lookup         1 种骨架，最常见的占 100.0%  ⚠️ 只有一种走法
        30 条  get_metrics

==============================================================================
4 · 工具调用频次（gold 里）
==============================================================================
  campaign.get_metrics              580  24.3%  ████████████████████
  memory.search                     320  13.4%  ███████████·········
  playbook.get_optimization         194   8.1%  ███████·············
  campaign.list                     182   7.6%  ██████··············
  benchmark.get_safety_line         170   7.1%  ██████··············
  memory.write_proposal             160   6.7%  ██████··············
  policy.get_budget_rule            140   5.9%  █████···············
  risk.check_account                140   5.9%  █████···············
  campaign.detect_anomalies         140   5.9%  █████···············
  calendar.get_seasonal_context      90   3.8%  ███·················
  approval.create_case               81   3.4%  ███·················
  creative.search_similar            62   2.6%  ██··················
  creative.upload                    50   2.1%  ██··················
  creative.poll_review               50   2.1%  ██··················
  campaign.update_budget             31   1.3%  █···················
  合计 2390 次调用，用到 15 个工具

==============================================================================
5 · 顶层行为 / 结局分布
==============================================================================
  expected_behavior: clarify=50(9%)  reject=50(9%)  tool_call=480(83%)
  结局(分支轴产物): -=260(45%)  block=62(11%)  default_plan=36(6%)  denied=28(5%)  escalated=81(14%)  executed=31(5%)  launch=28(5%)  switch_plan=54(9%)
  难度标签: L1=30(5%)  L2=50(9%)  L3=220(38%)  L4=230(40%)  L5=50(9%)

==============================================================================
6 · 控制轴取值分布
==============================================================================
  amount     above=158(27%)  below=211(36%)  boundary=211(36%)
  entry      id_given=298(51%)  must_discover=282(49%)
  mem        clean=232(40%)  repeated=232(40%)  risky=116(20%)
  outcome    block=62(19%)  default_plan=36(11%)  denied=28(9%)  escalated=81(25%)  executed=31(10%)  launch=28(9%)  switch_plan=54(17%)
  season     approaching=200(34%)  off=200(34%)  peak=180(31%)
  wrapup     memory_written=160(50%)  no_memory_write=160(50%)

==============================================================================
7 · train / val 切分 —— ★ 这一节决定怎么解读评测分数
==============================================================================
  train 507 条 / val 73 条（每 8 条取 1 条进 val）
  val 里**骨架在 train 中出现过**的: 73/73 (100%)

  ⚠️ **这意味着 val 上的高分只证明「模板内泛化」，不证明学会了业务。**
     val 和 train 来自同一批模板，只是实体 id / 数值 / 轴取值不同。
     模型只要认出模板，就知道该走哪条骨架——剩下的只是把参数填对。

     要测真正的泛化，需要 **hold out 整个模板或整条骨架**：
       · 留出某个意图完全不训（测跨意图迁移）
       · 留出某条骨架不训（测能否组合出没见过的路径）
       · 留出某个轴的某个取值不训（如只训 id_given，测 must_discover）
```
