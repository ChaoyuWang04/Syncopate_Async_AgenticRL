# 动态 padding 会改变 verl 的 clip_ratio（B03）

状态：DRAFT。已在当前安装版本复现，本项目的门禁误读已修；未完成上游排重和干净 main 验收，未提交 issue/PR。
目标仓库：`verl-project/verl`。当前验证版本：verl 0.9。

## 一句话

同样两条生成，只改变补齐张量的宽度，`response_length/clip_ratio` 就会从 0 变为 0.5。它在这种输入下表示“长度等于张量宽度”，不能直接表示“生成撞到了配置中的 token 上限”。我们自己的门禁把两者当成一回事，误报了截断。

## 最小触发条件

实际 response 长度为 `[2,4]`，登记的生成预算为 12。分别把这批数据补齐到 4 和 12，调用真实 `compute_data_metrics`。脚本 [repro.py](repro.py) 只需要 CPU、PyTorch 和 verl，不需要模型、数据集或本项目模块。

```bash
python repro.py
```

| 相同实际长度 | padding 宽度 | 返回的最大长度 | 返回的 clip_ratio |
|---|---:|---:|---:|
| 2、4 | 4 | 4 | 0.5 |
| 2、4 | 12 | 4 | 0.0 |

脚本里的 12 是这次对照登记的预算，并没有伪装成上游函数支持的参数。正是因为函数只看输入张量宽度，调用方不能在动态 padding 时从这个字段推断实际生成截断。

## 根因与范围

安装版本的 `verl/trainer/ppo/metric_utils.py::compute_data_metrics`：

- 459–460 行从 prompts/responses 的最后一维取最大长度。
- 576–578 行将实际 response 长度与该宽度比较，平均后输出 clip_ratio。
- 592 行的 prompt clip_ratio 使用同类边界。

这是读码与 CPU 调用的事实，不证明所有 verl trainer 都使用动态 padding。在固定补齐到真实预算的调用路径中，两种边界可能一致；多轮工具、模板和模型 token 的长度口径还要单独区分。当前只实测了 response 的最小复现；prompt 的同类结论来自读码与本项目日志，没有另做完整测试矩阵。

## 实际工程后果

本项目一次真实双卡 RL 的两个 step 都报 clip_ratio=0.125，但 response 最大长度分别为 1,592 和 2,981，预算为 12,288。16 条原始 artifact 中的 82 次生成全为 `stop`，没有模型 token 裁剪或 token 预算耗尽，另有 3 条轮数耗尽。

本项目旧门禁因此把健康训练标成“长度截断 WARN”。训练没有因该质量 WARN 被中断，没有据此修改预算。不能据此否认历史另一批真实撞上预算的轨迹。

## 本地修法与验证

不改上游函数，不覆盖旧日志。`syncopate/train/rl_evidence.py` 从本轮原始结束原因、丢弃的 token 和轨迹原因计数；`rl_run_gate.py` 分别报告 token、轮数和观测限制。原 clip_ratio 仍保留，但不决定截断灯。

固定 sync 运行从已保存的启动配置核对 artifact 数量；缺覆盖或异步口径未接通就明确未测，不能报 0% PASS。token/mask/logprob 或预算不符仍为硬失败。真实质量异常保持 WARN，不因为修指标而被抹掉。

新门禁回归在旧实现上 13 failed / 3 passed，失败发生在判读行为；修复后相关 27 项在 Mac 和 Modal CPU 均通过。用同一批真实产物重验，得到 0 条 token 截断、3 条轮数耗尽，整体仍为 WARN，旧门禁 SHA 前后不变。

## 证据

- [B03 REPORT](../../../_audit/infra/B03/REPORT.md)：固定源码、真实运行身份、派生检查及所有证据路径。
- [上游函数原文与匿名轨迹长度](../../../_audit/infra/B03/c_clipping_summary.json)。
- [真实 CPU 最小复现输出](../../../_audit/stack_probe/summary_2026-09-05_212237_exec_3c7a5d3e.json)。
- [门禁修前负例](../../../_audit/infra/B03/gate_regression_before.xml)、[修后回归](../../../_audit/infra/B03/gate_regression_after.xml)。
- [新门禁只读复验](../../../_audit/stack_probe/summary_2026-09-05_214355_exec_948c7901.json)。

这些是内部工程证据；对外材料需去除私有路径和实验身份后另做整理，本文件不直接提交。

## 没验过什么

- 没有在干净上游 main 上跑本复现或提交补丁。
- 没有覆盖所有 V1/旧 trainer、异步、abort/resume 和多轮工具长度定义。
- 没有证明这是新的、无人报告过的问题；没有提出已验收的通用上游 API 修法。
- 这只是测量与本地门禁修复，不产生模型质量提升或训练加速结论。

2026-09-05 只做了初步搜索，并读到 [上游 main 的同类计算](https://github.com/verl-project/verl/blob/main/verl/trainer/ppo/metric_utils.py)；没有固定该页面的 commit，也没有做完 issue/PR 排重。下一位处理上游材料时须重新核验，不继承“上游仍有 bug”的最终结论。
