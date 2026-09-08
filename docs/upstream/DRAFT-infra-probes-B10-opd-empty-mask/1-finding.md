# 提交包 · OPD全拒绝microbatch在指标归约处崩溃（infra-probes/B10）

> 状态：DRAFT，CPU02反例已复现并完成JSON SHA回读；未修生产代码，未发布issue/PR，未由upstream负责人接收。
> 目标仓库：verl-project/verl；类型：bug report。
> 版本：v0.9.0 `483b8a009ba3a97563edee3a19887e4862b8094a`；调查main `c8687a02e5b38172d6645cf120b71220f28cce88`。

## 一句话

真实rollout rejection允许把非空response_mask全部过滤成零；直接top-k OPD的指标读取没有空mask保护，会在loss/backward之前崩溃。即使其他microbatch/rank仍有有效token、global分母为正，空局部microbatch也受影响。这与已修复的#7200分母传递问题不同。

## 触发条件（最小配置）

```yaml
distillation.distillation_loss.loss_mode: forward_kl_topk
distillation.distillation_loss.use_policy_gradient: false
algorithm.rollout_correction.bypass_mode: false
algorithm.rollout_correction.rollout_rs: token_k2
algorithm.rollout_correction.rollout_rs_threshold: 0.1
```

构造合法旧/rollout logprob分别-1与-3，k2=2超过阈值，官方correction helper把四个有效token全部拒绝。old=rollout对照全部保留。本组合属于显式启用拒绝采样的路径，不宣称默认OPD一定遇到。另一已有线索是all-aborted输出，但本次不把历史#5894当成当前OPD all-aborted端到端证明。

## 根因（读码事实）

以下行号以v0.9.0 API读取源码为准。

| 位置 | 现状 |
|---|---|
| `verl/trainer/ppo/rollout_corr_helper.py` L836及L872、L1061 | 入口要求原mask至少一个有效token，拒绝之后没有要求仍非空，最终写回batch.response_mask |
| `verl/trainer/ppo/ray_trainer.py` L1622–1634、L1664 | rejection在actor更新前执行，随后调用_update_actor；该段没有全拒绝跳步 |
| `verl/trainer/distillation/losses.py` L352–361 | 对mask选择后的student/teacher mass直接min/max，空局部张量崩溃 |
| 同文件 `compute_distillation_loss_range` L109–120 | 同样对空mask直接min/max；仅修前一个指标仍不够 |
| `verl/workers/engine/fsdp/transformer_impl.py::forward_backward_batch` | 对loss_mask.sum做DP归约并填global token分母，随后拆microbatch，没有零分母跳步 |
| `verl/workers/engine/base.py::train_batch` | forward_backward_batch返回后直接optimizer_step；没有“监督token为零”的独立决策 |

最后两行是所查main源码事实，目标CPU未启动真实FSDP2 trainer。即便把指标保护并让空loss返回0，AdamW仍可能因weight decay/momentum改变参数；因此这不是安全的完整跳步修法。

## 判据

CPU02预注册并已验证：官方rejection helper正/负对照；其输出进入正式dispatcher；global token分母0和4两个场景必须捕获RuntimeError及student_mass.min的真实源码行。具体执行结果与证据以[REPORT](../../infra_exp/experiments/B10-OPD目标与归一化/REPORT.md)及其run索引为准，本文不复制运行数字。

## 证据与入口

`python -m syncopate.infra.opd_contract_probe --cpu --out DIR`，依赖正常安装verl和torch，不摘录AST、不mock官方函数。`opd-cpu.json`保存实际框架版本/源码SHA和两个异常栈，`empty-dispatcher-global-0.txt`与`empty-dispatcher-global-4.txt`保存完整traceback。根任务负责Volume持久化/回读索引。

## 本地修法与影响面

尚未修改生产实现。候选方案需分两层：局部零mask下指标的空集合处理与图连通的零loss；global有效token为零时由分布式一致的上层决策跳过optimizer和适用scheduler，而不是只用nan梯度防护。还需确认空microbatch是否应继续执行collective以免其他rank死锁。修法未经真实engine验证，不能作为现成补丁发布。

该问题是显式抛异常，不是静默错误更新。CPU最小复现成本很低；未启动GPU、未造成训练损失。零目标但非空mask（如同教师/学生）与零监督token不同，不主张所有零梯度都跳optimizer。

## 已验与未验

CPU01正式KL/mask/分母数学参考已通过；CPU02反例已复现并完成JSON SHA回读。尚未验真实Ray rollout、FSDP2跨rank、Megatron schedule、optimizer/scheduler全局跳步与修后恢复。本DRAFT只交付可达源码链和局部复现，不能冒充整套trainer回归。

## 背景调查

2026-09-08，查询`repo:verl-project/verl distillation empty in:title,body`命中13项；未见精确空OPD指标修法。#7200/#7225解决的是microbatch全局分母传递；#5894/#5899与#5859/#5860保护其他普通/debug metrics，不能据此认为OPD也已保护。open #7096更改标量指标materialization，其distillation patch仍保留student_mass.min等空归约，没有覆盖本问题。完整检索记录在[research.json](../../infra_exp/experiments/B10-OPD目标与归一化/research.json)。upstream负责人提交前仍需重查main与全面排重。
