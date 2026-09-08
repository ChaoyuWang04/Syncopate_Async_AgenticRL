# infra-probes/B12 · 异步收益与陈旧代价

状态：Q05暂停排期，保留准入调查与指标方案；真实训练画像显示重叠机会后才重开；未运行CPU/GPU实验，尚无收益/代价数字。2026-09-08。

## 问题与边界

同一资源摆放、同更新目标下，异步重叠省多少时间，又增加多少陈旧、积压、丢弃和更新敏感性？不预设一定加速，不以业务reward判断infra健康。指标唯一口径见[训练专题](../../03-TRAINING.md#4-async-与摆放)。

## 旧候选设置（仅参考，重开时按新主模型/资源范围重定）

优先使用同一个官方fully_async_main，比较staleness_threshold=0/0.5；固定trigger_parameter_sync_step=1、require_batches=1、partial_rollout=False及动态资源调度关闭。当时计划先选择受支持小dense公开base，真实FSDP2 trainer和vLLM各一张B200；本节是候选方案，模型revision/源码/预算/测量窗口未冻结，不构成GPU就绪。FSDP2 world_size1不能称为分布式分片认证。

官方[recipe](https://verl.readthedocs.io/en/latest/advance/fully_async.html)支持同入口同步/异步配置；[#6780](https://github.com/verl-project/verl/issues/6780)提醒不同入口的比较存在混杂，不能作为当前错误已复现。release/main身份来自本批共同上游核验，具体调用源码和当前版本行为仍待完整绑定，见[research.json](research.json)。

## 重开后按机制选择的证据

先证真实非零更新、权重交接和版本使用。同步臂应满足登记的零陈旧对照，异步臂若无实际重叠/陈旧如实记录。记录有效计算token/s、GPU秒、版本/秒龄尾分布、生成token守恒、丢弃/积压与同步等待。固定同评分后端分别量策略漂移与引擎残差，ESS/裁剪和同参数同数据梯度敏感性独立报告。时间包括统一终态drain；诊断重放不混入测速。后续开跑前登记数值闸、重复、统计区间与资源上限。

## 当前结论

调查和方案不等于实测；短程梯度差异不证明长期学习变差或统计梯度bias。没有checkpoint或云端产物。
