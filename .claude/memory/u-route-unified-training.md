---
name: u-route-unified-training
description: U 路（OPD+多轮+CoT）的 v14 历史终态与坑清单；当前状态只看 docs/syncopate/00-START.md 与 01-TASKS.md
metadata:
  type: project
---

> **历史记忆，不是当前队列。** 现行训练状态看 `docs/syncopate/04-TRAINING.md`，
> 当前任务只看 `docs/syncopate/01-TASKS.md`。

U 路（Chaoyu 08-28 融合裁定）=「数据合一、监督分家」的 v14 世代主线。**08-29 终态**：P0 ✅ P1 ✅（说人话 1.37 超底座）P2 ✅ v14.5 收官（六过一改期：四遍考场聚合 L1-iv 68.8/90 线改期 P3 出口·L1-oov 71.2 ✓·L2 78 ✓·盲评 1.460 vs P1 1.141·话术复读 66%→1%）；P3 首跑判定=任务分 +0.068 但**行为标签通道漂移**（defer 9/9→0/9、L2 78→52，机理=标签寄生自研壳弱通道+训练分布外无锚）不晋级 ⇒ **P3/P4 并入 [[v15-contract-refactor]] 的 R6/R5**。判定全史与教训在 `docs/archive/syncopate/pre-consolidation-v16/24-unified-conversation-training.md §4/§7`。

**v14 世代沉淀的方法论（v15 继续用）**：
- 数据：程序造事实·教师穿语言·判据把关；五闸（份额=监督token口径带宽·密度=收尾句/病句/distinct·OOV教学面·被判句泄漏·冻结）；对照对（判别行为的数据必须成对）；外部语料只走 模式A题库注入/模式B模式抽取（S1 题库 120/S2 句式库 42 模板/S3 revision 正则 14 条全在 data/u_route/ 可复用）。
- 评测：考场单遍方差实测 29pp ⇒ 四遍聚合口径；iv/oov 双词表测规则泛化vs记忆；机判首用必人核；判据必须负向认证「会红」；盲评闭卷带钥匙。
- 训练：SFT 默认四卡 DDP（手动 all_reduce+rank 权重一致断言+负向认证）·断点续训·epoch 谱选点 e1/1.5/2/2.5/3（老口径只评 e1/e2 在 v14 会错选）。
- RL 机理库：动态池 WEIGHT_FLOOR 保「再见到」但 GRPO 零方差保「见到也学不回」⇒ 起点越好无锚漂移越危险；fully_async save_freq 挂 param_version（16 步/版）非训练步；--test-freq 触发 verl 内置 validate 的翻倍断言 bug 勿开；守卫杀 launcher 必须连 ray stop --force 清集群（91 孤儿进程事故）；D 族连零门槛在新采样制度失配（真仪器=defer 题 ema_reward 从饱和位下跌）。

**接手人坑清单（沿用）**：eval_parallel 必须显式 MODEL=（SFT 贴裸基座/RL 贴 SFT 合并基座）；RL adapter 不许 merge（增量 0.05% 被 bf16 舍入淹没，保持 adapter 形态 serve=--enable-lora）；pkill -f 禁用（第四次自杀在案）；考场 worker 带 cost-cap；编辑运行中的 bash 脚本=字节偏移错乱（watch 换独立进程接管）；文本替换插方法可能拦腰截断 __init__（ast 抓不到，要结构断言）。

**dev mode 三模型栈**（Chaoyu 实测用，保持服务）：GPU0=RL-s12@8100·GPU1=base@8101·GPU2=sft-e3@8102；会话级模型锁定（conversations.model 列）+CoT 折叠（SYNCOPATE_RUNTIME_THINKING=1）+前端三段选择器。真人实测五发现（标签漂移/false_claim 空头支票/考卷越权盲区/summary 污染/CoT 触发压死）在 24 §4-P3。
