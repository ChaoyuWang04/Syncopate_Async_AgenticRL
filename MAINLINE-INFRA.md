# MAINLINE ⇄ INFRA — 两条线的唯一交互文档

> ## ⛔ 铁律（Chaoyu 2026-08-19 定）
>
> **两条线之间的一切往来只写在这一份文档里。禁止再写任何「信件」文档**
> （此前的 MAINLINE-HANDOFF / INFRA-REPLY / INFRA-TO-MAINLINE 系列已全部删除，
> 未闭合的事项都迁到了下面的表里）。
>
> 用法：
> - **只登记「还开着的事」**：方向 · 一句话 · 谁在打 · 判据/去处。
> - **办完就删行**：结论写进各自的权威文档（主线 `docs/syncopate/` · infra 的 E 报告），
>   本文不留历史 —— 要历史去 git log。
> - 长论证不进本文：写进权威文档，这里只留一行指针。
> - infra 只在乎**速度和正确性**，模型性能类的事不必抄送。

---

## 开着的事（办完删行）

| 方向 | 事项 | 谁在打 | 判据 / 去处 |
|---|---|---|---|
| infra→主线 | ℹ️ infra 文档已重组（对齐你们的 00/01 约定）：`infra_exp/{00-START,01-TASKS,02-DECISIONS,TRACKS}.md`，旧 ONBOARDING/00-INFRA-HANDOFF/TRACK-* 已删、E12 归档；**你们文档里指向旧名的指针已代为机械更新**（syncopate/00·01·18·22、archive/README、两份设计文档）——若有异议直接改回并在此留言 | — | 阅后可删本行 |
| infra→主线 | ✅ 「e26ab 两臂不同代码」已按你们给的方案二闭合：**Chaoyu 裁定不重跑**（采样器只改抽题，吞吐对比统计等价），前提已标注在 `E26 §6.6 口径说明二`；⛔ 正确性/学习类对比不跨那条代码边，**B5 两臂同代码跑** | — | 已闭合，主线阅后删本行 |
| 主线→infra | dump 已带 `case_id` + `failures_fp`（e26ab_off 起）⇒ 你们 §5.6 搁置的 **P4 组结构复查现在可做**（按 dump 文件号 × case_id） | infra | `check_pipeline_invariants --only rollout` 全绿 |
| 主线→infra | ⚠️ 采样器现在**排除上一批**（重复的根因是批边界错位，`docs/syncopate/18 §6` 更正）⇒ 新旧跑的采样序列**同 seed 也不再逐步可比**，严格重放对照别跨这条边 | 双方知悉 | — |
| 主线→infra | ⚠️ v13 SFT 数据重建过（131/503 条此前缺终答，`18 §12`）⇒ 臂对臂比较不受影响，但**下次 SFT 重训后基线会动，跨代比较别混用** | 双方知悉 | — |
| infra→主线 | P8 降级后的尾巴：`logprob_coverage` 有 ~0.1% 占位值（最低 0.9932），会污染那几条的 IS 权重 —— **归因无人认领** | **待认领**（引擎侧，建议 infra） | 找到占位值的来源并判定可否消除 |
| 主线→主线 | E23 的翻案条件挂账：**B-4 接上真模型服务后**，部署侧若硬性要求截尾采样（0.95/20），训练/评测/部署三方要重新对齐一次（`rollout_budget.py` 注释有全文） | 主线 | B-4 落地时核 |
| 主线→infra | ⚠️ **评测单轮上限默认已改**：256 → `MAX_RESPONSE_LENGTH`（2048/think 8192，E23「评测跟训练」——256 会把崩塌型长轮截掉，评测在最需要诚实时不诚实）。你们注释里「off=256 逐字节不变」的冻结被此取代；**审计头部现在记录 `gen`（max_new_tokens/温度/组大小）与 `data_version`**（分片合并器此前把这些键丢了——e27 两臂 label 一模一样分不清的缺口已堵）。E27 若要续跑，跨代配对看 `gen` 字段 | infra 知悉 | — |

| infra→主线 | ⚠️ **已删你们的 `_audit/v13_base.json`（+.done）**（Chaoyu 指示）：它是 256/轮上限下跑的裸基座 eval，截断 40.2%（其中 38.8% 撞轮数上限、parse_errors 909）——256 的砍断与真实弱分不开，不能当基线。**干净替身**：E27 A 臂将以 `--max-new-tokens 2048` 重产 `_audit/e27_base_off.json` 作为修复后管线的永久基线（logs/eval_v13_base/ 日志保留） | — | 阅后可删本行 |
| infra→主线 | 📊 **E27 thinking 三臂已定案**（E27 §5）：thinking 净 −0.057 但解锁探索空间（有梯度 170→233）；**红利路径 = 带思考的 SFT 数据**（B 反超 SFT 的题集中在 CHAT 判断类）—— 要不要立项归你们 | 主线 | 读 E27 §5 后此行可删 |
| infra→主线 | 🟠 **`fabricated_safety_line_cap` 建议升常驻观察**：两处独立信号汇合（E17 KL 臂 +2 · E27 SFT vs 基座 6→24）。compare 工具归你们 | 主线 | 加进常驻读数后删行 |
---

## ✅ infra 二次确认完毕（2026-08-19 晚）→ 可开 candidate 首跑

**最终命令（在拟用命令上改三处，其余照旧）**：

```
SYNCOPATE_PREFIX_GROUPER=1 \
launch_rl --model models/Qwen3-4B-sft-v13r2-e1 --lora-rank 32 \
  --mode fully_async --trainer-gpus 3 --rollout-gpus 1 \
  --micro-batch-size 8 --use-kl-loss False \
  --steps 400 --purpose candidate --experiment cand_v13r2_e1 \
  --save-path checkpoints/grpo/cand_v13r2_e1
# ① PG 开（Chaoyu 拍板）⇒ 必须配 --micro-batch-size 8（拟用命令漏了；mb=1 会"无组可分"）
# ② KL 关：--use-kl-loss False（E17 B 臂原样）
# ③ 其余走默认（lr 3e-5 · mini 6 · seed 1234 · sequence IS · bucket 512）
# 伴跑不变：rl_guard --kill · rl_ckpt_rolling_prune · SYNC_PAYLOAD/DDP_PROBE
# 预计步速 ~13 s/gstep（14.94 − ref 2.0），400 步 ≈ 1.5–2 h 步进 + 收尾
```

五项确认（原节已删，全文见 git log）：

| # | 结论 | 证据 |
|---|---|---|
| 1 | **PG 开 + mb=8**。B5 独立消融按 Chaoyu 裁定跳过，**由 candidate 晋级评测兜底**（若 candidate 不达标，PG-off 重跑是第一嫌疑，已登记 infra 02） | E26 §6.3–6.6（fp32 逐位等价 + 归约逐位同 + 四常驻判据） |
| 2 | **KL 关（`--use-kl-loss False`）**。⚠️ 你们的担心不成立：**判据③ `rollout_corr/kl` 不吃 ref** —— 它来自 rollout-IS 诊断（rollout logprob vs trainer 重算）。E17 B 臂（KL off）实证该指标出现 15 次、值在地板（3.7e-4 / 2.3e-4）⇒ **A1 三条腿全保** | logs/e17b_kl_off.log |
| 3 | **四个新默认值 infra 无暗依赖**：e26ab/e20h 显式钉了 mini/train-batch；吞吐脚本不吃 lr；max-turns 对自定义 loop 是 no-op（真上限 = per-case max_steps 经 extra_info 进 RolloutConfig，launch_rl:791 注释已核，verl 侧无消费路径） | grep 三个 run_*.sh |
| 4 | **分池接线已核**（launch_rl:1068 → main_ppo_pool:215）；rollouter 侧无已知批宽假设。判据 = 开跑 ~30 min 后 infra 跑 `check_pipeline_invariants --only rollout` + P4 case_id 复查（infra 认领） | — |
| 5 | **评测 256→2048 + gen 头：无异议**。E27 裸基座臂本来就 @2048 跑的，口径一致；gen 头正好堵了 label 撞名缺口 | — |

⚠️ 两条口径提醒：E26 的 14.94 s/gstep 是 **KL-on** 量的，KL off 后步速会更快，别把差异读成漂移；
PG/KL 的**库默认值今晚不动**（显式旗子跑）——candidate 过晋级评测后再切默认（"兜底必须是对的那个"要有这次的证据垫底）。

## 双方现状一句话（过期就改，不追加）

```
主线   B 路 runtime 收口，只剩要 GPU 的两件（B-4 模型端点 · 真压测）；A 路暂停等 infra；
       全管线起跑前判据已全部接上（22 §G），v13 SFT 数据已重建并过全部门禁
infra  E26 已定案：集成通（根因=绕过根 FSDP 的归约竞态，已修）+ A/B **生产→PG 2.31×**
       （采样器混淆经 Chaoyu 裁定不影响吞吐口径，注记在 E26 §6.6）；
       队首见 01-TASKS §1：B5 任务尺子（两臂同代码）→ KL 多种子 → token/seq 多种子
       （lr 1e-4 重测已按 Chaoyu 降级为可选上限基线）
```
