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
| 主线→infra | ⚠️ 采样器现在**排除上一批**（重复的根因是批边界错位，`docs/syncopate/18 §6` 更正）⇒ 新旧跑的采样序列**同 seed 也不再逐步可比**，严格重放对照别跨这条边 | 双方知悉 | — |
| 主线→infra | ⚠️ v13 SFT 数据重建过（131/503 条此前缺终答，`18 §12`）⇒ 臂对臂比较不受影响，但**下次 SFT 重训后基线会动，跨代比较别混用** | 双方知悉 | — |
| infra→主线 | P8 降级后的尾巴：`logprob_coverage` 有 ~0.1% 占位值（最低 0.9932），会污染那几条的 IS 权重 —— **归因无人认领** | **待认领**（引擎侧，建议 infra） | 找到占位值的来源并判定可否消除 |
| 双方 | 🆕 **CoT（thinking）训练支持立项**（Chaoyu 2026-08-20）：主线产**带思考的 SFT 数据**（E27 红利路径：拨开关净 −0.057 但探索格子 170→233）；infra 解训练侧（`SYNCOPATE_THINK` 目前 launch_rl 拦训练 · think 预算 8192 契约联动 · CoT 后 rollout 变慢，异步配比/陈旧度条件那时才真正出现）。任务细分挂 infra 01 §1 | 双方 | 设计先行，动代码前对齐口径 |
| 双方 | 🆕 **OPD 立项**（Chaoyu 08-20，`docs/syncopate/22 §J`）：把"说人话"的能力从**裸底座 Qwen3-4B**（与我们 SFT 同源）蒸回候选。O-1 探针已给绿灯：分歧集中在自然语言段（Δ=+1.91，58% 的 token \|Δ\|>3），**任务段几乎零漂（+0.01）**。主线出分布设计与 token 级 mask（分段器已写好），**训练侧实现可能要 infra**（on-policy：学生生成→老师在学生轨迹上给 token 级监督；只动 LoRA） | 双方 | 口径对齐后定分工；主线 01 §O |
| infra→主线 | ✅ **GPU0/GPU1 两个端点已于 08-20 12:47 关闭让卡**（Chaoyu 当面指令：主线侧测试已完，infra 起 4 卡采集）。按你们记的运维坑连 EngineCore 一起杀，显存已全部归还。要重起按 `docs/syncopate/09 §0` | 双方知悉 | 主线重起端点时删本行 |
| infra→主线 | 🆕 **serving 端点默认已加 `--kv-cache-dtype fp8`**（Chaoyu 08-21 拍板，`logs/runtime/start_vllm.sh` 已改）：KV 池 ×2 ⇒ 并发 +50%；质量 −0.009 恰在 MDE 界（E19 §8 五臂+归因）。压测口径注意：重起端点后 TTFT/TPOT 基线要按 fp8 KV 重记 | 主线知悉 | 主线确认后删本行 |
| infra→主线 | **「真压测」共建**（Chaoyu 08-20 JD 对齐，infra 01 §1-1）：infra 出压测框架与优化（SLO 画像/prefix cache/批调度/多 LoRA 热切换），主线出业务流量形状与验收口径。**主线侧交付已就位（08-20）**：驱动器 `scripts/runtime_loadtest.py`（分意图流量形状 + §19 全表判定）+ 首轮全量实测 24/25 达标（`docs/syncopate/11 §5`，含单流 TPOT/TTFT 基线）——可直接作为你们优化的 before 基线 | 双方 | infra 侧接手框架后各自记账 |
| 主线→infra | ⚠️ **评测单轮上限默认已改**：256 → `MAX_RESPONSE_LENGTH`（2048/think 8192，E23「评测跟训练」——256 会把崩塌型长轮截掉，评测在最需要诚实时不诚实）。你们注释里「off=256 逐字节不变」的冻结被此取代；**审计头部现在记录 `gen`（max_new_tokens/温度/组大小）与 `data_version`**（分片合并器此前把这些键丢了——e27 两臂 label 一模一样分不清的缺口已堵）。E27 若要续跑，跨代配对看 `gen` 字段 | infra 知悉 | — |

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
主线   A 路候选 cand_v13r2_e1/RL-100（+0.186）待晋级确认（22 §G-10）；
       B/F 路**已交付到人手上**：chatbox 在跑（会话+多轮+全量工具自选+档位推导），
       B-4 端点 + §19 压测 24/25（11 §5）。队首转向 **M 路多轮数据**（22 §J-8 有实测形状）；
       O 路 OPD 探针做完、范围缩一半，训练建议先观望（22 §J）
infra  PG+KL 已切库默认（08-20，cand 400 步全绿垫底）；token/seq 多种子与 R2 陈旧度
       按 Chaoyu 裁定撤销（ESS 中位 0.92 健康 · partial_ratio 恒 0 = 陈旧度条件不存在）；
       队首换成 CoT 训练支持（01 §1），复活条件都挂在它上面
```
