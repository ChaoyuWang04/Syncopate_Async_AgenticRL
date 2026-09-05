"""一条命令把 SFT / RL 的观测面板建到 W&B 上（可重跑，幂等覆盖同名 view）。

    python -m syncopate.train.wandb_panels            # 建 SFT + RL 两个 view
    python -m syncopate.train.wandb_panels --only sft

★ 为什么要脚本而不是在网页上拖：**面板是判据的一部分**。
手拖的面板换个人、换台机器就没了，而"该看哪几条线、红线在哪"是要跟着仓库走的。
这份脚本和 `docs/syncopate/04-TRAINING.md` 是同一件事的两种形态 ——
文档说为什么，脚本让它出现在屏幕上。

⚠️ 面板只负责**把该看的摆出来**，不负责判定。红线的具体数值在文档里，
因为它们会随基线变，而面板改起来比文档贵。
"""

from __future__ import annotations

import argparse
import os

# ⚠️ 垫片：`wandb-workspaces` 0.4.5 还在调 `wandb.util.generate_id`，
# 而 wandb 0.28 把它挪到了 `wandb.sdk.lib.runid`。不补的话 `save()` 直接 AttributeError。
# ⇒ 记在这：换 wandb 大版本后这行可能就不需要了，报错了先看这里。
import wandb
import wandb.sdk.lib.runid as _runid

if not hasattr(wandb.util, "generate_id"):
    wandb.util.generate_id = _runid.generate_id  # type: ignore[attr-defined]

import wandb_workspaces.workspaces as ws
import wandb_workspaces.reports.v2 as wr

ENTITY = os.environ.get("WANDB_ENTITY", "spaemtuerl-northwestern-university")
PROJECT = os.environ.get("WANDB_PROJECT", "syncopate")


def line(title: str, keys: list[str], *, log_y: bool = False) -> wr.LinePlot:
    return wr.LinePlot(title=title, x="Step", y=keys, log_y=log_y, smoothing_factor=0.0)


SFT_SECTIONS = [
    # ★ 第一屏放"训没训动"，不是放 loss —— M7 的教训：loss 好看不代表权重动了。
    ws.Section(
        name="① 训没训动（先看这一屏）",
        panels=[
            line("‖ΔW‖/‖W‖ 位移（正常 LoRA 0.5%–5%）", ["health/delta_w_ratio"]),
            line("grad_norm（突然跳两个数量级 = 停）", ["train/grad_norm"]),
            line("学习率", ["train/lr"]),
            line("被静默跳过的 micro-step（应恒为 0）", ["health/skipped_micro_steps"]),
        ],
        is_open=True,
    ),
    ws.Section(
        name="② 学得对不对",
        panels=[
            line("train / val loss", ["train/loss", "val/loss"]),
            line("val perplexity", ["val/ppl"]),
            line("val loss 按分组（看有没有某一类被牺牲）",
                 ["val/loss_all"]),
        ],
        is_open=True,
    ),
    ws.Section(
        name="③ 吞吐与资源",
        panels=[
            line("被监督 token/s", ["perf/supervised_tokens_per_sec"]),
            line("步/s", ["perf/steps_per_sec"]),
            line("显存峰值 GB", ["perf/peak_memory_gb"]),
            line("每 epoch 秒数", ["health/epoch_seconds"]),
        ],
    ),
]

RL_SECTIONS = [
    ws.Section(
        name="① 训没训动 + 停止条件（都优先于分数曲线）",
        panels=[
            # ★★★ M7 那次唯一真正解释了「什么都没测出」的数：位移只有 0.0093%。
            #     位移不动 = lr 太小；飙升 = lr 太大、熵会塌。**先看这条再看 reward。**
            line("‖ΔW‖/‖W‖ 位移（rl_report 补报）", ["syncopate/weight_shift"]),
            # ★★ 2026-08-18：ESS **不再是停机线**，降级成"该换聚合口径"的信号
            #   （上游原话是 "consider switching to geometric aggregation"，
            #     我们此前写成了"立即停"；见 06 §2.B）。
            line("ESS/N（<0.3 ⇒ 换 --rollout-is token，**不停机**）",
                 ["rollout_corr/rollout_is_eff_sample_size"]),
            # ★★★ 真正的停机线是**绝对有效条数**，不是比例：
            #   0.3 隐含了大 batch 的假设（上游 ~1024 条 ⇒ 有效 307；我们 48 条 ⇒ 有效 14）。
            #   ⚠️ W&B 不算派生列，这条由 `rl_report` 补报（= ESS/N × 每次更新的序列数）。
            line("★ 绝对有效条数 N×ESS/N（<24 停机）", ["syncopate/effective_seqs"]),
            # ★★ A1：权重同步真的发生了吗 —— 每次同步后 kl 必须回落到首步那个数量级。
            #   2026-08-18 那条 bug（权重从没推给 rollout engine）就是从这里漏过去的：
            #   它跑完了两轮训练，**没有任何指标报警**。
            line("★ kl vs 首步地板（不回落 = 权重同步没生效）",
                 ["rollout_corr/kl", "syncopate/kl_floor"]),
            # ★ 被截到上下界的比例：IS 修正退化成常数缩放的正面症状（研究问题 H3）
            line("★ 被截断比例 low+high（>40% 停机）",
                 ["rollout_corr/rollout_is_ratio_fraction_low",
                  "rollout_corr/rollout_is_ratio_fraction_high"]),
            line("grad_norm（跳两个数量级立即停）", ["actor/grad_norm"]),
            # ★★★ P 族 · 输入完整性（2026-08-19 第四条常驻判据）：
            #   3584 那次 100% 截断翻掉了一整条归因链（defer 崩塌 ≠ reward 教的）。
            #   rl_guard 非零即停机；这里是它的曲线版。v13 实测余量只剩 466 token。
            line("★ prompt 截断率（必须恒 0.0000，非零守卫停机）",
                 ["prompt_length/clip_ratio"]),
            # ⚠️ 长度上涨可能是学会推理，也可能是刷长度然后被 max_len 砍掉 ——
            #   后者的 reward 是假的。**必须和截断率一起看**，单看均值分不出来。
            #   response 侧截断 7–12% 是预算内的正常行为（三种成因见 P1-3），
            #   要盯的是**突然上台阶**，不是绝对值。
            line("response_length + response 截断率（分不出推理还是刷长度）",
                 ["response_length/mean", "response_length/clip_ratio",
                  "syncopate/truncated"]),
        ],
        is_open=True,
    ),
    ws.Section(
        name="② ★ 零梯度（GRPO 的头号杀手，默认指标里没有）",
        panels=[
            # ⚠️ `critic/rewards/std` 是 **batch 内** std，而 advantage 是**组内**归一化。
            # 某个 prompt 的一组 rollout 全对或全错 ⇒ σ_g→0 ⇒ 这条样本零梯度。
            # 这个数从 10% 涨到 60% 时有效样本已经塌一半，**而 reward 曲线还在涨**。
            line("组内 std=0 的占比（rl_report 补报）",
                 ["syncopate/zero_grad_group_ratio"]),
            # ★ 全对 / 全错分开：都零梯度，但处理方式相反 ——
            #   全对 = 学会了该降权；全错 = 够不着该补 SFT。
            line("全对 vs 全错（处理方式相反）",
                 ["syncopate/group_all_correct_ratio", "syncopate/group_all_wrong_ratio"]),
            line("组内 reward std 均值", ["syncopate/group_reward_std_mean"]),
            line("clip 比例（长度 hacking 的证据）", ["actor/pg_clipfrac"]),
            # ★ 动态分池的判据行：抽中的题里有梯度的占多少。
            #   分池没接上时这个数会退化成"题库里有梯度题的自然占比"。
            line("分池：抽中题的组内 std 分布", ["syncopate/group_reward_std_mean"]),
        ],
        is_open=True,
    ),
    ws.Section(
        name="③ reward 与护栏（必须同向）",
        panels=[
            line("reward", ["critic/rewards/mean"]),
            # ★ verl 的 compute_data_metrics 只认两个字段 ⇒ cap 分解由
            #   `rl_report` 补报，跑完必须执行，否则这一格是空的。
            line("cap 命中/条（rl_report 补报）", ["cap/total_per_case"]),
            line("截断率", ["cap/truncated_ratio"]),
        ],
        is_open=True,
    ),
    ws.Section(
        name="④ 动态分池 + 行为读数（都由 rl_report 补报）",
        panels=[
            line("零梯度组占比（分池应压住它；连续 3 窗口不创新高 = 完成判据）",
                 ["syncopate/zero_grad_group_ratio"]),
            # ★★ 覆盖率不到平台期就停 = 提前收工（e17a 实测 60 步只盖 22.7%）
            line("★ 池覆盖率（累计不同题 / 采样器够得着的池）", ["syncopate/pool_coverage"]),
            line("每步不同 prompt 数", ["syncopate/groups"]),
            line("轨迹步数均值（短任务被降权后应上升）", ["syncopate/num_steps"]),
            # ★ D 族的曲线版：看**连零**，不看水平（率分不开好坏）。
            #   连零 ≥25 步守卫会停机；曲线上贴着 0 走就是塌陷前兆。
            line("★ defer / reject 条数（连零 ≥25 停机，D 族）",
                 ["syncopate/defer_count", "syncopate/reject_count"]),
            # ★ P8 的尾巴：占位 logprob 只有 ~0.1%，**均值看不见，看最小值**
            line("logprob_coverage（min 掉离 1.0 = 占位值混进 IS）",
                 ["syncopate/logprob_coverage", "syncopate/logprob_coverage_min"]),
        ],
    ),
    ws.Section(
        name="⑤ 异步与漂移（★ 这套 harness 就是围绕它建的）",
        panels=[
            line("陈旧轨迹比例", ["fully_async/count/stale_trajectory_processed"]),
            line("partial_ratio（=0 说明没有漂移可测）",
                 ["fully_async/partial/partial_ratio"]),
            # ★ rollout 用的权重和 update 时的权重不是同一份 —— 两者 logprob 的偏离度
            #   是 async RL 的头号故障源。别当噪声删掉。
            line("模板混合漂移 TV（rl_report 补报）",
                 ["syncopate/mix_drift/total_variation"]),
            line("IS ratio 均值（应≈1.0，<0.5 或 >2.0 告警）",
                 ["rollout_corr/rollout_is_mean"]),
            line("rollouter 空闲率", ["fully_async/rollouter/idle_ratio"]),
            line("每步秒数", ["timing_s/step"]),
        ],
    ),
]


def build(name: str, sections: list[ws.Section]) -> str:
    view = ws.Workspace(
        name=name, entity=ENTITY, project=PROJECT, sections=sections,
        settings=ws.WorkspaceSettings(x_axis="_step", smoothing_type="none"),
    )
    view.save()
    return view.url


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["sft", "rl"], default=None)
    args = ap.parse_args()
    if args.only in (None, "sft"):
        print("SFT 面板:", build("Syncopate · SFT 健康度", SFT_SECTIONS))
    if args.only in (None, "rl"):
        print("RL  面板:", build("Syncopate · RL 停止条件", RL_SECTIONS))
    print("\n⚠️ 训练与晋级规则在 docs/syncopate/04-TRAINING.md —— "
          "面板只负责把该看的摆出来，不负责判定。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
