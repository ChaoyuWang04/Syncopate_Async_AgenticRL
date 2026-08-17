"""一条命令把 SFT / RL 的观测面板建到 W&B 上（可重跑，幂等覆盖同名 view）。

    python scripts/make_wandb_panels.py            # 建 SFT + RL 两个 view
    python scripts/make_wandb_panels.py --only sft

★ 为什么要脚本而不是在网页上拖：**面板是判据的一部分**。
手拖的面板换个人、换台机器就没了，而"该看哪几条线、红线在哪"是要跟着仓库走的。
这份脚本和 `docs/syncopate/14-sft-health-metrics.md` 是同一件事的两种形态 ——
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
        name="① 停止条件（优先级高于分数曲线）",
        panels=[
            line("ESS/N（跌破 0.3 立即停）",
                 ["rollout_corr/rollout_is_eff_sample_size"]),
            line("grad_norm（跳两个数量级立即停）", ["actor/grad_norm"]),
            line("response_length（暴涨 = 长度 hacking）",
                 ["response_length/mean"]),
            line("IS ratio 超界比例（最早的信号）",
                 ["rollout_corr/rollout_is_ratio_fraction_high"]),
        ],
        is_open=True,
    ),
    ws.Section(
        name="② reward 与护栏（必须同向）",
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
        name="③ 异步与吞吐",
        panels=[
            line("陈旧轨迹比例", ["fully_async/count/stale_trajectory_processed"]),
            line("partial_ratio（=0 说明没有漂移可测）",
                 ["fully_async/partial/partial_ratio"]),
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
    print("\n⚠️ 红线的具体数值在 docs/syncopate/14-sft-health-metrics.md —— "
          "面板只负责把该看的摆出来，不负责判定。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
