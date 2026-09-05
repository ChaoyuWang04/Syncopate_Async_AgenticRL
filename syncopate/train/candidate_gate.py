#!/usr/bin/env python
"""v16 候选运行的离线资格检查。

    python -m syncopate.train.candidate_gate checkpoints/grpo/<run>          # 查
    python -m syncopate.train.candidate_gate checkpoints/grpo/<run> --strict # 不够格就非零退出

它检查三件事：运行是否一开始就声明为 candidate、是否达到 v16 登记的
最少更新数、rollout 是否足以产生零梯度率轨迹。

短 smoke 和 infra 对照不需要通过候选资格检查；它们默认声明为 smoke。
candidate 的最少更新数同时在启动器入口阻止误跑，在这里阻止事后追认。

三条检查，缺一不可：

    ① purpose == "candidate"      这一跑当初就是奔着上线去的
    ② 步数 ≥ MIN_CANDIDATE_STEPS  **下限，不是目标**
    ③ 零梯度率**不再创新高**       ← 当前只作待复验的完成信号

重要：第 ③ 条来自旧栈，只说明“零方差组占比最近没有继续创新高”，不能单独证明
模型质量已经足够，也不能证明动态分池应当启用。v16 candidate 尚未开放；B05/B07
会用新栈数据重新冻结停止与晋级规则。在那之前，本模块不接入固定 runbook。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from syncopate.train import pool_readout
from syncopate.train.launch_rl_v1 import MIN_CANDIDATE_STEPS


def _readout():
    """复用 `pool_readout` 的实现，**不在这里另抄一份判据**。

    ⚠️ 两份实现慢慢漂开，是本项目付过多次钱的东西 ——
    尤其这两份都在判"学完了没有"，漂开的后果是同一条跑在两处结论不同。
    """
    return pool_readout


def min_candidate_steps() -> int:
    """从 v16 启动器取，**不在这里写第二个数**。"""
    return MIN_CANDIDATE_STEPS


def evaluate(run_dir: Path) -> tuple[bool, list[str]]:
    """返回 (够不够格, 逐条理由)。**理由永远给全**，不在第一条不过时短路。

    ⚠️ 短路的话，人修完第一条再跑一次才发现还有第二条 —— 一次只告诉一个坏消息，
      是让人反复上机的最快方式。
    """
    reasons: list[str] = []
    ok = True

    purpose_file = run_dir / "run_purpose.json"
    if not purpose_file.exists():
        ok = False
        reasons.append("🔴 没有 run_purpose.json ⇒ 这一跑没有声明用途"
                       "（老的跑都没有；用 --profile candidate 重跑）")
        purpose = None
    else:
        purpose = json.loads(purpose_file.read_text(encoding="utf-8")).get("purpose")
        if purpose != "candidate":
            ok = False
            reasons.append(f"🔴 purpose = {purpose!r}，不是 candidate "
                           f"⇒ 这一跑当初不是奔着上线去的，**不要事后追认**")
        else:
            reasons.append("✅ purpose = candidate")

    m = _readout()
    dumps = sorted((run_dir / "rollout_dumps").glob("*.jsonl"))
    need = min_candidate_steps()
    if len(dumps) < need:
        ok = False
        reasons.append(f"🔴 只跑了 {len(dumps)} 步 < 最少 {need} 步"
                       f"（⚠️ 这是**下限不是目标**）")
    else:
        reasons.append(f"✅ 跑了 {len(dumps)} 步 ≥ {need}")

    traj = m.zero_gradient_trajectory(run_dir)
    if not traj:
        ok = False
        reasons.append("🔴 读不到零梯度率轨迹（没有 rollout_dumps？）—— **无法判定，不是通过**")
    elif not m.plateaued(traj):
        ok = False
        reasons.append(
            f"🔴 零梯度率**还在创新高**（{traj[0]:.0%} → {traj[-1]:.0%}）"
            f" ⇒ 当前完成信号未满足；不能只因跑够步数就晋级")
    else:
        reasons.append(f"✅ 零梯度率连续 {m.PLATEAU_WINDOWS} 个窗口没有创新高 "
                       f"⇒ 当前诊断信号满足（仍须冻结质量门槛）")
    return ok, reasons


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="上线候选的晋级闸")
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--strict", action="store_true",
                    help="不够格就非零退出（给脚本用）")
    args = ap.parse_args(argv)

    ok, reasons = evaluate(args.run_dir)
    print(f"── 晋级闸 · {args.run_dir} ──")
    for r in reasons:
        print("  " + r)
    if ok:
        print("\n✅ 通过当前离线资格检查；这不等于质量晋级")
        return 0
    print("\n🔴 **不够格作为上线候选**")
    print("   ⚠️ 这不代表这一跑没用 —— smoke 和精度/吞吐实验本来就不需要过这道闸。")
    print("   ⇒ 它只拦一件事：别把不满足候选身份与当前完成信号的短跑事后当候选。")
    return 1 if args.strict else 0


if __name__ == "__main__":
    raise SystemExit(main())
