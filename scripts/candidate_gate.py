#!/usr/bin/env python
"""上线候选的**晋级闸**：一条跑够不够格被当成上线候选。

    python scripts/candidate_gate.py checkpoints/grpo/<run>          # 查
    python scripts/candidate_gate.py checkpoints/grpo/<run> --strict # 不够格就非零退出

★★★ 为什么约束加在**晋级**上，而不是**起跑**上（2026-08-19 定）

infra 一直在用 RL 跑**短的精度/吞吐实验**（60 步就够）。
把"必须跑到没梯度"加在起跑上，会**当场挡住他们**，而他们本来就不需要跑到没梯度。

⇒ 所以：**任何跑都随便跑；只有"声称自己是上线候选"的跑才过闸。**
  · `--purpose probe`（默认）  实验 —— 不受任何约束
  · `--purpose candidate`      上线候选 —— 受最少步数 + 完成判据约束

⚠️ 那"主线忘了声明 candidate"怎么办？**不靠记性**：
  忘了声明的后果是**晋级时被这道闸拦下**（`run_purpose.json` 里写着 probe），
  不是"悄悄拿一个 60 步的短跑当候选"。

★★ 三条判据，缺一不可：

    ① purpose == "candidate"      这一跑当初就是奔着上线去的
    ② 步数 ≥ MIN_CANDIDATE_STEPS  **下限，不是目标**
    ③ 零梯度率**不再创新高**       ← 真正的停止条件

⚠️⚠️ ③ 才是重点。②只是"分池能开始起作用"的下限：
  `[实测 e17a]` 60 步时零梯度率仍在创新高（15%→52%），RL 桶只覆盖 22.7%，
  56% 被抽到的题只抽到**一次** ⇒ 分池对它们无从降权。
  **跑到步数就停 = 在还有东西可学的时候停下。**
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _readout():
    """复用 `pool_readout` 的实现，**不在这里另抄一份判据**。

    ⚠️ 两份实现慢慢漂开，是本项目付过多次钱的东西 ——
    尤其这两份都在判"学完了没有"，漂开的后果是同一条跑在两处结论不同。
    """
    spec = importlib.util.spec_from_file_location(
        "_pool_readout_for_gate", ROOT / "scripts" / "pool_readout.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_pool_readout_for_gate"] = mod
    spec.loader.exec_module(mod)
    return mod


def min_candidate_steps() -> int:
    """从 `launch_rl` 取，**不在这里写第二个数**。"""
    src = (ROOT / "syncopate" / "train" / "launch_rl.py").read_text(encoding="utf-8")
    import re
    m = re.search(r"^MIN_CANDIDATE_STEPS\s*=\s*(\d+)", src, re.M)
    return int(m.group(1)) if m else 400


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
                       "（老的跑都没有；用 --purpose candidate 重跑）")
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
            f" ⇒ **还有东西在被学会，这一跑停早了**")
    else:
        reasons.append(f"✅ 零梯度率连续 {m.PLATEAU_WINDOWS} 个窗口没有创新高 "
                       f"⇒ 没有新东西被学会了")
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
        print("\n✅ 够格作为上线候选")
        return 0
    print("\n🔴 **不够格作为上线候选**")
    print("   ⚠️ 这不代表这一跑没用 —— 精度/吞吐实验本来就不需要过这道闸。")
    print("   ⇒ 它只拦一件事：**别把一个还没学完的跑当成要发给用户的那个**。")
    return 1 if args.strict else 0


if __name__ == "__main__":
    raise SystemExit(main())
