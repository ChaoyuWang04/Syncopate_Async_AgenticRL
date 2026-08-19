#!/usr/bin/env python
"""动态分池的读数：**还剩多少东西可学**，以及**剩下的是哪三类**。

    python scripts/pool_readout.py checkpoints/grpo/<run>              # 打报告
    python scripts/pool_readout.py checkpoints/grpo/<run> --export-triage out/  # 导三类清单

★★★ 为什么需要它（2026-08-19）

分池（`syncopate/train/pool.py`）**早就实现了**该实现的那套：
组内方差为主信号、降权到地板而不删除、`last_seen_step` 做回归保护。
⚠️ **但它的状态从来没有人看。** `pool_state.json` 落在 ckpt 目录里，
没有任何读数、没有任何判据消费它。
⇒ 又一次「机制在，但没接上」—— 只不过这次接的那一头是**观测**，不是执行。

★★ 一条实测（`e17a_kl_on`，60 步）说明这件事有多贵：

    RL 桶 824 条，60 步真的抽到过 **187 条 = 22.7%**，其中 56% 只抽到 **1 次**
    而 rollout dump 里 **35.9% 的组是零方差** —— 这些组的 8 次 rollout
    对梯度的贡献**精确等于 0**，不是"贡献小"

⇒ 分池要先"体检过"才能降权，而 60 步里大多数题只被抽到一次
  ⇒ **跑 60 步，等于建了分池然后不给它数据。**

★ 三类的出口**完全不同**，所以必须分开导（这是排查的输入）：

    饱和   分高、无方差    已经会了      ⇒ 降权，保留地板做回归体检
    死格   分低、无方差    从没探索到    ⇒ **RL 结构上救不了**，该由 SFT 覆盖去解
    卡死   中间分、无方差  在里面打转    ⇒ 查是缺工具还是缺信息；curriculum 只适用这一类

⚠️ 「死格」不是 curriculum 能解的：8 次采样一次都没碰到正确路径 ⇒
   RL 只能强化**已经出现过**的行为，没出现过的它**看不见**（`01 §P0-1.3` 的 GEO 同族）。
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import statistics
from pathlib import Path
from typing import Any

# 组内 std 低于它就算"没有梯度"。和 `compare.py` / `select_sft_ckpt.py` 用的是同一个数 ——
# ⚠️ 三处必须一致，否则同一条题在不同报告里会被分进不同的类。
FLAT_STD = 0.01
# 分高 / 分低的界，同 `select_sft_ckpt._grad_alive`
SATURATED_REWARD = 0.9
DEAD_REWARD = 0.15

# ★ 完成判据：连续多少个窗口"零梯度率**不再上升**"就算学到头了。
#
# ⚠️⚠️ 第一版我把方向写反了（写成"不再下降"），而且**读数当场把它照出来**：
#   实测轨迹是 15% → 17% → 37% → 39% → 55% → 52%，明明在**上升**，
#   判据却打印「仍在下降」。
#   ⇒ 想清楚了才对：**零梯度率上升 = 越来越多的题被学会**（或饱和）。
#     所以"学到头了"的信号是它**不再创新高**，不是它不再下降。
#   ★ 这正是记了很多次的那个形状：判据写满了、真的在跑、也真的报了个数，
#     **只是它量的方向和你以为的相反**。
#
# ⚠️⚠️ 这是**完成信号，不是停机信号** —— 和 `rl_guard.sh` 那一族性质相反：
#   停机   = 训练**坏了**，继续跑是在浪费或在把错的训进权重
#   完成   = 训练**做完了**，继续跑只是收益递减
#   ⇒ 两者混在一起的话，"跑完了"和"崩了"会长得一样（同 agent_loop 那四种停法）。
PLATEAU_WINDOWS = 3
WINDOW_STEPS = 10


def _step_no(p: Path) -> int:
    m = re.search(r"(\d+)\.jsonl$", p.name)
    return int(m.group(1)) if m else 0


def zero_gradient_trajectory(run_dir: Path, window: int = WINDOW_STEPS) -> list[float]:
    """按窗口统计零方差组的占比。**看轨迹，不看终值。**

    ⚠️ 终值说明不了问题：零梯度率高可能是"学完了"，也可能是"从来没学动过"。
      **是不是还在上升**才分得开这两件事。
    """
    dumps = sorted((run_dir / "rollout_dumps").glob("*.jsonl"), key=_step_no)
    per_step: list[tuple[int, int]] = []
    for f in dumps:
        groups: dict[int, list[float]] = collections.defaultdict(list)
        for line in f.open(encoding="utf-8"):
            try:
                r = json.loads(line)
            except Exception:
                continue
            groups[hash(r.get("input", ""))].append(r.get("reward", 0.0))
        flat = sum(1 for v in groups.values()
                   if len(v) > 1 and statistics.pstdev(v) <= FLAT_STD)
        per_step.append((flat, len(groups)))
    out = []
    for i in range(0, len(per_step), window):
        chunk = per_step[i:i + window]
        tot = sum(n for _, n in chunk)
        out.append(sum(f for f, _ in chunk) / tot if tot else 0.0)
    return out


def plateaued(traj: list[float], windows: int = PLATEAU_WINDOWS) -> bool:
    """零梯度率**连续 N 个窗口没有创新高** ⇒ 没有新东西被学会了。

    ★ 为什么用"没有创新高"而不是"低于某个值"：
      后者是「这个数应该在某范围里」型（守则①点名的那一类），
      而且我们**不知道**那个值该是多少 ——
      它取决于数据里有多少题是**结构上学不会**的（死格），那是数据的性质不是训练的。
    ⚠️ 也不能用"连续 N 个窗口单调不增"：单个窗口的抖动（实测 55%→52%）
      会把一个还在上升的趋势误判成到顶。**看最大值，不看逐点。**
    """
    if len(traj) < windows + 1:
        return False
    before = max(traj[:-windows])
    return max(traj[-windows:]) <= before + 1e-9


def classify(states: dict[str, dict[str, Any]]) -> dict[str, list[str]]:
    """把 case 分成四类。**只看被体检过的**（`seen ≥ 2`）。

    ⚠️ 只抽到过 1 次的题，`ema_std` 是**一次观测**不是估计 ——
      拿它分类会把"还没量准"说成"已经学会"。**宁可报"没量够"。**
    """
    out: dict[str, list[str]] = {"有梯度": [], "饱和": [], "死格": [], "卡死": [],
                                 "没量够": []}
    for cid, v in states.items():
        if v.get("seen", 0) < 2:
            if v.get("seen", 0) >= 1:
                out["没量够"].append(cid)
            continue
        std, rew = v.get("ema_std", 1.0), v.get("ema_reward", 0.0)
        if std > FLAT_STD:
            out["有梯度"].append(cid)
        elif rew > SATURATED_REWARD:
            out["饱和"].append(cid)
        elif rew < DEAD_REWARD:
            out["死格"].append(cid)
        else:
            out["卡死"].append(cid)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="动态分池读数 + 三类清单")
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--export-triage", type=Path, default=None,
                    help="把三类清单导到这个目录（排查的输入）")
    ap.add_argument("--total-cases", type=int, default=None,
                    help="RL 桶总条数，用来算覆盖率")
    args = ap.parse_args(argv)

    state_path = args.run_dir / "pool_state.json"
    if not state_path.exists():
        print(f"⏳ {args.run_dir} 还没有 pool_state.json")
        return 0
    states = json.loads(state_path.read_text(encoding="utf-8")).get("states", {})
    drawn = {k: v for k, v in states.items() if v.get("seen", 0) >= 1}

    print(f"── 覆盖率 ──")
    line = f"  被抽到过 {len(drawn)} 条"
    if args.total_cases:
        line += f" / {args.total_cases} 条 = {len(drawn)/args.total_cases:.1%}"
    print(line)
    once = sum(1 for v in drawn.values() if v.get("seen") == 1)
    print(f"  其中只抽到 1 次的 {once} 条 = "
          f"{once/max(len(drawn),1):.0%}   ← 这些**还没量准**，分池对它们无从降权")

    groups = classify(states)
    print(f"\n── 四类（只统计体检过 ≥2 次的）──")
    for k in ("有梯度", "饱和", "卡死", "死格", "没量够"):
        print(f"  {k:<6} {len(groups[k]):>5}")

    traj = zero_gradient_trajectory(args.run_dir)
    if traj:
        print(f"\n── 零梯度率轨迹（每 {WINDOW_STEPS} 步一个窗口）──")
        print("  " + "  ".join(f"{v:.0%}" for v in traj))
        print("  （零梯度率**上升 = 越来越多的题被学会**；到顶 = 学不动新东西了）")
        if plateaued(traj):
            # ★ 完成信号，**不是**停机信号
            print(f"  ✅ 连续 {PLATEAU_WINDOWS} 个窗口**没有创新高** ⇒ "
                  f"**没有新东西被学会了**，可以收工去排查三类清单")
        else:
            print(f"  🔄 还在创新高（{traj[0]:.0%} → {traj[-1]:.0%}）⇒ "
                  f"**还有东西在被学会，不该停**")

    if args.export_triage:
        args.export_triage.mkdir(parents=True, exist_ok=True)
        for k, ids in groups.items():
            (args.export_triage / f"{k}.json").write_text(
                json.dumps(sorted(ids), ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n三类清单 → {args.export_triage}")
        print("  ⚠️ 三类的出口**完全不同**：")
        print("     饱和 ⇒ 降权，保留地板做回归体检")
        print("     卡死 ⇒ 查缺工具还是缺信息；**curriculum 只适用这一类**")
        print("     死格 ⇒ **RL 结构上救不了**（8 次采样一次都没碰到正确路径）"
              "，该由 SFT 覆盖去解")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
