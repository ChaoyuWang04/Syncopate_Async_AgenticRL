#!/usr/bin/env python
"""跑中盯「拒绝能力有没有塌」—— 读 rollout dump，报最长连续零 defer 步数。

    python scripts/defer_watch.py checkpoints/grpo/<run>          # 打一行摘要
    python scripts/defer_watch.py checkpoints/grpo/<run> --streak # 只打数字（给守卫用）

★★ 为什么需要它（2026-08-19）

`defer` 塌陷是**不可逆**的：`[实测]` lr 1e-4 那跑，9 条该 defer 的 EVAL 题
全部掉到 0.000，**组内 std = 0** ⇒ GRPO 的 advantage 恒为 0
⇒ 这些题**再也不产生梯度**，RL 自己爬不出来。

而此前唯一能看到它的地方是 `compare`，那是**跑完之后**才跑的。
一个 137 步的 epoch 要几个小时，这几个小时里没有任何东西在看这件事。
⇒ 本探针的价值不是"防止塌陷"，是**让塌陷变便宜**：半路停，而不是烧完一整跑。

★ 走过的两条死路（留着，免得下一个人重走）

① `cap/premature_decision_cap` —— 看起来是天然的代理（"该 defer 却硬答"）。
   `[实测]` **不成立**：lr 1e-4 那跑 defer 已经 0%，而这条 cap 在 2880 条 rollout 上
   命中 **0 次**。因为它还要求"做了决定性写操作"，模型直接答不写就不触发。
   ⇒ **验之前别假设机制接上了。**
② 拿 defer **率**当阈值 —— 也不成立：`r1_seqis`（评测该 defer 率 97%，完全健康）
   的总体 defer 率从 5.1% 一路降到 1.6%，降的是**误 defer**（那是好事）。
   ⇒ 水平分不开好坏，**"持续为零"才分得开**。

⚠️ 只能拿到**总体** defer 率，拿不到双向拆分 —— rollout dump 里没有 `case_id`
（`gts` 是 null）⇒ 对不上 `expected_behavior`。双向拆分只有评测侧有（`compare`）。
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# ⚠️ 阈值是**按 4 个跑反填的工程值，不是推导出来的**（同 rl_guard 的 24 / 0.40）。
#
#   [实测 2026-08-19] 最长连续零 defer 步数：
#     r1_seqis         60 步   9 步   评测该 defer 97%   健康
#     seqis_long120   120 步  16 步   健康
#     r1_tokenis       60 步  18 步   评测 83%           健康（有代价但没塌）
#     e20f_lr1e4_keep  60 步  56 步   评测 **0%**        🔴 塌了，且连到结束
#
#   ⇒ 健康侧上界 18、崩塌侧 56 ⇒ 取 25：对健康跑有 1.4× 余量，
#     且能在 e20f 那跑的中段就触发（省下大约一半的跑）。
#   ⚠️ **第一次干净重跑之后要用实测反填**，别把这个数当成推导出来的。
#   [复核 2026-08-19] 5120 干净跑（infra 回信第 5 条要求的重反填）：
#     e17a 11 · e17b 4 · e17d×2 0（各 60 步）⇒ 健康侧上界降到 11，25 余量 ≥2.3×
#     ⇒ 维持 25。⚠️ 样本都是 60 步跑，137 步整 epoch 的自然连零可能更长，别急着收紧。
#   ⚠️ 基线本来就低：RL 桶里只有 3.4% 的题该 defer，每步 6 题
#     ⇒ 多数步天然就是 0 —— 所以门槛必须**远高于**直觉。
MAX_ZERO_STREAK = 25

_BEHAVIOR = re.compile(r'"behavior"\s*:\s*"(\w+)"')


def _step_no(p: Path) -> int:
    m = re.search(r"(\d+)\.jsonl$", p.name)
    return int(m.group(1)) if m else 0


def defer_counts(run_dir: Path) -> list[int]:
    """每一步的 defer 条数（按步号排序）。"""
    dumps = sorted((run_dir / "rollout_dumps").glob("*.jsonl"), key=_step_no)
    out = []
    for f in dumps:
        n = 0
        for line in f.open(encoding="utf-8"):
            try:
                text = json.loads(line).get("output", "")
            except Exception:
                continue
            # ⚠️ 取**最后一个** behavior：一条轨迹有多步，终答那步才是它的行为
            found = _BEHAVIOR.findall(text)
            if found and found[-1] == "defer":
                n += 1
        out.append(n)
    return out


def longest_zero_streak(counts: list[int]) -> int:
    best = cur = 0
    for c in counts:
        cur = cur + 1 if c == 0 else 0
        best = max(best, cur)
    return best


def trailing_zero_streak(counts: list[int]) -> int:
    """★ 守卫要看的是**当前**这一串，不是历史最长 ——
    历史最长发生在早期又恢复了的话，停机就是误报（`r1_tokenis` 正是这样）。"""
    n = 0
    for c in reversed(counts):
        if c:
            break
        n += 1
    return n


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="盯 defer 塌陷（跑中）")
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--streak", action="store_true", help="只打当前连零步数（给守卫用）")
    ap.add_argument("--threshold", type=int, default=MAX_ZERO_STREAK)
    args = ap.parse_args(argv)

    if not (args.run_dir / "rollout_dumps").is_dir():
        # ⚠️ 报"没有"，不猜 —— 跑刚起来还没 dump 是正常的，不该当成塌陷
        print("" if args.streak else f"⏳ {args.run_dir} 还没有 rollout_dumps")
        return 0

    counts = defer_counts(args.run_dir)
    cur = trailing_zero_streak(counts)
    if args.streak:
        print(cur)
        return 0

    total = sum(counts)
    print(f"步数 {len(counts)} · defer 共 {total} 条 · "
          f"当前连零 {cur} 步 · 历史最长连零 {longest_zero_streak(counts)} 步 "
          f"（门槛 {args.threshold}）")
    if cur >= args.threshold:
        print(f"🔴 连续 {cur} 步没有任何 defer —— 拒绝能力可能已经塌陷，**且不可逆**")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
