"""两次评测的**配对**比较，并把这把尺子的精度一起报出来。

★★★ 为什么必须配对，以及为什么加采样次数没用

用已有的三次评测实测（64 条冻结 EVAL）：

    case 之间的 reward 标准差       0.326   ← 主导项：题目难度差异
    非配对比较的最小可检出差异        0.115
    配对比较的最小可检出差异         0.050   ← 好 2.3 倍

    组内 8 次采样的标准差            0.153
    但它贡献的标准误只有             0.0068
    采样 8 → 32 次                 0.0068 → 0.0034

⇒ **尺子的粗细来自「同一道题在两个模型下行为差异有多大」，不是采样噪声。**
  把每题采样从 8 加到 32 是花 4 倍 GPU 时间买 0.003 的改善，几乎白花；
  而换成配对比较，一分钱不花就把 0.115 降到 0.050。

★★ 最重要的一条使用纪律

观测差异小于最小可检出差异时，结论是「**没测出差异**」，
不是「没有差异」。所以这个模块每次都把 MDE 一起打印出来 ——
不打印的话，人一定会把 +0.02 读成"略有提升"。

    python -m syncopate.train.compare _audit/M1_base_4b.json _audit/M1_ctrl_epoch2.json
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import statistics
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


def load(path: Path) -> tuple[str, dict[str, dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("label", path.stem), {r["case_id"]: r for r in payload["rows"]}


def paired_stats(a: dict, b: dict, ids: list[str], key: str = "reward") -> dict[str, float]:
    """配对差值的统计量。分母是 case 数，不是 rollout 数——case 才是独立单位。"""
    diffs = [b[c][key] - a[c][key] for c in ids]
    n = len(diffs)
    mean = statistics.mean(diffs)
    sd = statistics.stdev(diffs) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n else 0.0
    return {
        "mean_diff": mean,
        "sd_diff": sd,
        "se": se,
        # 2σ 口径。样本量小时用 t 更严谨，但 n=64 时 t≈2.0，差别可以忽略
        "mde": 2 * se,
        "t": mean / se if se else 0.0,
        "wins": sum(d > 1e-9 for d in diffs),
        "losses": sum(d < -1e-9 for d in diffs),
        "ties": sum(abs(d) <= 1e-9 for d in diffs),
    }


def verdict(stats: dict[str, float]) -> str:
    if abs(stats["mean_diff"]) < stats["mde"]:
        return f"没测出差异（|{stats['mean_diff']:+.3f}| < MDE {stats['mde']:.3f}）—— 不等于没有差异"
    return f"{'提升' if stats['mean_diff'] > 0 else '退化'} {abs(stats['mean_diff']):.3f}（t={stats['t']:+.1f}）"


def defer_rates(rows: dict[str, dict[str, Any]]) -> tuple[float, float] | None:
    """defer 的双向准确率。老的评测文件没有这两个字段，返回 None。"""
    yes = [r for r in rows.values() if r.get("expected_behavior") == "defer"]
    no = [r for r in rows.values() if r.get("expected_behavior") not in (None, "defer")]
    if not yes or "behaviors" not in next(iter(rows.values())):
        return None
    def rate(group):
        total = sum(len(r["behaviors"]) for r in group)
        return sum(b == "defer" for r in group for b in r["behaviors"]) / total if total else 0.0
    return rate(yes), rate(no)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="两次评测的配对比较")
    parser.add_argument("baseline")
    parser.add_argument("candidate")
    parser.add_argument("--by-template", action="store_true", default=True)
    args = parser.parse_args(argv)

    label_a, a = load(Path(args.baseline))
    label_b, b = load(Path(args.candidate))
    ids = sorted(set(a) & set(b))
    if not ids:
        print("两次评测没有共同的 case —— 配对比较无从谈起")
        return 1
    only_a, only_b = len(a) - len(ids), len(b) - len(ids)
    print(f"基线  {label_a}")
    print(f"候选  {label_b}")
    print(f"共同 case {len(ids)}" + (f"（基线独有 {only_a} / 候选独有 {only_b}，已排除）"
                                     if only_a or only_b else ""))

    s = paired_stats(a, b, ids)
    print(f"\n★ 配对比较（reward）")
    print(f"  均值           {statistics.mean(a[c]['reward'] for c in ids):.3f} → "
          f"{statistics.mean(b[c]['reward'] for c in ids):.3f}")
    print(f"  配对差值        {s['mean_diff']:+.3f}   （标准差 {s['sd_diff']:.3f}，标准误 {s['se']:.3f}）")
    print(f"  最小可检出差异   {s['mde']:.3f}")
    print(f"  逐题            赢 {s['wins']} / 平 {s['ties']} / 输 {s['losses']}")
    print(f"  结论           {verdict(s)}")

    # 非配对口径也报一次，让「配对到底值多少」是看得见的
    across = statistics.stdev([a[c]["reward"] for c in ids])
    print(f"  （若不配对，MDE 会是 {2 * across / math.sqrt(len(ids)) * math.sqrt(2):.3f}）")

    print(f"\n★ 零梯度构成 —— SFT 看这个，不看均值")
    for label, rows in ((label_a, a), (label_b, b)):
        g = sum(rows[c]["reward_std"] > 0.01 for c in ids)
        d = sum(rows[c]["reward_std"] <= 0.01 and rows[c]["reward"] < 0.15 for c in ids)
        sat = sum(rows[c]["reward_std"] <= 0.01 and rows[c]["reward"] > 0.9 for c in ids)
        print(f"  {label[:44]:<46} 有梯度 {g:>3}  全灭 {d:>3}  饱和 {sat:>3}  卡死 {len(ids)-g-d-sat:>3}")

    print(f"\n★ cap 命中（reward 涨但 cap 不降 = reward hacking 的指纹）")
    ca = collections.Counter(x for c in ids for x in a[c]["caps"])
    cb = collections.Counter(x for c in ids for x in b[c]["caps"])
    for name in sorted(set(ca) | set(cb)):
        delta = cb[name] - ca[name]
        print(f"  {name:<32}{ca[name]:>6}{cb[name]:>7}{delta:>+7}")

    for label, rows in ((label_a, a), (label_b, b)):
        rates = defer_rates({c: rows[c] for c in ids})
        if rates:
            print(f"\n  defer 双向  {label[:40]:<42} 该 defer {rates[0]:.0%} / 误 defer {rates[1]:.1%}")

    if args.by_template:
        print(f"\n★ 按模板")
        by = collections.defaultdict(list)
        for c in ids:
            by[c.split("_")[0]].append(c)
        print(f"  {'模板':<8}{'n':>4}{'基线':>9}{'候选':>9}{'差值':>9}")
        for tpl, group in sorted(by.items()):
            ma = statistics.mean(a[c]["reward"] for c in group)
            mb = statistics.mean(b[c]["reward"] for c in group)
            print(f"  {tpl:<8}{len(group):>4}{ma:>9.3f}{mb:>9.3f}{mb - ma:>+9.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
