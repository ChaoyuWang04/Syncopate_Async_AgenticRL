#!/usr/bin/env python3
"""E08-c / B4 · 异步分布漂移：**发出 → 完成 → 训练到** 三段各丢了什么。

★ 为什么要三段而不是两段（这是 E08 §6 的教训）：
上一轮只有「完成」和「训练到」两端，量到逐桶零差（7200=7200），读起来像结论 ——
其实是**仪器装在了下游**：被中途杀掉的长轨迹**在「完成」那一端根本不存在**，
所以它们的消失不可能显示为差。⇒ 现在 `dispatched.jsonl` 写三类事件：

    dispatch  交给 agent loop 的那一刻     ← 长任务在这里就已经存在
    complete  正常跑完
    abort     被取消 / 抛异常              ← partial_rollout 的 abort 走这条

    发出 − 完成 − 中止 = 还在飞
    ★ **发出 − 完成 = 漂移的上界**；完成 − 训练到 = 下游丢弃

判据（比的是**分布**，不是均值 —— 均值会把长尾平掉）：
    按 num_steps 分桶，逐桶对比「完成」与「训练到」的条数；
    未完成的那批单独按 case 列出来（它们没有 num_steps，只能看是谁）。

用法：
    python scripts/infra/analyze_drift.py checkpoints/grpo/<exp>
    python scripts/infra/analyze_drift.py checkpoints/grpo/<exp> --json _audit/infra/<exp>_drift.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def read_jsonl(path: Path):
    with path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue        # 并发追加可能写出半行


def analyze(exp_dir: Path) -> dict:
    dispatch_log = exp_dir / "dispatched.jsonl"
    dumps_dir = exp_dir / "rollout_dumps"
    if not dispatch_log.exists():
        raise SystemExit(f"没有 {dispatch_log} —— 这一跑没开 SYNCOPATE_DISPATCH_LOG？")

    events = Counter()
    dispatched: dict[str, dict] = {}
    completed: dict[str, dict] = {}
    aborted: dict[str, dict] = {}
    legacy = 0
    for row in read_jsonl(dispatch_log):
        ev = row.get("event")
        if ev is None:
            legacy += 1                      # 旧格式：每行都是完成行，没有 rollout_id
            completed.setdefault(f"__legacy_{legacy}", row)
            continue
        events[ev] += 1
        rid = row.get("rollout_id") or f"__norid_{events[ev]}"
        {"dispatch": dispatched, "complete": completed, "abort": aborted}.get(ev, {})[rid] = row

    # 训练到的那一端
    trained_steps = Counter()
    trained_total = 0
    if dumps_dir.exists():
        for f in sorted(dumps_dir.glob("*.jsonl")):
            for row in read_jsonl(f):
                trained_total += 1
                trained_steps[int(row.get("num_steps") or 0)] += 1

    completed_steps = Counter(int(r.get("num_steps") or 0) for r in completed.values())

    missing = sorted(set(dispatched) - set(completed) - set(aborted))
    return {
        "exp": str(exp_dir),
        "counts": {
            "dispatch": len(dispatched),
            "complete": len(completed),
            "abort": len(aborted),
            "trained": trained_total,
            "legacy_rows": legacy,
        },
        "gap_dispatch_minus_complete": len(dispatched) - len(completed),
        "gap_complete_minus_trained": len(completed) - trained_total,
        "in_flight_or_lost": missing[:50],
        "in_flight_or_lost_n": len(missing),
        "by_num_steps": {
            str(k): {"complete": completed_steps.get(k, 0), "trained": trained_steps.get(k, 0)}
            for k in sorted(set(completed_steps) | set(trained_steps))
        },
        "abort_reasons": dict(Counter(r.get("reason") for r in aborted.values())),
        "lost_by_case": dict(
            Counter(dispatched[r].get("case_id") for r in missing).most_common(20)
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("exp_dir", type=Path)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    res = analyze(args.exp_dir)
    c = res["counts"]
    print(f"# {res['exp']}")
    if c["legacy_rows"]:
        print(f"  ⚠️ 有 {c['legacy_rows']} 行是**旧格式**（没有 event 键）—— "
              f"那一跑的仪器还在下游，只能看「完成 vs 训练到」这一段")
    print(f"  发出 {c['dispatch']} · 完成 {c['complete']} · 中止 {c['abort']} · 训练到 {c['trained']}")
    if c["dispatch"]:
        print(f"  ★ 发出 − 完成 = {res['gap_dispatch_minus_complete']}   （漂移的上界）")
    print(f"    完成 − 训练到 = {res['gap_complete_minus_trained']} （下游丢弃）")
    # ⚠️⚠️ 跑**还没结束**时这两个差天然为正：最后一批已完成的还没进 dump，
    #    而且长轨迹更容易落在这一批里 ⇒ 逐桶看会**伪造出「长的被丢得多」**。
    #    ⇒ 只在训练进程退出之后读这份报告。（本项目为「把在飞当成丢失」付过一次钱。）
    print("  ⚠️ 判据前提：**训练进程已退出**。跑到一半读，在飞的会被记成丢失，且长轨迹偏多。")
    if res["abort_reasons"]:
        print(f"    中止原因：{res['abort_reasons']}")
    if res["in_flight_or_lost_n"]:
        print(f"  ⚠️ 发出但既没完成也没中止：{res['in_flight_or_lost_n']} 条"
              f"（还在飞，或**被静默杀掉** —— 跑结束后再看这个数才有意义）")
        if res["lost_by_case"]:
            print(f"    集中在这些 case：{list(res['lost_by_case'].items())[:8]}")

    print(f"\n  {'轮数':>6}{'完成':>10}{'训练到':>10}{'差':>8}")
    for k, v in res["by_num_steps"].items():
        diff = v["trained"] - v["complete"]
        flag = "" if diff == 0 else "  ←★ 有差"
        print(f"  {k:>6}{v['complete']:>10}{v['trained']:>10}{diff:>8}{flag}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(res, indent=2, ensure_ascii=False))
        print(f"  → {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
