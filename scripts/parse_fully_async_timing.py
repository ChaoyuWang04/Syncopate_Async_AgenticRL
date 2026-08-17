#!/usr/bin/env python3
"""从 verl 的训练日志里把 timing 行解析成「每 global step」的口径。

★ 为什么需要它（2026-08-17 建）：
   fully_async 的每条 timing 行**覆盖多个 global step**（本项目 sync_every=4 ⇒ 4 步一行），
   直接读日志里的绝对秒数会**报错 4 倍** —— E18 §7-5 记着这个坑，此前已经犯过一次。
   ⇒ 把「除以覆盖步数」这件事固化进工具，别再靠人记得。

用法：
    python scripts/parse_fully_async_timing.py logs/rl_v13e1.log
    python scripts/parse_fully_async_timing.py logs/xxx.log --json logs/xxx_timing.json
    python scripts/parse_fully_async_timing.py logs/xxx.log --tail 3   # 只看最后 3 条（稳态）

口径说明：
  * 覆盖步数由相邻两条 timing 行的 `training/global_step` 差分**实测**得出，不写死。
  * `param_sync` 每个同步周期只发生一次 ⇒ 同时给「单次真实时长」和「摊到每步」两个数。
  * 占比按同一条 timing 行内的 `timing_s/step` 为分母（不跨行混算）。
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

# 一条 timing 行长这样：
#   step:3 - training/global_step:11.0 - ... - timing_s/gen:24.6 - timing_s/step:107.3 - ...
_STEP_RE = re.compile(r"training/global_step:([0-9.]+)")
_TIMING_RE = re.compile(r"timing_s/([a-zA-Z_0-9]+):([0-9.eE+-]+)")
# 独立的 param_sync 打点（FullyAsyncTrainer 自己打的，含 param_version）
_PARAM_SYNC_RE = re.compile(
    r"param_sync: ([0-9.]+) seconds self\.current_param_version: ([0-9]+)"
)

# 这三项是同一批数据的三次前向 —— 占空比成因③（E17 / 队列 B12）
TRIPLE_FORWARD = ("update_actor", "old_log_prob", "ref")


def parse(path: Path) -> dict:
    text = path.read_text(errors="replace")

    rows: list[dict] = []
    for line in text.splitlines():
        if "timing_s/step" not in line:
            continue
        gs = _STEP_RE.search(line)
        timings = {k: float(v) for k, v in _TIMING_RE.findall(line)}
        if not timings:
            continue
        rows.append({"global_step": float(gs.group(1)) if gs else None, "timings": timings})

    # 覆盖步数 = 相邻 timing 行的 global_step 差分（实测，不写死）
    steps = [r["global_step"] for r in rows if r["global_step"] is not None]
    diffs = [b - a for a, b in zip(steps, steps[1:])] if len(steps) > 1 else []
    span = int(statistics.median(diffs)) if diffs else 1
    span_uniform = len(set(diffs)) <= 1

    param_sync = [
        {"seconds": float(s), "param_version": int(v)}
        for s, v in _PARAM_SYNC_RE.findall(text)
    ]

    return {
        "log": str(path),
        "rows": rows,
        "step_span": span,
        "step_span_uniform": span_uniform,
        "step_span_diffs": diffs,
        "param_sync_events": param_sync,
    }


def summarize(parsed: dict, tail: int | None) -> dict:
    rows = parsed["rows"]
    if tail:
        rows = rows[-tail:]
    if not rows:
        return {"error": "日志里没有 timing 行"}

    span = parsed["step_span"] or 1
    keys: list[str] = []
    for r in rows:
        for k in r["timings"]:
            if k not in keys and k not in ("start_profile", "stop_profile"):
                keys.append(k)

    out = {}
    total = statistics.median([r["timings"]["step"] for r in rows if "step" in r["timings"]])
    for k in keys:
        vals = [r["timings"][k] for r in rows if k in r["timings"]]
        if not vals:
            continue
        med = statistics.median(vals)
        out[k] = {
            "n": len(vals),
            "median_per_timing_row": round(med, 3),
            "median_per_global_step": round(med / span, 3),
            "share_of_step": round(med / total, 4) if total else None,
        }

    triple = sum(out[k]["median_per_timing_row"] for k in TRIPLE_FORWARD if k in out)
    return {
        "n_rows_used": len(rows),
        "step_span": span,
        "step_span_uniform": parsed["step_span_uniform"],
        "per_key": out,
        "triple_forward_share": round(triple / total, 4) if total else None,
        "param_sync_seconds": [e["seconds"] for e in parsed["param_sync_events"]],
        "param_sync_steady_median": (
            round(statistics.median([e["seconds"] for e in parsed["param_sync_events"][1:]]), 3)
            if len(parsed["param_sync_events"]) > 1
            else None
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("log", type=Path)
    ap.add_argument("--tail", type=int, default=None, help="只用最后 N 条 timing 行（取稳态）")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    if not args.log.exists():
        print(f"没有这个文件：{args.log}", file=sys.stderr)
        return 2

    parsed = parse(args.log)
    summary = summarize(parsed, args.tail)
    if "error" in summary:
        print(summary["error"], file=sys.stderr)
        return 1

    span = summary["step_span"]
    print(f"# {args.log}")
    print(f"  timing 行 {summary['n_rows_used']} 条 · 每行覆盖 {span} 个 global step"
          f"{'' if summary['step_span_uniform'] else '  ⚠️ 覆盖步数不均匀，绝对秒数存疑'}")
    print(f"  {'项':<26}{'每 timing 行':>14}{'每 global step':>16}{'占步':>9}")
    for k, v in sorted(summary["per_key"].items(), key=lambda kv: -kv[1]["median_per_timing_row"]):
        print(f"  {k:<26}{v['median_per_timing_row']:>14.2f}{v['median_per_global_step']:>16.2f}"
              f"{(v['share_of_step'] or 0) * 100:>8.1f}%")
    if summary["triple_forward_share"] is not None:
        print(f"  ⇒ 三次前向（update_actor+old_log_prob+ref）占步 "
              f"{summary['triple_forward_share'] * 100:.1f}%   ← 占空比成因③ / 队列 B12")
    if summary["param_sync_seconds"]:
        print(f"  param_sync 各次：{summary['param_sync_seconds']}"
              f"   稳态中位 {summary['param_sync_steady_median']} s（单次真实时长，非摊薄）")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({"summary": summary, "raw": parsed}, indent=2))
        print(f"  → {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
