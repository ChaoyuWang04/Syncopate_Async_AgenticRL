#!/usr/bin/env python3
"""在**修好之后的干净基线**上复查几条管线不变量（不吃 GPU，读已有产物）。

★ 为什么要在新基线上再查一遍：主线 18 那 8 条探针跑在**坏基线**的产物上
（E21 只学 1/3 + E22 rollout 策略恒为 π₀）。前提变了，至少这几条值得复查。

⚠️ 原计划这一格是 Q4（失败注入在组内是否确定性），**做不了** ——
`dispatched.jsonl` 只有 `case_id / rollout_id / param_version_approx`，
`rollout_dumps` 里也没有注入字段 ⇒ **需要主线先在 dump 里加字段**。
⇒ 老实换成用现有数据真能测的四条，而不是假装测了。

四条：
  I1  题目覆盖：每个 param_version 内**抽到多少条不同的题**
      ⛔ **这不是主线 P4 的复查** —— P4 按 **fit step** 分组（每步应当 6 条不同的题），
         而 `dispatched.jsonl` 只有 `param_version_approx`（跨约 4 个 fit step）。
         **两个不同的对象，不许当同一件事读。** 要复查 P4 需要 dump 里带 case_id。
  I2  `logprob_coverage` 全样本 = 1.0000（P8 复查 —— 有占位值就会污染 IS）
  I3  `parse_errors` 的量级与趋势 ★ 与 2026-08-18 修的解析器崩溃直接相关
  I4  截断率（`truncated`）—— 它是 E20 §2 那条"长度"叙事的输入
"""
from __future__ import annotations

import glob
import json
import sys
from collections import Counter
from pathlib import Path


def load_dumps(run: Path) -> list[dict]:
    rows = []
    for f in sorted(glob.glob(str(run / "rollout_dumps" / "*.jsonl"))):
        step = int(Path(f).stem)
        for line in open(f, encoding="utf-8", errors="replace"):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            r["_step"] = step
            rows.append(r)
    return rows


def check(run: Path) -> None:
    print(f"\n{'='*78}\n  {run}\n{'='*78}")
    rows = load_dumps(run)
    if not rows:
        print("  🔴 没有 rollout_dumps ⇒ 这一跑没有产物可查")
        return
    print(f"  样本 {len(rows)} 条 / {len({r['_step'] for r in rows})} 步")

    # I1 · 组结构（用 dispatched 的 case_id 按 param_version 分组）
    disp = run / "dispatched.jsonl"
    if disp.exists():
        by_ver: dict = {}
        for line in open(disp, encoding="utf-8", errors="replace"):
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("event") != "dispatch":
                continue
            by_ver.setdefault(d.get("param_version_approx"), []).append(d.get("case_id"))
        sizes = Counter(len(set(v)) for v in by_ver.values())
        print(f"  I1 题目覆盖 每个 param_version 的**不同题数**分布: {dict(sorted(sizes.items()))}")
        print("     ⛔ 这**不是** P4（P4 按 fit step 分组，这里按 param_version ≈ 4 个 fit step）")
        dup = {k: len(v) - len(set(v)) for k, v in by_ver.items() if len(v) != len(set(v))}
        print(f"     含重复 case 的 version 数: {len(dup)}/{len(by_ver)}"
              "（跨 4 个 fit step 出现重复是**预期内**的，不是 P4 那个现象）")
    else:
        print("  I1 组结构   🔴 没有 dispatched.jsonl")

    # I2 · logprob 覆盖
    cov = [r.get("logprob_coverage") for r in rows if r.get("logprob_coverage") is not None]
    if cov:
        bad = [c for c in cov if c < 0.9999]
        print(f"  I2 logprob  覆盖率 min={min(cov):.4f} · 低于 1.0 的样本 {len(bad)}/{len(cov)}"
              + ("  ✅" if not bad else "  🔴 有占位 logprob ⇒ 会污染 IS"))
    else:
        print("  I2 logprob  ⚠️ dump 里没有 logprob_coverage 字段 ⇒ **判据无效，不当通过读**")

    # I3 · 解析错误（★ 与 2026-08-18 修的解析器崩溃直接相关）
    pe = [r.get("parse_errors", 0) or 0 for r in rows]
    steps = sorted({r["_step"] for r in rows})
    if steps:
        half = len(steps) // 2 or 1
        early = [r.get("parse_errors", 0) or 0 for r in rows if r["_step"] in steps[:half]]
        late = [r.get("parse_errors", 0) or 0 for r in rows if r["_step"] in steps[half:]]
        m = lambda a: sum(a) / len(a) if a else 0.0            # noqa: E731
        print(f"  I3 解析错误 总计 {sum(pe)} · 每样本均值 {m(pe):.3f}"
              f" · 前半 {m(early):.3f} → 后半 {m(late):.3f}")
        print("     ⚠️ 解析器崩溃已于 2026-08-18 修（畸形 payload 丢弃而非崩溃）"
              " ⇒ 这里的非零值现在是**被扣分的行为**，不再是崩溃")

    # I4 · 截断率
    tr = [bool(r.get("truncated")) for r in rows]
    if tr:
        early = [bool(r.get("truncated")) for r in rows if r["_step"] in steps[: len(steps) // 2 or 1]]
        late = [bool(r.get("truncated")) for r in rows if r["_step"] in steps[len(steps) // 2 or 1:]]
        pct = lambda a: 100 * sum(a) / len(a) if a else 0.0    # noqa: E731
        print(f"  I4 截断率   {pct(tr):.1f}% · 前半 {pct(early):.1f}% → 后半 {pct(late):.1f}%")

    # 顺带：长度与得分的趋势（E20 §2 那条叙事的输入）
    ln = [(r["_step"], r.get("num_steps") or 0) for r in rows]
    if ln:
        first = [v for s, v in ln if s in steps[: len(steps) // 4 or 1]]
        last = [v for s, v in ln if s in steps[-(len(steps) // 4 or 1):]]
        f = sum(first) / len(first) if first else 0
        l = sum(last) / len(last) if last else 0
        print(f"  ─ 平均步数  {f:.2f} → {l:.2f}（前 1/4 → 后 1/4）")


def main() -> int:
    runs = sys.argv[1:] or sorted(
        str(p) for p in Path("checkpoints/grpo").glob("*")
        if (p / "rollout_dumps").exists() and p.name.startswith(("r1_", "e20"))
    )
    print("管线不变量 · 在**修好之后的干净基线**上复查")
    for r in runs:
        check(Path(r))
    print("\n⚠️ 原计划的 Q4（失败注入组内确定性）**做不了** —— "
          "dispatched/dump 里都没有注入字段，需要主线先加。**不假装测了。**")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
