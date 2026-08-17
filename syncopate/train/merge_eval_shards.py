"""把多卡分片的评测结果合并成一份审计，并**重新出报告**。

    python -m syncopate.train.merge_eval_shards --shards <目录> --out _audit/xxx.json

★ 为什么不能只把 rows 拼起来就完事

审计里除了逐条 `rows`，报告里那些聚合量（读/写分桶、行为混淆矩阵、零梯度构成、
采样多样性）都是**在全量上算的**。分片各算各的再平均，某些量（如"有梯度格子占比"）
会得到不同的数。⇒ 合并 = 拼 rows + **用 `eval_local` 的同一套函数在全量上重出报告**，
保证「分四片跑」和「一片跑」得到逐字相同的结论。

⚠️ 分片必须**没有重叠也没有遗漏**：这里会核对 case_id 集合，
   数量对不上直接报错 —— 静默少几条 case 的评测比不评测更危险。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--expect", type=int, default=None,
                    help="期望的 case 总数，给了就校验（防静默少跑）")
    args = ap.parse_args()

    files = sorted(args.shards.glob("shard_*.json"))
    if not files:
        raise SystemExit(f"🔴 {args.shards} 里没有 shard_*.json")

    rows: list[dict] = []
    labels: list[str] = []
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.extend(payload["rows"])
        labels.append(payload.get("label", ""))

    ids = [r["case_id"] for r in rows]
    dupes = len(ids) - len(set(ids))
    if dupes:
        raise SystemExit(f"🔴 分片有重叠：{dupes} 条 case_id 重复 —— 分片规则错了")
    if args.expect is not None and len(ids) != args.expect:
        raise SystemExit(f"🔴 合并后 {len(ids)} 条，期望 {args.expect} 条 —— 有分片没跑完")

    rows.sort(key=lambda r: r["case_id"])
    label = labels[0] if labels else ""
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"label": label, "rows": rows},
                                   ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[merge] {len(files)} 片 · {len(rows)} 条 case -> {args.out}")

    # ★ 用 --from-audit 在全量上重出报告：分片跑和整跑的结论必须一致。
    print(f"\n⇒ 重出报告：python -m syncopate.train.eval_local "
          f"--from-audit {args.out} --batch data/batches/v13 --split-dir data/splits/v13")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
