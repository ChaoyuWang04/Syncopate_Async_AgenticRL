"""独立复核器：训练集 parquet 的每一行底题都在它该在的桶里（三桶隔离硬机制的第三层，2026-09-04）。

    python -m syncopate.pipeline.split_isolation data/sft/v16/train.parquet data/sft/v16/val.parquet --pool sft [--split-dir data/splits/v16]

退出码 0 = 全部在桶内；1 = 有越桶行（逐条打印）。派生行按 `source_case_ids` 列核，没有该列按编号后缀反推底题。
被谁调：建库探针（build_v16 判据）· syncopate/pipeline/invariants.py · 起训自查。
"""
from __future__ import annotations

import argparse
from pathlib import Path

from syncopate.pipeline.split import DEFAULT_SPLIT_DIR, split_isolation_report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("parquets", nargs="+")
    ap.add_argument("--pool", default="sft", choices=["eval", "sft", "rl"])
    ap.add_argument("--split-dir", default=DEFAULT_SPLIT_DIR)
    a = ap.parse_args()
    import pandas as pd
    bad = 0
    for pq in a.parquets:
        df = pd.read_parquet(pq)
        if "case_id" not in df.columns and "extra_info" in df.columns:
            df = pd.DataFrame({"case_id": [e["case_id"] for e in df["extra_info"]]})
        rep = split_isolation_report(df, Path(a.split_dir), a.pool)
        c = rep["counts"]
        print(f"[隔离] {pq}: 行 {len(df)} · 底题 eval={c['eval']} sft={c['sft']} rl={c['rl']} 无底题={c['none']} · 越桶 {len(rep['offenders'])}"
              + ("  ✅" if rep["ok"] else "  🔴"))
        for cid, base, bucket in rep["offenders"][:20]:
            print(f"   ✗ {cid} ← {base} 在 {bucket} 桶")
        bad += len(rep["offenders"])
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
