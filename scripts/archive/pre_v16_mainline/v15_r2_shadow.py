#!/usr/bin/env python
"""v15 · R2 S5 影子重建 —— 同一份配置重跑一遍，产物必须**逐字节一致**。

    SYNCOPATE_CONTRACT=v15 .venv/bin/python scripts/v15_r2_shadow.py

★ 这一步是「数据可复现」的唯一证据，不许因为赶时间跳过（P4-1 同法）。
⚠️ 口径说明：v15 的教师生成**不是**确定性的（vLLM 批处理会影响数值），
   所以"可复现"的范围是**给定冻结的教师物料之后的装配路径**——
   这也正是 v14.5 的口径（缓存 = 冻结的物料）。装配路径不确定的话，
   同一份物料两次会产出不同的训练数据，那才是真正危险的那种不可复现。

判据：train.parquet / val.parquet / manifest.json 的 SHA256 两次一致。
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

OUT = Path("data/sft/v15")
SHADOW = Path("data/sft/v15_shadow")


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    if not (OUT / "train.parquet").exists():
        print(f"🔴 先得有一份正式产物：{OUT}/train.parquet")
        return 1
    before = {f: sha(OUT / f) for f in ("train.parquet", "val.parquet", "manifest.json")}
    print("第一份产物 SHA256:")
    for k, v in before.items():
        print(f"  {k:16s} {v[:32]}…")

    # 影子跑：把正式产物挪开，重跑一次（缓存=冻结物料，教师不会被再调用）
    if SHADOW.exists():
        shutil.rmtree(SHADOW)
    OUT.rename(SHADOW)
    env = dict(os.environ, SYNCOPATE_CONTRACT="v15")
    print("\n影子重建中（复用同一份冻结物料）…", flush=True)
    p = subprocess.run([sys.executable, "scripts/v16_build_sft.py"],
                       capture_output=True, text=True, env=env)
    if not (OUT / "train.parquet").exists():
        print(f"🔴 影子重建失败:\n{p.stdout[-3000:]}\n{p.stderr[-2000:]}")
        shutil.rmtree(OUT, ignore_errors=True)
        SHADOW.rename(OUT)
        return 1
    after = {f: sha(OUT / f) for f in ("train.parquet", "val.parquet", "manifest.json")}

    bad = 0
    print("\n════ S5 影子重建（逐字节）════")
    for k in before:
        ok = before[k] == after[k]
        bad += int(not ok)
        print(f"  {k:16s} {'✅ 一致' if ok else '🔴 不一致'}")
    Path("_audit/v15_r2").mkdir(parents=True, exist_ok=True)
    Path("_audit/v15_r2/shadow.json").write_text(json.dumps(
        {"before": before, "after": after, "identical": bad == 0}, indent=2))
    shutil.rmtree(SHADOW, ignore_errors=True)
    print(f"产物 → _audit/v15_r2/shadow.json")
    return bad


if __name__ == "__main__":
    raise SystemExit(main())
