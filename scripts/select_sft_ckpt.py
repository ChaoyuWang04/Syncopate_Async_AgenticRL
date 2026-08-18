#!/usr/bin/env python
"""SFT 选点：按正确的尺子排序候选，**并把删除做成流程的收尾动作**。

    python scripts/select_sft_ckpt.py checkpoints/sft/v14              # 只排序，不删
    python scripts/select_sft_ckpt.py checkpoints/sft/v14 --keep epoch1 --prune

★★★ 为什么删除必须内建在这个脚本里

这些 `sel_f*` 是**选点用的临时产物**（每个 ~126 MB），选完就没用了。
但"跑完记得去删"是一个**手动步骤** —— 而本项目的第一失效形状就是
「机制建好了，然后假设它会自动生效」。手动的清理一定会被忘，
然后下一个人看到一堆 `sel_f0.25/` 也不敢删（"万一还有用？"）。

⇒ **让删除是"选点"这个动作的一部分**，而不是一件独立的、要记得做的事。

★★ 选点的判据（`docs/syncopate/14-sft-health-metrics.md §1-②`）

    **决策位熵高 + 有梯度格子多**，**不看 val_loss**。

⚠️ 为什么不看 val_loss：训得越狠输出熵越低，接上 GRPO 就探索不动
   —— 我们踩过一次（选了 val loss 最低的那版，零梯度格子 63%）。
⚠️⚠️ 而且在 v13 这份数据上 val_loss **根本不含信息**：
   val 的 21 个模板家族 **100% 与 train 重合**（全库仅 160 种句式）
   ⇒ 它降到 0.011 只说明背下来了。
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _candidates(out_dir: Path) -> list[Path]:
    """所有候选：epoch* + sel_f*（后者是临时选点产物）。"""
    return sorted([d for d in out_dir.iterdir()
                   if d.is_dir() and (d.name.startswith("epoch") or d.name.startswith("sel_f"))])


def _metric(kind: str, cand_dir: Path) -> dict | None:
    """从 `_audit/` 里找这个候选的审计 —— **只按 label 里的 adapter 路径精确匹配**。

    ⚠️⚠️ 第一版用了文件名的模糊包含（`_epoch1.` in name），当场匹配到了
    **v3 时代**的 `M1_ctrl_epoch1.json`，并打出 `熵 0.483 / 有梯度 44` ——
    一组**看起来完全合理、其实是另一个模型**的数字。
    这正是本项目反复栽的那个形状：**判据看起来在量，量的却是另一件事，
    而且它产出了一个具体的数字**（`19 §2.1` 第三层，最贵的那一层）。

    ⇒ 改成：只认 label 里出现**这个候选的相对路径**；认不出就返回 None。
      **宁可报"没有"，也不要猜。**
    """
    want = str(cand_dir.relative_to(ROOT)) if cand_dir.is_absolute() else str(cand_dir)
    for p in sorted((ROOT / "_audit").glob("*.json")):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        label = str(d.get("label", ""))
        if want not in label:
            continue
        is_entropy = "decision_mean_entropy" in d
        if (kind == "entropy") == is_entropy:
            return d
    return None


def _grad_alive(audit: dict) -> tuple[int, int, int]:
    """(有梯度, 饱和, 卡死) —— 判据同 `compare.py`：组内 std 与均分。"""
    rows = audit.get("rows") or []
    alive = sum(r.get("reward_std", 0) > 0.01 for r in rows)
    sat = sum(r.get("reward_std", 0) <= 0.01 and r.get("reward", 0) > 0.9 for r in rows)
    stuck = len(rows) - alive - sat - sum(
        r.get("reward_std", 0) <= 0.01 and r.get("reward", 0) < 0.15 for r in rows)
    return alive, sat, stuck


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="SFT 选点 + 清理临时产物")
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--keep", default=None, help="选中的候选名（如 epoch1 / sel_f0.25）")
    ap.add_argument("--prune", action="store_true", help="删掉所有未选中的 sel_f*")
    args = ap.parse_args(argv)

    out = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    cands = _candidates(out)
    if not cands:
        print(f"🔴 {out} 下没有候选（epoch* / sel_f*）")
        return 1

    print(f"{'候选':<14}{'大小MB':>8}{'决策位熵':>10}{'有梯度':>8}{'饱和':>7}{'卡死':>7}   判据：熵高 + 有梯度多")
    rows = []
    for d in cands:
        mb = sum(f.stat().st_size for f in d.glob("*")) / 1048576
        ent = _metric("entropy", d)
        e = ent.get("decision_mean_entropy") if ent else None
        ev = _metric("eval", d)
        alive = sat = stuck = None
        if ev:
            alive, sat, stuck = _grad_alive(ev)
        rows.append((d, mb, e, alive))
        f = lambda v, w, p=3: (f"{v:>{w}.{p}f}" if isinstance(v, float)
                               else (f"{v:>{w}}" if v is not None else "  —".rjust(w)))
        print(f"{d.name:<14}{mb:>8.0f}{f(e, 10)}{f(alive, 8)}{f(sat, 7)}{f(stuck, 7)}")

    missing = [r[0].name for r in rows if r[2] is None and r[3] is None]
    if missing:
        print(f"\n⚠️ 这些候选还没有审计（**没找到就报没有，不猜**）：{missing}"
              "\n   先跑："
              "\n   python -m syncopate.train.entropy    --adapter <候选> --limit 24"
              "\n   python -m syncopate.train.eval_local --adapter <候选> ...")
        print("⚠️ **不要拿 val_loss 选** —— 它在这份数据上不含信息（见本脚本文档串）")

    if not args.prune:
        if args.keep:
            print(f"\n（未加 --prune，不会删任何东西。要清理：--keep {args.keep} --prune）")
        return 0

    if not args.keep:
        print("\n🔴 --prune 必须配 --keep：不指定选中谁就删，等于随机丢弃")
        return 1
    victims = [d for d in cands if d.name.startswith("sel_f") and d.name != args.keep]
    if not victims:
        print("\n✅ 没有可删的临时产物")
        return 0
    freed = sum(f.stat().st_size for d in victims for f in d.glob("*")) / 1048576
    for d in victims:
        shutil.rmtree(d)
        print(f"  删 {d.name}")
    print(f"✅ 回收 {freed:.0f} MB（保留 {args.keep} 与所有 epoch*）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
