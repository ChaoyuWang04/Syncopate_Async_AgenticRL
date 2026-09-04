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
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ★ 数据版本从**共用常量**取，不在这里再写一份（本项目为"同一件事两份实现"付过多次钱）
from syncopate.pipeline.split import (  # noqa: E402
    DATA_VERSION, DEFAULT_BATCH_DIR, DEFAULT_SPLIT_DIR,
)


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

    ★★ 2026-08-18 晚补的第二道门：**数据版本也要对得上。**
    上面那道门只管"是不是同一个模型"，管不了"是不是同一份数据"。
    而 `entropy.py` 的 `--batch/--split-dir` 是必须同时动的一对、默认值又是陈旧的 v3
    ⇒ 一份在 v3 eval 桶上量出来的熵，模型路径完全对得上、会被这里收下。
      （v3 eval 64 条 vs v13 343 条、交集仅 49；且其中有落在 v13 训练桶里的题。）
    ⇒ 所以：审计**写着** `data_version` 而不等于期望版本的，一律当"没有"。

    ⚠️⚠️ 但**没写**这个字段的要区别对待 —— 那是这次改造**之前**的老产物
      （`_audit/v13_sft_e1.json` 等，按 `21 §3.5` 是**已知有效**的 v13 审计）。
      把它们也判成"没有"就是过度作废：会扔掉有效证据、还逼一次 15 分钟的 GPU 重评，
      而「过度作废的代价和引用错数字一样大」（`21 §3` 开头那条）。
    ⇒ 分界：**身份**（是哪个模型）由 label 定，本来就是确定的；缺的只是**出处**。
      ⇒ 收下，但由调用方**显式标出"版本未记录"**，让人自己看见，而不是静默当成对的。
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
        if (kind == "entropy") != is_entropy:
            continue
        ver = d.get("data_version")
        if ver != DATA_VERSION:
            continue                      # 写着别的版本或没写版本 ⇒ 不是这次要的，跳过（09-05：此前没写版本的旧审计也会被接受）
        return d
    return None


def _grad_alive(audit: dict) -> tuple[int, int, int, float | None]:
    """(有梯度, 饱和, 卡死, **零梯度占比**) —— 判据同 `compare.py`：组内 std 与均分。

    零梯度的格子分成**三类**（std ≤ 0.01，8 次采样几乎同分）：

        饱和   分高（> 0.9）    已经会了，没梯度**不是问题**
        死格   分低（< 0.15）   全错且稳定错 —— RL 够不着，该由 SFT 覆盖去解
        卡死   中间分           在里面打转（GEO 那一类，`01 §P0-1.3`）

    ⚠️ 旧版**只返回前三个**，而"死格"那一类是用减法隐式算掉的、从没打印过 ——
      于是这张表加起来不等于总数（v13-e1 实测 222+96+22=340，少 3 条），
      而少掉的正是"全错且稳定错"那一类。
    ⚠️⚠️ 更要紧的是：**M6 的毕业条件问的是"零梯度占比 < 30%"**，
      而这张表从来没有把这个数打出来 —— 人得自己拿 1−有梯度/总数 心算。
      判据不在屏幕上，就等于判据没接上（`01 §P2-2` 记的 e1 = 35.3% **不达标**）。
    """
    rows = audit.get("rows") or []
    if not rows:
        return 0, 0, 0, None
    flat = [r for r in rows if r.get("reward_std", 0) <= 0.01]
    alive = len(rows) - len(flat)
    sat = sum(r.get("reward", 0) > 0.9 for r in flat)
    dead = sum(r.get("reward", 0) < 0.15 for r in flat)
    stuck = len(flat) - sat - dead
    return alive, sat, stuck, len(flat) / len(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="SFT 选点 + 清理临时产物")
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--keep", default=None, help="选中的候选名（如 epoch1 / sel_f0.25）")
    ap.add_argument("--prune", action="store_true", help="删掉所有未选中的 sel_f*")
    ap.add_argument("--auto", action="store_true",
                    help="固定管线用：所有候选都有审计时按（有梯度多 → 决策位熵高）自动选，写 <out>/SELECTED 软链；缺审计就红，不猜")
    args = ap.parse_args(argv)

    out = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    cands = _candidates(out)
    if not cands:
        print(f"🔴 {out} 下没有候选（epoch* / sel_f*）")
        return 1

    print(f"{'候选':<14}{'大小MB':>8}{'决策位熵':>10}{'有梯度':>8}{'饱和':>7}{'卡死':>7}"
          f"{'零梯度%':>9}   判据：熵高 + 有梯度多（数据版本 {DATA_VERSION}）")
    rows = []
    unversioned = []
    for d in cands:
        mb = sum(f.stat().st_size for f in d.glob("*")) / 1048576
        ent = _metric("entropy", d)
        e = ent.get("decision_mean_entropy") if ent else None
        ev = _metric("eval", d)
        alive = sat = stuck = dead_pct = None
        if ev:
            alive, sat, stuck, dead_pct = _grad_alive(ev)
        for a in (ent, ev):
            # ⚠️ 老产物没有 `data_version`：**收下但要说出来**，不静默当成对的。
            if a is not None and a.get("data_version") is None:
                unversioned.append(d.name)
        rows.append((d, mb, e, alive))
        f = lambda v, w, p=3: (f"{v:>{w}.{p}f}" if isinstance(v, float)
                               else (f"{v:>{w}}" if v is not None else "  —".rjust(w)))
        pct = f"{dead_pct:>8.1%}" if dead_pct is not None else "       —"
        print(f"{d.name:<14}{mb:>8.0f}{f(e, 10)}{f(alive, 8)}{f(sat, 7)}{f(stuck, 7)}{pct}")

    if unversioned:
        print(f"\nℹ️ 这些候选的审计**没记数据版本**（本次改造之前的老产物）："
              f"{sorted(set(unversioned))}"
              f"\n   已按 label 里的模型路径收下，但它量在哪份数据上**无从确认** ——"
              f"\n   要它确定，重跑一次即可（下面的命令已带上版本参数）。")

    missing = [r[0].name for r in rows if r[2] is None and r[3] is None]
    if missing:
        # ★ 打**照抄就能跑**的完整命令。此前这里印的是
        #   `--adapter <候选> --limit 24`，缺 `--batch/--split-dir/--out` 三样：
        #   缺 --out ⇒ 根本不落审计文件，这张表永远显示"—"；
        #   缺版本参数 ⇒ 默认值是陈旧的 v3（`data/batches/v3` 本机都不存在）。
        #   ⇒ 「打印出来的指令本身就是接口」，印不全就等于机制没接上。
        print(f"\n⚠️ 这些候选还没有审计（**没找到就报没有，不猜**）：{missing}")
        for name in missing:
            cand = f"{out.relative_to(ROOT)}/{name}"
            print(f"\n   # {name}")
            print(f"   python -m syncopate.train.entropy --adapter {cand} --limit 24 \\")
            print(f"       --batch {DEFAULT_BATCH_DIR} --split-dir {DEFAULT_SPLIT_DIR} \\")
            print(f"       --out _audit/{DATA_VERSION}_entropy_{name}.json")
            print(f"   python -m syncopate.train.eval_local --adapter {cand} "
                  f"--samples-per-case 8 \\")
            print(f"       --batch {DEFAULT_BATCH_DIR} --split-dir {DEFAULT_SPLIT_DIR} \\")
            print(f"       --out _audit/{DATA_VERSION}_eval_{name}.json")
        print("\n⚠️ **不要拿 val_loss 选** —— 它在这份数据上不含信息（见本脚本文档串）")
        print("⚠️ 先只跑熵（≈1 分钟/点）排序，前二才跑 eval_local（15 分钟/点）")

    if args.auto:
        if missing:
            print("\n🔴 --auto 要求每个候选都有 entropy + eval 审计（上面列了缺的）—— 先跑 sft-eval")
            return 1
        ranked = sorted(rows, key=lambda r: (-(r[3] or 0), -(r[2] or 0.0)))
        best = ranked[0][0]
        args.keep = best.name
        link = out / "SELECTED"
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(best.name)
        print(f"\n✅ --auto 选中 {best.name}（有梯度 {ranked[0][3]} · 熵 {ranked[0][2]}）→ {link}")
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
