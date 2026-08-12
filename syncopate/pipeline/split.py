"""三桶切分：EVAL（冻结）/ SFT / RL，互斥且用 SHA-256 实测。

    EVAL   冻结，永不训练      ← 先切，写死 case_id 列表，单独文件
      ↓
    剩余池
      ├── SFT 桶（最难的一批）
      └── RL 桶（其余）

★ 为什么 EVAL 要先切、要冻结、要写成独立文件

之前只有 train/val 两桶，且 val 来自同一批模板——所有超参和数据设计决策
都是在 val 上做的，**val 已经被反复看过，等于已污染**。
再加上实测出的 6 条内容级泄漏，评测数字整体不可信。

冻结的含义是：`eval_cases.json` 一旦生成，**case_id 列表不再变**。
后续换数据版本时，同名 case 必须仍然存在且内容一致，否则前后评测不可比。

★ 为什么 SFT 该吃"最难的"

    简单 case RL 自己能搜到；只有死格（p≈0）是 RL 永远够不着的。
    SFT 冷启动的职责是把 p 从 0 抬到 5–10%，不是抬到 90%。

所以 SFT 桶的正确选法是「base 上 p≈0 的那批」。但这需要先跑 base 评测——
鸡生蛋问题。本模块因此提供两种选法：

    difficulty  —— M0 用。按难度标签 + 链长的启发式代理
    dead_grid   —— base 评测出来后用。按实测死格清单精确选

⚠️ 两者切出来的 SFT 桶不同，切换时必须重新训练，不能混用。

★ dead_grid 模式为什么不是「按 case_id 精确匹配」

死格是在**冻结 EVAL** 上测出来的，而 EVAL 永不进训练——精确匹配的结果是空桶。
所以要走一次外推：EVAL 里的死 case → 它所在的格子 → 池子里同格的 case，
每格按死因配额取。死因的划分与配额见 `dead_grid.py`。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from syncopate.core.schemas import CaseBundle
from syncopate.train.rollout_loop import build_messages


def content_hash(bundle: CaseBundle) -> str:
    """内容指纹：模型看到的 prompt + 该给的答案。

    ⚠️ 用 SHA-256，不用 Python 的 `hash()`——后者从 3.3 起每次进程启动会加盐随机化，
    跨进程比对得到的"无重叠"是假的。
    """
    payload = json.dumps({
        "messages": build_messages(bundle, bundle.case.tool_menu),
        "tools": sorted(bundle.case.tool_menu or []),
        "answer": bundle.gold.final_answer if bundle.gold else None,
        "behavior": bundle.verifier.expected_behavior,
    }, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class Buckets:
    eval: list[str] = field(default_factory=list)
    sft: list[str] = field(default_factory=list)
    rl: list[str] = field(default_factory=list)

    def counts(self) -> dict[str, int]:
        return {"eval": len(self.eval), "sft": len(self.sft), "rl": len(self.rl)}


def _difficulty_rank(bundle: CaseBundle) -> tuple[int, int, str]:
    """M0 的难度代理：难度标签 → 链长 → case_id（稳定排序）。

    真正的难度应该按 base 实测的 p 值排（见模块 docstring），
    这里只是没有 base 数据时的替代。降序排在前面的进 SFT 桶。
    """
    level = int((bundle.case.metadata.difficulty or "L1")[1:])
    chain = len(bundle.gold.actions) if bundle.gold else 0
    return (level, chain, bundle.case_id)


# ⚠️ 分层维度必须细到"分支"，不能只到"模板"。
# 早期用 (模板, behavior, 难度) 分层只得到 9 个格子——因为每个模板的
# behavior 和难度基本是固定的，等于只按模板分了一次。
# 结果 EVAL 只有 18 条，且**每种结局分支只有 0-2 条**，
# 某个分支上的失败在评测里根本看不见。
# 现在按 (模板, behavior, 结局, entry_mode) 分——结局是控制轴的产物，
# 它才是"这条 case 走哪条路"的真正标识。
def stratum(cid: str, b: CaseBundle) -> tuple[str, ...]:
    tags = {t.split(":", 1)[0]: t.split(":", 1)[1]
            for t in b.case.metadata.tags if ":" in t}
    return (cid.split("_")[0], b.verifier.expected_behavior,
            tags.get("outcome", "-"), tags.get("entry", "-"))


def load_bundles(batch_dir: Path) -> dict[str, CaseBundle]:
    manifest = json.loads((batch_dir / "manifest.json").read_text(encoding="utf-8"))
    entries = sorted(manifest["entries"], key=lambda e: e["case_id"])
    return {e["case_id"]: CaseBundle.read(batch_dir, e["case_id"]) for e in entries}


# ★ 稀有行为在 EVAL 里要加厚，否则那个指标没有分辨力。
#
# 实测（v10）：EVAL 里 tool_call 156 条，而 clarify / defer / reject **各只有 4 条**。
# M6 的毕业条件之一是「行为分类准确率 ≥ 90%」—— 4 条样本下，**答错一条就掉 25 个
# 百分点**，这个数根本区分不出 0.88 和 0.92。
#
# 尤其是 defer：M1 花了整整一个里程碑建它（数据成熟度 + premature_decision_cap），
# 考试时只有 4 道题。「训练里没有 ⇒ 评测里也没有 ⇒ 失败模式不可见」这句话，
# 对**量太少**同样成立。
#
# ⚠️ 加厚只动 EVAL 的取样数，不造新数据 —— 池子里本来就有，只是没被取上来。
RARE_BEHAVIOR_EVAL_QUOTA = {"clarify": 4, "reject": 4, "defer": 4, "answer": 3}


def split(
    batch_dir: Path,
    *,
    eval_per_stratum: int = 2,
    sft_ratio: float = 0.20,
    dead_grids: dict[tuple[str, ...], Any] | None = None,
    quota_by_kind: dict[str, int] | None = None,
) -> tuple[Buckets, dict[str, Any]]:
    """切三桶。

    EVAL 按 (模板 × behavior × 结局 × entry_mode) 分层取样，保证每个格子都有代表——
    否则某类失败模式在评测里完全不可见。

    `dead_grids` 给了就走 dead_grid 模式：值是 `dead_grid.DeadGrid`，
    按其 `kind` 到 `quota_by_kind` 查这一格取几条。
    """
    bundles = load_bundles(batch_dir)

    # ---- 1. EVAL：分层冻结 ----
    strata: dict[tuple[str, ...], list[str]] = {}
    for cid, b in bundles.items():
        strata.setdefault(stratum(cid, b), []).append(cid)
    buckets = Buckets()
    for key in sorted(strata):
        # key[1] 是 expected_behavior。稀有行为多取几条，见 RARE_BEHAVIOR_EVAL_QUOTA。
        take = max(eval_per_stratum, RARE_BEHAVIOR_EVAL_QUOTA.get(key[1], 0))
        buckets.eval.extend(sorted(strata[key])[:take])
    eval_set = set(buckets.eval)

    # ---- 2. 剩余池 → SFT（最难的）/ RL ----
    pool = [cid for cid in sorted(bundles) if cid not in eval_set]
    selection: dict[str, Any] | None = None
    if dead_grids:
        # 死格模式：EVAL 测出来的死格 → 池子里同格的 case，每格按死因配额取。
        # 取样用 case_id 排序而不是随机——切分必须可复现，随机化会让
        # 「换了配额」和「换了随机种子」两件事在结果里分不开。
        quota = dict(quota_by_kind or {})
        by_stratum: dict[tuple[str, ...], list[str]] = {}
        for cid in pool:
            by_stratum.setdefault(stratum(cid, bundles[cid]), []).append(cid)

        chosen: list[str] = []
        selection = {}
        for key in sorted(dead_grids):
            grid = dead_grids[key]
            available = sorted(by_stratum.get(key, []))
            cap = quota.get(grid.kind, len(available))
            taken = available[:cap]
            chosen.extend(taken)
            selection["|".join(key)] = {
                "kind": grid.kind,
                "eval_evidence": f"{len(grid.eval_case_ids)}/{grid.eval_seen}",
                "available": len(available),
                "taken": len(taken),
            }
        buckets.sft = sorted(chosen)
        sft_set = set(buckets.sft)
        buckets.rl = [c for c in pool if c not in sft_set]
        mode = "dead_grid"
    else:
        ranked = sorted(pool, key=lambda c: _difficulty_rank(bundles[c]), reverse=True)
        cut = max(1, int(len(pool) * sft_ratio))
        buckets.sft = sorted(ranked[:cut])
        buckets.rl = sorted(ranked[cut:])
        mode = "difficulty_proxy"

    # ---- 3. 互斥性实测（不看代码猜）----
    hashes = {cid: content_hash(b) for cid, b in bundles.items()}
    sets = {name: {hashes[c] for c in ids} for name, ids in
            (("eval", buckets.eval), ("sft", buckets.sft), ("rl", buckets.rl))}
    overlaps = {
        "eval∩sft": len(sets["eval"] & sets["sft"]),
        "eval∩rl": len(sets["eval"] & sets["rl"]),
        "sft∩rl": len(sets["sft"] & sets["rl"]),
    }
    # 全局内容去重检查：同一份内容出现在两个 case_id 下也算泄漏
    dupes = len(hashes) - len(set(hashes.values()))

    report = {
        "batch_dir": str(batch_dir),
        "mode": mode,
        "counts": buckets.counts(),
        "strata_count": len(strata),
        "overlaps_by_content_sha256": overlaps,
        "duplicate_content_pairs": dupes,
        "eval_strata_coverage": {"|".join(k): len(v) for k, v in sorted(strata.items())},
    }
    if selection is not None:
        report["dead_grid_selection"] = selection
        report["quota_by_kind"] = dict(quota_by_kind or {})
    return buckets, report


def write(buckets: Buckets, report: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # EVAL 单独成文件并标 frozen —— 它是唯一不该被随便重生成的东西
    (out_dir / "eval_cases.json").write_text(json.dumps({
        "frozen": True,
        "note": "冻结集。case_id 列表不得变更；换数据版本时同名 case 内容必须一致，否则前后评测不可比。",
        "count": len(buckets.eval),
        "case_ids": buckets.eval,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    (out_dir / "sft_cases.json").write_text(json.dumps(
        {"count": len(buckets.sft), "case_ids": buckets.sft}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    (out_dir / "rl_cases.json").write_text(json.dumps(
        {"count": len(buckets.rl), "case_ids": buckets.rl}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    (out_dir / "split_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")


def load_bucket(split_dir: Path, name: str) -> list[str]:
    path = split_dir / f"{name}_cases.json"
    return json.loads(path.read_text(encoding="utf-8"))["case_ids"]
