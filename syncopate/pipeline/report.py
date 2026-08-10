"""训练数据分布体检报告。

回答的问题是：**我们到底在训练什么？**

不是"有多少条数据"，而是：
  - 按**意图**分，每类各有多少？（意图 = 用户想干什么，比模板名更贴近业务）
  - 每个意图的**链长**铺开了吗？还是全挤在同一个长度上？
  - 工具的使用是不是极度偏斜（几个工具吃掉大部分调用）？
  - 有没有哪个意图只有一种走法（= 模型认出意图就赢了）？
  - **val 是不是真的"没见过"**——同模板不同参数，算不算泛化？

最后一条尤其重要：我们的 val 和 train 来自同一批模板，只是 index 不同。
所以 val 上的高分只能说明"模板内泛化"，**不能说明模型学会了这类业务**。
这份报告会把这件事显式标出来，免得被漂亮的数字误导。

    python -m syncopate data report --batch data/batches/v2
"""

from __future__ import annotations

import collections
import json
import statistics
from pathlib import Path
from typing import Any

# 模板 -> 业务意图。意图是"用户想干什么"，比模板名更贴近真实业务分类。
TEMPLATE_INTENT = {
    "BUD": "budget_change",       # 调预算
    "CRE": "creative_launch",     # 素材投放决策
    "DIA": "anomaly_diagnosis",   # 异常诊断
    "LOW": "portfolio_review",    # 大盘复盘
    "LONG": "creative_upload",    # 素材上传 + 等审核
    "MISS": "anomaly_diagnosis",  # 同诊断，但工具缺失
    "HIGH": "metric_lookup",      # 单指标查询
    "CLAR": "clarify_boundary",   # 信息不足
    "REJ": "reject_boundary",     # 越权/离题
}


def load_cases(batch_dir: Path) -> list[dict[str, Any]]:
    manifest = json.loads((batch_dir / "manifest.json").read_text(encoding="utf-8"))
    rows = []
    for entry in manifest["entries"]:
        cid = entry["case_id"]
        case = json.loads((batch_dir / "cases" / f"{cid}.json").read_text(encoding="utf-8"))
        gold = json.loads((batch_dir / "gold_paths" / f"{cid}.gold.json").read_text(encoding="utf-8"))
        spec = json.loads((batch_dir / "verifier_specs" / f"{cid}.verifier.json").read_text(encoding="utf-8"))
        tools = [a["tool"] for a in gold["actions"]]
        prefix = cid.split("_")[0]
        tags = case.get("metadata", {}).get("tags", [])
        rows.append({
            "case_id": cid,
            "template": prefix,
            "intent": TEMPLATE_INTENT.get(prefix, prefix),
            "behavior": spec.get("expected_behavior", "tool_call"),
            "difficulty": case.get("metadata", {}).get("difficulty"),
            "chain": tuple(tools),
            "chain_len": len(tools),
            "n_read": len(spec.get("required_read_tools") or []),
            "n_write": len(spec.get("required_side_effects") or []),
            "n_answer": len(spec.get("required_answer_fields") or []),
            "n_caps": len(spec.get("active_caps") or []),
            "outcome": next((t.split(":", 1)[1] for t in tags if t.startswith("outcome:")), "-"),
            "axes": {t.split(":", 1)[0]: t.split(":", 1)[1] for t in tags if ":" in t},
        })
    return rows


def _bar(n: int, total: int, width: int = 26) -> str:
    filled = round(width * n / max(1, total))
    return "█" * filled + "·" * (width - filled)


def render(rows: list[dict[str, Any]], val_every: int) -> str:
    out: list[str] = []
    total = len(rows)
    add = out.append

    add(f"共 {total} 条 case\n")

    # ---------------- 1. 按意图 ----------------
    add("=" * 78)
    add("1 · 按业务意图")
    add("=" * 78)
    by_intent: dict[str, list[dict]] = collections.defaultdict(list)
    for r in rows:
        by_intent[r["intent"]].append(r)
    add(f"{'意图':<20}{'条数':>5}{'占比':>7}  {'链长 min/中位/max':<18}{'骨架数':>6}  分布")
    for intent, group in sorted(by_intent.items(), key=lambda kv: -len(kv[1])):
        lens = [g["chain_len"] for g in group]
        skel = len({g["chain"] for g in group})
        add(f"{intent:<20}{len(group):>5}{len(group)/total:>7.1%}  "
            f"{min(lens):>3} /{statistics.median(lens):>4.0f} /{max(lens):>4}      {skel:>6}  "
            f"{_bar(len(group), total)}")

    # ---------------- 2. 链长分布 ----------------
    add("")
    add("=" * 78)
    add("2 · 链长分布（每个意图内部是不是从短到长都有）")
    add("=" * 78)
    add(f"{'意图':<20}" + "".join(f"{n:>5}" for n in range(0, 9)) + "   ← gold 步数")
    for intent, group in sorted(by_intent.items()):
        counts = collections.Counter(g["chain_len"] for g in group)
        add(f"{intent:<20}" + "".join(f"{counts.get(n, 0) or '·':>5}" for n in range(0, 9)))
    add("")
    all_lens = collections.Counter(r["chain_len"] for r in rows)
    add(f"  全局：{dict(sorted(all_lens.items()))}")
    add("  ⚠️ 若某个意图只集中在 1-2 个长度上，说明它内部没有「由简到繁」的梯度，")
    add("     模型认出意图 = 知道要走几步，curriculum 也无从下手。")

    # ---------------- 3. 骨架集中度 ----------------
    add("")
    add("=" * 78)
    add("3 · 骨架集中度（同一意图有几种走法）")
    add("=" * 78)
    for intent, group in sorted(by_intent.items(), key=lambda kv: -len(kv[1])):
        skeletons = collections.Counter(g["chain"] for g in group)
        top = skeletons.most_common(1)[0]
        flag = "  ⚠️ 只有一种走法" if len(skeletons) == 1 else ""
        add(f"  {intent:<20} {len(skeletons):>2} 种骨架，最常见的占 {top[1]/len(group):>5.1%}{flag}")
        for chain, n in skeletons.most_common(3):
            short = " → ".join(t.split(".")[-1] for t in chain) or "(不调工具)"
            add(f"      {n:>4} 条  {short[:88]}")

    # ---------------- 4. 工具使用 ----------------
    add("")
    add("=" * 78)
    add("4 · 工具调用频次（gold 里）")
    add("=" * 78)
    tool_counts = collections.Counter(t for r in rows for t in r["chain"])
    calls = sum(tool_counts.values())
    for tool, n in tool_counts.most_common():
        add(f"  {tool:<32}{n:>5}{n/calls:>7.1%}  {_bar(n, tool_counts.most_common(1)[0][1], 20)}")
    add(f"  合计 {calls} 次调用，用到 {len(tool_counts)} 个工具")

    # ---------------- 5. 行为与结局 ----------------
    add("")
    add("=" * 78)
    add("5 · 顶层行为 / 结局分布")
    add("=" * 78)
    for key, label in (("behavior", "expected_behavior"), ("outcome", "结局(分支轴产物)"),
                       ("difficulty", "难度标签")):
        counts = collections.Counter(r[key] for r in rows)
        add(f"  {label}: " + "  ".join(f"{k}={v}({v/total:.0%})" for k, v in sorted(counts.items())))

    # ---------------- 6. 控制轴 ----------------
    add("")
    add("=" * 78)
    add("6 · 控制轴取值分布")
    add("=" * 78)
    axis_names = sorted({a for r in rows for a in r["axes"]})
    for axis in axis_names:
        counts = collections.Counter(r["axes"].get(axis, "-") for r in rows)
        counts.pop("-", None)
        if not counts:
            continue
        n = sum(counts.values())
        add(f"  {axis:<10} " + "  ".join(f"{k}={v}({v/n:.0%})" for k, v in sorted(counts.items())))

    # ---------------- 7. train/val 切分的真实含义 ----------------
    add("")
    add("=" * 78)
    add("7 · train / val 切分 —— ★ 这一节决定怎么解读评测分数")
    add("=" * 78)
    val = [r for i, r in enumerate(sorted(rows, key=lambda x: x["case_id"])) if i % val_every == 0]
    train = [r for i, r in enumerate(sorted(rows, key=lambda x: x["case_id"])) if i % val_every != 0]
    add(f"  train {len(train)} 条 / val {len(val)} 条（每 {val_every} 条取 1 条进 val）")
    train_chains = {r["chain"] for r in train}
    unseen = [r for r in val if r["chain"] not in train_chains]
    add(f"  val 里**骨架在 train 中出现过**的: {len(val)-len(unseen)}/{len(val)}"
        f" ({(len(val)-len(unseen))/max(1,len(val)):.0%})")
    add("")
    add("  ⚠️ **这意味着 val 上的高分只证明「模板内泛化」，不证明学会了业务。**")
    add("     val 和 train 来自同一批模板，只是实体 id / 数值 / 轴取值不同。")
    add("     模型只要认出模板，就知道该走哪条骨架——剩下的只是把参数填对。")
    add("")
    add("     要测真正的泛化，需要 **hold out 整个模板或整条骨架**：")
    add("       · 留出某个意图完全不训（测跨意图迁移）")
    add("       · 留出某条骨架不训（测能否组合出没见过的路径）")
    add("       · 留出某个轴的某个取值不训（如只训 id_given，测 must_discover）")

    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="训练数据分布体检")
    parser.add_argument("--batch", default="data/batches/v2")
    parser.add_argument("--val-every", type=int, default=8)
    parser.add_argument("--out", default=None, help="同时写成 markdown 文件")
    args = parser.parse_args(argv)

    rows = load_cases(Path(args.batch))
    text = render(rows, args.val_every)
    print(text)
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# 训练数据分布报告\n\n> 数据源：`{args.batch}`\n\n```\n{text}\n```\n",
                        encoding="utf-8")
        print(f"\n[OK] -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
