"""检索阈值标定工具：把 `MATCH_THRESHOLD` 从"拍一个"变成"扫出来"。

★ 为什么需要它

`corpus.MATCH_THRESHOLD` 直接决定「检索为空」发生的频率，而**「无检索幻觉率」
这项验收（§14，要求趋近 0）整个挂在它上面**：
定高了空洞遍地、定低了空洞永不发生 —— **两种都会让那项验收失去意义**。
第一版这个数是拿两个字符串试出来的（样本量 2），这个工具是它的替代品。

★ 两种用法

    # ① 种子评测集（17 查询 × 10 条款，2026-08-14 自建）—— 换打分函数时用
    python -m syncopate.authoring.calibrate_retrieval

    # ② 真语料 + 真 gold 查询 —— ★ 造完题之后必须跑这个重标定
    python -m syncopate.authoring.calibrate_retrieval --from-batch data/batches/v12

⚠️ 种子集是**我自己写的**，比"两个字符串"强，但**不足以定案**。
真模板出来后必须用 ② 重标。

★ 判读：两类错误的代价**不对称**，别只看总分

    误召回  该空的返回了东西  ⇒ 那条 case 的「空洞」轴失效，考不了检索幻觉
    漏召回  该中的返回了空    ⇒ gold 要求作答但检索给不出
                              ⇒ 模型转人工被判错、编造反而蒙对
                              ★★ **等于在训练我们正要消灭的那个行为**

⇒ **漏召回更毒**。而且为了把漏掉的自然口语查询救回来，造题人只能去凑文档原词 ——
那又正好训出「把查询词凑成文档原词」这个只在沙盒里有用的错技能。

★★★ 但真正的保证不在阈值，在**造题时的双向断言**：

    应命中的 case  →  断言 gold 查询确实命中，且命中的是**对的那条**
    空洞的 case    →  断言该查询确实返回空

有这两条断言，阈值选偏一点也不会让哪条轴静默失效；**没有它们，阈值调到天上也没用**。
⇒ 所以本工具**只报权衡，不替你拍板**。

★ 平台期只是个参考信号：一段"阈值变动而指标不变"的区间说明该处不敏感。
若根本没有平台期（指标随阈值单调滑动），说明**相关与不相关之间没有分离带** ——
那不是阈值的问题，是打分函数不行，换阈值救不回来。
（2026-08-14 实测：Qwen3-0.6B mean-pool 向量就是这种情况，正确留空恒为 0 或全漏。）
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

from syncopate.domains.adcampaign.corpus import (  # noqa: E402
    MATCH_THRESHOLD, overlap_score,
)

# --------------------------------------------------------------------------
# 种子评测集（2026-08-14）
#
# 查询按「模型真的会怎么问」写：原词 / 换说法 / 只给关键词 / 以及**同主题但另一条款**。
# ⚠️ 刻意不用 tests/domains/test_corpus_retrieval.py 里的句子 ——
# 拿被测对象自己的样例证明自己，等于没测。
# --------------------------------------------------------------------------

SEED_DOCS = {
    "BUDGET_DAILY": "单日预算涨幅上限 单日预算上调不得超过前一日的 20%，超过需提交审批。 Meta 广告政策 / 4. 预算与竞价 / 4.2 单日涨幅",
    "BUDGET_MONTH": "月度预算上限 账户月度总消耗不得超过合同约定的月度上限，触顶后自动暂停投放。 Meta 广告政策 / 4. 预算与竞价 / 4.5 月度封顶",
    "BID_STRATEGY": "出价策略变更频率 同一广告组的出价策略 24 小时内最多变更一次，频繁变更会重置学习期。 Meta 广告政策 / 4. 预算与竞价 / 4.7 出价",
    "CREATIVE_REV": "素材审核时长 新上传素材需经过人工审核，通常 24 小时内完成，节假日可能延长至 72 小时。 Meta 广告政策 / 6. 素材 / 6.1 审核",
    "CREATIVE_TXT": "素材文案限制 素材文案不得包含绝对化用语、医疗功效承诺、以及未经证实的收益宣称。 Meta 广告政策 / 6. 素材 / 6.3 文案",
    "PRIVACY_IOS": "iOS 隐私标签要求 投放 iOS 应用需在应用商店声明数据收集类型，未声明的应用不得投放个性化广告。 平台合规 / 3. 隐私 / 3.2 iOS",
    "GEO_SEA": "东南亚地域投放限制 印尼、越南市场需额外提供本地化落地页，且素材需通过本地语言审核。 平台合规 / 5. 地域 / 5.4 东南亚",
    "AUDIENCE_AGE": "受众年龄下限 面向未成年人的定向投放需单独申请资质，默认最低定向年龄为 18 岁。 平台合规 / 4. 受众 / 4.1 年龄",
    "ATTR_WINDOW": "归因窗口设置 默认归因窗口为点击后 7 天、展示后 1 天，修改归因窗口会影响历史数据可比性。 MMP 接入 / 2. 归因 / 2.3 窗口",
    "REFUND_POLICY": "广告费退款规则 因平台系统故障导致的无效消耗可在 30 天内申请退款，需提供投放日志。 商务条款 / 8. 结算 / 8.2 退款",
}

SEED_QUERIES: list[tuple[str, str | None]] = [
    ("单日预算涨幅上限", "BUDGET_DAILY"),
    ("素材审核时长", "CREATIVE_REV"),
    ("iOS 隐私标签", "PRIVACY_IOS"),
    ("每天预算最多能加多少", "BUDGET_DAILY"),      # ★ 已知失败：词汇失配
    ("新素材要审多久", "CREATIVE_REV"),
    ("出价能不能一天改好几次", "BID_STRATEGY"),
    ("投印尼需要注意什么", "GEO_SEA"),
    ("归因窗口默认是多少天", "ATTR_WINDOW"),
    ("未成年人能不能定向", "AUDIENCE_AGE"),
    ("文案里能不能写疗效", "CREATIVE_TXT"),
    ("月度封顶", "BUDGET_MONTH"),
    ("退款", "REFUND_POLICY"),
    ("竞品品牌词能不能投", None),
    ("视频素材最长多少秒", None),
    ("代理商返点比例", None),
    ("像素安装失败怎么排查", None),
    ("Android 渠道包如何配置", None),
]


def load_from_batch(batch_dir: Path) -> list[tuple[str, dict[str, str], str | None]]:
    """从真实 batch 抽出**逐 case** 的 (查询, 该 case 自己的语料, 期望命中)。

    ★★ 必须逐 case，不能汇成一个全局库 —— 语料是**逐 case** 放在
    `env_snapshot.readonly_tables` 里的（见 corpus.py 的模块 docstring：
    GRPO 并发跑同一条 case 8 遍，任何跨 rollout 的共享状态都是污染）。

    ⚠️ 第一版就是汇成全局库的，结果所有 POL 条款文本完全相同 ⇒
    每条查询都命中同一条，失败明细刷了满屏。**这个工具的 --from-batch 路径
    自己就没被接上过** —— 正是它在 docstring 里警告的那个形状。

    ★ 期望有**三种**，不是两种（第一版只写了两种，把 insight 侧 62 条全误标成负例）：

        具体 id  gold 的 final_answer.cited_clause_id —— 必须命中且中这一条
        "*"      该档有语料但 gold 没指定哪一条 —— 只要求"查得到"
        None     空洞档（rag_state:empty / insight_state:absent）—— 必须查不到

    档位从 case 的 tags 取，**不是我事后补标的**：标定基准和训练目标同源。
    """
    items: list[tuple[str, dict[str, str], str | None]] = []
    for gold_file in sorted((batch_dir / "gold_paths").glob("*.json")):
        case_id = gold_file.name.split(".")[0]
        env_file = batch_dir / "env_snapshots" / f"{case_id}.env.json"
        if not env_file.exists():
            continue
        gold = json.loads(gold_file.read_text(encoding="utf-8"))
        calls = [c for c in gold.get("actions", [])
                 if c.get("tool") in ("policy.search", "insight.search_claims")
                 and (c.get("arguments") or {}).get("query")]
        if not calls:
            continue
        env = json.loads(env_file.read_text(encoding="utf-8"))
        tables = env.get("readonly_tables", {})
        docs: dict[str, str] = {}
        for row in (tables.get("policy_clauses") or {}).values():
            docs[row["clause_id"]] = " ".join(
                filter(None, [row.get("title"), row.get("body"), row.get("section_path")]))
        for row in (tables.get("insights") or {}).values():
            scope = row.get("scope") or {}
            docs[row["claim_id"]] = " ".join(
                [str(row.get("claim") or ""), *(str(v) for v in scope.values())])
        case_file = batch_dir / "cases" / f"{case_id}.json"   # ⚠️ 不是 .case.json
        tags = set()
        if case_file.exists():
            tags = set(json.loads(case_file.read_text(encoding="utf-8"))
                       .get("metadata", {}).get("tags", []))
        want = (gold.get("final_answer") or {}).get("cited_clause_id")
        if want is None:
            want = None if {"rag_state:empty", "insight_state:absent"} & tags else "*"
        for call in calls:
            items.append((call["arguments"]["query"], docs, want))
    return items


def sweep(items, thresholds, scorer=overlap_score):
    """items: [(query, 该条自带的语料, 期望命中或 None)]。逐条独立打分。"""
    n_pos = sum(1 for _, _, g in items if g is not None)
    n_neg = len(items) - n_pos
    raw = [{k: scorer(q, d) for k, d in docs.items()} for q, docs, _ in items]
    rows = []
    for th in thresholds:
        tp = fp = fn = tn = 0
        for (q, _docs, gold), scores in zip(items, raw):
            hits = sorted(((s, k) for k, s in scores.items() if s >= th), reverse=True)
            top = hits[0][1] if hits else None
            if gold is None:
                tn += int(not hits)
                fp += int(bool(hits))
            elif gold == "*":                 # 只要求查得到，不指定哪一条
                tp += int(bool(hits)); fn += int(not hits)
            elif top == gold:
                tp += 1
            elif hits:
                fp += 1
            else:
                fn += 1
        rows.append({"threshold": round(th, 3), "tp": tp, "fp": fp, "fn": fn, "tn": tn})
    return rows, raw, n_pos, n_neg


def find_plateau(rows: list[dict]) -> list[dict]:
    """找最长的一段「指标完全不变」的连续区间。没有平台期是个坏信号，见模块 docstring。"""
    best: list[dict] = []
    cur: list[dict] = []
    for row in rows:
        key = (row["tp"], row["fp"], row["fn"], row["tn"])
        if cur and (cur[-1]["tp"], cur[-1]["fp"], cur[-1]["fn"], cur[-1]["tn"]) == key:
            cur.append(row)
        else:
            cur = [row]
        if len(cur) > len(best) or (len(cur) == len(best) and cur[0]["tp"] + cur[0]["tn"] >
                                    (best[0]["tp"] + best[0]["tn"] if best else -1)):
            best = list(cur)
    return best


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="检索阈值标定")
    p.add_argument("--from-batch", type=Path, default=None,
                   help="用真实 batch 的语料和 gold 查询标定（造完题之后必须跑这个）")
    p.add_argument("--lo", type=float, default=0.10)
    p.add_argument("--hi", type=float, default=0.75)
    p.add_argument("--step", type=float, default=0.05)
    p.add_argument("--json-out", type=Path, default=None)
    args = p.parse_args(argv)

    if args.from_batch:
        items = load_from_batch(args.from_batch)
        source = f"真实 batch {args.from_batch}（逐 case 语料）"
        if not items:
            print(f"⚠️ {args.from_batch} 里没有检索类 gold 调用 —— 还没造 RAG 题？")
            return 1
    else:
        items = [(q, SEED_DOCS, g) for q, g in SEED_QUERIES]
        source = "种子集（自建，2026-08-14）"

    ths = [args.lo + i * args.step for i in range(int((args.hi - args.lo) / args.step) + 1)]
    rows, raw, n_pos, n_neg = sweep(items, ths)

    corpus_sizes = {len(d) for _, d, _ in items}
    print(f"评测条目 {len(items)}（应命中 {n_pos} / 应留空 {n_neg}） · "
          f"每条语料 {min(corpus_sizes)}–{max(corpus_sizes)} 篇 · {source}")
    print(f"当前 MATCH_THRESHOLD = {MATCH_THRESHOLD}")
    print()
    print(f"{'阈值':>6} {'命中且排首':>10} {'误召回':>8} {'漏召回':>8} {'正确留空':>10}")
    for r in rows:
        mark = "  ←现值" if abs(r["threshold"] - MATCH_THRESHOLD) < args.step / 2 else ""
        print(f"{r['threshold']:>6.2f} {r['tp']:>10} {r['fp']:>8} {r['fn']:>8} {r['tn']:>10}{mark}")

    plateau = find_plateau(rows)
    print()
    if len(plateau) < 2:
        print("🔴 **没有平台期** —— 指标随阈值单调滑动，说明相关与不相关之间没有分离带。")
        print("   这不是阈值的问题，是打分函数不行，换阈值救不回来。")
    else:
        lo, hi = plateau[0]["threshold"], plateau[-1]["threshold"]
        print(f"平台期 {lo:.2f}–{hi:.2f}（命中 {plateau[0]['tp']}/{n_pos} · "
              f"误召回 {plateau[0]['fp']} · 正确留空 {plateau[0]['tn']}/{n_neg}）")

    # ★ 不给单一推荐值，给两端 —— 两类错误的代价不对称，取舍要人来做（见模块 docstring）。
    zero_fn = [r for r in rows if r["fn"] == 0]
    zero_fp = [r for r in rows if r["fp"] == 0]
    print()
    print("两端参考（本工具只报权衡，不替你拍板）：")
    if zero_fn:
        r = max(zero_fn, key=lambda r: (r["tn"], r["threshold"]))
        print(f"  · 零漏召回的最高阈值 {r['threshold']:.2f}"
              f"（误召回 {r['fp']}、正确留空 {r['tn']}/{n_neg}）"
              f" ← 偏这端：自然口语查得到，代价是少数该空的会被填上")
    if zero_fp:
        r = min(zero_fp, key=lambda r: r["threshold"])
        print(f"  · 零误召回的最低阈值 {r['threshold']:.2f}"
              f"（漏召回 {r['fn']}、命中 {r['tp']}/{n_pos}）"
              f" ← 偏这端：空洞可靠，代价是造题得凑文档原词")
    print(f"  当前 MATCH_THRESHOLD = {MATCH_THRESHOLD}")
    print("  ⚠️ 无论取哪个，**造题脚本必须双向断言**（该中的真中、该空的真空），"
          "否则哪条轴静默失效都看不出来。")

    # 逐条明细：失败的那几条才是有信息量的。★ 按**当前配置值**报，不按平台期报 ——
    # 拿一个你没在用的阈值列失败，会让人以为线上就是这个表现。
    th = MATCH_THRESHOLD
    print()
    print(f"当前阈值 {th} 下的失败明细（成功的不列）：")
    any_fail = 0
    for (q, _docs, gold), scores in zip(items, raw):
        hits = sorted(((s, k) for k, s in scores.items() if s >= th), reverse=True)
        top = hits[0] if hits else None
        ok = ((gold is None and not hits)
              or (gold == "*" and bool(hits))
              or (top is not None and top[1] == gold))
        if ok:
            continue
        any_fail += 1
        if any_fail <= 10:          # 只列前 10 条，不刷屏
            got = "（空）" if not hits else f"{top[1]} {top[0]:.2f}"
            print(f"  ❌ {q:24} 期望={str(gold):20} 实得={got}")
    if any_fail > 10:
        print(f"  …… 共 {any_fail} 条失败")
    if not any_fail:
        print("  （无）")

    if args.json_out:
        args.json_out.write_text(json.dumps(
            {"source": source, "rows": rows, "plateau": plateau,
             "n_pos": n_pos, "n_neg": n_neg}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n明细 -> {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
