"""★ 数据门禁：每个大版本重建之前跑一遍，不过就不许开训。

    python scripts/check_data_gates.py --batch data/batches/v13 --split-dir data/splits/v13
    python scripts/check_data_gates.py --batch data/batches/v13 --verbose        # 打全表

两组判据，性质不同：
    **D1–D11 多样性**（只要 --batch）    见 docs/syncopate/13-diversity-gates.md
    **L1–L2  泄露**（要 --split-dir）    见 docs/syncopate/15-leakage-gates.md

退出码 0 = 全部达标；非 0 = 有门禁没过（可直接接进 CI / Makefile）。

===========================================================================
为什么需要这个东西
===========================================================================

2026-08-17 量了一次才发现：**1670 条 case 只有 63 种句式**，7 个模板 90–150 条题
共用同一句话（只有 ID 和数字不同），最高频那句占全库 17.6%。

后果**不是"数据少"**，是模型可以**靠表层特征路由** —— 看到「能不能铺到 US、GB、JP」
就走那条 9 步链，根本不用理解任务。上线后用户话术千变万化，这套就塌了。

⚠️⚠️ 而**冻结 EVAL 用的是同一批句式，所以它测不出这件事** ——
评测对这类风险天生是盲的。这就是为什么多样性必须有**独立的门禁**，
而不是"看评测分数好不好"。

===========================================================================
门禁清单（每条都对应一种具体的失效）
===========================================================================

    D1  题面覆盖      每格 ≥5 种句式            防「一个模板一句话」
    D2  题面区分度    平均 ≤0.35 且最大 ≤0.75   防「5 句其实是同义词替换」
    D3  题面不塌缩    最高频句式 ≤5%            防「某模板条数多把全局带偏」
    D4  表达结构      每格 ≥4 种风格            防「只换词不换说法的方式」
    D5 ★无泄露       各档句式集合必须相同       防「句式泄露这是哪一档」
    D6  实体多样性    每轴 ≥5 个取值、最高频 ≤40%  防「模型记住那几个 ID」
    D7  工具菜单      ≥5 种菜单组合             防「菜单恒定 ⇒ 剪枝这条轴是死的」
    D8  gold 轨迹     ≥5 种工具序列、最高频 ≤35%  防「所有题一条链，背下来就行」
    D9  行为分布      5 种 behavior 都要有       防「某种行为样本太少学不会」
    D10 结局分布      每模板 ≥2 种结局           防「同一模板结局恒定 ⇒ 不用看世界」
    D11 长度分布      题面长度变异系数 ≥0.25     防「长度本身成为路由信号」

★ D5 是这里面最反直觉的一条，也是唯一一条**已经真的抓到过 bug** 的：
  改写上线第一版按哈希选句式，而档位也由 index 推 ⇒ 两者相关，
  实测 `empty` 档独有 2 条句式 —— **模型看到那句话就知道这题会查空**，
  根本不用等检索结果。判据是「每一档用到的句式集合必须完全相同」。

===========================================================================
怎么加新指标
===========================================================================

每个 `check_*` 函数返回 `(通过与否, 打印用的行列表)`，在 `CHECKS` 里注册。
⚠️ 新指标必须能回答两个问题，否则不要加：
   ① 它防的是**哪一种具体失效**（要能举出失效长什么样）
   ② 阈值是**怎么定的**（实测反填 / 从设计文档来 / 还是拍的 —— 拍的要写明）
"""

from __future__ import annotations

import argparse
import collections
import itertools
import json
import re
import statistics
from pathlib import Path
from typing import Any, Callable

from syncopate.domains.adcampaign.corpus import tokenize

# ---------------------------------------------------------------------------
# 阈值。★ 改这里之前先想清楚是"数据真的变好了"还是"把门槛挪到数据那儿去了"。
# ---------------------------------------------------------------------------
MIN_PHRASINGS_PER_GRID = 5      # D1
MEAN_SIM_MAX = 0.35             # D2 ——「铺开程度」
PAIR_SIM_MAX = 0.75             # D2 ——「只挡近乎重复」，短请求会天然偏高
TOP_SKELETON_SHARE_MAX = 0.05   # D3
MIN_STYLES_PER_GRID = 4         # D4（池子里定义了 5 种，允许有一格差一种）
MIN_ENTITY_VALUES = 5           # D6
MAX_ENTITY_SHARE = 0.40         # D6
MIN_TOOL_MENUS = 5              # D7
MIN_GOLD_SHAPES = 5             # D8
MAX_GOLD_SHAPE_SHARE = 0.35     # D8
MIN_OUTCOMES_PER_TEMPLATE = 2   # D10
MIN_LENGTH_CV = 0.25            # D11 变异系数 = 标准差 / 均值

STYLES = ("terse", "context", "question", "multi", "casual")
# 这些轴的取值本来就少（如 entry_mode 只有 2 个），不该按 D6 判
ENTITY_AXES = ("campaign_id", "account_id", "product_id", "region", "creative_name")

_NUM = re.compile(r"\d+(?:\.\d+)?%?")


# ---------------------------------------------------------------------------
# 载入
# ---------------------------------------------------------------------------


class Batch:
    def __init__(self, batch_dir: Path) -> None:
        self.dir = batch_dir
        self.cases = [json.loads(p.read_text(encoding="utf-8"))
                      for p in sorted((batch_dir / "cases").glob("*.json"))]
        self.gold = {p.stem: json.loads(p.read_text(encoding="utf-8"))
                     for p in sorted((batch_dir / "gold_paths").glob("*.json"))}
        self.verifier = {p.stem: json.loads(p.read_text(encoding="utf-8"))
                         for p in sorted((batch_dir / "verifier_specs").glob("*.json"))}

    def template(self, case: dict) -> str:
        return case["case_id"].split("_")[0]

    def grid(self, case: dict) -> str:
        """格子 = 模板 × entry_mode。两种入口的题面本来就不同，
        混在一起算会**虚高**多样性。"""
        return f"{self.template(case)}/{case['metadata'].get('entry_mode') or '-'}"

    def tag(self, case: dict, prefix: str) -> str | None:
        for t in case["metadata"].get("tags", []):
            if t.startswith(prefix + ":"):
                return t.split(":", 1)[1]
        return None


def skeleton(case: dict) -> str:
    """抹掉实体和数值，剩下的才是**句式**。

    ★ 用 case 自己的 `entities` / `context` 去抹，**不猜正则** ——
    模板往题面里填的东西本来就都在这两个字典里。
    第一版靠正则，`halloween_hook_e_v1` 这种没抓到，CRE 报 43 种句式，
    而最相似的一对其实是 0.91（同一句换了素材名）——**多样性虚高**。
    """
    out = case["user_message"]
    values: list[Any] = []
    for source in (case.get("entities", {}), case.get("context", {})):
        for v in source.values():
            values.extend(v if isinstance(v, list) else [v])
    for v in sorted((str(x) for x in values if x not in (None, "")), key=len, reverse=True):
        out = out.replace(v, "§S")
    for word in _feature_vocab():
        out = out.replace(word, "§F")
    return re.sub(r"\s+", " ", _NUM.sub("§N", out)).strip()


def _feature_vocab() -> tuple[str, ...]:
    """★ 从模板那边 import，**不许手抄** —— 手抄漏过两个词，
    ATTR 的相似度就虚低成 0.88 判不达标，而问题在指标不在数据。"""
    try:
        from syncopate.authoring.templates import _FEATURE_CN
        return tuple(_FEATURE_CN.values())
    except Exception:
        return ()


def jaccard(a: str, b: str) -> float:
    sa, sb = set(tokenize(a)), set(tokenize(b))
    return len(sa & sb) / len(sa | sb) if sa and sb else 0.0


# ---------------------------------------------------------------------------
# D1–D4 · 题面
# ---------------------------------------------------------------------------


def check_phrasing(b: Batch, verbose: bool) -> tuple[bool, list[str]]:
    """D1 覆盖 / D2 区分度 / D3 不塌缩 / D4 表达结构。"""
    by_grid: dict[str, list[dict]] = collections.defaultdict(list)
    for case in b.cases:
        by_grid[b.grid(case)].append(case)

    lines, bad = [], []
    if verbose:
        lines.append(f"  {'格子':22} {'条数':>5} {'句式':>5} {'均相似':>7} {'最相似':>7} {'风格':>5}")
    for grid in sorted(by_grid):
        group = by_grid[grid]
        sks = sorted({skeleton(c) for c in group})
        sims = [jaccard(x, y) for x, y in itertools.combinations(sks, 2)]
        avg = statistics.mean(sims) if sims else 0.0
        mx = max(sims, default=0.0)
        styles = {b.tag(c, "phrasing") for c in group} - {None}
        fails = []
        if len(sks) < MIN_PHRASINGS_PER_GRID:
            fails.append(f"句式 {len(sks)}<{MIN_PHRASINGS_PER_GRID}")
        if avg > MEAN_SIM_MAX or mx > PAIR_SIM_MAX:
            fails.append(f"相似 均{avg:.2f}/最{mx:.2f}")
        if len(styles) < MIN_STYLES_PER_GRID:
            fails.append(f"风格 {len(styles)}<{MIN_STYLES_PER_GRID}")
        if fails:
            bad.append(f"    🔴 {grid}: {' · '.join(fails)}")
        if verbose:
            lines.append(f"  {grid:22} {len(group):>5} {len(sks):>5} {avg:>7.2f} {mx:>7.2f} "
                         f"{len(styles):>5} {'✅' if not fails else '🔴'}")

    share = collections.Counter(skeleton(c) for c in b.cases)
    top, n = share.most_common(1)[0]
    top_share = n / len(b.cases)
    collapse_ok = top_share <= TOP_SKELETON_SHARE_MAX
    lines.append(f"  全库句式 {len(share)} 种 · 最高频占比 {top_share*100:.1f}%"
                 f"（≤{TOP_SKELETON_SHARE_MAX*100:.0f}%）{'✅' if collapse_ok else '🔴'}")
    if not collapse_ok:
        lines.append(f"    └ {top[:70]}")
    lines.extend(bad)
    return (not bad and collapse_ok), lines


# ---------------------------------------------------------------------------
# D5 · ★ 无泄露
# ---------------------------------------------------------------------------


def check_no_leak(b: Batch, verbose: bool) -> tuple[bool, list[str]]:
    """★★★ 句式不许泄露「这是哪一档」。

    **唯一一条已经真的抓到过 bug 的门禁。** 题面改写第一版按哈希选句式，
    而档位也由 index 推 ⇒ 两者相关，实测 `empty` 档独有 2 条句式
    ⇒ 模型看到那句话就知道会查空，**根本不用等检索结果**。

    ⚠️⚠️ **判据分两档，强度不同，理由是数学上的不是妥协**：

    D5a 严格（集合必须完全相同）——只对**真·分档轴**：
        rag_state / insight_state / safety_line_state / data_maturity
        这些是「同一句话、不同世界、不同正确动作」，档数少（3 档）、
        每格样本数远大于句式数 ⇒ 轮转能做到完全均匀，做不到就是 bug。
        `_phrase(..., stratum=...)` 就是为这几条实现的。

    D5b 宽松（不许有句式**只在一个档出现**）——其它轴，尤其 `outcome`：
        FAIL 有 **10 种结局**而句式只有 7 种，每格约 7.5 条
        ⇒ **集合相等在数学上不可能**（鸽笼原理）。
        但真正要挡的那个利用方式是「看到这句话就知道是哪一档」，
        判据写成「没有任何句式是某一档独占的」就够，而且做得到。

    ★ 第一版对所有轴一刀切用 D5a，把 FAIL/INJ 判成不达标 ——
      **那是判据不可能被满足，不是数据有问题**。记在这里免得下次又改回去。
    """
    strict_axes = ("rag_state", "insight_state", "safety_line_state", "data_maturity")
    loose_axes = ("outcome",)
    lines, bad = [], []

    def groups_of(cases: list[dict], axis: str) -> dict[str, dict[str, set[str]]]:
        out: dict[str, dict[str, set[str]]] = collections.defaultdict(dict)
        buckets: dict[tuple[str, str], set[str]] = collections.defaultdict(set)
        for case in cases:
            arm = b.tag(case, axis)
            if arm is None:
                continue
            buckets[(arm, case["metadata"].get("entry_mode") or "-")].add(skeleton(case))
        for (arm, entry), sks in buckets.items():
            out[entry][arm] = sks
        return out

    for template in sorted({b.template(c) for c in b.cases}):
        cases = [c for c in b.cases if b.template(c) == template]
        for axis in strict_axes:
            for entry, arms in groups_of(cases, axis).items():
                if len(arms) < 2:
                    continue
                union = set().union(*arms.values())
                leaky = [a for a, s in arms.items() if s != union]
                if leaky:
                    bad.append(f"    🔴 [D5a] {template}/{entry} 的 {axis}: "
                               f"{leaky} 档句式集合不全 ⇒ 句式能预测档位")
                elif verbose:
                    lines.append(f"  [D5a] {template}/{entry:14} {axis:20} "
                                 f"{len(arms)} 档 × {len(union)} 句式 ✅")
        for axis in loose_axes:
            for entry, arms in groups_of(cases, axis).items():
                if len(arms) < 2:
                    continue
                owner: dict[str, set[str]] = collections.defaultdict(set)
                for arm, sks in arms.items():
                    for sk in sks:
                        owner[sk].add(arm)
                exclusive = {sk: next(iter(a)) for sk, a in owner.items() if len(a) == 1}
                if exclusive:
                    bad.append(f"    🔴 [D5b] {template}/{entry} 的 {axis}: "
                               f"{len(exclusive)} 种句式被某一档独占 ⇒ 看到它就知道结局")
                elif verbose:
                    lines.append(f"  [D5b] {template}/{entry:14} {axis:20} "
                                 f"{len(arms)} 档 · 无独占句式 ✅")

    lines.extend(bad)
    if not bad:
        lines.append("  D5a 真·分档轴：各档句式集合完全相同 ✅")
        lines.append("  D5b 结局轴：没有任何句式被单一结局独占 ✅")
    return (not bad), lines


# ---------------------------------------------------------------------------
# D6 · 实体
# ---------------------------------------------------------------------------


def check_entities(b: Batch, verbose: bool) -> tuple[bool, list[str]]:
    """实体取值不能太集中 —— 否则模型记住那几个 ID 就能过关，
    而真实世界里 ID 是无穷的。"""
    lines, bad = [], []
    for axis in ENTITY_AXES:
        values = [str(c.get("entities", {}).get(axis) or c.get("context", {}).get(axis))
                  for c in b.cases]
        values = [v for v in values if v and v != "None"]
        if not values:
            continue
        counter = collections.Counter(values)
        top, n = counter.most_common(1)[0]
        share = n / len(values)
        ok = len(counter) >= MIN_ENTITY_VALUES and share <= MAX_ENTITY_SHARE
        if not ok:
            bad.append(f"    🔴 {axis}: {len(counter)} 种取值 · 最高频 {top} 占 {share*100:.0f}%")
        if verbose:
            lines.append(f"  {axis:16} {len(counter):>4} 种 · 最高频占 {share*100:>5.1f}% "
                         f"{'✅' if ok else '🔴'}")
    lines.extend(bad)
    return (not bad), lines


# ---------------------------------------------------------------------------
# D7 · 工具菜单
# ---------------------------------------------------------------------------


def check_tool_menus(b: Batch, verbose: bool) -> tuple[bool, list[str]]:
    """菜单要是恒定的，「按意图剪枝」（设计 §11-④）这条轴就是死的 ——
    而它同时是研究实验的混淆变量，做异步对照时必须固定。"""
    menus = collections.Counter(tuple(sorted(c["tool_menu"])) for c in b.cases)
    sizes = [len(c["tool_menu"]) for c in b.cases]
    ok = len(menus) >= MIN_TOOL_MENUS
    lines = [f"  菜单组合 {len(menus)} 种（≥{MIN_TOOL_MENUS}）· "
             f"大小 {min(sizes)}–{max(sizes)}（中位 {int(statistics.median(sizes))}）"
             f" {'✅' if ok else '🔴'}"]
    if verbose:
        for menu, n in menus.most_common(6):
            lines.append(f"    {n:>5} 条 · {len(menu)} 个工具")
    return ok, lines


# ---------------------------------------------------------------------------
# D8 · gold 轨迹
# ---------------------------------------------------------------------------


def check_gold_shapes(b: Batch, verbose: bool) -> tuple[bool, list[str]]:
    """gold 的**工具序列**不能太集中。

    所有题一条链的话，模型背下那条链就能拿分 —— 而那不是"学会了"，
    是"记住了"。和 D1 是同一种病的不同部位：D1 管输入，这条管输出。
    """
    shapes = collections.Counter(
        tuple(a["tool"] for a in g.get("actions", [])) for g in b.gold.values())
    top, n = shapes.most_common(1)[0]
    share = n / max(1, len(b.gold))
    ok = len(shapes) >= MIN_GOLD_SHAPES and share <= MAX_GOLD_SHAPE_SHARE
    lens = [len(s) for s in shapes.elements()]
    lines = [f"  工具序列 {len(shapes)} 种（≥{MIN_GOLD_SHAPES}）· 最高频占 {share*100:.1f}%"
             f"（≤{MAX_GOLD_SHAPE_SHARE*100:.0f}%）· 链长 {min(lens)}–{max(lens)}"
             f"（中位 {int(statistics.median(lens))}）{'✅' if ok else '🔴'}"]
    if not ok:
        lines.append(f"    └ 最高频：{' → '.join(top) if top else '(空链)'}")
    if verbose:
        for shape, cnt in shapes.most_common(6):
            lines.append(f"    {cnt:>5} 条 · {' → '.join(shape) or '(空链)'}")
    return ok, lines


# ---------------------------------------------------------------------------
# D9 / D10 · 行为与结局
# ---------------------------------------------------------------------------


def check_behavior_outcome(b: Batch, verbose: bool) -> tuple[bool, list[str]]:
    """五种 behavior 都得有样本；每个模板至少两种结局。

    ★ 结局恒定的模板 = **不用看世界就能答对** —— M0 那条「只装死格的 SFT 桶
    让 defer 从 97% 掉到 0%」是同一个病：桶里没有对照档，模型学的是常数。
    """
    behaviors = collections.Counter(v.get("expected_behavior") for v in b.verifier.values())
    missing = [x for x in ("tool_call", "clarify", "reject", "defer", "answer")
               if behaviors.get(x, 0) == 0]
    lines = [f"  behavior 分布 " + " · ".join(f"{k}={v}" for k, v in behaviors.most_common())]
    bad = []
    if missing:
        bad.append(f"    🔴 这些 behavior 一条样本都没有：{missing}")

    by_tpl: dict[str, set[str]] = collections.defaultdict(set)
    for case in b.cases:
        outcome = b.tag(case, "outcome")
        if outcome:
            by_tpl[b.template(case)].add(outcome)
    thin = {t: s for t, s in by_tpl.items() if len(s) < MIN_OUTCOMES_PER_TEMPLATE}
    if thin:
        lines.append(f"  🟡 结局恒定的模板（可能是刻意的单一局面）：{sorted(thin)}")
    if verbose:
        for t in sorted(by_tpl):
            lines.append(f"    {t:8} {len(by_tpl[t])} 种结局 {sorted(by_tpl[t])}")
    lines.extend(bad)
    return (not bad), lines


# ---------------------------------------------------------------------------
# D11 · 长度
# ---------------------------------------------------------------------------


def check_length_spread(b: Batch, verbose: bool) -> tuple[bool, list[str]]:
    """题面长度不能全都差不多 —— 否则**长度本身**就是一个路由信号：
    模型学到「短的是查指标、长的是地域铺开」，而不是读懂内容。"""
    lengths = [len(c["user_message"]) for c in b.cases]
    cv = statistics.pstdev(lengths) / statistics.mean(lengths)
    ok = cv >= MIN_LENGTH_CV
    lines = [f"  题面长度 {min(lengths)}–{max(lengths)} 字（中位 "
             f"{int(statistics.median(lengths))}）· 变异系数 {cv:.2f}"
             f"（≥{MIN_LENGTH_CV}）{'✅' if ok else '🔴'}"]
    return ok, lines


CHECKS: list[tuple[str, str, Callable[[Batch, bool], tuple[bool, list[str]]]]] = [
    ("D1-D4", "题面：覆盖 / 区分度 / 不塌缩 / 表达结构", check_phrasing),
    ("D5",    "★ 无泄露：句式不能预测档位",              check_no_leak),
    ("D6",    "实体多样性",                              check_entities),
    ("D7",    "工具菜单多样性",                          check_tool_menus),
    ("D8",    "gold 轨迹多样性",                         check_gold_shapes),
    ("D9-D10", "行为与结局分布",                          check_behavior_outcome),
    ("D11",   "题面长度分布",                            check_length_spread),
]


# ---------------------------------------------------------------------------
# L1 / L2 · 三桶泄露（要 --split-dir）
# ---------------------------------------------------------------------------


def check_leakage(batch: Batch, split_dir, verbose: bool) -> tuple[bool, list[str]]:
    """★★★ 三桶之间不许有「同一道题」跨桶。

    ⚠️ **和 split_report 里那条 `overlaps_by_content_sha256` 不是一回事**：
    那条算的是**完整渲染后的 prompt** 的哈希，两条 case 只要 campaign_id 不同
    哈希就不同、检查就过 —— 而它们要答的那件事完全一样。
    2026-08-17 实测 v13：**62/278 = 22.3% 的 EVAL 有这样的孪生**。

    判据与模板豁免见 `syncopate/pipeline/leakage.py`。
    """
    import json as _json
    from pathlib import Path as _Path

    from syncopate.core.schemas import CaseBundle
    from syncopate.pipeline import leakage as _leak

    buckets = {b: _json.loads((_Path(split_dir) / f"{b}_cases.json").read_text("utf-8"))["case_ids"]
               for b in ("eval", "sft", "rl")}
    wanted = {c for ids in buckets.values() for c in ids}
    bundles = {c: CaseBundle.read(_Path(batch.dir), c) for c in sorted(wanted)}
    report = _leak.audit(bundles, buckets)

    l1, l2 = report["L1_exact_message_and_answer"], report["L2_skeleton_and_answer_non_exempt"]
    lines = [
        f"  L1 题面原文+答案 跨桶组 {l1['cross_bucket_groups']:>4} · "
        f"受影响 EVAL {l1['affected_eval']:>3} ({l1['affected_eval_ratio']*100:.1f}%)",
        f"  L2 题面句式+答案 跨桶组 {l2['cross_bucket_groups']:>4} · "
        f"受影响 EVAL {l2['affected_eval']:>3} ({l2['affected_eval_ratio']*100:.1f}%)"
        f"   （豁免 {sorted(report['L2_exempt_templates'])}）",
    ]
    if l1["by_template"] or l2["by_template"]:
        lines.append(f"    受影响模板 L1={l1['by_template']} L2={l2['by_template']}")
    # ★ 训练池内的重复是**效率问题不是有效性问题**：RL 在学 SFT 已经教会的东西
    #   ⇒ 组内方差为 0 ⇒ 零梯度。报出来但不阻塞。
    lines.append(f"  🟡 训练池内(SFT∩RL)重复组 {report['train_pool_overlap_groups']} "
                 f"—— 效率问题（会推高 RL 零梯度率），不阻塞门禁")
    return report["passed"], lines


def main() -> int:
    ap = argparse.ArgumentParser(description="数据多样性门禁（大版本重建前必跑）")
    ap.add_argument("--batch", type=Path, default=Path("data/batches/v13"))
    ap.add_argument("--split-dir", type=Path, default=None,
                    help="给了才跑 L1/L2 泄露门禁（它要三桶名单）")
    ap.add_argument("--verbose", action="store_true", help="打全表，不只打不达标的")
    args = ap.parse_args()

    batch = Batch(args.batch)
    print(f"多样性门禁 · {len(batch.cases)} 条 case ← {args.batch}\n")
    failed = []
    checks = list(CHECKS)
    if args.split_dir:
        checks.append(("L1-L2", "★ 三桶泄露（同一道题不许跨桶）",
                       lambda b, v: check_leakage(b, args.split_dir, v)))
    else:
        print("⚠️ 没给 --split-dir ⇒ **跳过 L1/L2 泄露门禁**。"
              "跳过不是通过 —— 大版本重建前必须带上它。\n")
    for code, title, fn in checks:
        ok, lines = fn(batch, args.verbose)
        print(f"{'✅' if ok else '🔴'} {code:8} {title}")
        for line in lines:
            print(line)
        print()
        if not ok:
            failed.append(code)

    if failed:
        print(f"🔴 未通过：{', '.join(failed)}")
        print("⚠️ 改阈值之前先想清楚：是数据真的变好了，还是把门槛挪到数据那儿去了。")
        return 1
    print("✅ 全部门禁通过 —— 可以进入重建 / 训练")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
