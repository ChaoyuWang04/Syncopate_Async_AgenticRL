"""量「题面多样性」。**先有指标，再谈扩充** —— 否则改完不知道有没有变好。

★ 为什么不能只看"去重后有几条"

`user_message` 里带 ID 和数字，换个 campaign_id 就算"不同的一条"，
唯一率能到 50% —— 而实际上 90 条 GEO 用的是**同一句话**。
⇒ 必须先把**实体和数值抹掉**，剩下的才是"句式"。

★★ 四条指标，缺一不可（一条过了另一条塌，等于没改善）

    ① 覆盖    每个格子（模板 × entry_mode）**≥ 5 种句式**
    ② 区分度  同格子内任意两句的 Jaccard ≤ 0.60
              —— 只有 ①没有② 的话，5 句可以是同义词替换，模型照样按表层路由
    ③ 不塌缩  全库最高频句式占比 ≤ 5%
              —— 前两条都是格子内的，这条防"某个模板条数特别多把全局带偏"
    ④ 风格    每个格子要覆盖 ≥ 4 种**表达结构**（不是换词，是换说法的方式）

★★★ ④ 是这套指标里最不常规、也最要紧的一条

真实用户不会都用一种句法。他们会：一句话命令、带一段背景再问、直接抛疑问、
**一次夹好几个诉求**、或者口语省略主语。只做同义词替换，前三条指标都能刷过去，
而模型学到的还是"看到这个句型就走这条链"。
⇒ 风格轴：terse（短命令）/ context（带背景铺垫）/ question（疑问式）
          / multi（多诉求夹杂）/ casual（口语省略）
"""

from __future__ import annotations

import argparse
import collections
import itertools
import json
import re
from pathlib import Path

from syncopate.domains.adcampaign.corpus import tokenize

# 实体与数值的归一：把它们抹成占位符，剩下的才是句式。
# ⚠️ 顺序有讲究：先抹长的（带前缀的 ID），再抹裸数字，否则 CMP_4001 会被拆成 CMP_#
_NORM = [
    (re.compile(r"[A-Za-z][A-Za-z0-9]*_\d+"), "§ID"),        # CMP_4001 / fresh_12
    (re.compile(r"\b[A-Z]{2,}\b"), "§CODE"),                  # US / GB / JP / ROAS
    (re.compile(r"\d+(?:\.\d+)?%?"), "§N"),                   # 480 / 12.5 / 20%
]

STYLES = ("terse", "context", "question", "multi", "casual")


# 特征词是闭集，正则抓不到，只能按词表抹。
# ⚠️ **直接从模板那边 import，不许手抄** —— 第一版手抄漏了「暗色调」和
# 「前后对比开场」，ATTR 的相似度就虚低成 0.88 判不达标，而问题在指标不在数据。
def _feature_vocab() -> tuple[str, ...]:
    try:
        from syncopate.authoring.templates import _FEATURE_CN
        return tuple(_FEATURE_CN.values())
    except Exception:
        return ()


_SLOT_VOCAB = _feature_vocab()


def skeleton(message: str, case: dict | None = None) -> str:
    """抹掉实体和数值，剩下的才是**句式**。

    ★★ 正则抓不干净：`halloween_hook_e_v1`、`MERGE_FARM`、`真人出镜` 都是槽位值，
    留着它们会让"同一句话换个素材名"被算成两种句式 —— 多样性**虚高**。
    第一版就栽在这：CRE 报 43 种句式，其实最相似的一对是 0.91（同一句，换了素材名）。

    ⇒ 正确做法是**用 case 自己的 entities/context 去抹**，而不是猜正则：
      模板往题面里填的东西，本来就都在这两个字典里。
    """
    out = message
    if case:
        values = list(case.get("entities", {}).values()) + list(case.get("context", {}).values())
        flat = []
        for v in values:
            flat.extend(v if isinstance(v, list) else [v])
        # 长的先替换，避免 CMP_4000 被 CMP_40 的前缀截断
        for v in sorted((str(x) for x in flat if x not in (None, "")), key=len, reverse=True):
            out = out.replace(v, "§S")
    for word in _SLOT_VOCAB:
        out = out.replace(word, "§F")
    for pattern, repl in _NORM:
        out = pattern.sub(repl, out)
    return re.sub(r"\s+", " ", out).strip()


def jaccard(a: str, b: str) -> float:
    """句式之间的相似度。用 token 集合而不是编辑距离：

    编辑距离对"换个语序"很敏感但对"换个说法"不敏感，而我们要量的正好是后者。
    """
    sa, sb = set(tokenize(a)), set(tokenize(b))
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def load_cases(batch_dir: Path) -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8"))
            for p in sorted((batch_dir / "cases").glob("*.json"))]


def grid_key(case: dict) -> tuple[str, str]:
    """格子 = 模板 × entry_mode。

    ★ 为什么按 entry_mode 分：`id_given` 和 `must_discover` 的题面本来就不同
    （一个带 campaign_id 一个不带），混在一起算会**虚高**多样性。
    """
    template = case["case_id"].split("_")[0]
    entry = case["metadata"].get("entry_mode") or "-"
    return template, entry


def report(batch_dir: Path, *, min_styles: int = 5, max_sim: float = 0.60,
           max_share: float = 0.05, min_style_kinds: int = 4) -> int:
    cases = load_cases(batch_dir)
    by_grid: dict[tuple[str, str], list[dict]] = collections.defaultdict(list)
    for case in cases:
        by_grid[grid_key(case)].append(case)

    all_sk = [skeleton(c["user_message"], c) for c in cases]
    share = collections.Counter(all_sk)
    top_sk, top_n = share.most_common(1)[0]

    print(f"语料 {len(cases)} 条 · 格子 {len(by_grid)} 个 ← {batch_dir}\n")
    print(f"{'格子':22} {'条数':>5} {'句式':>5} {'最大两两相似':>12} {'风格':>5}  判定")
    fail_cover, fail_sim, fail_style = [], [], []
    for key in sorted(by_grid):
        group = by_grid[key]
        sks = sorted({skeleton(c["user_message"], c) for c in group})
        worst = max((jaccard(a, b) for a, b in itertools.combinations(sks, 2)), default=0.0)
        kinds = len({t.split(":", 1)[1] for c in group for t in c["metadata"].get("tags", [])
                     if t.startswith("phrasing:")})
        ok_cover, ok_sim = len(sks) >= min_styles, worst <= max_sim
        ok_style = kinds >= min_style_kinds
        if not ok_cover:
            fail_cover.append(key)
        if not ok_sim:
            fail_sim.append(key)
        if not ok_style:
            fail_style.append(key)
        mark = "✅" if (ok_cover and ok_sim and ok_style) else "🔴"
        print(f"{key[0]+'/'+key[1]:22} {len(group):>5} {len(sks):>5} "
              f"{worst:>12.2f} {kinds:>5}  {mark}")

    print(f"\n① 覆盖  每格 ≥{min_styles} 种句式        不达标 {len(fail_cover)}/{len(by_grid)} 格")
    print(f"② 区分  同格两两相似 ≤{max_sim}         不达标 {len(fail_sim)}/{len(by_grid)} 格")
    print(f"④ 风格  每格 ≥{min_style_kinds} 种表达结构       不达标 {len(fail_style)}/{len(by_grid)} 格")
    print(f"③ 塌缩  最高频句式占比 {top_n/len(cases)*100:.1f}%  "
          f"（门槛 ≤{max_share*100:.0f}%）{'✅' if top_n/len(cases) <= max_share else '🔴'}")
    print(f"        └ {top_sk[:70]}")
    print(f"\n全库句式总数 {len(share)}")
    bad = len(set(fail_cover) | set(fail_sim) | set(fail_style))
    passed = bad == 0 and top_n / len(cases) <= max_share
    print("\n" + ("✅ 全部达标" if passed else f"🔴 {bad} 个格子不达标"))
    return 0 if passed else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=Path, default=Path("data/batches/v13"))
    ap.add_argument("--min-styles", type=int, default=5)
    ap.add_argument("--max-sim", type=float, default=0.60)
    raise SystemExit(report(ap.parse_args().batch,
                            min_styles=ap.parse_args().min_styles,
                            max_sim=ap.parse_args().max_sim))
