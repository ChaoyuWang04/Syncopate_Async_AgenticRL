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
from syncopate.pipeline import leakage
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


def _tags(b: CaseBundle) -> dict[str, str]:
    return {t.split(":", 1)[0]: t.split(":", 1)[1] for t in b.case.metadata.tags if ":" in t}


# ★★★ SFT 桶的三条保底线（2026-08-16 由一次验收审计推出）
#
# 背景：v12 的 SFT 桶里 M8 新增的两个模板 **POL=0 / CONF=0**，一条都没有。
# 机制不是"选得不好"，是**保底只写在 dead_grid 分支里，而 v12 走的是
# difficulty_proxy 分支**（v11 走 dead_grid，所以以前没暴露）。
# `_difficulty_rank` 按 (难度, gold 链长) 降序取前 20%，而 POL 的 gold 只有
# 「一次 policy.search + 终答」，链最短 ⇒ 整个模板被挤到桶外。
#
# 后果不是"少学一点"，是**M8 的验收路径整个断掉**：
# 07 文档开头那句「p=0 的格子 RL 永远够不着」⇒ POL/CONF 没进 SFT，
# 冻结 EVAL 上量出来的一定是"没学会"，**而这个结果和 RAG 实现好坏无关** ——
# 会把一个分桶 bug 误判成 RAG 设计问题，然后去改根本没错的地方。
#
# ⇒ 所以保底从 dead_grid 分支里提出来，**两个分支走同一个函数** `_apply_sft_floors`。
#   「换一条代码分支，保底就悄悄没了」这个形状，正是本项目反复栽的那个
#   （00-START §6「机制在，但没接上」）。
#   ⚠️ 共用的是**函数**，不是**策略**：在场保底只对 difficulty_proxy 开，
#      理由写在 `_apply_sft_floors` 的 docstring 里（一句话：dead_grid 说得出
#      「这个模板是活的」这个理由，difficulty_proxy 说不出任何理由）。
#
# 三个轴各挡一种形态，少一个都不行：
#   behavior  某个**行为**太稀薄 —— 实测 defer 只有 4 条 ⇒ SFT 后从 77% 崩到 36%
#   template  某个**模板整个没进桶** —— 就是这次 POL/CONF 撞的
#   outcome   进了桶但**只进了一档** —— 坑 #6 的第一种形态（只装死格 ⇒ defer 97%→0%）
TEMPLATE_SFT_FLOOR = 12      # 每个模板至少这么多条（对齐 RARE_BEHAVIOR_SFT_FLOOR）
OUTCOME_SFT_FLOOR = 4        # 且每个 (模板, 结局) 档至少这么多 —— 只喂一档等于没喂


def _apply_sft_floors(
    pool: list[str], chosen: list[str], bundles: dict[str, CaseBundle],
    *, presence_floors: bool,
) -> tuple[list[str], dict[str, dict[str, int]]]:
    """把"某一类在 SFT 桶里太稀薄"补上来。**两个分支共用这一个函数。**

    ★★ `presence_floors` 为什么按分支给，而不是一律开：

        dead_grid 模式      有 base 实测数据。某个模板没进桶，是因为**它在 EVAL 上是活的**
                           —— 这是一个正当理由，而且 SFT 的职责本来就是
                           「把 p 从 0 抬到 5–10%，不是抬到 90%」。
                           新模板若真的不会做，它的格子必然是死格 ⇒ 自然会被选中。
                           ⇒ 这条分支**不需要**在场保底，硬加反而把活格卷进来。

        difficulty_proxy   **没有任何 base 数据**，纯按 (难度标签, gold 链长) 排序。
                           某个模板没进桶，说不出任何理由 —— v12 就是这么把
                           POL/CONF 整个丢掉的。⇒ 这条分支必须有在场保底。

    ⇒ 判据是「说不说得出为什么这个模板不需要 SFT」：说得出就跳过，说不出就保底。
      （同 infra 线那条纪律：答不上「服务哪条兑现物」的实验一律显式停放。）

    ★ 补进去的明细一并返回并写进 `split_report.json` —— 保底如果是静默发生的，
    下次它没生效同样是静默的，而这正是本项目最贵的那类 bug。

    ⚠️ 取样一律走 `pool` 的既有顺序（= case_id 排序），不随机：
    切分必须可复现，随机化会让「换了配额」和「换了种子」在结果里分不开。
    """
    from syncopate.pipeline.dead_grid import RARE_BEHAVIOR_SFT_FLOOR

    chosen = list(chosen)
    chosen_set = set(chosen)
    added: dict[str, dict[str, int]] = {}

    # 每组在池子里总共有多少条 —— 保底的上限要按它算，见下面 allowance。
    supply: dict[Any, int] = {}

    def top_up(axis: str, group_of, floors: dict[Any, int], *, share: float | None) -> None:
        # 每个轴开始前重算一次 have：前一个轴补进来的 case 在这个轴上同样算数。
        have: dict[Any, int] = {}
        for cid in chosen_set:
            key = group_of(cid)
            have[key] = have.get(key, 0) + 1
        supply.clear()
        for cid in pool:
            key = group_of(cid)
            supply[key] = supply.get(key, 0) + 1
        detail: dict[str, int] = {}
        for group, floor in sorted(floors.items(), key=lambda kv: str(kv[0])):
            # ★★ 两类保底的**上限策略刻意不同** —— 因为它们要的东西不是一回事：
            #
            #   behavior（share=None，不设上限）
            #       要的是**饱和**：某个行为得有足够的量才学得会。
            #       实测依据 —— defer 只有 4 条 ⇒ SFT 后从 77% 崩到 36%；
            #       M1 那轮从 0 学到 97% 用了 9 条。这是绝对量要求，不能按比例打折。
            #
            #   template / outcome（share=0.5，最多吃一半）
            #       要的是**在场**：这个模板/这一档不能一条都没有。
            #       在场不需要饱和，所以要给 RL 留口粮 —— 否则该模板在 RL 阶段
            #       一条梯度来源都没有，等于把这次修的坑原样搬到下一个阶段。
            #
            # ⚠️ 真实规模上 share 这条根本不绑定（v12 的 POL 池子 78 条，
            #    一半 39 ≫ 保底 12），只有小批量/冒烟批次才会撞到。
            allowance = floor if share is None else max(1, int(supply.get(group, 0) * share))
            missing = min(floor, allowance) - have.get(group, 0)
            if missing <= 0:
                continue
            # 同一个轴内各组互斥（一条 case 只有一个 behavior / 模板 / 结局），
            # 所以循环里不必刷新 have，只需刷新 chosen_set。
            spare = [c for c in pool if c not in chosen_set and group_of(c) == group]
            take = spare[:missing]
            chosen.extend(take)
            chosen_set.update(take)
            if take:
                detail[str(group)] = len(take)
        if detail:
            added[axis] = detail

    # ① 饱和保底：两个分支都要。行为学不会就是学不会，和有没有 base 数据无关。
    top_up("behavior", lambda c: bundles[c].verifier.expected_behavior,
           RARE_BEHAVIOR_SFT_FLOOR, share=None)
    # ②③ 在场保底：只有 difficulty_proxy 需要，理由见 docstring。
    if presence_floors:
        top_up("template", lambda c: c.split("_")[0],
               {c.split("_")[0]: TEMPLATE_SFT_FLOOR for c in pool}, share=0.5)
        top_up("outcome", lambda c: (c.split("_")[0], _tags(bundles[c]).get("outcome", "-")),
               {(c.split("_")[0], _tags(bundles[c]).get("outcome", "-")): OUTCOME_SFT_FLOOR
                for c in pool}, share=0.5)

    return sorted(set(chosen)), added


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


# ★★ 同一条道理的第二个轴：**按结局档加厚**（2026-08-16）。
#
# 上面那条按 behavior 加厚，而 M8 两项验收（过期检出率 / 无检索幻觉率，§14 都要求
# 趋近 0）考的档位 behavior 恰好是最常见的 `tool_call` ⇒ 它们只拿到
# `eval_per_stratum=2`。实测 v12 的冻结 EVAL：
#
#     rag_state:superseded   4 条   ← 过期检出率的**全部**分母
#     rag_state:empty        4 条   ← 无检索幻觉率的**全部**分母
#
# **4 条题，错一条就是 25%** —— 用它去判定一个"趋近 0"的指标，
# 和 M7 那个「配对 MDE 0.013 比效应量本身还大」是同一种病：尺子比要量的东西粗。
#
# ⚠️⚠️ 这个加厚是**严格增量**的，这一点是构造保证的、不是事后 diff 出来的：
# 取样写法是 `sorted(strata[key])[:take]`，**调大 take 只会在尾部追加**，
# 已有成员一个都不动 ⇒ 冻结 EVAL 的老 case_id 全部保留，
# `_audit/v11_sft_e1_m2.json` 这类历史基线对它们仍然逐条配对可比。
# （同 `set_tool_menus.py --freeze-from` 那条纪律：「哪些基线仍可比」要由构造保证。）
OUTCOME_EVAL_QUOTA = {
    # POL（政策演练）—— 半结构化侧
    "answered_v2_from_superseded": 6,   # 过期检出率
    "escalated_no_policy": 6,           # 无检索幻觉率（policy 侧）
    # CONF（结论冲突）—— 非结构化侧
    "answered_from_absent": 6,          # 无检索幻觉率（insight 侧）
    "conflict_reported": 6,             # 历史结论与实测数据打架时怎么办
    # v13 · 检索契约的两个新局面。★ 加厚的理由和上面四条**一模一样**：
    # 不加厚的话，(模板 × behavior × 结局 × entry_mode) 这个分层下每格只拿
    # eval_per_stratum=2，两个 entry_mode 合计 **4 条** —— 错一题就是 25%，
    # 而这两项要判的是"该转人工的时候有没有转"，同样是趋近某个硬值。
    # **尺子比要量的东西粗，量出来的数谁也不是**（M8 §6-② 那次已经栽过一回）。
    "escalated_policy_unavailable": 6,  # 查不了 ≠ 没有限制
    "escalated_policy_not_applicable": 6,  # 查到了但答非所问
}


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
    sft_floors: dict[str, dict[str, int]] = {}

    # ---- 1. EVAL：分层冻结 ----
    strata: dict[tuple[str, ...], list[str]] = {}
    for cid, b in bundles.items():
        strata.setdefault(stratum(cid, b), []).append(cid)
    # ★★★ 分层取样在**泄露组**上做，不在单条 case 上做。
    #
    # 泄露组 = 同一个泄露键的所有 case（`leakage.grouping_key`）。**组是原子的**：
    # 要么整组进 EVAL，要么整组不进。
    #
    # 为什么不能"先按 case 选完再把孪生拉进来"：实测那样 EVAL 会从 278 膨胀到 507
    # （拉进来 229 条），SFT/RL 被抽干。按组取样则 EVAL 规模基本不变。
    #
    # ⇒ 这是**构造保证**：EVAL 里任何一条的孪生都不可能留在训练桶里。
    #   判据定义见 `syncopate/pipeline/leakage.py`。
    key_groups: dict[tuple[str, str], list[str]] = {}
    for cid, b in bundles.items():
        key_groups.setdefault(leakage.grouping_key(b), []).append(cid)
    group_of = {cid: key for key, cids in key_groups.items() for cid in cids}

    buckets = Buckets()
    eval_set: set[str] = set()
    taken_groups: set[tuple[str, str]] = set()
    for key in sorted(strata):
        # key[1] 是 expected_behavior，key[2] 是结局档。
        # 两条加厚规则取**较大者**，见 RARE_BEHAVIOR_EVAL_QUOTA / OUTCOME_EVAL_QUOTA。
        take = max(eval_per_stratum,
                   RARE_BEHAVIOR_EVAL_QUOTA.get(key[1], 0),
                   OUTCOME_EVAL_QUOTA.get(key[2], 0))
        picked = 0
        for cid in sorted(strata[key]):
            if picked >= take:
                break
            gkey = group_of[cid]
            if gkey in taken_groups:
                continue
            members = sorted(key_groups[gkey])
            taken_groups.add(gkey)
            eval_set.update(members)
            picked += len(members)
    buckets.eval = sorted(eval_set)


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
        # ★★ 保底：dead_grid 只按「哪些格子死了」选，做得好的行为一个格子都不会进桶。
        # 三个轴见 _apply_sft_floors 的说明。
        buckets.sft, sft_floors = _apply_sft_floors(pool, chosen, bundles, presence_floors=False)
        sft_set = set(buckets.sft)
        buckets.rl = [c for c in pool if c not in sft_set]
        mode = "dead_grid"
    else:
        ranked = sorted(pool, key=lambda c: _difficulty_rank(bundles[c]), reverse=True)
        cut = max(1, int(len(pool) * sft_ratio))
        # ★ 保底和 dead_grid 分支**共用同一个函数**。
        # 这条分支以前一条保底都没有，v12 因此把 POL / CONF 整个挤出了 SFT 桶。
        buckets.sft, sft_floors = _apply_sft_floors(pool, ranked[:cut], bundles, presence_floors=True)
        sft_set = set(buckets.sft)
        buckets.rl = sorted(c for c in ranked[cut:] if c not in sft_set)
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
        # ★ 泄露审计：content_hash 的盲区（同题同答但 ID 不同）由它兜住
        "leakage": leakage.audit(bundles, {"eval": buckets.eval, "sft": buckets.sft,
                                           "rl": buckets.rl}),
        "eval_strata_coverage": {"|".join(k): len(v) for k, v in sorted(strata.items())},
        # ★ 保底补了什么必须可见。静默的保底，下次它没生效同样是静默的 ——
        #   而"机制在但没接上"正是本项目最贵的那类 bug（00-START §6）。
        "sft_floors_applied": sft_floors,
        # 每个模板在 SFT 桶里最终有几条。**0 就是这次要修的那个 bug 又回来了。**
        "sft_template_counts": _count_by_template(buckets.sft),
    }
    if selection is not None:
        report["dead_grid_selection"] = selection
        report["quota_by_kind"] = dict(quota_by_kind or {})
    return buckets, report


def _count_by_template(case_ids: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for cid in case_ids:
        t = cid.split("_")[0]
        out[t] = out.get(t, 0) + 1
    return dict(sorted(out.items()))


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


# ══════════════════════════════════════════════════════════════════════════
# 数据版本：**一份值**，`entropy` / `eval_local` 共用
# ══════════════════════════════════════════════════════════════════════════
#
# ★★ 为什么要有这一节（2026-08-18 晚查出来的）
#
# `--batch` 和 `--split-dir` 是**必须同时动的一对**，而它们此前各写各的默认值，
# 且都是陈旧的：`entropy.py` 默认 v3、`eval_local.py` 默认 v2 ——
# 而 `data/batches/v2` 与 `data/batches/v3` **在本机根本不存在**。
#
# ⚠️ 两个失效方向，严重程度差很远：
#   ① 两个都用默认 ⇒ `FileNotFoundError`，**是响的**，只浪费一次上机
#   ② 🔴 只传 `--batch data/batches/v13`、忘了 `--split-dir`
#      ⇒ 拿 **v3 的 eval 桶 id** 去 v13 的 batch 里读，**24/24 全部读成功，不报错**
#        v3 eval 64 条 vs v13 343 条、交集仅 49 ⇒ 量的是另一个 case 集；
#        且那 24 条里 **4 条落在 v13 的 sft/rl 桶**（模型训过的题）⇒ 熵被记忆压低。
#      而决策位熵**正是决定 RL 起点的那把尺子**。
#
# ⇒ 这就是记过多次的第七形态：**默认值指向了另一件事，且不报错**。
#   修法两条，缺一不可：
#     ① 默认值收成**一份**（下面这个常量），换版本改一处
#     ② `assert_same_data_version` —— 「两个东西应当相同」型判据，
#        非黑即白、不需要阈值（守则①）。只改一个参数就硬失败。
#
# ⚠️ 换数据版本时**只改这一行**。不要在调用方各自写死。
# ★ 2026-09-03 Chaoyu 裁定⑩：全部口径 v16，case 库/切分/训练集从零重生成（B200 + 新栈），v13/v15 冻结读数不再是任何比较的一端。
DATA_VERSION = "v16"
DEFAULT_BATCH_DIR = f"data/batches/{DATA_VERSION}"
DEFAULT_SPLIT_DIR = f"data/splits/{DATA_VERSION}"
DEFAULT_SFT_DIR = f"data/sft/{DATA_VERSION}"      # SFT 训练集（原 data/sft/v15；契约协议名仍是 v15，数据版本是 v16）
DEFAULT_RL_DIR = f"data/rl/{DATA_VERSION}"


def data_version_of(path: str | Path) -> str | None:
    """从 `data/batches/v13` / `data/splits/v13` 这类路径里取出版本段。

    ⚠️ 认不出就返回 `None`，**不猜**（守则④）—— 调用方据此决定是"放过"还是"报没有"。
    """
    name = Path(str(path).rstrip("/")).name
    return name or None


def assert_same_data_version(batch: str | Path, split_dir: str | Path) -> str:
    """`--batch` 与 `--split-dir` 必须是同一个数据版本，不是就**硬失败**。

    返回那个共同的版本串（给调用方写进审计产物，见 `entropy.py`）。

    ⚠️ 刻意做成 `ValueError` 而不是 warning：这条的整个价值就在于
    「只改了一个参数」这件事**必须停下来**，而 warning 会被滚过去
    —— 上游那两个 bug 各自都只给了一行 UserWarning，我们一处都没接住。
    """
    vb, vs = data_version_of(batch), data_version_of(split_dir)
    if vb != vs:
        raise ValueError(
            f"数据版本不一致：--batch 是 {vb!r}，--split-dir 是 {vs!r}。\n"
            f"这两个参数**必须同时动** —— 只改一个不会报错，但会拿一个版本的 case_id "
            f"去另一个版本的 batch 里读，量出一个看起来合理的错数字。\n"
            f"⇒ 两个都传 {DATA_VERSION}，或者两个都不传（默认已是 {DATA_VERSION}）。")
    return vb
