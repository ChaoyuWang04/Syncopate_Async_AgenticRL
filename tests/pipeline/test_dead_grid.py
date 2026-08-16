"""死格提取 + dead_grid 模式切桶。

这里守住的是一条容易被 reward 数字骗过去的判据：
**「全灭」和「卡死」的死因不同，混在一起 SFT 就会去修最便宜的那批。**
"""

from __future__ import annotations

import json

import pytest

from syncopate.core.schemas import Case, CaseBundle, CaseMetadata, EnvSnapshot, GoldPath, VerifierSpec
from syncopate.pipeline import dead_grid as dg
from syncopate.pipeline.split import split, stratum


def row(case_id: str, reward: float, std: float = 0.0, caps: list[str] | None = None) -> dict:
    return {"case_id": case_id, "reward": reward, "reward_std": std, "caps": caps or []}


# --------------------------------------------------------------------------
# classify
# --------------------------------------------------------------------------


@pytest.mark.parametrize("r, expected", [
    (row("A", 0.30, std=0.12, caps=["false_claim_cap"]), dg.GRADIENT),   # ★ std 优先于一切
    (row("B", 1.00), dg.SATURATED),
    (row("C", 0.00), dg.CONVENTION),
    (row("D", 0.30, caps=["missing_memory_check_cap"]), dg.SHORTCUT),
    (row("E", 0.75), dg.SUBSCORE),
    (row("F", 0.75, caps=["premature_decision_cap"]), dg.SUBSCORE),      # 不在指纹里的 cap 不算走捷径
])
def test_classify(r, expected):
    assert dg.classify(r) == expected


def test_gradient_wins_even_when_caps_hit():
    """有梯度就该留给 RL —— 哪怕它同时在犯「跳过前置」的错。

    反过来做（先看 cap 再看 std）会把 RL 自己能学会的 case 也搬进 SFT 桶，
    SFT 桶被稀释，而这些 case 的探索空间反而被 SFT 压掉了。
    """
    assert dg.classify(row("X", 0.3, std=0.2, caps=list(dg.SHORTCUT_CAPS))) == dg.GRADIENT


# --------------------------------------------------------------------------
# analyze
# --------------------------------------------------------------------------


def test_analyze_records_partial_evidence():
    """同格里 2 条 EVAL 只死了 1 条 → confidence 0.5，外推不可靠，报告里必须看得见。"""
    strata = {"a": ("REJ", "reject", "-", "id_given"), "b": ("REJ", "reject", "-", "id_given")}
    grids, kinds = dg.analyze([row("a", 0.0), row("b", 0.95)], strata)

    grid = grids[("REJ", "reject", "-", "id_given")]
    assert grid.kind == dg.CONVENTION
    assert grid.eval_case_ids == ["a"]
    assert grid.eval_seen == 2
    assert grid.confidence == 0.5
    assert kinds[dg.SATURATED] == 1


def test_analyze_mixed_cause_takes_the_expensive_one():
    """一格里两种死因都出现 → 按 shortcut 算（配额更大，宁可多给样本）。"""
    key = ("BUD", "tool_call", "denied", "id_given")
    grids, _ = dg.analyze(
        [row("a", 0.0), row("b", 0.3, caps=["false_claim_cap"])],
        {"a": key, "b": key},
    )
    assert grids[key].kind == dg.SHORTCUT


def test_controls_pull_in_the_sibling_grids_the_model_already_gets_right():
    """★ 实测打脸后加的：只喂难例会把模型已经做对的行为抹掉。

    I02 有三档，base 只在 mature 那格是死的 → 桶里全是「要回答，不要等」
    → 模型学成「见到这类题就回答」，defer 从 97% 掉到 0%。
    """
    dead = ("FRESH", "tool_call", "mature", "must_discover")
    sibling = ("FRESH", "defer", "immature", "id_given")
    other_template = ("BUD", "tool_call", "denied", "id_given")
    strata = {"a": dead, "b": sibling, "c": other_template}
    rows = [row("a", 0.3, caps=["false_claim_cap"]),   # 死格
            row("b", 0.9, std=0.2),                    # 同模板、模型做得对 → 该进对照
            row("c", 0.9, std=0.2)]                    # 别的模板 → **也该进**，见下一个测试
    grids, _ = dg.analyze(rows, strata)
    out = dg.add_controls(grids, rows, strata)

    assert out[dead].kind == dg.SHORTCUT
    assert out[sibling].kind == dg.CONTROL


def test_controls_cover_templates_that_have_no_dead_grid_at_all():
    """★ 2026-08-12：第一版只在「有死格的模板」下找对照，DIA/HIGH/LONG/MISS 因此各 0 条。

    而那几个模板**正是上一次退化里掉得最狠的**（HIGH -0.12 / LONG -0.11 / DIA -0.09）——
    对照防的是全局退化，一个完全没进过训练数据的模板照样会被带跑偏。
    """
    dead = ("FRESH", "tool_call", "mature", "must_discover")
    untouched = ("LONG", "tool_call", "-", "id_given")
    strata = {"a": dead, "b": untouched}
    rows = [row("a", 0.3, caps=["false_claim_cap"]), row("b", 0.95)]
    grids, _ = dg.analyze(rows, strata)
    out = dg.add_controls(grids, rows, strata)

    assert out[untouched].kind == dg.CONTROL


@pytest.mark.parametrize("r, ok", [
    (row("saturated", 0.95), True),
    (row("gradient", 0.24, std=0.14, caps=["false_claim_cap"]), True),
    (row("DIA-ish", 0.90), True),        # 卡在 0.9 边界外的 subscore，一条 cap 都没打中
    (row("MISS-ish", 0.745), True),
    (row("FRESH_0014", 0.683), False),   # 无 cap 但分数不够
    (row("FAIL_0009", 0.15, caps=["abandoned_without_escalation_cap"]), False),
    (row("clean-but-low", 0.5), False),
])
def test_control_eligible_lets_in_high_subscore_but_not_the_broken_ones(r, ok):
    """subscore 区间 0.15–0.9 两头是完全不同的东西，不能整类放开也不能整类挡死。"""
    assert dg.control_eligible(r) is ok


def test_control_grid_needs_only_one_eligible_row():
    """判据是「有」不是「全部」——用「全部」会丢掉 FRESH|defer|immature（0.598/0.683），
    而那正是本机制当初为之而生的那一格。"""
    key = ("FRESH", "defer", "immature", "must_discover")
    strata = {"a": key, "b": key}
    rows = [row("a", 0.598, std=0.226, caps=["behavior_mismatch"]), row("b", 0.683)]
    out = dg.add_controls({}, rows, strata)

    assert out[key].kind == dg.CONTROL


def test_controls_never_override_a_dead_grid():
    key = ("CLAR", "clarify", "-", "id_given")
    strata = {"a": key, "b": key}
    grids, _ = dg.analyze([row("a", 0.0), row("b", 0.95)], strata)
    out = dg.add_controls(grids, [row("a", 0.0), row("b", 0.95)], strata)
    assert out[key].kind == dg.CONVENTION      # 死格身份优先，不被对照档覆盖


def test_control_quota_is_smaller_than_dead_quota():
    """对照档是**锚定**，不是重新教一遍——给多了就把死格挤掉了。"""
    assert dg.DEFAULT_QUOTA[dg.CONTROL] < dg.DEFAULT_QUOTA[dg.CONVENTION]
    assert dg.DEFAULT_QUOTA[dg.CONTROL] < dg.DEFAULT_QUOTA[dg.SHORTCUT]


def test_analyze_ignores_live_grids():
    strata = {"a": ("BUD", "tool_call", "-", "-"), "b": ("CRE", "tool_call", "-", "-")}
    grids, kinds = dg.analyze([row("a", 0.5, std=0.2), row("b", 0.98)], strata)
    assert grids == {}
    assert kinds == {dg.GRADIENT: 1, dg.SATURATED: 1}


# --------------------------------------------------------------------------
# split(dead_grids=...)
# --------------------------------------------------------------------------


def make_batch(tmp_path, spec: dict[str, int]):
    """按 {模板前缀: 条数} 造一个最小 batch。outcome/entry 走 tags，和真实数据同源。"""
    entries = []
    for prefix, count in spec.items():
        template, behavior, outcome, entry = prefix.split("|")
        for i in range(count):
            cid = f"{template}_{i:04d}"
            bundle = CaseBundle(
                case=Case(case_id=cid, user_message="x",
                          metadata=CaseMetadata(signal_class="graded", bucket="tool_confusion",
                                                tags=[f"outcome:{outcome}", f"entry:{entry}"])),
                env=EnvSnapshot(case_id=cid),
                verifier=VerifierSpec(expected_behavior=behavior),
                gold=GoldPath(final_answer={"summary": cid}),
            )
            bundle.write(tmp_path)
            entries.append({"case_id": cid, "signal_class": "graded", "bucket": "tool_confusion",
                            "topology": "standard", "difficulty": "L2"})
    (tmp_path / "manifest.json").write_text(
        json.dumps({"version": "manifest_v1", "entries": entries}), encoding="utf-8")
    return tmp_path


def test_dead_grid_applies_quota_and_keeps_buckets_disjoint(tmp_path):
    batch = make_batch(tmp_path, {
        "BUD|tool_call|denied|id_given": 20,   # 死格 · shortcut
        "CLAR|clarify|-|id_given": 20,         # 死格 · convention
        "LOW|tool_call|-|id_given": 20,        # 活格，不该进 SFT
    })
    grids = {
        ("BUD", "tool_call", "denied", "id_given"): dg.DeadGrid(
            ("BUD", "tool_call", "denied", "id_given"), dg.SHORTCUT, ["BUD_0000"], 2),
        ("CLAR", "clarify", "-", "id_given"): dg.DeadGrid(
            ("CLAR", "clarify", "-", "id_given"), dg.CONVENTION, ["CLAR_0000"], 2),
    }
    buckets, report = split(batch, dead_grids=grids, quota_by_kind=dg.DEFAULT_QUOTA)

    assert report["mode"] == "dead_grid"
    # 配额取到的 = shortcut 10 + convention 6；再加上 clarify 的保底补足到 12
    # （CLAR 那格是 convention，只取了 6 条，稀有行为保底会把它补到 RARE_BEHAVIOR_SFT_FLOOR）
    quota_take = dg.DEFAULT_QUOTA[dg.SHORTCUT] + dg.DEFAULT_QUOTA[dg.CONVENTION]
    floor_top_up = dg.RARE_BEHAVIOR_SFT_FLOOR["clarify"] - dg.DEFAULT_QUOTA[dg.CONVENTION]
    assert report["counts"]["sft"] == quota_take + floor_top_up
    assert not any(c.startswith("LOW") for c in buckets.sft)   # 活格没被卷进来
    assert set(buckets.eval) & set(buckets.sft) == set()
    assert set(buckets.sft) & set(buckets.rl) == set()
    assert report["overlaps_by_content_sha256"] == {"eval∩sft": 0, "eval∩rl": 0, "sft∩rl": 0}
    # 池子里没被选中的同格 case 要落回 RL，不能凭空消失
    assert len(buckets.eval) + len(buckets.sft) + len(buckets.rl) == 60


def test_dead_grid_never_touches_frozen_eval(tmp_path):
    """★ EVAL 冻结：死格是在 EVAL 上测出来的，但那些 case 本身绝不能进 SFT。"""
    batch = make_batch(tmp_path, {"BUD|tool_call|denied|id_given": 20})
    key = ("BUD", "tool_call", "denied", "id_given")
    buckets, _ = split(batch, dead_grids={key: dg.DeadGrid(key, dg.SHORTCUT, ["BUD_0000"], 2)},
                       quota_by_kind=dg.DEFAULT_QUOTA)
    assert "BUD_0000" in buckets.eval
    assert "BUD_0000" not in buckets.sft


def test_quota_falls_back_to_whole_stratum(tmp_path):
    batch = make_batch(tmp_path, {"BUD|tool_call|denied|id_given": 20})
    key = ("BUD", "tool_call", "denied", "id_given")
    buckets, _ = split(batch, dead_grids={key: dg.DeadGrid(key, dg.SHORTCUT, ["BUD_0000"], 2)},
                       quota_by_kind={})
    assert len(buckets.sft) == 18 and buckets.rl == []


def test_stratum_reads_outcome_from_tags(tmp_path):
    bundle = CaseBundle(
        case=Case(case_id="BUD_0001", user_message="x",
                  metadata=CaseMetadata("graded", "tool_confusion",
                                        tags=["outcome:escalated", "entry:must_discover", "noise"])),
        env=EnvSnapshot(case_id="BUD_0001"),
        verifier=VerifierSpec(expected_behavior="tool_call"),
    )
    assert stratum("BUD_0001", bundle) == ("BUD", "tool_call", "escalated", "must_discover")


def test_rare_behavior_floor_tops_up_a_starved_behavior(tmp_path):
    """★ 2026-08-13：dead_grid 只按「哪些格子死了」选桶，做得好的行为一个都不进 —— 
    而"训练集里太稀薄"照样会被挤掉。

    实测：base 的 defer 双向已有 77%，于是 defer 一个格子都没进 SFT 桶，
    训练集里只剩 4 条，**epoch1 之后 defer 从 77% 崩到 36%**。
    这是 v3 那次「defer 97%→0%」的同一个坑，另一个维度：
        add_controls 保证的是「同意图的其它**档**」进桶
        这条保证的是「某个**行为**」有最低量
    """
    batch = make_batch(tmp_path, {
        "BUD|tool_call|denied|id_given": 20,      # 死格
        "FRESH|defer|immature|id_given": 20,      # 活格 —— dead_grid 一条都不会选
    })
    grids = {("BUD", "tool_call", "denied", "id_given"): dg.DeadGrid(
        ("BUD", "tool_call", "denied", "id_given"), dg.SHORTCUT, ["BUD_0000"], 2)}
    buckets, report = split(batch, dead_grids=grids, quota_by_kind=dg.DEFAULT_QUOTA)

    defer_in_sft = [c for c in buckets.sft if c.startswith("FRESH")]
    assert len(defer_in_sft) >= dg.RARE_BEHAVIOR_SFT_FLOOR["defer"], \
        "defer 没被保底补上 —— 它会在 SFT 后崩掉"
    # 2026-08-16：保底提成三个轴、两个分支共用后，明细挪到了 report["sft_floors_applied"]。
    # ★ behavior 轴**不设比例上限**：它要的是饱和（学得会），不是在场。
    assert report["sft_floors_applied"]["behavior"]["defer"] > 0


# --------------------------------------------------------------------------
# ★ difficulty_proxy 的在场保底（2026-08-16 验收审计推出）
# --------------------------------------------------------------------------


def test_difficulty_proxy_never_leaves_a_template_out_of_sft(tmp_path):
    """★★ v12 实际发生过：M8 新增的 POL / CONF 两个模板，SFT 桶里**各 0 条**。

    机制不是"选得不好"，是 `difficulty_proxy` 分支**一条保底都没有**
    （保底当时只写在 dead_grid 分支里，而 v11 走的正好是那条，所以没暴露）。
    `_difficulty_rank` 按 (难度, gold 链长) 降序取前 20%，POL 的 gold 只有
    「一次检索 + 终答」，链最短 ⇒ 整个模板被挤出桶外。

    后果不是"少学一点"，是 **M8 的验收路径整个断掉**：
    「p=0 的格子 RL 永远够不着」⇒ 没进 SFT ⇒ 冻结 EVAL 上必然量出"没学会"，
    **而这个结论和 RAG 实现好坏无关** —— 会把分桶 bug 误判成 RAG 设计问题。

    这里用「长链模板 × 大量」+「短链模板 × 少量」复现那个形状：
    没有保底时，短链的那个模板会被完全挤掉。
    """
    batch = make_batch(tmp_path, {
        "ATTR|tool_call|conclusive|id_given": 60,   # 链长，排前面，会吃满配额
        "POL|tool_call|answered|id_given": 40,      # 链短，正是被挤掉的那种
    })
    # gold 链长决定排序：给 ATTR 造长链，POL 保持最短
    for i in range(60):
        p = tmp_path / "gold_paths" / f"ATTR_{i:04d}.gold.json"
        g = json.loads(p.read_text(encoding="utf-8"))
        g["actions"] = [{"tool": "campaign.get_metrics", "arguments": {}} for _ in range(6)]
        p.write_text(json.dumps(g, ensure_ascii=False), encoding="utf-8")

    buckets, report = split(batch)

    assert report["mode"] == "difficulty_proxy"
    pol_in_sft = [c for c in buckets.sft if c.startswith("POL")]
    assert pol_in_sft, "POL 整个模板没进 SFT 桶 —— 这正是 v12 那个 bug"
    # 每个模板在报告里都要有数，0 就是 bug 回来了
    assert all(n > 0 for n in report["sft_template_counts"].values())
    # ★ 在场保底不能把 RL 吃光：那等于把坑搬到下一个阶段
    assert [c for c in buckets.rl if c.startswith("POL")], "POL 在 RL 桶里一条不剩"


def test_eval_thickening_is_strictly_additive(tmp_path):
    """★★ 冻结 EVAL 的加厚必须是**严格增量**的 —— 这是构造保证，不是事后 diff。

    取样写法是 `sorted(strata[key])[:take]`，调大 take 只会在尾部追加。
    只要这一点成立，历史基线（如 `_audit/v11_sft_e1_m2.json`）对老 case_id
    就仍然逐条配对可比 —— 而"哪些基线仍可比"必须由构造保证，
    否则每次改数据都要重新跑一遍全量 diff 才敢说话。
    """
    # ⚠️ 用一个**不在 OUTCOME_EVAL_QUOTA 里**的档，否则两边都被 clamp 到同一个数，
    #    测试会因为"两边相等"而失败 —— 那不是冻结被破坏，是这个用例没测到东西。
    batch = make_batch(tmp_path, {"BUD|tool_call|executed|id_given": 30})
    thin, _ = split(batch, eval_per_stratum=2)
    thick, _ = split(batch, eval_per_stratum=6)
    assert set(thin.eval) < set(thick.eval), "加厚把老成员挤掉了 —— 冻结被破坏"


def test_outcome_quota_thickens_the_m8_acceptance_tiers(tmp_path):
    """M8 两项验收（过期检出率 / 无检索幻觉率）的分母就是这几个档。

    实测 v12：每档只有 4 条 ⇒ 错一题 25%，判不了一个"趋近 0"的指标。
    根因是 `RARE_BEHAVIOR_EVAL_QUOTA` 只按 behavior 加厚，
    而这些档的 behavior 恰好是最常见的 `tool_call`。
    """
    batch = make_batch(tmp_path, {
        "POL|tool_call|answered_v2_from_superseded|id_given": 30,
        "BUD|tool_call|executed|id_given": 30,
    })
    buckets, _ = split(batch, eval_per_stratum=2)
    pol = [c for c in buckets.eval if c.startswith("POL")]
    bud = [c for c in buckets.eval if c.startswith("BUD")]
    assert len(pol) == 6, f"过期检出率的分母没被加厚：{len(pol)}"
    assert len(bud) == 2, "不该动的档位被顺手改了"
