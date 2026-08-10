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
            row("c", 0.9, std=0.2)]                    # 别的模板 → 不该进
    grids, _ = dg.analyze(rows, strata)
    out = dg.add_controls(grids, rows, strata)

    assert out[dead].kind == dg.SHORTCUT
    assert out[sibling].kind == dg.CONTROL
    assert other_template not in out


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
    assert report["counts"]["sft"] == dg.DEFAULT_QUOTA[dg.SHORTCUT] + dg.DEFAULT_QUOTA[dg.CONVENTION]
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
