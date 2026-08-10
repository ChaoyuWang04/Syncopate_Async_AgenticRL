"""控制轴测试：轴必须真的改变正确动作，不只是改数字。"""

from __future__ import annotations

import asyncio
import collections

import pytest

from syncopate.authoring.axes import Params, axis_summary, params_for
from syncopate.authoring.generate import verify_gold
from syncopate.authoring.templates import TEMPLATES
from syncopate.domains.adcampaign import build_domain

DOMAIN = build_domain()


@pytest.fixture(autouse=True)
def no_latency():
    original = DOMAIN.registry.latency_scale
    DOMAIN.registry.latency_scale = 0.0
    yield
    DOMAIN.registry.latency_scale = original


def _skeleton(bundle) -> tuple[str, ...]:
    return tuple(a["tool"] for a in bundle.gold.actions)


# --------------------------------------------------------------------------
# 1. 轴本身要铺开
# --------------------------------------------------------------------------


def test_every_axis_takes_all_its_values():
    summary = axis_summary([params_for(i) for i in range(120)])
    assert set(summary["entry_mode"]) == {"id_given", "must_discover"}
    assert set(summary["memory_state"]) == {"clean", "repeated", "risky"}
    assert set(summary["season_phase"]) == {"off", "approaching", "peak"}
    assert set(summary["amount_band"]) == {"below", "boundary", "above"}
    assert set(summary["memory_action"]) == {"none", "propose"}
    # 每个取值都不能太边缘，否则组合是虚的
    for axis, counts in summary.items():
        least = min(counts.values())
        assert least >= 120 / (len(counts) * 3), f"{axis} 分布太偏: {counts}"


def test_axes_are_decorrelated():
    """★ 两条轴不能同步变化——否则组合数是假的。

    `index * k % n` 在模数相同时只是重排，仍然一一对应。
    """
    ps = [params_for(i) for i in range(120)]
    pairs = {(p.entry_mode, p.memory_state) for p in ps}
    assert len(pairs) == 6, f"entry_mode × memory_state 应有 6 种组合，实际 {pairs}"
    pairs2 = {(p.season_phase, p.amount_band) for p in ps}
    assert len(pairs2) == 9


# --------------------------------------------------------------------------
# 2. ★ 轴要真的改变骨架
# --------------------------------------------------------------------------


def test_entry_mode_changes_the_skeleton():
    """must_discover 必须多出一步 campaign.list。"""
    for name in ("budget_change", "diagnosis", "creative_launch", "portfolio_review"):
        given = next(TEMPLATES[name](params_for(i)) for i in range(40)
                     if params_for(i).entry_mode == "id_given")
        discover = next(TEMPLATES[name](params_for(i)) for i in range(40)
                        if params_for(i).entry_mode == "must_discover")
        assert "campaign.list" not in _skeleton(given), name
        assert _skeleton(discover)[0] == "campaign.list", name


def test_memory_state_changes_the_outcome():
    """★ 记忆状态决定结局：直接执行 / 走审批 / 拒绝。"""
    outcomes = {}
    for state in ("clean", "repeated", "risky"):
        p = next(params_for(i) for i in range(200)
                 if params_for(i).memory_state == state
                 and params_for(i).amount_band == "below"
                 and params_for(i).entry_mode == "id_given")
        bundle = TEMPLATES["budget_change"](p)
        outcomes[state] = [t for t in bundle.case.metadata.tags if t.startswith("outcome:")][0]
    assert outcomes["clean"] == "outcome:executed"
    assert outcomes["repeated"] == "outcome:escalated"
    assert outcomes["risky"] == "outcome:denied"
    assert len(set(outcomes.values())) == 3


def test_season_phase_flips_creative_recommendation():
    """★ 同一条素材、同样超安全线，时令到了就从 block 变 launch。"""
    verdicts = {}
    for phase in ("off", "peak"):
        p = next(params_for(i) for i in range(200)
                 if params_for(i).season_phase == phase
                 and params_for(i).entry_mode == "id_given")
        bundle = TEMPLATES["creative_launch"](p)
        verdicts[phase] = [t for t in bundle.case.metadata.tags if t.startswith("outcome:")][0]
    assert verdicts["off"] == "outcome:block"
    assert verdicts["peak"] == "outcome:launch"


def test_memory_action_adds_write_proposal_step():
    """memory_action=propose 让写提案机制真的进 gold。"""
    for name in ("budget_change", "diagnosis", "creative_launch"):
        with_write = next(TEMPLATES[name](params_for(i)) for i in range(60)
                          if params_for(i).memory_action == "propose")
        without = next(TEMPLATES[name](params_for(i)) for i in range(60)
                       if params_for(i).memory_action == "none")
        assert _skeleton(with_write)[-1] == "memory.write_proposal", name
        assert "memory.write_proposal" not in _skeleton(without), name


def test_gold_proposals_meet_the_quality_bar():
    """gold 里的写提案必须自己合规（confidence≥0.7 且证据≥2），否则是在教坏。"""
    from syncopate.domains.adcampaign.memory import MIN_CONFIDENCE, MIN_EVIDENCE_REFS

    checked = 0
    for i in range(60):
        for name in ("budget_change", "diagnosis", "creative_launch"):
            bundle = TEMPLATES[name](params_for(i))
            for action in bundle.gold.actions:
                if action["tool"] != "memory.write_proposal":
                    continue
                checked += 1
                assert action["arguments"]["confidence"] >= MIN_CONFIDENCE
                assert len(action["arguments"]["evidence_refs"]) >= MIN_EVIDENCE_REFS
    assert checked > 0


# --------------------------------------------------------------------------
# 3. 骨架总数
# --------------------------------------------------------------------------


def test_skeleton_count_is_much_higher_than_v1():
    """★ v1 只有 7 种骨架，模型认出模板就赢了。v2 要显著多于它。"""
    skeletons = set()
    for i in range(80):
        p = params_for(i)
        for name in TEMPLATES:
            skeletons.add((name, _skeleton(TEMPLATES[name](p))))
    unique = {s for _, s in skeletons}
    assert len(unique) >= 25, f"骨架只有 {len(unique)} 种，分支轴没起作用"


@pytest.mark.parametrize("name", sorted(TEMPLATES))
def test_every_branch_of_every_template_has_working_gold(name):
    """每个模板的每条分支，gold 都要真跑通。"""
    for i in range(12):
        bundle = TEMPLATES[name](params_for(i))
        ok, reason = asyncio.run(verify_gold(bundle, DOMAIN))
        assert ok, f"{name} index={i}: {reason}"
