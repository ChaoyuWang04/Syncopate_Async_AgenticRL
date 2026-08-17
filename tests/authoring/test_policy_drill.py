"""M8 · `policy_drill` 模板的验收测试。

★ 这一份的核心是**双向断言**（`scripts/calibrate_retrieval.py` 那条警告要求的）：

    应命中的档  →  gold 的自然查询确实命中，且拿到的是**对的那条**
    空洞的档    →  该查询确实返回空，且 no_match 置位

没有这两条，哪条轴静默失效都看不出来 —— 阈值调到天上也没用。

★ 还有一条更隐蔽的：**过期版必须排在现行版前面**（陷阱）。
若现行版天然排第一，模型只要"取第一条"就能满分、根本不用看 expired，
「过期检出」这条轴就变成一个不需要那项能力就能通过的题。
"""

from __future__ import annotations

import pytest

from syncopate.authoring.axes import params_for
from syncopate.authoring.templates import TEMPLATES, _POLICY_QUERY
from syncopate.core.sandbox import Sandbox
from syncopate.core.tool_registry import ToolContext
from syncopate.domains.adcampaign import build_domain

DOMAIN = build_domain()
MAKE = TEMPLATES["policy_drill"]
N = 120                       # 覆盖到每档都有几十条


def _bundles():
    return [(params_for(i), MAKE(params_for(i))) for i in range(N)]


def _search(bundle, **kw):
    ctx = ToolContext(case=bundle.case, env=bundle.env, sandbox=Sandbox(bundle.env, "ns"),
                      step=1, tool_call_id="c1")
    args = {"query": _POLICY_QUERY, "platform": "meta", **kw}
    return DOMAIN.registry.get("policy.search").handler(args, ctx)


def _by_state(state: str):
    return [(p, b) for p, b in _bundles() if p.rag_state == state]


# --------------------------------------------------------------------------
# 双向断言
# --------------------------------------------------------------------------


@pytest.mark.parametrize("state", ["present", "superseded"])
def test_gold_query_actually_hits_the_right_clause(state: str) -> None:
    """★ 该中的真中，而且中的是对的那条。

    ⚠️ 若这条挂了，**不要去改断言，去改语料或查询词** —— gold 要求作答而检索给空的
    case 会让模型"转人工被判错、编造反而蒙对"，那是在训练我们正要消灭的行为。
    """
    cases = _by_state(state)
    assert cases, f"{state} 档一条都没有，轴的取模可能被削平了"
    for p, b in cases:
        hits = _search(b).data["hits"]
        ids = [h["clause_id"] for h in hits]
        want = b.case.entities["cited_clause_id"]
        assert want in ids, f"{b.case.case_id}: 应命中 {want}，实得 {ids}"


def test_empty_arm_really_returns_nothing() -> None:
    """★ 该空的真空。这是「无检索幻觉率」这项验收成立的前提。"""
    cases = _by_state("empty")
    assert cases
    for p, b in cases:
        data = _search(b).data
        assert data["hits"] == [], f"{b.case.case_id}: empty 档却召回了 {data['hits']}"
        assert data["no_match"] is True


def test_empty_arm_corpus_is_not_actually_empty() -> None:
    """空洞档装的是**别的主题**，不是空表。

    空表会让"查不到"退化成"库坏了"，模型可以靠"表是空的"这个旁证蒙对，
    而不是真的从"检索结果为空"这件事做判断。
    """
    for p, b in _by_state("empty"):
        assert len(b.env.readonly_tables["policy_clauses"]) >= 2


# --------------------------------------------------------------------------
# 陷阱：过期版排第一
# --------------------------------------------------------------------------


def test_expired_clause_ranks_first_so_take_the_top_hit_fails() -> None:
    """★★ 若这条挂了，「过期检出」这条轴就废了（详见模板里那段注释）。

    它目前依赖 `_V1` < `_V2` 的字母序 —— 改 id 命名会让陷阱**静默消失**，
    所以这里显式钉死。
    """
    cases = _by_state("superseded")
    assert cases
    for p, b in cases:
        ids = [h["clause_id"] for h in _search(b).data["hits"]]
        assert ids[0].endswith("_V1"), \
            f"{b.case.case_id}: 过期版没排第一，取第一条就能蒙对 ⇒ 这道题不再考过期检出"


def test_superseded_arm_marks_expiry_and_points_at_successor() -> None:
    for p, b in _by_state("superseded"):
        hits = {h["clause_id"]: h for h in _search(b).data["hits"]}
        want = b.case.entities["cited_clause_id"]
        old = next(k for k in hits if k.endswith("_V1"))
        assert hits[old]["expired"] is True
        assert hits[old]["superseded_by"] == want
        assert hits[want]["expired"] is False


def test_old_and_new_versions_state_different_numbers() -> None:
    """★ 新旧两版的**数值必须真的不同**（20% vs 50%）。

    数值一样的话，引用旧版和引用新版得出同一个结论，判据分辨不出模型有没有看有效期
    —— 那就成了"能被什么都不做骗过"的指标。（安全线那条轴记着同一件事。）
    """
    for p, b in _by_state("superseded"):
        rows = b.env.readonly_tables["policy_clauses"]
        old = next(v for k, v in rows.items() if k.endswith("_V1"))
        new = next(v for k, v in rows.items() if k.endswith("_V2"))
        assert "20%" in old["body"] and "50%" in new["body"]


# --------------------------------------------------------------------------
# 结构纪律
# --------------------------------------------------------------------------


def test_prompt_is_identical_across_arms_for_same_entry_mode() -> None:
    """★★ 「同一句话、不同世界、不同正确动作」——**问法不能泄露这是哪一档**。

    ⚠️ **判据在 2026-08-17 改过一次**：原来断言三档的问法**逐字相同**，
    那是"每个模板只有一句话"时代的写法。上了题面改写之后逐字相同不再成立，
    但要守的东西没变 —— **句式不许携带任何关于档位的信息**。

    ⇒ 新判据：**每一档用到的句式集合必须完全相同**（不是"差不多"）。
      某个句式只在 empty 档出现，模型看到它就知道这题会查空，
      根本不用等检索结果 —— 那正是这条测试要挡的。

    ★ 而这是**构造保证**的，不是碰巧：`_phrase` 按 (档 × entry_mode) 轮转选变体。
      第一版用哈希选，实测就漏了 —— empty 独有 2 条、superseded 独有 2 条，
      这条测试当场把它判红了。
    """
    import re as _re

    N = 150
    for mode in ("id_given", "must_discover"):
        by_arm: dict[str, set[str]] = {}
        for i in range(N):
            p = params_for(i)
            if p.entry_mode != mode:
                continue
            bundle = MAKE(p)
            msg = bundle.case.user_message
            for value in (p.campaign_id, p.product, p.region, p.account_id):
                msg = msg.replace(str(value), "§")
            by_arm.setdefault(getattr(p, "rag_state"), set()).add(_re.sub(r"\d+", "#", msg))
        union = set().union(*by_arm.values())
        for arm, phrasings in sorted(by_arm.items()):
            missing = union - phrasings
            assert not missing, (
                f"{mode} 下 {arm} 档缺了 {len(missing)} 种句式 —— "
                f"句式泄露了档位：看到这些说法就知道是哪一档。缺的：{sorted(missing)[:2]}")


def test_outcomes_do_not_collapse_into_one_grid_cell() -> None:
    """三档的 outcome 必须各不相同，否则 EVAL 分不出是哪种失败。

    FAIL 模板用 `outcome:{变体名}` 就是这个先例：写成统一的 "escalated"，
    stale 和 missing 会塌进同一个格子。
    """
    outcomes = {p.rag_state: {t for t in b.case.metadata.tags if t.startswith("outcome:")}
                for p, b in _bundles()}
    assert len(outcomes) == 3
    flat = [next(iter(v)) for v in outcomes.values()]
    assert len(set(flat)) == 3, f"outcome 塌成了同一个格子: {flat}"


def test_cited_clause_id_is_declared_so_the_cap_can_fire() -> None:
    """★★★ `cited_expired_policy_cap` 的**唯一激活开关**。

    缺了这个字段，那条 cap 自动闭合 ⇒ 永远是 0 ⇒ 看起来"模型没犯这个错"，
    实际是"这个错不可能被观测到"。这正是本项目反复栽的那个形状。
    """
    for p, b in _bundles():
        keys = {f.key for f in b.verifier.required_answer_fields}
        if p.rag_state == "empty":
            assert "cited_clause_id" not in keys      # 没东西可引用
            continue
        assert "cited_clause_id" in keys, f"{b.case.case_id}: cap 会永远沉默"
        assert b.gold.final_answer["cited_clause_id"] == b.case.entities["cited_clause_id"]


def test_both_m8_caps_are_active_on_this_template() -> None:
    for p, b in _bundles():
        active = set(b.verifier.active_caps or [])
        assert {"cited_expired_policy_cap", "no_retrieval_hallucination_cap"} <= active
