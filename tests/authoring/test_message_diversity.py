"""题面改写的回归守卫。

★ 分工：**批次级的多样性门禁**在 `scripts/check_data_gates.py`（D1–D11，
  大版本重建前必跑，见 `docs/syncopate/13-diversity-gates.md`）；
  **这一份**守的是「改写有没有破坏任务本身」—— 门禁看的是分布，这里看的是正确性。

★★★ 改写唯一不能碰的三样：**gold / env / verifier**

改写的目的是降低表层单调性，不是改变任务。所以判据不是"读着像不像"，
而是 —— **拿原来的 gold 重跑一遍，reward 必须一模一样**。
这把"改写有没有破坏题目"从人工检查变成了构造保证，和 `--freeze-from` 同一个思路。

★ 三条最容易被改写悄悄破坏的东西，各有一条测试盯着：
    CLAR  缺失的槽位**不能被改写补上** —— 补上了就不用问了，题目自相矛盾
    REJ   越权点 / 越域点**必须原样保留** —— 改没了就成了一道普通题
    multi 风格只许把 gold 链里本来就有的步骤显式说出来 ——
          加一个要额外调工具的诉求会触发 unnecessary_tool_call_cap
"""

from __future__ import annotations

import asyncio
import itertools
import statistics

import pytest

from syncopate.authoring.axes import params_for
from syncopate.authoring.templates import _PHRASINGS, TEMPLATES
from syncopate.core.runner import PlannedCall, run_plan
from syncopate.core.verifier_engine import score_trajectory
from syncopate.domains.adcampaign import build_domain
from syncopate.domains.adcampaign.corpus import tokenize

DOMAIN = build_domain()
# ⚠️ **必须压掉延迟**：超时注入是**真的 sleep**（沙盒刻意这么建的 ——
# 不计时的话超时在吞吐指标上就是免费的，异步收益会被系统性低估）。
# 不压的话 FAIL 那 150 条每条要等 30 秒，测试跑不完。
DOMAIN.registry.latency_scale = 0.0
N = 8


def _score(bundle):
    calls = [PlannedCall(tool=a["tool"], arguments=a["arguments"]) for a in bundle.gold.actions]
    trajectory, sandbox = asyncio.run(
        run_plan(bundle, DOMAIN.registry, calls, final_answer=bundle.gold.final_answer,
                 behavior=bundle.verifier.expected_behavior or "tool_call"))
    return score_trajectory(bundle, trajectory, sandbox,
                            policy_scorer=DOMAIN.policy_scorer,
                            decision_fn=DOMAIN.decision_fn, caps=DOMAIN.caps)


@pytest.mark.parametrize("name", sorted(TEMPLATES))
def test_gold_still_scores_full_after_paraphrasing(name: str) -> None:
    """★★★ 最重要的一条：**改了题面，gold 必须还是满分**。

    改写要是动了意图、实体、数值或缺失的槽位，这里会立刻红 ——
    而不是等到训练跑完发现学歪了。
    """
    make = TEMPLATES[name]
    for i in range(N):
        bundle = make(params_for(i))
        result = _score(bundle)
        assert result.reward >= bundle.gold.expected_reward_min, (
            f"{bundle.case.case_id}: 改写后 gold 只拿到 {result.reward:.3f}"
            f"（门槛 {bundle.gold.expected_reward_min}）· cap={[h.name for h in result.cap_hits]}"
            f"\n题面: {bundle.case.user_message}")


def test_clarify_paraphrases_never_fill_the_missing_slot() -> None:
    """★ CLAR 的题眼是「关键信息缺失且工具查不到」。

    改写要是顺手把 campaign_id 或地域补进去，模型就不用问了 —— 题目自相矛盾，
    而 `expected_behavior=clarify` 还在那儿，等于逼模型答一道已经没有必要问的题。
    """
    make = TEMPLATES["clarify"]
    for i in range(40):
        bundle = make(params_for(i))
        missing = bundle.gold.final_answer.get("missing_field", "")
        value = bundle.case.entities.get(missing) or bundle.case.context.get(missing)
        if value:
            assert str(value) not in bundle.case.user_message, (
                f"{bundle.case.case_id}: 改写把缺失的 {missing}={value} 补进题面了")


def test_reject_paraphrases_keep_the_violation() -> None:
    """★ REJ 的题眼是那个越权/越域的诉求本身。改写只许换说法，不许换那件事。"""
    make = TEMPLATES["reject"]
    # ⚠️ **按变体逐个校验，不按 reason 笼统查关键词。**
    # 第一版按 reason 给了一张宽表，结果把「RIVAL_0 那边预算给它调低点呗」判成
    # "改没了" —— 而那条的越权点是**那个竞品 ID**，不是"竞品"两个字。
    # ⇒ 判据要盯的是**这一条具体违反了什么**，不是一类里的常见词。
    markers = {
        0: lambda m, p: f"RIVAL_{p.index % 9}" in m,          # 动竞品的预算
        1: lambda m, p: "内部" in m,                           # 外泄内部文档
        2: lambda m, p: "诗" in m,                             # 越域：写诗
        3: lambda m, p: any(k in m for k in ("股", "行情", "大盘")),   # 越域：股市
        4: lambda m, p: p.account_id in m and any(              # 删光整个账户
            k in m for k in ("删", "清")),
        # 2026-09-04 新增 9 种越权/越域点（同样逐条盯"那件事"）
        5: lambda m, p: p.account_id in m and any(k in m for k in ("手机号", "设备")),   # 导出用户个人数据
        6: lambda m, p: f"RIVAL_{p.index % 9}" in m and "停" in m,                       # 停竞品的广告（暂停/停了）
        7: lambda m, p: any(k in m for k in ("刷", "假")),                                  # 刷量作弊
        8: lambda m, p: p.account_id in m and any(k in m for k in ("卡", "付款", "扣款")),   # 动账户财务
        9: lambda m, p: "管理员" in m and "审批" in m,                                       # 绕过审批
        10: lambda m, p: "安全线" in m and "代理" in m,                                      # 外泄内部数据
        11: lambda m, p: "工资" in m,                                                        # 越域：他人薪资
        12: lambda m, p: "请假" in m,                                                        # 越域：私人事务
        13: lambda m, p: "合同" in m and any(k in m for k in ("翻译", "翻")),                # 越域：法律翻译
    }
    from syncopate.authoring.templates import REJECT_REQUESTS
    assert len(markers) == len(REJECT_REQUESTS), "每种拒绝请求都要有自己的越权点判据"
    for i in range(60):
        p = params_for(i)
        bundle = make(p)
        idx = i % len(REJECT_REQUESTS)
        assert markers[idx](bundle.case.user_message, p), (
            f"{bundle.case.case_id}(变体{idx}): 改写把越权点改没了 —— "
            f"{bundle.case.user_message}")


def test_every_pool_covers_at_least_four_styles() -> None:
    """★ 五种表达结构（terse/context/question/multi/casual）是这套改写的核心。

    只做同义词替换的话，"句式数"和"相似度"两条都能刷过去，
    而模型学到的还是"看到这个句型就走这条链"。
    """
    for key, variants in _PHRASINGS.items():
        styles = {v["style"] for v in variants}
        assert len(styles) >= 4, f"{key} 只有 {len(styles)} 种风格：{sorted(styles)}"
        assert len(variants) >= 5, f"{key} 只有 {len(variants)} 条变体"


def test_pools_are_internally_diverse() -> None:
    """★ 池子内部两两相似度：平均 ≤0.40、最大 ≤0.75。

    ⚠️ 用平均管铺开、最大只挡近乎重复 —— **最大值这个统计量会结构性地惩罚短请求**
    （8 个词改 2 个词，Jaccard 就是 0.75），第一版只看最大值时判错过三个格子。

    ⚠️⚠️ **这条的阈值(0.40)比格子级指标(0.35)松，是有理由的，不是凑数据**：
    池子比的是**原始模板文本**，里面必须保留"这件事本身"的内容词
    （`reject.2` 的「诗」「夏天」，`reject.3` 的「股票」）—— 那些是**语义不是句式**，
    换掉就不是同一个请求了。而 `scripts/check_data_gates.py` 比的是
    **渲染并归一后的句式**，才是模型真正看到的分布。
    ⇒ **权威指标是那个脚本**；这条测试只负责挡"有人往池子里塞 5 条几乎一样的变体"。
    """
    import re

    for key, variants in _PHRASINGS.items():
        # ⚠️ 先把 `{slot}` 占位符抹掉再比：留着的话 `{name}`/`{dur}` 这些**共有的**
        # 花括号内容会把相似度顶高，和格子级指标（比的是渲染并归一后的句式）口径不一致。
        # 第一版没抹，creative_upload 报 0.38 不达标，而格子级量出来是 0.33。
        texts = [re.sub(r"\{[^}]*\}", "", v["id_given"]) for v in variants]
        sims = [len(set(tokenize(a)) & set(tokenize(b))) / len(set(tokenize(a)) | set(tokenize(b)))
                for a, b in itertools.combinations(texts, 2)]
        assert statistics.mean(sims) <= 0.40, f"{key} 平均相似 {statistics.mean(sims):.2f}"
        assert max(sims) <= 0.75, f"{key} 最大相似 {max(sims):.2f}"


def test_diversity_gate_script_is_the_source_of_truth() -> None:
    """★ 门禁脚本必须能 import 且判据齐 —— 它是 D1–D11 的真相来源。

    ⚠️ 这条不跑完整门禁（那要先生成批次，太慢），只保证脚本没烂掉、
    判据没被人悄悄删掉。**完整门禁在大版本重建前手动跑**：
        python scripts/check_data_gates.py --batch data/batches/<版本>
    见 `docs/syncopate/13-diversity-gates.md`。
    """
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "scripts" / "check_data_gates.py"
    spec = importlib.util.spec_from_file_location("check_data_gates", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    codes = [code for code, _, _ in module.CHECKS]
    assert codes == ["D1-D4", "D5", "D6", "D7", "D8", "D9-D10", "D11"], (
        f"门禁清单变了：{codes} —— 加减判据要同步更新 13-diversity-gates.md")
