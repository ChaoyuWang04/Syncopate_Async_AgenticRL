"""三桶之间的**泄露判据**。切分时用它构造保证，门禁脚本用它复查。

===========================================================================
为什么现有的 content_hash 不够
===========================================================================

`split.py` 的 `content_hash` 算的是「**完整渲染后的 prompt** + gold 答案 + behavior」
的 SHA-256，三桶实测 0 重叠。**那个检查是真的，但它有个盲区**：

    ATTR_0001 [EVAL]  题面「我们在 GB 想扩量。真人出镜这个素材特点到底有没有用？」
    ATTR_0048 [SFT ]  题面**一字不差**，gold 答案也一字不差（lift 0.1791）
    ⇒ 但两条的 context 里 campaign_id 不同 ⇒ **完整 prompt 不同 ⇒ 哈希不同 ⇒ 检查通过**

而模型要答的那件事完全一样：**把 SFT 那条背下来，不读世界也能答对 EVAL 这条。**
2026-08-17 实测 v13：**62 / 278 = 22.3% 的 EVAL 有这样的孪生。**

⇒ 教训和 D5 是同一条：**一个判据只覆盖了它作者当时想到的那种形状。**
  content_hash 想的是「两条 case 完全一样」，没想到「只有无关的 ID 不一样」。

===========================================================================
两级判据
===========================================================================

    L1  (题面原文, gold 答案) 跨桶重复          ⇒ **所有模板都是硬门禁**
        输入文字一模一样、答案一模一样，没有任何正当理由。

    L2  (题面句式, gold 答案) 跨桶重复          ⇒ **只对"该读世界"的模板是硬门禁**
        句式 = 抹掉实体和数值之后的题面。命中意味着「答案完全由问法决定」，
        模型不看世界也能答。

★★ **L2 必须分模板豁免，否则是误报**：

    CHAT / CLAR / REJ  的答案**本来就该由问法决定** ——
      「帮我把竞品预算调低」永远该是 reject/unauthorized，
      「帮我把日预算提到 400」（没给 campaign_id）永远该是 clarify。
      **这三类的可记忆性是我们要训的能力，不是漏洞。**

    其余模板（ATTR / CONF / SCALE / FRESH …）的答案**必须来自工具返回**。
      L2 命中 = 这条 case 根本没在考"读世界"，不管它在哪个桶。

⚠️ 豁免必须**显式列出来**（`L2_EXEMPT`），不许靠"反正跑出来是红的就调阈值"——
   空着的门槛应读作"无法判定"，不是"通过"。
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Iterable

from syncopate.core.schemas import CaseBundle

# ★ 答案本就由问法决定的模板。加进来要写清为什么，不许"因为它红了"。
L2_EXEMPT: dict[str, str] = {
    "CHAT": "不需要查数据的能力询问/闲聊，答案就该由问法决定",
    "CLAR": "关键槽位缺失，答案是「问哪个字段」，由问法决定",
    "REJ": "越权/越域，答案是「拒绝 + 类别」，由问法决定",
}


def _answer_key(bundle: CaseBundle) -> str:
    return json.dumps(bundle.gold.final_answer if bundle.gold else None,
                      sort_keys=True, ensure_ascii=False)


def skeleton(bundle: CaseBundle) -> str:
    """抹掉实体和数值后的题面。

    ★ 用 case 自己的 `entities` / `context` 去抹，不猜正则 ——
    模板往题面里填的东西本来就都在这两个字典里（同 `check_data_gates.py`）。
    """
    text = bundle.case.user_message
    values: list[Any] = []
    for source in (bundle.case.entities or {}, bundle.case.context or {}):
        for v in source.values():
            values.extend(v if isinstance(v, list) else [v])
    for v in sorted((str(x) for x in values if x not in (None, "")), key=len, reverse=True):
        text = text.replace(v, "§")
    return re.sub(r"\d+", "#", text)


def l1_key(bundle: CaseBundle) -> tuple[str, str]:
    """L1：题面原文 + 答案。**所有模板适用。**"""
    return (bundle.case.user_message, _answer_key(bundle))


def l2_key(bundle: CaseBundle) -> tuple[str, str]:
    """L2：题面句式 + 答案。**豁免模板不参与。**"""
    return (skeleton(bundle), _answer_key(bundle))


def template_of(case_id: str) -> str:
    return case_id.split("_")[0]


def grouping_key(bundle: CaseBundle) -> tuple[str, str]:
    """切分时用的**分组键**：同一个键的所有 case **必须落进同一个桶**。

    ★ 这是**构造保证**，不是事后 diff —— 和 `--freeze-from` 同一个思路：
      「哪些基线仍可比」要由构造决定，不能靠跑完再查。

    用 L1 还是 L2 做分组，取决于模板豁免与否：
      豁免模板（CHAT/CLAR/REJ）用 L1 —— 它们的可记忆性是特性，只挡完全相同的输入
      其余模板用 L2 —— 连"只换个 ID"的孪生也不许跨桶
    """
    if template_of(bundle.case_id) in L2_EXEMPT:
        return l1_key(bundle)
    return l2_key(bundle)


def audit(bundles: dict[str, CaseBundle],
          buckets: dict[str, Iterable[str]]) -> dict[str, Any]:
    """复查三桶之间有没有泄露。返回结构化报告（门禁脚本和切分报告共用）。"""
    where = {cid: name for name, ids in buckets.items() for cid in ids}

    def scan(key_fn: Callable[[CaseBundle], tuple[str, str]],
             skip_exempt: bool) -> dict[str, Any]:
        groups: dict[tuple[str, str], list[str]] = {}
        for cid, bundle in bundles.items():
            if cid not in where:
                continue
            if skip_exempt and template_of(cid) in L2_EXEMPT:
                continue
            groups.setdefault(key_fn(bundle), []).append(cid)
        cross = [cids for cids in groups.values()
                 if len({where[c] for c in cids}) > 1]
        affected = sorted({c for cids in cross for c in cids if where[c] == "eval"})
        by_template: dict[str, int] = {}
        for cid in affected:
            by_template[template_of(cid)] = by_template.get(template_of(cid), 0) + 1
        return {"cross_bucket_groups": len(cross),
                "affected_eval": len(affected),
                "affected_eval_ids": affected[:20],
                "by_template": dict(sorted(by_template.items()))}

    n_eval = len(list(buckets.get("eval", [])))
    l1 = scan(l1_key, skip_exempt=False)
    l2 = scan(l2_key, skip_exempt=True)
    l1["affected_eval_ratio"] = round(l1["affected_eval"] / max(1, n_eval), 4)
    l2["affected_eval_ratio"] = round(l2["affected_eval"] / max(1, n_eval), 4)
    # ★★ 判据分两类，性质不同，不能混成一个 passed：
    #
    #   **涉及 EVAL 的泄露 = 有效性问题**（评测数不可信）⇒ 硬门禁，必须 0
    #   **SFT ∩ RL 的重复 = 效率问题**（RL 在学 SFT 已经教会的东西 ⇒ 组内方差为 0
    #     ⇒ 零梯度）⇒ 报出来，不阻塞
    return {"L1_exact_message_and_answer": l1,
            "L2_skeleton_and_answer_non_exempt": l2,
            "L2_exempt_templates": L2_EXEMPT,
            "eval_leakage_free": l1["affected_eval"] == 0 and l2["affected_eval"] == 0,
            "train_pool_overlap_groups": {
                "L1": l1["cross_bucket_groups"], "L2": l2["cross_bucket_groups"]},
            "passed": l1["affected_eval"] == 0 and l2["affected_eval"] == 0}
