"""人话字段**永远不走机器通道**（v15 契约，`25 §3.1` / `§7㉗`）。

⛔ 判据来源是考场实测，不是设计推演：`multiturn_l1` 那 150 行把 `reply` 塞进
  `session.report`、下一步再原样抄一遍，模型学成了"先往机器通道写一句再复制"。
  复制模式一断就落回背下来的 reject —— 50 道概念题 41 道被拒（08-30 v15r2 考场）。

三条一起查，因为它们是**同一条规则的三个消费点**，少一条就会各自漂：
  ① 构造侧不写进 report   ② 教学面/生产面清单里看不见   ③ 判分侧从终答文本取
"""
from __future__ import annotations

import pytest

from syncopate.core.contract import IS_V15, PROSE_FIELDS, visible_answer_fields
from syncopate.core.schemas import AnswerField
from syncopate.pipeline.split import DEFAULT_BATCH_DIR, DEFAULT_SPLIT_DIR, DEFAULT_SFT_DIR, DEFAULT_RL_DIR

pytestmark = pytest.mark.skipif(not IS_V15, reason="v15 契约专有")


def _l1_bundle():
    """L1 形态：概念追问，终答只有一句人话（`summary` 已废除但仍留在旧 spec 里）。

    直接取一条真 bundle 改造 —— 手搓一个假 bundle 就等于再写一份 spec 副本。
    """
    import copy
    from pathlib import Path

    from syncopate.pipeline.split import load_bundles

    src = next(b for b in load_bundles(Path(DEFAULT_BATCH_DIR)).values() if b.gold)
    b = copy.deepcopy(src)
    b.verifier.required_answer_fields = [
        AnswerField(key="summary", description="结论的机器可校验形式"),
        AnswerField(key="reply", description="给用户读的完整回复"),
    ]
    b.gold.actions = []
    b.gold.final_answer = {"summary": "ROAS 释义", "reply": "ROAS 就是广告花的钱能带回多少收入。"}
    return b


def test_report_step_disappears_for_prose_only_case() -> None:
    """只有人话要给 ⇒ **不该有 report 这一步**（否则就是"写一句再抄一遍"）。"""
    from syncopate.pipeline.sft_replay import _machine_fields, _v15_tail

    b = _l1_bundle()
    assert _machine_fields(b) == {}, "人话字段混进了机器通道"
    steps = _v15_tail(b, "answer")
    assert not any("session.report" in s for s in steps), f"仍有 report 步：{steps}"
    assert steps[-1].strip() == "ROAS 就是广告花的钱能带回多少收入。"


def test_prose_fields_hidden_from_the_field_list() -> None:
    """清单是"要填的表"，人话不是表格里的一格 —— 模型照着表填才会把人话写进机器通道。"""
    fields = _l1_bundle().verifier.required_answer_fields
    assert [f.key for f in visible_answer_fields(fields)] == []
    assert PROSE_FIELDS >= {"reply", "summary"}


def test_verifier_reads_prose_from_the_final_text() -> None:
    """不写进 report 之后，判分必须去终答文本里看 —— 否则等于白丢分。"""
    from syncopate.core.trajectory import Trajectory
    from syncopate.core.verifier_engine import _score_answer_fields

    b = _l1_bundle()
    traj = Trajectory(case_id=b.case.case_id, rollout_id="t", namespace_id="t",
                      behavior="answer", final_answer={},
                      final_text="ROAS 就是广告花的钱能带回多少收入。")
    traj.parse_ok = True
    score, detail = _score_answer_fields(b.verifier, b, traj, None)
    assert score == 1.0, detail
