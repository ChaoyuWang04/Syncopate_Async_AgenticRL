"""读/写分桶这把尺子（`eval_local._report_read_write`）。

★ 它为什么存在：设计文档 §21 把「读操作 ⊥ 写操作」列为三条**绝对不能合并**
的指标之一，而 §31.2 的毕业条件里也写着「E2E 成功率按读/写分桶均达标」——
但 2026-08-16 审计时发现**全项目没有任何代码按读/写分过桶**，
于是那一条从 M7 至今从来没有被判定过。

补上之后第一次量就看见了：只读桶 reward 0.909 / 满分成功率 69%，
含写动作桶 0.722 / **20%** —— 总均分 0.803 是被读操作撑起来的。
这正是 §21 那句「大量读操作会稀释掉写操作的风险」的实物证据。
"""

from __future__ import annotations

from syncopate.core.schemas import (
    Case, CaseBundle, CaseMetadata, EnvSnapshot, GoldPath, VerifierSpec,
)
from syncopate.train.eval_local import _report_read_write, is_write_case


def bundle(case_id: str, tools: list[str]) -> CaseBundle:
    return CaseBundle(
        case=Case(case_id=case_id, user_message="x", metadata=CaseMetadata("graded", "reasoning")),
        env=EnvSnapshot(case_id=case_id),
        verifier=VerifierSpec(expected_behavior="tool_call"),
        gold=GoldPath(actions=[{"tool": t, "arguments": {}} for t in tools],
                      final_answer={"summary": case_id}),
    )


def test_write_case_is_decided_by_the_registry_not_a_second_list():
    """★ 判据取自工具注册表的 `kind`，不另立一套「哪些算写」的清单。

    同一份工具在两个地方有两种说法，迟早会分叉 —— 沙盒和 runtime 的契约必须同源
    （见 09-runtime-handoff §1-③）。所以这里断言的是"跟着注册表走"，
    而不是断言某个写死的工具名单。
    """
    assert is_write_case(bundle("W_0000", ["campaign.get_metrics", "campaign.update_budget"]))
    assert not is_write_case(bundle("R_0000", ["campaign.get_metrics"]))


def test_read_only_case_with_no_gold_is_not_counted_as_write():
    """没有 gold 的 case 不该被算进写桶 —— 写桶是"要花钱"的那一桶，
    宁可漏算也不能虚报，虚报会把写桶的成功率稀释回去，正好抵消这把尺子的作用。"""
    b = bundle("X_0000", [])
    b.gold = None
    assert not is_write_case(b)


def test_report_separates_the_two_buckets(capsys):
    rows = [
        {"is_write": False, "reward": 1.0, "group": [1.0, 1.0], "caps": []},
        {"is_write": True, "reward": 0.5, "group": [0.5, 0.5], "caps": ["unauthorized_write_cap"]},
    ]
    _report_read_write(rows)
    out = capsys.readouterr().out
    assert "只读" in out and "含写动作" in out
    # 读−写 差值必须报出来：它是"总分被读操作撑起来了多少"的直接读数
    assert "+0.500" in out


def test_report_is_silent_when_there_is_no_write_case(capsys):
    """一条写 case 都没有时不出这段 —— 打一个分母为 0 的表比不打更容易误导。"""
    _report_read_write([{"is_write": False, "reward": 1.0, "group": [1.0], "caps": []}])
    assert capsys.readouterr().out == ""
