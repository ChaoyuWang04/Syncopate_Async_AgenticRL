from __future__ import annotations

from syncopate.train.eval_local import extract_thinking_texts


def test_eval_reads_qwen_prompt_side_think_opener():
    assert extract_thinking_texts([
        "先查指标</think>\n\n<tool_call>{}</tool_call>",
        "整理结论</think>\n\n给用户的答案",
    ], implicit_open=True) == ["先查指标", "整理结论"]


def test_eval_counts_unclosed_reasoning_but_not_visible_answer():
    assert extract_thinking_texts(
        ["重复思考直到被截断"], implicit_open=True
    ) == ["重复思考直到被截断"]
