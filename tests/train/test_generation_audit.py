from syncopate.train.generation_audit import (
    assistant_spans, generation_shape, quantiles, repeated_span,
)


def test_periodic_run_after_nonrepeating_prefix():
    assert repeated_span([99, 98] + [1, 2, 3] * 400 + [97]) == {
        "span_tokens": 1200, "period_tokens": 3, "copies": 400}
    assert repeated_span(list(range(100)))["copies"] == 0


def test_missing_close_and_repeat_are_independent():
    unclosed = generation_shape("budget " * 100, [1] * 100, implicit_think_open=True)
    assert unclosed["think_unclosed"]
    assert unclosed["repeat"]["span_tokens"] == 100
    normal = generation_shape("已确认条件。\n</think>\n结果完成。", list(range(200)), implicit_think_open=True)
    assert not normal["think_unclosed"]
    assert normal["repeat"]["copies"] == 0
    long_unclosed = generation_shape("还在推理", list(range(12239)), implicit_think_open=True)
    assert long_unclosed["think_unclosed"] and not long_unclosed["repeat"]["copies"]


def test_spans_keep_first_and_post_tool_assistant_turn():
    # 9=end, [7,8]=assistant header；工具区不算 assistant。
    ids = [0, 0, 1, 2, 9, 4, 5, 9, 7, 8, 3, 6, 9]
    assert list(assistant_spans(ids, 2, header=[7, 8], end_id=9)) == [(2, 4, True), (10, 12, True)]
    assert list(assistant_spans([0, 1, 2], 1, header=[7, 8], end_id=9)) == [(1, 3, False)]


def test_empty_quantile_is_not_a_pass():
    assert quantiles([])["p99"] is None
    assert quantiles(range(1, 101))["p99"] == 99
