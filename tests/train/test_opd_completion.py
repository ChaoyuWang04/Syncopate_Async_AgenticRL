import pytest

from syncopate.train.opd import prior_result, prioritize_smoke_routes, training_completed


def test_skips_do_not_count_as_real_opd_updates():
    assert training_completed(real_steps=0, target_real_steps=1) is False


def test_smoke_requires_all_registered_real_updates():
    assert training_completed(real_steps=1, target_real_steps=2) is False
    assert training_completed(real_steps=2, target_real_steps=2) is True


def test_unbounded_candidate_still_requires_one_real_update():
    assert training_completed(real_steps=0, target_real_steps=0) is False
    assert training_completed(real_steps=1, target_real_steps=0) is True


def test_smoke_prefix_covers_task_and_chat_without_changing_the_remainder():
    rows = [
        {"id": "c1", "family": "chat"},
        {"id": "t1", "family": "task"},
        {"id": "t2", "family": "task"},
        {"id": "c2", "family": "chat"},
    ]
    got = prioritize_smoke_routes(rows)
    assert [row["id"] for row in got] == ["t1", "c1", "c2", "t2"]
    assert rows[0]["id"] == "c1", "helper must not mutate the shuffled candidate list"


def test_smoke_route_prefix_refuses_a_one_family_dataset():
    with pytest.raises(ValueError, match="task.*chat"):
        prioritize_smoke_routes([{"id": "t1", "family": "task"}])


def test_prior_result_restores_qwen_implicit_think_open():
    assert prior_result("内部推理</think>给用户的人话", thinking_enabled=True) == {
        "text": "给用户的人话"
    }


def test_prior_result_rejects_unclosed_thinking_and_nonterminal_tool_call():
    with pytest.raises(ValueError, match="没有闭合"):
        prior_result("只有内部推理", thinking_enabled=True)
    with pytest.raises(ValueError, match="没有形成可回灌终态"):
        prior_result(
            '<tool_call>{"name":"x.y","arguments":{}}</tool_call>',
            thinking_enabled=False,
        )
