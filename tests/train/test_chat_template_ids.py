"""transformers 4.x/5.x 下 apply_chat_template(tokenize=True) 返回类型不同（list vs BatchEncoding）——
chat_template_ids 必须两种都还原成 list[int]，且与 tokenize=False 再 encode 的结果一致（守则①：两个东西应当相同）。"""
from pathlib import Path

import pytest

from syncopate.core.model_paths import TEST_TOKENIZER
from syncopate.train.rollout_loop import chat_template_ids


@pytest.fixture(scope="module")
def tok():
    if not Path(TEST_TOKENIZER, "tokenizer.json").exists():
        pytest.skip("no tokenizer locally")
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(TEST_TOKENIZER)


def test_returns_plain_list_of_ints(tok):
    msgs = [{"role": "user", "content": "hi"}]
    ids = chat_template_ids(tok, msgs, add_generation_prompt=True)
    assert isinstance(ids, list) and ids and all(isinstance(i, int) for i in ids)


def test_equals_text_then_encode(tok):
    msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "hi"}]
    ids = chat_template_ids(tok, msgs, add_generation_prompt=True)
    text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    assert ids == tok.encode(text, add_special_tokens=False)


def test_list_arithmetic_works(tok):
    """5.x 的 BatchEncoding + list 会 TypeError（09-03 新栈 39 红的根因）——helper 之后必须能直接拼。"""
    ids = chat_template_ids(tok, [{"role": "user", "content": "x"}], add_generation_prompt=True)
    assert len(ids + [1, 2]) == len(ids) + 2
