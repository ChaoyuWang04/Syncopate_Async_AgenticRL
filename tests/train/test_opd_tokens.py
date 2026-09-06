"""原始生成 ID 的 CPU 对照；不需要 Torch，也不重新分词生成文本。"""
from dataclasses import replace
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def real_tokenizer():
    from transformers import AutoTokenizer
    from syncopate.core.model_paths import STUDENT_MODEL
    model = Path(STUDENT_MODEL)
    if not model.is_absolute():
        model = Path(__file__).resolve().parents[2] / model
    return AutoTokenizer.from_pretrained(model, local_files_only=True)


@pytest.fixture
def mapper(real_tokenizer):
    from syncopate.train.opd_tokens import ByteLevelTokenMapper
    return ByteLevelTokenMapper(real_tokenizer)


def capture(mapper, response, **kwargs):
    from syncopate.train.opd_tokens import capture_generated_sample
    return capture_generated_sample(mapper, prompt_ids=(0, 1, 2),
        prompt_attention_mask=(0, 1, 1), prompt_position_ids=(0, 0, 1),
        generated_ids=(0, 1, 2, *response), eos_token_ids=(248046, 248044),
        pad_token_id=248044, implicit_think_open=False, **kwargs)


def test_noncanonical_generated_ids_survive_without_text_reencoding(mapper, real_tokenizer, record_property):
    import json
    record_property("tokenizer_runtime", json.dumps(mapper.evidence(), sort_keys=True))
    sample = capture(mapper, [64, 65])
    assert real_tokenizer.encode("ab", add_special_tokens=False) == [365]
    assert sample.response_text == "ab"
    assert sample.response_ids == (64, 65)
    assert sample.alignment.labels == ("text", "text")
    assert sample.input_ids == (0, 1, 2, 64, 65)
    assert sample.attention_mask == (0, 1, 1, 1, 1)
    assert sample.position_ids == (0, 0, 1, 2, 3)


@pytest.mark.parametrize("ids,text", [([160, 116, 255], "中"), ([68, 136, 223], "e\u0301")])
def test_byte_fragments_and_nfc_are_not_retokenized(mapper, real_tokenizer, ids, text):
    sample = capture(mapper, ids)
    assert sample.response_text == text
    assert tuple(real_tokenizer.encode(text, add_special_tokens=False)) != sample.response_ids
    assert sample.alignment.labels == ("text",) * len(ids)
    assert not sample.alignment.warnings


def test_invalid_utf8_masks_only_uncertain_bytes(mapper):
    sample = capture(mapper, [160, 116, 64])  # e4 b8 61: incomplete 中, then valid a
    assert sample.response_text == "\ufffda"
    assert sample.alignment.labels == ("format", "format", "text")
    assert sample.alignment.invalid_utf8_spans == ((0, 2),)
    assert "invalid_utf8" in sample.alignment.warnings


def test_same_replacement_text_does_not_erase_byte_identity(mapper):
    bad = capture(mapper, [187])  # ff
    literal = capture(mapper, [171, 123, 121])  # ef bf bd, genuine U+FFFD
    assert bad.response_text == literal.response_text == "\ufffd"
    assert bad.alignment.labels == ("format",)
    assert literal.alignment.labels == ("text", "text", "text")


def test_mixed_think_text_token_cannot_be_blessed_by_majority(mapper):
    sample = capture(mapper, [27, 14, 83, 71, 72, 77, 74, 76759])
    assert sample.response_text == "</think>Hello"
    assert sample.alignment.labels == ("think",) * 7 + ("format",)
    assert "mixed_label_span" in sample.alignment.warnings


def test_backend_only_special_tokens_are_masked_and_absent_from_visible_text(mapper, real_tokenizer):
    assert 248045 not in real_tokenizer.all_special_ids
    sample = capture(mapper, [64, 248045, 65])
    assert sample.response_text == "ab"
    assert sample.alignment.labels == ("text", "format", "text")
    assert sample.alignment.special_tokens_mask == (0, 1, 0)
    assert sample.alignment.byte_spans == ((0, 1), (1, 1), (1, 2))


def test_first_eos_is_kept_and_only_following_batch_padding_is_trimmed(mapper):
    sample = capture(mapper, [64, 248046, 248044, 248044])
    assert sample.response_ids == (64, 248046)
    assert sample.generated_ids == (0, 1, 2, 64, 248046, 248044, 248044)
    assert sample.alignment.labels == ("text", "format")
    with pytest.raises(ValueError, match="padding"):
        capture(mapper, [64, 248046, 65])


def test_eos_equal_to_padding_is_kept_once(mapper):
    sample = capture(mapper, [64, 248044, 248044, 248044])
    assert sample.response_ids == (64, 248044)
    assert sample.alignment.special_tokens_mask == (0, 1)


def test_padding_id_zero_is_not_treated_as_missing(mapper):
    from syncopate.train.opd_tokens import capture_generated_sample
    sample = capture_generated_sample(mapper, prompt_ids=(1,), prompt_attention_mask=(1,),
        prompt_position_ids=(0,), generated_ids=(1, 64, 248046, 0),
        eos_token_ids=(248046,), pad_token_id=0, implicit_think_open=False)
    assert sample.response_ids == (64, 248046)
    no_eos = capture_generated_sample(mapper, prompt_ids=(1,), prompt_attention_mask=(1,),
        prompt_position_ids=(0,), generated_ids=(1, 64, 0),
        eos_token_ids=(248046,), pad_token_id=0, implicit_think_open=False)
    assert no_eos.response_ids == (64, 0)  # not EOS: never trim a token by value alone


@pytest.mark.parametrize("field,value,match", [
    ("generated_ids", (0, 1, 3, 64), "prefix"),
    ("prompt_attention_mask", (1, 0, 1), "left"),
    ("prompt_attention_mask", (0, 0, 0), "nonempty"),
    ("prompt_position_ids", (0, 1, 2), "position"),
    ("generated_ids", (0, 1, 2, 999999), "unknown"),
])
def test_bad_generation_identity_or_interface_is_fatal(mapper, field, value, match):
    from syncopate.train.opd_tokens import capture_generated_sample
    args = dict(prompt_ids=(0, 1, 2), prompt_attention_mask=(0, 1, 1),
        prompt_position_ids=(0, 0, 1), generated_ids=(0, 1, 2, 64),
        eos_token_ids=(248046,), pad_token_id=248044, implicit_think_open=False)
    args[field] = value
    with pytest.raises(ValueError, match=match):
        capture_generated_sample(mapper, **args)


def test_mapper_never_calls_string_encode(mapper, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("sampled text must not be re-encoded")
    monkeypatch.setattr(mapper.tokenizer, "encode", forbidden)
    monkeypatch.setattr(type(mapper.tokenizer), "__call__", forbidden)
    assert capture(mapper, [64, 65]).response_ids == (64, 65)


def test_hf_decode_backend_mismatch_is_fatal(mapper, monkeypatch):
    monkeypatch.setattr(mapper.tokenizer, "decode", lambda *args, **kwargs: "corrupted")
    with pytest.raises(ValueError, match="decode"):
        capture(mapper, [64, 65])


def test_record_revalidation_rejects_forged_labels_or_truncated_masks(mapper):
    sample = capture(mapper, [64, 248045, 65])
    with pytest.raises(ValueError, match="alignment"):
        replace(sample, alignment=replace(sample.alignment, labels=("text",) * 3)).validate()
    with pytest.raises(ValueError, match="length"):
        replace(sample, prompt_attention_mask=(1,)).validate()


def test_record_revalidation_requires_integer_immutable_positions(mapper):
    sample = capture(mapper, [64, 65])
    with pytest.raises(ValueError, match="position"):
        replace(sample, prompt_position_ids=(0, False, 1)).validate()


def test_unknown_decoder_is_rejected_without_guessing_bytes():
    from tokenizers import Tokenizer, models, decoders
    from transformers import PreTrainedTokenizerFast
    from syncopate.train.opd_tokens import ByteLevelTokenMapper
    backend = Tokenizer(models.WordLevel({"a": 0, "<unk>": 1}, unk_token="<unk>"))
    backend.decoder = decoders.WordPiece(prefix="##")
    with pytest.raises(ValueError, match="decoder"):
        ByteLevelTokenMapper(PreTrainedTokenizerFast(tokenizer_object=backend))


def test_unlocatable_utf8_alignment_masks_whole_response_and_warns(mapper, monkeypatch):
    import syncopate.train.opd_tokens as tokens
    def cannot_locate(_payload):
        raise ValueError("decoder range unavailable")
    monkeypatch.setattr(tokens, "_utf8_units", cannot_locate)
    sample = capture(mapper, [64, 65])
    assert sample.response_text == "ab"
    assert sample.alignment.labels == ("format", "format")
    assert "utf8_alignment_unlocated" in sample.alignment.warnings


def test_token_audit_roundtrip_preserves_raw_arrays_and_counts(mapper, tmp_path):
    from syncopate.train.opd_tokens import TokenAuditWriter, GeneratedSample
    sample = capture(mapper, [64, 65])
    writer = TokenAuditWriter(tmp_path, run_token="0123456789abcdef", rank=0, runtime={"test": True})
    gen_id = writer.generation(sample, attempt=1, sample_index=0, turn_index=0, family="chat", final_turn=True)
    assert gen_id == "0:1"
    summary = writer.summary()
    assert summary["generated_records"] == 1
    import json
    rows = [json.loads(line) for line in Path(summary["path"]).read_text().splitlines()]
    assert rows[0]["event"] == "runtime"
    assert GeneratedSample.from_dict(rows[1]["sample"], mapper=mapper) == sample


@pytest.mark.parametrize('fault', ['eos', 'pad', 'post_eos_padding'])
def test_external_mapper_checks_ids_outside_the_distilled_response(mapper, fault):
    from syncopate.train.opd_tokens import GeneratedSample
    sample = capture(mapper, [64, 248046, 248044])
    if fault == 'eos':
        forged = replace(sample, eos_token_ids=sample.eos_token_ids + (999999,))
    elif fault == 'pad':
        forged = replace(sample, generated_ids=sample.generated_ids[:-1], pad_token_id=999999)
    else:
        forged = replace(sample, generated_ids=sample.generated_ids[:-1] + (999999,), pad_token_id=999999)
    # The local tuple relationships remain self-consistent, but the frozen
    # tokenizer has no such ID, even if it never receives an active loss mask.
    with pytest.raises(ValueError, match='unknown'):
        GeneratedSample.from_dict(forged.to_dict(), mapper=mapper)


def test_deserialization_has_no_self_certifying_default(mapper):
    from syncopate.train.opd_tokens import GeneratedSample
    sample = capture(mapper, [64, 65])
    with pytest.raises(TypeError):
        GeneratedSample.from_dict(sample.to_dict())


def test_external_validation_does_not_encode_sample_text(mapper, monkeypatch):
    from syncopate.train.opd_tokens import GeneratedSample
    sample = capture(mapper, [64, 65])
    def forbidden(*args, **kwargs):
        raise AssertionError('external validation must never encode sampled strings')
    monkeypatch.setattr(mapper.tokenizer, 'encode', forbidden)
    monkeypatch.setattr(type(mapper.tokenizer), '__call__', forbidden)
    assert GeneratedSample.from_dict(sample.to_dict(), mapper=mapper).response_ids == (64, 65)
