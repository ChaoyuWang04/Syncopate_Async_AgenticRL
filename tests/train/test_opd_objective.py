"""真实 kl_step 的小模型数学对照；不把合成 token 当真实 tokenizer 验收。"""
import copy

import pytest

torch = pytest.importorskip("torch")


@pytest.fixture
def scenario():
    from transformers import Qwen3Config, Qwen3ForCausalLM, PreTrainedTokenizerFast
    from tokenizers import Tokenizer, models, decoders, pre_tokenizers, AddedToken
    from syncopate.train.opd_tokens import ByteLevelTokenMapper, capture_generated_sample

    config = Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                         num_attention_heads=2, num_key_value_heads=1, head_dim=8, attention_dropout=0.0,
                         pad_token_id=0, eos_token_id=1, bos_token_id=2)
    config._attn_implementation = "eager"
    torch.manual_seed(17)
    student = Qwen3ForCausalLM(config).eval()
    torch.manual_seed(23)
    teacher = Qwen3ForCausalLM(config).eval().requires_grad_(False)
    vocab = {token: i for i, token in enumerate(
        ["<pad>", "<eos>", "P", "<think>", "</think>", "a", "b", "c", "ab",
         "<tool_call>", "</tool_call>"] + [f"<unused_{i}>" for i in range(11, 32)])}
    backend = Tokenizer(models.BPE(vocab=vocab, merges=[("a", "b")]))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    backend.add_tokens([AddedToken(token, special=False) for token in ("<think>", "</think>", "<tool_call>", "</tool_call>")])
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="<pad>", eos_token="<eos>",
                                       model_input_names=["input_ids", "attention_mask"])
    mapper = ByteLevelTokenMapper(tokenizer)
    rows = {}
    for name, prompt, mask, positions, response in [
        ("short", (0, 2, 2), (0, 1, 1), (0, 0, 1), (3, 5, 4, 5, 6)),
        ("long", (2, 2, 2), (1, 1, 1), (0, 1, 2), (9, 5, 10, 5, 6, 7)),
    ]:
        rows[name] = capture_generated_sample(mapper, prompt_ids=prompt, prompt_attention_mask=mask,
            prompt_position_ids=positions, generated_ids=prompt + response, eos_token_ids=(1,),
            pad_token_id=0, implicit_think_open=False)
    return student, teacher, tokenizer, rows


def flat_grad(model):
    return torch.cat([param.grad.flatten() for param in model.parameters() if param.grad is not None])


def test_real_kl_sum_and_token_mean_match_independent_formula(scenario, record_property):
    from syncopate.train.opd import kl_step, finish_token_window

    student, teacher, tokenizer, rows = scenario
    reference = copy.deepcopy(student)
    terms = []
    # 独立参考：完整前向，不使用 kl_step 的 logits_to_keep 或其 mask 构造。
    for sample in rows.values():
        tokens = torch.tensor([sample.prompt_ids + sample.response_ids])
        attention = torch.tensor([sample.prompt_attention_mask + (1,) * len(sample.response_ids)])
        positions = (attention.long().cumsum(-1) - 1).masked_fill(attention == 0, 0)
        start = len(sample.prompt_ids) - 1
        left = reference(tokens, attention_mask=attention, position_ids=positions).logits[0, start:-1].float().log_softmax(-1)
        with torch.no_grad():
            right = teacher(tokens, attention_mask=attention, position_ids=positions).logits[0, start:-1].float().log_softmax(-1)
        # Independently registered targets: two prose tokens / three prose tokens.
        active = torch.tensor([False] * 3 + [True] * (len(sample.response_ids) - 3))
        terms.append((left.exp() * (left - right)).sum(-1)[active])
    expected = torch.cat(terms).mean()
    expected.backward()
    target = flat_grad(reference)
    loss_sum, count = 0., 0
    for sample in rows.values():
        loss, tokens = kl_step(student, teacher, sample, "cpu")
        loss_sum += loss
        count += tokens
    assert count == 5
    summed = flat_grad(student).clone()
    old_relative = float((summed - target).norm() / target.norm())
    record_property("sum_vs_mean_relative_gradient_l2", old_relative)
    assert old_relative == pytest.approx(count - 1, rel=1e-4)
    result = finish_token_window(student.parameters(), local_tokens=count, local_loss_sum=loss_sum, device="cpu")
    corrected_relative = float((flat_grad(student) - target).norm() / target.norm())
    record_property("normalized_relative_gradient_l2", corrected_relative)
    assert corrected_relative < 1e-5
    assert result["loss"] == pytest.approx(float(expected.detach()), rel=1e-5)
    assert all(param.grad is None for param in teacher.parameters())


def test_zero_mask_is_real_forward_backward_not_empty_reply(scenario):
    from syncopate.train.opd import kl_step
    student, teacher, tokenizer, rows = scenario
    calls = []
    handles = [model.register_forward_hook(lambda *args: calls.append("forward")) for model in (student, teacher)]
    try:
        total, count = kl_step(student, teacher, rows["short"], "cpu", zero_mask=True)
    finally:
        for handle in handles: handle.remove()
    assert calls == ["forward", "forward"]
    assert total == 0 and count == 0
    gradients = [param.grad for param in student.parameters() if param.grad is not None]
    assert gradients and all(bool((grad == 0).all()) for grad in gradients)


def test_equal_student_and_teacher_have_zero_reverse_kl(scenario):
    from syncopate.train.opd import kl_step
    student, _, tokenizer, rows = scenario
    teacher = copy.deepcopy(student).requires_grad_(False)
    loss, count = kl_step(student, teacher, rows["long"], "cpu")
    assert count == 3 and loss == pytest.approx(0., abs=1e-9)
    assert flat_grad(student).norm() < 1e-6


def test_zero_control_requires_actual_graph_and_catches_removed_mask(scenario, monkeypatch):
    import syncopate.train.opd as opd
    student, teacher, tokenizer, rows = scenario
    from dataclasses import replace
    from syncopate.train.opd_tokens import ByteLevelTokenMapper
    empty = replace(rows['short'], generated_ids=rows['short'].prompt_ids, response_ids=(),
        alignment=ByteLevelTokenMapper(tokenizer).align((), implicit_think_open=False))
    with pytest.raises(ValueError, match='完整执行'):
        opd.zero_mask_control(student, teacher, empty, 'cpu')
    observed = opd.zero_mask_control(student, teacher, rows['short'], 'cpu')
    assert observed['student_forward'] == observed['aux_forward'] == observed['backward'] == 1
    assert observed['gradients'] > 0
    original = opd.kl_step
    def no_mask(*args, **kwargs):
        kwargs['zero_mask'] = False
        return original(*args, **kwargs)
    monkeypatch.setattr(opd, 'kl_step', no_mask)
    with pytest.raises(ValueError, match='非零'):
        opd.zero_mask_control(student, teacher, rows['short'], 'cpu')


@pytest.mark.parametrize("use_lora", [False, True])
def test_actual_generate_and_kl_forward_inputs_match_with_left_padding(scenario, record_property, use_lora):
    import json
    from syncopate.train.opd import gen_batch, kl_step, generation_runtime_evidence
    from syncopate.train.opd_tokens import ByteLevelTokenMapper
    student, teacher, tokenizer, _ = scenario
    backbone = student.model
    if use_lora:
        from peft import LoraConfig, get_peft_model
        student = get_peft_model(student, LoraConfig(r=2, lora_alpha=4, lora_dropout=0.0,
            target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")).eval()
    record_property("generation_runtime", json.dumps(generation_runtime_evidence(student), sort_keys=True))
    generation_calls, student_calls, teacher_calls = [], [], []

    def capture(calls):
        def hook(module, args, kwargs):
            calls.append({name: kwargs[name].detach().cpu().clone()
                          for name in ("input_ids", "attention_mask", "position_ids")})
        return hook

    # Hook the actual backbone.  A PEFT wrapper may call .forward directly on its
    # causal-LM wrapper, bypassing that wrapper's __call__ hooks.
    handle = backbone.register_forward_pre_hook(capture(generation_calls), with_kwargs=True)
    try:
        records = gen_batch(student, tokenizer, ["P", "PP"], max_new=3, temp=0.7, top_p=0.9,
                            top_k=20, mapper=ByteLevelTokenMapper(tokenizer))
    finally:
        handle.remove()
    assert len(records) == 2
    assert records[0].prompt_attention_mask == (0, 1)
    for i, sample in enumerate(records):
        assert generation_calls[0]["input_ids"][i].tolist() == list(sample.prompt_ids)
        assert generation_calls[0]["attention_mask"][i].tolist() == list(sample.prompt_attention_mask)
        assert generation_calls[0]["position_ids"][i].tolist() == list(sample.prompt_position_ids)
        # The last sampled target has no later generation forward.  Earlier
        # targets are the cached input to the next real generate iteration.
        for step in range(1, len(sample.response_ids)):
            call = generation_calls[step]
            assert call["input_ids"][i].tolist() == [sample.response_ids[step - 1]]
            assert call["position_ids"][i].tolist() == [sample.prompt_position_ids[-1] + step]
            assert call["attention_mask"][i].tolist() == list(sample.prompt_attention_mask + (1,) * step)
        hooks = [backbone.register_forward_pre_hook(capture(student_calls), with_kwargs=True),
                 teacher.model.register_forward_pre_hook(capture(teacher_calls), with_kwargs=True)]
        audit = {}
        try:
            kl_step(student, teacher, sample, "cpu", zero_mask=True, audit=audit)
        finally:
            for hook in hooks:
                hook.remove()
        for actual in (student_calls[-1], teacher_calls[-1]):
            assert actual["input_ids"].tolist() == [list(sample.input_ids)]
            assert actual["attention_mask"].tolist() == [list(sample.attention_mask)]
            assert actual["position_ids"].tolist() == [list(sample.position_ids)]
        assert audit["student_input_ids"] == audit["aux_input_ids"] == list(sample.input_ids)


def test_no_text_reencoding_api_or_early_zero_label_success(scenario):
    from syncopate.train.opd import kl_step
    from dataclasses import replace
    student, teacher, _tokenizer, rows = scenario
    with pytest.raises(TypeError):
        kl_step(student, teacher, "prompt", "reply", "cpu")
    bad = replace(rows['short'], prompt_position_ids=(0, 1, 2))
    with pytest.raises(ValueError, match="position"):
        kl_step(student, teacher, bad, "cpu")
