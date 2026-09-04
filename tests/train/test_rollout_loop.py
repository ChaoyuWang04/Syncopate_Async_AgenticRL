"""agent 主循环测试：真 tokenizer + 假引擎。

用真的 Qwen3-0.6B tokenizer（token 对齐问题只有真 tokenizer 才暴露得出来），
生成侧用脚本化的假引擎——这样不需要 GPU、不需要 verl，秒级跑完，
但 prompt 渲染 / 解析 / 工具执行 / mask 对齐全都是真的。
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from syncopate.authoring.seed_cases import SEED_BUILDERS
from syncopate.core.parsing import render_final_answer, render_tool_call
from syncopate.core.verifier_engine import score_trajectory
from syncopate.domains.adcampaign import build_domain
from syncopate.train.rollout_budget import assistant_turn_budget
from syncopate.core.model_paths import TEST_TOKENIZER, STUDENT_MODEL, TEACHER_MODEL
from syncopate.train.rollout_loop import (
    CHAT_TEMPLATE_KWARGS, RolloutConfig, build_messages, observation_message, run_rollout,
)

MODEL_DIR = Path(TEST_TOKENIZER)
DOMAIN = build_domain()

pytestmark = pytest.mark.skipif(not MODEL_DIR.exists(), reason="需要 models/Qwen3-0.6B 软链")


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(MODEL_DIR))


@pytest.fixture(autouse=True)
def fast_latency():
    original = DOMAIN.registry.latency_scale
    DOMAIN.registry.latency_scale = 0.0002
    yield
    DOMAIN.registry.latency_scale = original


class ScriptedEngine:
    """假引擎：按剧本一步步吐出预设的文本。

    它同时记录每次收到的 prompt 长度——用来验证「上下文是不是在正确增长」。
    """

    def __init__(self, tokenizer, script: list[str]) -> None:
        self.tokenizer = tokenizer
        self.script = list(script)
        self.prompt_lengths: list[int] = []

    async def __call__(self, prompt_ids, sampling_params):
        self.prompt_lengths.append(len(prompt_ids))
        if not self.script:
            return []
        return self.tokenizer.encode(self.script.pop(0), add_special_tokens=False)


def _gold_script(bundle) -> list[str]:
    """把 gold 轨迹翻译成「模型该输出什么」的剧本 = **直接用生产那一份**。

    ⛔ 2026-08-30：这里原本自己抄了一份实现（tool_call + render_final_answer）。
      think-off 的 v14 下两份**恰好逐字节相等**，所以副本存在了很久没人发现；
      一开 think-on（`25 §3.2` 修法 A），生产的 gold_script 会给每个 assistant 轮
      加 think 段、副本不会 ⇒ 同构测试 6/6 红。
      而 25 §3.2 恰恰写着「同构靠"只有一条代码路径"保证，与开关无关」——
      **那句话当时是错的：一共有两条，另一条在测试里。**
    ★ 一般化（守则⑨）：凡是「这个值/这段逻辑应该和那边一致」的地方，
      正确的判据是「这里根本不该有第二份」。**测试里的副本也是副本。**
    """
    from syncopate.pipeline.sft_replay import gold_script
    return gold_script(bundle)


def _final(fields: dict, behavior: str = "tool_call") -> list[str]:
    """按当前契约产「终答段」的剧本片段（v14=壳一步 / v15=report + 一句人话）。

    ★ 测试也必须按契约分家 —— 用 v14 的输出去考 v15，量的是另一件事
      （`25 §7⑦` 同一形状：跨契约的判据要先问「新口径在旧产物里存不存在」）。
    """
    from syncopate.core.contract import IS_V15
    if not IS_V15:
        return [render_final_answer(behavior, fields)]
    from syncopate.core.parsing_v15 import render_report
    return [render_report(fields), "已按上面的结果处理完了。"]


async def _run(bundle, tokenizer, script, config=None):
    engine = ScriptedEngine(tokenizer, script)
    output = await run_rollout(
        bundle, registry=DOMAIN.registry, tokenizer=tokenizer, generate=engine,
        config=config or RolloutConfig(
            max_assistant_turns=assistant_turn_budget(bundle.case.max_steps)),
    )
    return output, engine


# --------------------------------------------------------------------------
# 1. token 对齐 —— 多轮 RL 最容易错的地方
# --------------------------------------------------------------------------


def test_mask_length_matches_and_marks_env_tokens_zero(tokenizer):
    """★ response_mask 必须和 response_ids 等长，且工具 token 必须是 0。

    把工具返回的 token 也算进梯度，等于训练模型「复述环境给它的东西」，
    会严重污染训练信号。这是多轮 agentic RL 最经典的坑。
    """
    bundle = SEED_BUILDERS["SIG_GRADED_001"]()
    output, _ = asyncio.run(_run(bundle, tokenizer, _gold_script(bundle)))

    assert len(output.response_ids) == len(output.response_mask)
    assert set(output.response_mask) == {0, 1}

    # 用 token_trace 交叉验证：assistant 段的 token 数应等于 mask=1 的总数
    assistant_tokens = sum(s["token_count"] for s in output.token_trace["segments"] if s["mask"] == 1)
    assert assistant_tokens == sum(output.response_mask)

    env_tokens = sum(s["token_count"] for s in output.token_trace["segments"] if s["mask"] == 0)
    assert env_tokens == len(output.response_mask) - sum(output.response_mask)
    assert env_tokens > 0, "应该有工具 observation 被插进来"


def test_context_grows_monotonically(tokenizer):
    """每一步看到的上下文都应该比上一步长——工具返回被正确回灌了。"""
    bundle = SEED_BUILDERS["SIG_GRADED_001"]()
    _, engine = asyncio.run(_run(bundle, tokenizer, _gold_script(bundle)))
    assert engine.prompt_lengths == sorted(engine.prompt_lengths)
    assert len(set(engine.prompt_lengths)) == len(engine.prompt_lengths)


def test_token_trace_maps_every_token_to_a_step(tokenizer):
    """★ token -> step 的映射必须完整覆盖，这是步级信用分配的地基。"""
    bundle = SEED_BUILDERS["SIG_GRADED_001"]()
    output, _ = asyncio.run(_run(bundle, tokenizer, _gold_script(bundle)))
    traced = sum(s["token_count"] for s in output.token_trace["segments"])
    assert traced == len(output.response_ids)
    assert all(s["step"] >= 1 for s in output.token_trace["segments"])


# --------------------------------------------------------------------------
# 2. 端到端：gold 剧本走完必须拿高分
# --------------------------------------------------------------------------


@pytest.mark.parametrize("case_id", sorted(SEED_BUILDERS))
def test_gold_script_through_full_loop_scores_high(case_id, tokenizer):
    """★ 这是真正的端到端：prompt 渲染 -> 生成 -> 解析 -> 执行 -> 判分。

    和 test_seed_cases 的区别：那边直接喂 PlannedCall 跳过了模型输出格式，
    这里走的是完整的「文本 -> 解析 -> 工具」链路。
    """
    bundle = SEED_BUILDERS[case_id]()
    output, _ = asyncio.run(_run(bundle, tokenizer, _gold_script(bundle)))

    assert output.trajectory.parse_ok, f"{case_id} 终答解析失败"
    bad = [(o.tool, o.error) for o in output.trajectory.observations if not o.ok]
    assert not bad, f"{case_id} 工具报错: {bad}"

    result = score_trajectory(
        bundle, output.trajectory, output.sandbox,
        policy_scorer=DOMAIN.policy_scorer, decision_fn=DOMAIN.decision_fn, caps=DOMAIN.caps,
    )
    assert result.reward >= bundle.gold.expected_reward_min, (
        f"{case_id} reward={result.reward} subscores={result.subscores} "
        f"caps={[h.name for h in result.cap_hits]}"
    )


# --------------------------------------------------------------------------
# 3. 失败模式
# --------------------------------------------------------------------------


def test_unparseable_output_gets_feedback_and_retries(tokenizer):
    """输出格式崩了 -> 把错误喂回去 -> 模型有机会自己修。"""
    bundle = SEED_BUILDERS["SIG_HIGH_001"]()
    from syncopate.core.contract import IS_V15
    # ★ "格式崩了"的形态**本身就是契约的一部分**：
    #   v14 里「既不是 tool_call 也不是 json」= 崩；
    #   v15 里纯文本是**合法终答** ⇒ 同一句话不再是错误。能崩的只剩坏 tool_call。
    #   ⚠️ 这是换契约的一个真实代价，已记进 25 §6：v15 少了一层"胡言乱语"的网。
    broken = ('<tool_call>\n{"name": "campaign.get_metrics", "arguments": {,,,}\n</tool_call>'
              if IS_V15 else "我觉得 CPI 大概是 2.1 左右吧")
    script = [
        broken,
        render_tool_call("campaign.get_metrics", {"campaign_id": "CMP_1024"}),
        *_final({"cpi": 2.10}),
    ]
    output, _ = asyncio.run(_run(bundle, tokenizer, script))

    assert output.metrics["parse_errors"] == 1
    assert output.trajectory.parse_ok is True          # 最终修好了
    assert output.trajectory.final_answer == {"cpi": 2.10}


def test_thinking_block_is_stripped(tokenizer):
    """Qwen3 的 <think> 块必须剥掉再解析，否则 JSON 抽不出来。"""
    bundle = SEED_BUILDERS["SIG_HIGH_001"]()
    script = [
        "<think>用户要查 CPI，先调 get_metrics</think>"
        + render_tool_call("campaign.get_metrics", {"campaign_id": "CMP_1024"}),
        *[("<think>拿到了，CPI 是 2.10</think>" + x) if i == 0 else x
          for i, x in enumerate(_final({"cpi": 2.10}))],
    ]
    output, _ = asyncio.run(_run(bundle, tokenizer, script))
    assert output.trajectory.final_answer == {"cpi": 2.10}
    assert output.trajectory.actions[0].name == "campaign.get_metrics"


def test_tool_outside_menu_is_refused(tokenizer):
    """★ tool_missing case：菜单外的工具即使注册表里有，也必须拒绝执行。

    否则模型可以靠调用「隐藏工具」绕过能力缺口，这条 case 就白设计了。
    """
    from syncopate.core.contract import IS_V15
    if IS_V15:
        pytest.skip("v15：训练与线上同为全量菜单（守则⑮ #6，contract.effective_tool_menu），不再靠裁菜单造缺工具")
    bundle = SEED_BUILDERS["SIG_TOOLMISS_001"]()
    script = [
        render_tool_call("campaign.detect_anomalies", {"campaign_id": "CMP_6144"}),   # 不在菜单里
        render_tool_call("campaign.get_metrics", {"campaign_id": "CMP_6144"}),
        render_final_answer("tool_call", {"anomaly_type": "cpi_spike", "campaign_cpi": 2.90}),
    ]
    output, _ = asyncio.run(_run(bundle, tokenizer, script))

    first = output.trajectory.observations[0]
    assert first.ok is False
    assert "tool_not_available" in first.error
    assert DOMAIN.registry.get("campaign.detect_anomalies") is not None   # 工具本身存在


def test_max_turns_truncates(tokenizer):
    """一直不给终答 -> 撞上步数上限 -> truncated + max_steps_cap。"""
    bundle = SEED_BUILDERS["SIG_HIGH_001"]()
    script = [render_tool_call("campaign.get_metrics", {"campaign_id": "CMP_1024"})] * 6
    output, _ = asyncio.run(_run(bundle, tokenizer, script, RolloutConfig(max_assistant_turns=3)))

    assert output.trajectory.truncated is True
    result = score_trajectory(
        bundle, output.trajectory, output.sandbox,
        policy_scorer=DOMAIN.policy_scorer, decision_fn=DOMAIN.decision_fn, caps=DOMAIN.caps,
    )
    assert "max_steps_cap" in result.cap_steps


def test_false_claim_is_caught(tokenizer):
    """★ 补上的漏洞：不查审核就直接声称 approved。

    值是对的（literal:approved），outcome 会给满分——但 false_claim_cap 会封顶。
    小模型最擅长猜高频词蒙混过关，这条必须堵死。
    """
    bundle = SEED_BUILDERS["SIG_LONGTAIL_001"]()
    script = [
        render_tool_call("campaign.get_metrics", {"campaign_id": "CMP_3072"}),
        render_tool_call("creative.upload", {"campaign_id": "CMP_3072", "creative_name": "hook_b_v1",
                                             "asset_type": "video", "duration_seconds": 45}),
        # 直接跳过 poll_review，猜一个 approved
        *_final({"asset_id": "ASSET_CMP_3072_hook_b_v1", "review_status": "approved"}),
    ]
    output, _ = asyncio.run(_run(bundle, tokenizer, script))
    result = score_trajectory(
        bundle, output.trajectory, output.sandbox,
        policy_scorer=DOMAIN.policy_scorer, decision_fn=DOMAIN.decision_fn, caps=DOMAIN.caps,
    )
    assert result.subscores["outcome"] == pytest.approx(1.0), "值本身是对的，所以 outcome 满分"
    assert "false_claim_cap" in result.cap_steps, "但必须被 false_claim_cap 抓住"
    assert result.reward <= 0.30


# --------------------------------------------------------------------------
# 4. prompt 一致性
# --------------------------------------------------------------------------


def test_prompt_hash_is_stable_and_menu_sensitive(tokenizer):
    """prompt 指纹必须稳定；换了工具菜单必须变。

    老师包里这个 hash 算了、落盘了，但全仓库没有一处做跨阶段比对（T10）。
    """
    graded = SEED_BUILDERS["SIG_GRADED_001"]()
    missing = SEED_BUILDERS["SIG_TOOLMISS_001"]()

    a, _ = asyncio.run(_run(graded, tokenizer, _gold_script(graded)))
    b, _ = asyncio.run(_run(SEED_BUILDERS["SIG_GRADED_001"](), tokenizer, _gold_script(graded)))
    c, _ = asyncio.run(_run(missing, tokenizer, _gold_script(missing)))

    assert a.token_trace["prompt_hash"] == b.token_trace["prompt_hash"]
    from syncopate.core.contract import IS_V15
    if IS_V15:
        # v15：菜单一律全量（contract.effective_tool_menu）⇒ 指纹只随 system+工具块变，两条 case 相同是**对的**
        assert a.token_trace["prompt_hash"] == c.token_trace["prompt_hash"]
    else:
        assert a.token_trace["prompt_hash"] != c.token_trace["prompt_hash"]


def test_prompt_includes_tools_and_required_fields(tokenizer):
    """system 规则 + 工具菜单 + 要求的终答字段，三样都得进 prompt。"""
    bundle = SEED_BUILDERS["SIG_RISK_001"]()
    messages = build_messages(bundle, bundle.case.tool_menu)
    rendered = tokenizer.apply_chat_template(
        messages, tools=DOMAIN.registry.menu(bundle.case.tool_menu),
        add_generation_prompt=True, tokenize=False, **CHAT_TEMPLATE_KWARGS,
    )
    assert "campaign.update_budget" in rendered
    assert "policy.get_budget_rule" in rendered
    from syncopate.core.contract import IS_V15
    if not IS_V15:
        assert "approved_budget" in rendered      # 要求的终答字段（v14）
    else:
        # v15（Chaoyu 08-31 裁定①，守则⑮ #3）：训练也不给字段清单——线上没有 gold，从不列字段
        assert "本次结论需要给出的字段" not in rendered.rsplit("<|im_start|>user", 1)[-1]  # 说明书里提到这几个字不算
    assert "每步只输出一个 tool call" in rendered   # system 规则


# --------------------------------------------------------------------------
# 5. ★ SFT / RL 同分布 —— 靠"只有一条代码路径"来保证
# --------------------------------------------------------------------------


def test_full_render_differs_from_incremental_by_design(tokenizer):
    """★ 记录一个反直觉的事实：整段渲染和增量拼接**天生不相等**。

    Qwen3 模板只给最后一个 assistant 轮加空 `<think>` 块，历史轮不加；
    而增量拼接时每一轮都是"当前最后一轮"。无论 enable_thinking 设什么都对不齐。

    所以我们**从不**用整段渲染造 SFT 数据。这条测试守着这个前提——
    哪天上游模板改了行为、两者真的相等了，它会失败，提醒我们重新评估。
    """
    bundle = SEED_BUILDERS["SIG_GRADED_001"]()
    output, _ = asyncio.run(_run(bundle, tokenizer, _gold_script(bundle)))

    messages = list(build_messages(bundle, bundle.case.tool_menu))
    for action in output.trajectory.actions:
        obs = output.trajectory.observation_for(action.tool_call_id)
        messages.append({"role": "assistant", "content": render_tool_call(action.name, action.arguments)})
        messages.append(observation_message(action.name, obs.data if obs.ok else {"error": obs.error}))
    messages.append({"role": "assistant",
                     "content": render_final_answer(output.trajectory.behavior,
                                                    output.trajectory.final_answer)})
    full = tokenizer.apply_chat_template(
        messages, tools=DOMAIN.registry.menu(bundle.case.tool_menu),
        add_generation_prompt=False, tokenize=True, **CHAT_TEMPLATE_KWARGS,
    )
    incremental = output.prompt_ids + output.response_ids
    assert incremental != full, "上游模板行为变了，重新评估 pipeline/sft_replay.py 的前提"


@pytest.mark.parametrize("case_id", sorted(SEED_BUILDERS))
def test_sft_sample_is_token_identical_to_rl_rollout(case_id, tokenizer):
    """★ SFT 样本和 RL rollout 逐 token 相等——因为走的是同一个循环。

    这是"同分布"的硬保证：不是靠对齐两套渲染逻辑碰运气，
    而是根本只有一套。
    """
    from syncopate.pipeline.sft_replay import build_sft_sample

    bundle = SEED_BUILDERS[case_id]()
    sample = asyncio.run(build_sft_sample(bundle, tokenizer=tokenizer, registry=DOMAIN.registry))

    rl_output, _ = asyncio.run(_run(bundle, tokenizer, _gold_script(bundle)))
    assert sample.input_ids == rl_output.prompt_ids + rl_output.response_ids
    # ★ 09-02（Chaoyu 裁定）：SFT 对**空 think 块**不监督；RL 的 response_mask 仍标全部模型 token。
    #   两者的差**只能**是空块那几段 —— 用生产同一份 _mask_empty_think 算期望值（不另抄一份）。
    from syncopate.pipeline.sft_replay import _mask_empty_think
    expect = [0] * len(rl_output.prompt_ids) + list(rl_output.response_mask)
    _mask_empty_think(tokenizer, sample.input_ids, expect, start=len(rl_output.prompt_ids))
    assert sample.loss_mask == expect


@pytest.mark.parametrize("case_id", sorted(SEED_BUILDERS))
def test_sft_loss_mask_only_covers_model_tokens(case_id, tokenizer):
    """loss 只能加在模型该生成的 token 上：prompt 段和工具返回段都必须是 0。"""
    from syncopate.pipeline.sft_replay import build_sft_sample

    bundle = SEED_BUILDERS[case_id]()
    sample = asyncio.run(build_sft_sample(bundle, tokenizer=tokenizer, registry=DOMAIN.registry))

    assert len(sample.input_ids) == len(sample.loss_mask)
    assert sum(sample.loss_mask[: sample.prompt_length]) == 0, "prompt 段不该被监督"
    assert sample.supervised_tokens > 0

    # 被监督的 token 解出来必须含 gold 的工具调用和终答
    supervised = [t for t, m in zip(sample.input_ids, sample.loss_mask) if m == 1]
    text = tokenizer.decode(supervised)
    for action in bundle.gold.actions:
        assert action["tool"] in text
    # ★ 终答形态按契约分家 —— 这条断言本身就是「被测契约」的一部分。
    #   ⛔ 2026-08-30：原来只写死了 v14 的 ```json，于是**整个训练侧测试只验过 v14**；
    #     真拿 v15 去训练时，守着 SFT 数据形态的判据一条都不在（「测试全绿但那条路
    #     从没被真的走过」同族）。
    from syncopate.core.contract import IS_V15
    if IS_V15:
        # v15：机器字段走 session.report；行为走信令或纯文本终答
        has_report = '"name": "session.report"' in text
        has_signal = any(f'"name": "session.{k}"' in text
                         for k in ("defer", "clarify", "reject"))
        assert has_report or has_signal or text.rstrip().endswith("<|im_end|>"), \
            f"v15 监督段既没有 report、也没有信令、也没有纯文本终答：{text[-300:]}"
        assert "```json" not in text, f"v15 监督段出现壳残留：{text[-300:]}"
    else:
        assert "```json" in text
    # 工具返回的内容不该出现在被监督的 token 里
    assert "tool_response" not in text


def test_sft_target_behavior_matches_the_spec(tokenizer):
    """★ SFT 的监督目标里，behavior 必须等于 verifier 期望的行为。

    这条守的是一个真实踩过的坑：`gold_script` 曾经把 behavior 硬编码成
    "tool_call"，于是 clarify / reject 的监督目标是错的。
    症状极具迷惑性——分组 val_loss 降到 0.0000（完美学会了错误标签），
    但生成时 behavior 恒为 tool_call。

    **loss 降到 0 只说明学到了标签，不说明标签是对的。**
    """
    import json as _json

    from syncopate.authoring.axes import params_for
    from syncopate.authoring.templates import TEMPLATES
    from syncopate.pipeline.sft_replay import build_sft_sample

    for name in ("clarify", "reject", "budget_change"):
        bundle = TEMPLATES[name](params_for(0))
        sample = asyncio.run(build_sft_sample(bundle, tokenizer=tokenizer,
                                              registry=DOMAIN.registry))
        supervised = [t for t, m in zip(sample.input_ids, sample.loss_mask) if m == 1]
        text = tokenizer.decode(supervised)
        from syncopate.core.contract import IS_V15
        if IS_V15:
            # v15 没有 behavior 字段了 —— 行为**由形态推导**（`25 §3.1`）。
            # 这条测试的价值不变：监督目标教的行为必须等于 spec 期望的那个。
            expected = bundle.verifier.expected_behavior
            if expected in ("defer", "clarify", "reject"):
                # 线格式无关（JSON `"name": "session.x"` 或 Qwen3.5 XML `<function=session.x>` 都含工具名）
                assert f"session.{expected}" in text, (
                    f"{name} 的 v15 监督目标里没有 session.{expected}：{text[-300:]}")
            else:
                assert not any(f"session.{k}" in text
                               for k in ("defer", "clarify", "reject")), (
                    f"{name} 期望 {expected} 却调了终止信令：{text[-300:]}")
        else:
            block = text[text.rindex("```json") + 7: text.rindex("```")]
            assert _json.loads(block)["behavior"] == bundle.verifier.expected_behavior, (
                f"{name} 的监督目标 behavior 和 spec 不一致：{block}")


def test_stops_before_exhausting_response_budget(tokenizer):
    """★ response 预算耗尽必须**主动收工**，不能再发一次生成请求。

    2026-08-13 实测：预算用满时送进 vLLM 的上下文正好等于 max_model_len，
    引擎抛 `leaves no room to generate` 并**杀掉整个训练任务**。
    v11 的长轨迹（GEO 14 步）第一次把预算真吃满，才暴露出来。
    """
    from syncopate.train.rollout_loop import MIN_GENERATION_HEADROOM

    bundle = SEED_BUILDERS["SIG_HIGH_001"]()
    # 每步都吐一个不给终答的工具调用，快速烧干预算
    script = [render_tool_call("campaign.get_metrics", {"campaign_id": "CMP_1024"})] * 30
    budget = 512
    output, engine = asyncio.run(_run(
        bundle, tokenizer, script,
        RolloutConfig(max_assistant_turns=30, max_prompt_length=4096,
                      max_response_length=budget),
    ))

    assert len(output.response_ids) <= budget
    assert output.trajectory.truncated is True
    # 关键断言：每次发出的请求，剩余预算都够真的生成点东西
    prompt_len = len(output.prompt_ids)
    for context_len in engine.prompt_lengths:
        assert context_len - prompt_len <= budget - MIN_GENERATION_HEADROOM


# ── 09-02（26 §W2①）：多轮行的 prior 进 prompt 后，SFT 与 RL 仍逐 token 同构 ────────────
def test_sft_sample_with_prior_is_token_identical_to_rl_rollout(tokenizer):
    """历史消息对由 build_messages 一处渲染，SFT（gold 回放）与 RL（真循环）走同一条路径。"""
    from syncopate.pipeline.sft_replay import build_sft_sample
    bundle = SEED_BUILDERS["FRESH_0001"]() if "FRESH_0001" in SEED_BUILDERS else \
        next(iter(SEED_BUILDERS.values()))()
    bundle.prior = [{"user_message": "CMP_1 最近消耗多少", "result": {"text": "CMP_1 近 7 天消耗 31500。"}},
                    {"user_message": "能扩量吗", "result": {"text": "", "signal": "defer",
                                                            "arguments": {"reason": "数据还没收敛。"}}}]
    config = RolloutConfig(max_assistant_turns=assistant_turn_budget(bundle.case.max_steps))
    sample = asyncio.run(build_sft_sample(bundle, tokenizer=tokenizer, registry=DOMAIN.registry, config=config))
    engine = ScriptedEngine(tokenizer, _gold_script(bundle))
    out = asyncio.run(run_rollout(bundle, registry=DOMAIN.registry, tokenizer=tokenizer,
                                  generate=engine, config=config, rollout_id="r", run_id="t"))
    assert sample.input_ids == out.prompt_ids + out.response_ids
    prompt_text = tokenizer.decode(out.prompt_ids)
    assert "CMP_1 近 7 天消耗 31500。" in prompt_text and "数据还没收敛。" in prompt_text
    assert "[上一轮]" not in prompt_text, "历史不许再折成题面文本"
    assert prompt_text.index("数据还没收敛。") < prompt_text.index(bundle.case.user_message), "历史必须在本轮 user 之前"


# ── 09-02（Chaoyu 裁定）：空 think 块**不监督**，非空 think 照常监督 ────────────────────
def test_empty_think_blocks_are_masked_but_nonempty_think_is_supervised(tokenizer):
    from syncopate.core.contract import IS_V15
    if not IS_V15:
        pytest.skip("v15 契约专有")
    from syncopate.pipeline.sft_replay import EMPTY_THINK, EMPTY_THINK_RESP, build_sft_sample, think_opener_in_prompt
    bundle = next(iter(SEED_BUILDERS.values()))()
    n_turns = len(bundle.gold.actions) + 2
    config = RolloutConfig(max_assistant_turns=assistant_turn_budget(bundle.case.max_steps))
    # 第 0 步给教师思考，其余轮为空块
    s = asyncio.run(build_sft_sample(bundle, tokenizer=tokenizer, registry=DOMAIN.registry, config=config,
                                     thinking={0: "先查指标再判断成熟度。"}))
    resp = s.input_ids[s.prompt_length:]
    resp_mask = s.loss_mask[s.prompt_length:]
    opener = think_opener_in_prompt(tokenizer)
    pat = tokenizer.encode(EMPTY_THINK_RESP if opener else EMPTY_THINK, add_special_tokens=False)
    empties = [i for i in range(len(resp) - len(pat) + 1)
               if resp[i:i + len(pat)] == pat and ((not opener) or i == 0 or resp_mask[i - 1] == 0)]
    assert empties, "样本里应有空 think 块（简单轮）"
    for i in empties:
        assert not any(resp_mask[i:i + len(pat)]), "空 think 块不许有梯度"
    # ★ 09-04 run22 出厂体检抓到的双开头：模板写了 "<think>\n" 后 attach_think 又写一次
    full = tokenizer.decode(s.input_ids[:s.total_length])
    assert "<think>\n<think>" not in full, "think 开头写了两次（模板 + attach_think）"
    assert full.count("<think>") == full.count("</think>"), "think 开/闭标签数不等"
    supervised = tokenizer.decode([t for t, m in zip(resp, resp_mask) if m == 1])
    assert "先查指标再判断成熟度" in supervised, "非空 think 必须仍被监督"
    assert "<think>\n\n</think>" not in supervised, "空 think 块不该出现在监督段里"
    assert sum(resp_mask) > 0
