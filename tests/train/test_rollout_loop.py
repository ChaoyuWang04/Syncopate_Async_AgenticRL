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
from syncopate.train.rollout_loop import (
    CHAT_TEMPLATE_KWARGS, RolloutConfig, build_messages, observation_message, run_rollout,
)

MODEL_DIR = Path("models/Qwen3-0.6B")
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
    """把 gold 轨迹翻译成「模型该输出什么」的剧本。

    这同时也是构造 SFT 训练数据的方式——同一个函数两用，保证
    SFT 教的格式和 RL 解析的格式绝对一致。
    """
    steps = [render_tool_call(a["tool"], a.get("arguments", {})) for a in bundle.gold.actions]
    steps.append(render_final_answer(bundle.verifier.expected_behavior, bundle.gold.final_answer))
    return steps


async def _run(bundle, tokenizer, script, config=None):
    engine = ScriptedEngine(tokenizer, script)
    output = await run_rollout(
        bundle, registry=DOMAIN.registry, tokenizer=tokenizer, generate=engine,
        config=config or RolloutConfig(max_assistant_turns=8),
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
    script = [
        "我觉得 CPI 大概是 2.1 左右吧",                       # 既不是 tool_call 也不是 json
        render_tool_call("campaign.get_metrics", {"campaign_id": "CMP_1024"}),
        render_final_answer("tool_call", {"cpi": 2.10}),
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
        "<think>拿到了，CPI 是 2.10</think>" + render_final_answer("tool_call", {"cpi": 2.10}),
    ]
    output, _ = asyncio.run(_run(bundle, tokenizer, script))
    assert output.trajectory.final_answer == {"cpi": 2.10}
    assert output.trajectory.actions[0].name == "campaign.get_metrics"


def test_tool_outside_menu_is_refused(tokenizer):
    """★ tool_missing case：菜单外的工具即使注册表里有，也必须拒绝执行。

    否则模型可以靠调用「隐藏工具」绕过能力缺口，这条 case 就白设计了。
    """
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
        render_final_answer("tool_call", {"asset_id": "ASSET_CMP_3072_hook_b_v1",
                                          "review_status": "approved"}),
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
    assert "approved_budget" in rendered          # 要求的终答字段
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
    assert sample.loss_mask == [0] * len(rl_output.prompt_ids) + rl_output.response_mask


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
        block = text[text.rindex("```json") + 7: text.rindex("```")]
        assert _json.loads(block)["behavior"] == bundle.verifier.expected_behavior, (
            f"{name} 的监督目标 behavior 和 spec 不一致：{block}")
