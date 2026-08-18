"""截断的**原因**必须分得开：tokens / observation / turns。

★ 为什么（2026-08-18）

`truncated` 有四个出口、三种成因，而**修法方向完全不同甚至相反**：

    tokens       模型自己把 token 预算写满了        ⇒ 加 token 预算
    observation  工具返回塞不进剩余预算（不是模型的锅）⇒ 截断 observation
    turns        轮数用完还没给终答                  ⇒ 缩链路 / 加轮数 / 查为什么打转

它们此前共用一个布尔值 ⇒ **数据里根本不存在这个区分**。
后果不是"少一个字段"：2026-08-18 就因此按错的假设回溯分类
（用 `num_steps >= 8`，而真实上限逐 case 是 **4–14**），得出过一个整个反了的结论。

★★ 而且 `max_steps_cap` 判的是 `truncated`、报的却是「撞上 max_steps=N 被截断」
—— 一条 token 用完的轨迹会被报成撞步数上限，**那个数字是编的**。
本文件同时钉住修好之后的判据。
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from syncopate.authoring.seed_cases import SEED_BUILDERS
from syncopate.core.parsing import render_tool_call
from syncopate.domains.adcampaign import build_domain, rules
from syncopate.train.rollout_loop import RolloutConfig, run_rollout

MODEL_DIR = Path("models/Qwen3-0.6B")
DOMAIN = build_domain()

pytestmark = pytest.mark.skipif(not MODEL_DIR.exists(), reason="需要 models/Qwen3-0.6B")


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


class _Scripted:
    def __init__(self, tokenizer, script):
        self.tokenizer, self.script = tokenizer, list(script)

    async def __call__(self, prompt_ids, sampling_params):
        if not self.script:
            return []
        return self.tokenizer.encode(self.script.pop(0), add_special_tokens=False)


def _run(bundle, tokenizer, script, **cfg):
    engine = _Scripted(tokenizer, script)
    return asyncio.run(run_rollout(
        bundle, registry=DOMAIN.registry, tokenizer=tokenizer, generate=engine,
        config=RolloutConfig(**cfg),
    ))


def _calls(bundle, n):
    out = []
    for a in bundle.gold.actions[:n]:
        out.append(render_tool_call(a["tool"], a.get("arguments", {})))
    return out


# --------------------------------------------------------------------------

def test_turns_exhausted_is_tagged_turns(tokenizer):
    """轮数用完还没给终答 ⇒ "turns"（**只有这一种**才是真的撞步数上限）。"""
    bundle = SEED_BUILDERS["SIG_GRADED_001"]()
    out = _run(bundle, tokenizer, _calls(bundle, 3),
               max_assistant_turns=2, max_prompt_length=5120, max_response_length=2048)
    assert out.trajectory.truncated
    assert out.trajectory.truncation_reason == "turns"


def test_token_budget_exhausted_is_tagged_tokens(tokenizer):
    """token 预算用完 ⇒ "tokens"，**不是** "turns"。

    ⚠️ 这正是此前被 `max_steps_cap` 误报成「撞上 max_steps=N」的那一类。
    """
    bundle = SEED_BUILDERS["SIG_GRADED_001"]()
    out = _run(bundle, tokenizer, _calls(bundle, 3),
               max_assistant_turns=8, max_prompt_length=5120, max_response_length=120)
    assert out.trajectory.truncated
    assert out.trajectory.truncation_reason in ("tokens", "observation"), \
        out.trajectory.truncation_reason
    assert out.trajectory.truncation_reason != "turns", "预算用完不能被记成撞轮数上限"


def test_flag_and_reason_never_disagree(tokenizer):
    """`truncated` 必须恒等于 `truncation_reason is not None`。

    ⚠️ 守的是"将来有人只改一处" —— 两个字段各说各话比没有字段更难查。
    """
    bundle = SEED_BUILDERS["SIG_GRADED_001"]()
    for cfg in [dict(max_assistant_turns=2, max_response_length=2048),
                dict(max_assistant_turns=8, max_response_length=120),
                dict(max_assistant_turns=8, max_response_length=2048)]:
        out = _run(bundle, tokenizer, _calls(bundle, 3) + ["done"],
                   max_prompt_length=5120, **cfg)
        tr = out.trajectory
        assert tr.truncated == (tr.truncation_reason is not None), (cfg, tr.truncation_reason)


def test_max_steps_cap_only_fires_on_turns():
    """★ 判据收窄之后：token/observation 用完**不再**触发 max_steps_cap。"""
    from syncopate.core.trajectory import Trajectory

    bundle = SEED_BUILDERS["SIG_GRADED_001"]()
    for reason, should_fire in [("turns", True), ("tokens", False),
                                ("observation", False), (None, False)]:
        tr = Trajectory(case_id="c", rollout_id="r", namespace_id="n")
        tr.truncated = reason is not None
        tr.truncation_reason = reason
        hit = rules.max_steps_hit(bundle, tr, None)
        assert (hit is not None) is should_fire, (reason, hit)


def test_cap_message_no_longer_invents_a_number():
    """报错必须说的是它真的判了的那件事。

    旧版：判 `truncated`，报「撞上 max_steps=N 被截断」—— N 是编的。
    """
    from syncopate.core.trajectory import Trajectory

    bundle = SEED_BUILDERS["SIG_GRADED_001"]()
    tr = Trajectory(case_id="c", rollout_id="r", namespace_id="n")
    tr.truncated, tr.truncation_reason = True, "turns"
    hit = rules.max_steps_hit(bundle, tr, None)
    assert "轮数上限" in hit.reason


def test_reason_reaches_the_dump(tokenizer):
    """字段必须真的落进 dump —— 否则又是"机制在但没接上"。"""
    bundle = SEED_BUILDERS["SIG_GRADED_001"]()
    out = _run(bundle, tokenizer, _calls(bundle, 3),
               max_assistant_turns=2, max_prompt_length=5120, max_response_length=2048)
    assert out.metrics["truncation_reason"] == "turns"
