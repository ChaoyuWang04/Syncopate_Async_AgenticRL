"""一步多调用：**只执行第一个**，其余当协议错误退回；cap 从最狠一档退回轻罚。

★ 为什么（实测 5280 条训练 rollout，`docs/syncopate/01 §P0-2`）

18.8% 的 rollout 一步发了多个 tool call，而 `system.txt` 第 8 行明确禁止。
**而截尾采样（评测口径 top_p 0.95 / top_k 20）下这个数是 0%**
⇒ 它是**采样尾巴**，不是模型不懂规矩。

它此前被罚成 `ceiling=0.0`（全项目最狠，和提示词注入同级），后果：
    那 990 条的 reward 全部恰好 0
    **29% 的组内方差**来自它，其中 45 个组剔掉它之后方差归零
⇒ GRPO 的梯度完全来自组内方差 ⇒ 近三成梯度在教"别进采样尾巴"，与任务无关。

⚠️ 两处改动是**一对**：拦在发生点 ⇒ 真正的危害消失 ⇒ cap 才有理由降档。
   只改一处都不对：只降 cap 会留着"没看到 observation 就动手"的危险；
   只拦不降 cap 会让格式噪声继续吃掉三成梯度。
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from syncopate.authoring.seed_cases import SEED_BUILDERS
from syncopate.core.parsing import render_tool_call
from syncopate.domains.adcampaign import build_domain, rules
from syncopate.train.rollout_loop import RolloutConfig, run_rollout
from syncopate.core.model_paths import TEST_TOKENIZER, STUDENT_MODEL, TEACHER_MODEL

MODEL_DIR = Path(TEST_TOKENIZER)
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


def _two_calls(bundle) -> str:
    """把 gold 的前两个动作塞进**同一步** —— 这正是那 18.8% 的形状。"""
    a, b = bundle.gold.actions[0], bundle.gold.actions[1]
    return (render_tool_call(a["tool"], a.get("arguments", {})) + "\n"
            + render_tool_call(b["tool"], b.get("arguments", {})))


def _run(bundle, tokenizer, script):
    engine = _Scripted(tokenizer, script)
    return asyncio.run(run_rollout(
        bundle, registry=DOMAIN.registry, tokenizer=tokenizer, generate=engine,
        config=RolloutConfig(max_assistant_turns=4),
    ))


# --------------------------------------------------------------------------
# 行为：非 parallel 的 case，第二个调用不被执行
# --------------------------------------------------------------------------

def test_second_call_is_not_executed(tokenizer):
    bundle = SEED_BUILDERS["SIG_GRADED_001"]()
    assert bundle.case.metadata.topology != "parallel"
    out = _run(bundle, tokenizer, [_two_calls(bundle)])

    obs = out.trajectory.observations
    assert len(obs) == 2, "两个调用都要留痕，否则 cap 看不见"
    assert obs[0].ok, "第一个必须真的执行"
    assert not obs[1].ok, "第二个必须被拦下"
    assert "protocol_violation" in (obs[1].error or "")
    assert "未执行" in (obs[1].error or ""), "错误必须说清楚它没被执行 —— 别静默丢弃"


def test_cap_still_fires_after_rejection(tokenizer):
    """⚠️ 拦下来之后 cap 仍要命中，否则就是"机制在但没接上"（本项目第一失效形状）。"""
    bundle = SEED_BUILDERS["SIG_GRADED_001"]()
    out = _run(bundle, tokenizer, [_two_calls(bundle)])
    assert out.trajectory.multi_tool_steps() == [1]
    assert rules.multi_tool_per_step(bundle, out.trajectory, None) is not None


def test_parallel_topology_is_exempt(tokenizer):
    """topology == parallel 的 case 本来就要求同一步发多个 —— 不能误伤（GEO 有 90 条）。"""
    bundle = SEED_BUILDERS["SIG_GRADED_001"]()
    bundle.case.metadata.topology = "parallel"
    out = _run(bundle, tokenizer, [_two_calls(bundle)])
    obs = out.trajectory.observations
    assert len(obs) == 2 and obs[0].ok and obs[1].ok, "并行 case 两个都该执行"
    assert rules.multi_tool_per_step(bundle, out.trajectory, None) is None


def test_single_call_path_is_untouched(tokenizer):
    """回归：正常的一步一个调用，行为一个字都不能变。"""
    bundle = SEED_BUILDERS["SIG_GRADED_001"]()
    a = bundle.gold.actions[0]
    out = _run(bundle, tokenizer, [render_tool_call(a["tool"], a.get("arguments", {}))])
    assert len(out.trajectory.observations) == 1
    assert out.trajectory.observations[0].ok
    assert out.trajectory.multi_tool_steps() == []


# --------------------------------------------------------------------------
# 刻度：协议问题不该比"越权花钱"罚得更狠
# --------------------------------------------------------------------------

def test_cap_is_no_longer_the_harshest_tier():
    """★ 方向判据。改之前 multi_tool=0.0 而 unauthorized_write=0.30 —— 狠一个量级。"""
    assert rules.MULTI_TOOL_CEILING > rules.UNAUTHORIZED_WRITE_CEILING
    assert rules.MULTI_TOOL_CEILING > 0.0
