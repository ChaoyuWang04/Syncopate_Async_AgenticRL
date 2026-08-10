"""骨架冒烟测试：四件套 → 工具执行 → 沙盒入账 → 判分，整条链走通。

这里用一个极简的假场景（不是真的广告域实现），只为证明引擎本身是通的。
真实的广告工具在 syncopate/domains/adcampaign/ 下。
"""

from __future__ import annotations

import asyncio

import pytest

from syncopate.core.sandbox import Sandbox
from syncopate.core.schemas import (
    AnswerField,
    Case,
    CaseBundle,
    CaseMetadata,
    EnvSnapshot,
    SideEffectReq,
    VerifierSpec,
)
from syncopate.core.tool_registry import ToolContext, ToolRegistry, ToolResult
from syncopate.core.trajectory import Action, Observation, Trajectory
from syncopate.core.verifier_engine import CapHit, CapRegistry, score_trajectory


# --------------------------------------------------------------------------
# 假场景：查 campaign 指标 -> 改预算
# --------------------------------------------------------------------------


@pytest.fixture
def registry() -> ToolRegistry:
    reg = ToolRegistry()

    @reg.tool(
        name="campaign.get_metrics",
        description="查询 campaign 投放指标",
        parameters={"type": "object", "properties": {"campaign_id": {"type": "string"}}, "required": ["campaign_id"]},
        kind="read",
    )
    def get_metrics(args, ctx: ToolContext):
        row = ctx.env.row("campaigns", args.get("campaign_id"))
        if row is None:
            return ToolResult(ok=False, error="campaign_not_found")
        return ToolResult(ok=True, data=row)

    @reg.tool(
        name="campaign.update_budget",
        description="调整 campaign 日预算",
        parameters={
            "type": "object",
            "properties": {"campaign_id": {"type": "string"}, "new_budget": {"type": "number"}},
            "required": ["campaign_id", "new_budget"],
        },
        kind="write",
        fact_key="budget_updated",
        latency_seconds=0.0,
    )
    def update_budget(args, ctx: ToolContext):
        return ToolResult(ok=True, data={"campaign_id": args["campaign_id"], "budget": args["new_budget"]})

    return reg


@pytest.fixture
def bundle() -> CaseBundle:
    case = Case(
        case_id="SMOKE_001",
        user_message="把 CMP_1024 的日预算提到 800",
        context={"campaign_id": "CMP_1024"},
        entities={"campaign_id": "CMP_1024"},
        metadata=CaseMetadata(signal_class="graded", bucket="critical_args", topology="sequential"),
        max_steps=6,
    )
    env = EnvSnapshot(
        case_id="SMOKE_001",
        readonly_tables={"campaigns": {"CMP_1024": {"campaign_id": "CMP_1024", "cpi": 2.35, "daily_budget": 500}}},
    )
    spec = VerifierSpec(
        expected_behavior="tool_call",
        required_read_tools=["campaign.get_metrics"],
        allowed_write_tools=["campaign.update_budget"],
        required_side_effects=[SideEffectReq(tool="campaign.update_budget", required_args={"new_budget": 800})],
        required_answer_fields=[
            AnswerField(key="old_budget", value_source="campaigns.daily_budget"),
            AnswerField(key="new_budget", value_source="literal:800"),
        ],
        # 明确关掉全部 cap：这些测试验的是引擎本身，
        # 不该被「碰巧 import 了哪个域」影响（空列表 ≠ None，见 schemas.VerifierSpec）
        active_caps=[],
    )
    return CaseBundle(case=case, env=env, verifier=spec)


async def _run(registry: ToolRegistry, bundle: CaseBundle, plan: list[tuple[str, dict]]):
    """按给定计划跑一条轨迹，返回 (trajectory, sandbox)。"""
    sandbox = Sandbox(bundle.env, namespace_id=f"test:{bundle.case_id}:r0")
    traj = Trajectory(case_id=bundle.case_id, rollout_id="r0", namespace_id=sandbox.namespace_id)
    for index, (tool, args) in enumerate(plan, start=1):
        tool_call_id = f"tc_{index}"
        ctx = ToolContext(case=bundle.case, env=bundle.env, sandbox=sandbox, step=index, tool_call_id=tool_call_id)
        result = await registry.execute(tool, args, ctx)
        traj.actions.append(Action(step=index, tool_call_id=tool_call_id, name=tool, arguments=args))
        traj.observations.append(
            Observation(tool_call_id=tool_call_id, tool=tool, ok=result.ok, data=result.data, error=result.error)
        )
    return traj, sandbox


# --------------------------------------------------------------------------
# 测试
# --------------------------------------------------------------------------


def test_perfect_run_scores_high(registry, bundle):
    """走对路 + 终答字段全对 -> 接近满分。"""
    traj, sandbox = asyncio.run(
        _run(registry, bundle, [
            ("campaign.get_metrics", {"campaign_id": "CMP_1024"}),
            ("campaign.update_budget", {"campaign_id": "CMP_1024", "new_budget": 800}),
        ])
    )
    traj.final_answer = {"old_budget": 500, "new_budget": 800}

    score = score_trajectory(bundle, traj, sandbox)
    assert score.reward == pytest.approx(1.0)
    assert sandbox.facts() == {"budget_updated"}
    # 写动作被记了步号——这是步级信用分配的地基
    assert sandbox.records_for("campaign.update_budget")[0].step == 2


def test_missing_read_tool_loses_evidence(registry, bundle):
    """跳过调查直接动手 -> evidence 子分归零。"""
    traj, sandbox = asyncio.run(
        _run(registry, bundle, [("campaign.update_budget", {"campaign_id": "CMP_1024", "new_budget": 800})])
    )
    traj.final_answer = {"old_budget": 500, "new_budget": 800}

    score = score_trajectory(bundle, traj, sandbox)
    assert score.subscores["evidence"] == 0.0
    assert score.reward == pytest.approx(0.8)   # 丢掉 evidence 的 0.20


def test_empty_answer_cannot_hack_reward(registry, bundle):
    """★ 防 reward hacking：终答空着 / 字段值错，answer 分必须掉下来。

    老师那套 heuristic 兜底是 `covered = bool(final_text.strip())`，
    随便说句话就算全覆盖。我们这里必须字段在**且**值对。
    """
    traj, sandbox = asyncio.run(
        _run(registry, bundle, [
            ("campaign.get_metrics", {"campaign_id": "CMP_1024"}),
            ("campaign.update_budget", {"campaign_id": "CMP_1024", "new_budget": 800}),
        ])
    )
    traj.final_answer = {"blah": "已为你调整完毕"}   # 说了话，但没给要求的字段

    score = score_trajectory(bundle, traj, sandbox)
    assert score.subscores["outcome"] == pytest.approx(0.75)   # write 满分，answer 零分
    assert score.reward < 1.0


def test_wrong_arg_fails_side_effect(registry, bundle):
    """工具调对了但关键参数错 -> 写动作不算完成。"""
    traj, sandbox = asyncio.run(
        _run(registry, bundle, [
            ("campaign.get_metrics", {"campaign_id": "CMP_1024"}),
            ("campaign.update_budget", {"campaign_id": "CMP_1024", "new_budget": 999}),
        ])
    )
    traj.final_answer = {"old_budget": 500, "new_budget": 800}

    score = score_trajectory(bundle, traj, sandbox)
    assert score.details["outcome"]["write"][0]["args_ok"] is False
    assert score.subscores["outcome"] == pytest.approx(0.25)   # write=0, answer=1


def test_behavior_mismatch_is_zero(registry, bundle):
    """该执行却去 clarify -> 直接零分。"""
    traj, sandbox = asyncio.run(_run(registry, bundle, []))
    traj.behavior = "clarify"

    score = score_trajectory(bundle, traj, sandbox)
    assert score.reward == 0.0
    assert score.cap_hits[0].name == "behavior_mismatch"


def test_cap_carries_step_numbers(registry, bundle):
    """★ cap 必须带责任步号，不能只返回一个布尔值。"""
    bundle.verifier.active_caps = ["duplicate_write_cap"]
    caps = CapRegistry()

    @caps.rule(name="duplicate_write_cap", ceiling=0.30)
    def _detect(bundle_, traj_, sandbox_):
        dupes = sandbox_.duplicate_writes()
        if not dupes:
            return None
        steps = sorted(step for steps in dupes.values() for step in steps)
        return CapHit("", 0.0, f"重复写: {sorted(dupes)}", steps)

    traj, sandbox = asyncio.run(
        _run(registry, bundle, [
            ("campaign.get_metrics", {"campaign_id": "CMP_1024"}),
            ("campaign.update_budget", {"campaign_id": "CMP_1024", "new_budget": 800}),
            ("campaign.update_budget", {"campaign_id": "CMP_1024", "new_budget": 800}),
        ])
    )
    traj.final_answer = {"old_budget": 500, "new_budget": 800}

    score = score_trajectory(bundle, traj, sandbox, caps=caps)
    assert score.reward <= 0.30
    assert score.cap_steps["duplicate_write_cap"] == [2, 3]   # 知道是第 2、3 步重复了


def test_multi_tool_step_is_detectable(bundle):
    """同一步发多个工具 -> 步号拿得到（老师那套把它丢了）。"""
    traj = Trajectory(case_id="X", rollout_id="r0", namespace_id="n")
    traj.actions = [
        Action(step=1, tool_call_id="tc_1", name="a"),
        Action(step=2, tool_call_id="tc_2_0", name="b"),
        Action(step=2, tool_call_id="tc_2_1", name="c"),
    ]
    assert traj.multi_tool_steps() == [2]


def test_real_latency_is_actually_slow():
    """★ 延迟必须是真的——假的慢暴露不了阻塞问题，异步对照就白做了。"""
    reg = ToolRegistry()

    @reg.tool(name="slow.poll", description="轮询审核状态", parameters={"type": "object", "properties": {}},
              kind="read", latency_seconds=0.20)
    def poll(args, ctx):
        return ToolResult(ok=True, data={"review_status": "approved"})

    async def timed():
        loop = asyncio.get_running_loop()
        env = EnvSnapshot(case_id="X")
        case = Case(case_id="X", user_message="")
        ctx = ToolContext(case=case, env=env, sandbox=Sandbox(env, "n"), step=1, tool_call_id="tc_1")
        start = loop.time()
        await reg.execute("slow.poll", {}, ctx)
        return loop.time() - start

    assert asyncio.run(timed()) >= 0.19

    # latency_scale 让调试时能把 480 秒压成秒级
    reg.latency_scale = 0.05

    async def timed_scaled():
        loop = asyncio.get_running_loop()
        env = EnvSnapshot(case_id="X")
        ctx = ToolContext(case=Case(case_id="X", user_message=""), env=env,
                          sandbox=Sandbox(env, "n"), step=1, tool_call_id="tc_1")
        start = loop.time()
        await reg.execute("slow.poll", {}, ctx)
        return loop.time() - start

    assert asyncio.run(timed_scaled()) < 0.05


def test_bundle_roundtrip(tmp_path, bundle):
    """四件套写盘再读回来，内容不丢。"""
    bundle.write(tmp_path)
    restored = CaseBundle.read(tmp_path, bundle.case_id)
    assert restored.case.user_message == bundle.case.user_message
    assert restored.case.metadata.signal_class == "graded"
    assert restored.verifier.required_side_effects[0].tool == "campaign.update_budget"
    assert restored.verifier.required_answer_fields[0].key == "old_budget"
