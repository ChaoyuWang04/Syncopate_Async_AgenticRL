"""沙盒保真度：写完之后读得到，但世界本身不变。

★ 这条是实测出来的缺口（历史见 docs/archive/syncopate/pre-consolidation-v16/07-toolbox-and-runtime-design.md §3 P0-1）：
  改预算 500 → 900 之后再查 campaign.get_metrics，读到的还是 500。
  因为 env 只读、写只进账本，而读工具**根本不看账本**。

  真实平台改完再读会读到新值（Meta 明确支持 read-after-write）。
  读不到的后果是模型学不会「改完要确认」，还可能因为一直读到旧值而反复改同一个对象。
"""

from __future__ import annotations

import asyncio

import pytest

from syncopate.core.runner import PlannedCall, run_plan
from syncopate.core.schemas import Case, CaseBundle, CaseMetadata, VerifierSpec
from syncopate.domains.adcampaign import build_domain
from syncopate.domains.adcampaign.world import WorldBuilder

DOMAIN = build_domain()


@pytest.fixture
def bundle():
    env = (WorldBuilder("FID_0001")
           .account("ACC_11", tier="standard")
           .campaign("CMP_1", account_id="ACC_11", daily_budget=500.0)
           .campaign("CMP_2", account_id="ACC_11", daily_budget=300.0)
           .build())
    return CaseBundle(
        case=Case(case_id="FID_0001", user_message="x",
                  entities={"campaign_id": "CMP_1", "account_id": "ACC_11"},
                  metadata=CaseMetadata("high_risk", "critical_args"), max_steps=8),
        env=env,
        verifier=VerifierSpec(allowed_write_tools=["campaign.update_budget"], max_steps=8),
    )


def run(bundle, calls):
    DOMAIN.registry.latency_scale = 0.0
    return asyncio.run(run_plan(bundle, DOMAIN.registry, calls, final_answer={"summary": "x"}))


def test_write_then_read_sees_the_new_value(bundle):
    trajectory, _ = run(bundle, [
        PlannedCall("campaign.update_budget", {"campaign_id": "CMP_1", "new_budget": 900.0, "client_request_id": "tk1"}),
        PlannedCall("campaign.get_metrics", {"campaign_id": "CMP_1"}),
    ])
    assert trajectory.observations[-1].data["daily_budget"] == 900.0


def test_list_view_also_sees_it(bundle):
    """整张表的视图也要叠加——只修 row() 不修 table()，list 出来的还是旧值。"""
    trajectory, _ = run(bundle, [
        PlannedCall("campaign.update_budget", {"campaign_id": "CMP_1", "new_budget": 900.0, "client_request_id": "tk2"}),
        PlannedCall("campaign.list", {"account_id": "ACC_11"}),
    ])
    rows = {r["campaign_id"]: r for r in trajectory.observations[-1].data["campaigns"]}
    assert rows["CMP_1"]["daily_budget"] == 900.0
    assert rows["CMP_2"]["daily_budget"] == 300.0      # 没动过的不受影响


def test_world_itself_is_never_mutated(bundle):
    """★ 叠加不等于可变。EnvSnapshot 必须原样不动 ——

    同一条 case 会被并发跑 N 条 rollout（GRPO 的组），env 是它们**共享**的。
    一旦某条 rollout 真的改了 env，其它 rollout 就被污染了，而且不会报错。
    """
    run(bundle, [PlannedCall("campaign.update_budget", {"campaign_id": "CMP_1", "new_budget": 900.0, "client_request_id": "tk3"})])
    assert bundle.env.table("campaigns")["CMP_1"]["daily_budget"] == 500.0


def test_two_rollouts_do_not_leak_into_each_other(bundle):
    """两条 rollout 各改各的，互相看不见对方的改动。"""
    DOMAIN.registry.latency_scale = 0.0
    a, _ = asyncio.run(run_plan(bundle, DOMAIN.registry, [
        PlannedCall("campaign.update_budget", {"campaign_id": "CMP_1", "new_budget": 900.0, "client_request_id": "tk4"}),
        PlannedCall("campaign.get_metrics", {"campaign_id": "CMP_1"})],
        final_answer={}, rollout_id="r1"))
    b, _ = asyncio.run(run_plan(bundle, DOMAIN.registry, [
        PlannedCall("campaign.get_metrics", {"campaign_id": "CMP_1"})],
        final_answer={}, rollout_id="r2"))
    assert a.observations[-1].data["daily_budget"] == 900.0
    assert b.observations[-1].data["daily_budget"] == 500.0      # r1 的改动没串台


def test_last_write_wins_in_the_view(bundle):
    """写两次，视图给最后一次；但审计日志两条都在（归因要用）。"""
    trajectory, sandbox = run(bundle, [
        PlannedCall("campaign.update_budget", {"campaign_id": "CMP_1", "new_budget": 700.0, "client_request_id": "tk5"}),
        PlannedCall("campaign.update_budget", {"campaign_id": "CMP_1", "new_budget": 900.0, "client_request_id": "tk6"}),
        PlannedCall("campaign.get_metrics", {"campaign_id": "CMP_1"}),
    ])
    assert trajectory.observations[-1].data["daily_budget"] == 900.0
    assert len(sandbox.records_for("campaign.update_budget")) == 2
    assert "campaign.update_budget" in sandbox.duplicate_writes()


def test_failed_write_does_not_enter_the_view(bundle):
    """写失败了就不该出现在视图里——否则模型会以为它改成功了。"""
    trajectory, _ = run(bundle, [
        PlannedCall("campaign.update_budget", {"campaign_id": "CMP_1", "new_budget": -1, "client_request_id": "tk7"}),
        PlannedCall("campaign.get_metrics", {"campaign_id": "CMP_1"}),
    ])
    assert trajectory.observations[-1].data["daily_budget"] == 500.0


# --------------------------------------------------------------------------
# 幂等键：F1（超时后重试）的正确形态
# --------------------------------------------------------------------------


def test_write_requires_an_idempotency_key(bundle):
    """没带 client_request_id 的写直接拒绝。

    ⚠️ 真实的 Meta Marketing API **没有**幂等机制（实查文档确认）。
    我们在沙盒里把它建成"平台支持"，是为了让模型养成**每次写都带唯一键**的习惯——
    真实接入时这层保证由我们自己的 runtime 兑现（设计文档 §38 三层幂等的第三层）。
    """
    trajectory, sandbox = run(bundle, [
        PlannedCall("campaign.update_budget", {"campaign_id": "CMP_1", "new_budget": 90_000}),
    ])
    assert not trajectory.observations[-1].ok
    assert "client_request_id" in trajectory.observations[-1].error
    assert sandbox.records_for("campaign.update_budget") == []


def test_same_request_id_is_deduplicated_not_reapplied(bundle):
    """★ 同一个键重试 → 返回上次结果 + deduplicated，**不产生第二次写**。

    这是超时重试的正确形态：带同一个键重试是安全的。
    如果去重后还入账，duplicate_write_cap 会把**正确的重试**判成违规。
    """
    same = {"campaign_id": "CMP_1", "new_budget": 90_000, "client_request_id": "r-1"}
    trajectory, sandbox = run(bundle, [PlannedCall("campaign.update_budget", dict(same)),
                                       PlannedCall("campaign.update_budget", dict(same))])
    assert trajectory.observations[-1].data.get("deduplicated") is True
    assert len(sandbox.records_for("campaign.update_budget")) == 1
    assert sandbox.duplicate_writes() == {}


def test_different_request_id_really_writes_twice(bundle):
    """换了键就是两次不同的写——去重不能把真正的重复操作也盖掉。"""
    _, sandbox = run(bundle, [
        PlannedCall("campaign.update_budget",
                    {"campaign_id": "CMP_1", "new_budget": 70_000, "client_request_id": "a"}),
        PlannedCall("campaign.update_budget",
                    {"campaign_id": "CMP_1", "new_budget": 90_000, "client_request_id": "b"}),
    ])
    assert len(sandbox.records_for("campaign.update_budget")) == 2
    assert "campaign.update_budget" in sandbox.duplicate_writes()


# --------------------------------------------------------------------------
# 调用配额：Meta 每个 ad set 每小时最多改 4 次预算
# --------------------------------------------------------------------------


def test_fifth_budget_change_is_rejected(bundle):
    """Meta 实况：613 / 1487632，并冻结该对象一小时。"""
    calls = [PlannedCall("campaign.update_budget",
                         {"campaign_id": "CMP_1", "new_budget": 60_000 + i * 1000,
                          "client_request_id": f"k{i}"}) for i in range(5)]
    trajectory, sandbox = run(bundle, calls)
    assert trajectory.observations[-1].ok is False
    assert "613" in trajectory.observations[-1].error
    assert len(sandbox.records_for("campaign.update_budget")) == 4


def test_quota_is_per_object_not_global(bundle):
    """配额按对象计，改另一个 campaign 不受影响。"""
    calls = [PlannedCall("campaign.update_budget",
                         {"campaign_id": "CMP_1", "new_budget": 60_000 + i * 1000,
                          "client_request_id": f"k{i}"}) for i in range(4)]
    calls.append(PlannedCall("campaign.update_budget",
                             {"campaign_id": "CMP_2", "new_budget": 40_000, "client_request_id": "other"}))
    trajectory, _ = run(bundle, calls)
    assert trajectory.observations[-1].ok is True


def test_budget_is_in_minor_units(bundle):
    """★ 单位。真实 Meta API 的 daily_budget 就是最小货币单位，**字段名不告诉你这件事**。

    沙盒如果用"元"，模型学会填 900，到真实 API 上填的是 9 块钱——
    而且不会报错，只会安静地把预算改成 1/100。
    """
    schema = DOMAIN.registry.get("campaign.update_budget")
    assert schema.parameters["properties"]["new_budget"]["type"] == "integer"
    assert "分" in schema.description and "90000" in schema.description
    assert schema.api_ref == "meta:POST /{campaign_id}"
