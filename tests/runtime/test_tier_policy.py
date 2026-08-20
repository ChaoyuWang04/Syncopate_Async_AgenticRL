"""档位推导（2026-08-20）：**档位是动作的属性，不是声明、也不是模型自选**。

★ 这一族钉的是一条信任边界，判据分两类：
  ① 推导本身对不对（读免闸 · 写要人点头 · 大额永不自动 · 升级通道免闸）
  ② **能升不能降** —— 任何来源都只能把档位往严了拉，这是安全性的全部
"""

from __future__ import annotations

import asyncio

from syncopate.runtime.action_gate import ActionGate, ToolBinding
from syncopate.runtime.gateway import DecisionContext
from syncopate.runtime.tier_policy import (NEVER_AUTOMATE_AMOUNT, derive_tier,
                                           more_cautious)
from syncopate.runtime.tools import WRITE_TOOLS


# ── ① 推导 ────────────────────────────────────────────────────────────────

def test_reads_never_pass_through_the_tier_gate():
    """★ 读不改变世界 ⇒ 完全不过闸。灰测期间也必须能查（降级 ≠ 失明）。"""
    for tool in ("campaign.get_metrics", "policy.search", "memory.search"):
        assert derive_tier(tool, {}).tier is None


def test_every_write_tool_requires_at_least_approval():
    """★★ 除升级通道外，**每一个**写工具都必须推导出要人点头。

    ⚠️ 判据写成"遍历 WRITE_TOOLS"而不是列举几个：
      将来有人加写工具却忘了配策略，这条会红 —— 否则新工具会静默地免闸。
    """
    for tool in WRITE_TOOLS:
        d = derive_tier(tool, {"campaign_id": "C", "new_budget": 1})
        if tool == "approval.create_case":
            continue
        assert d.tier in ("C", "D"), f"{tool} 推出了 {d.tier}，写动作不该免审批"
        assert d.reason, "判定必须带理由（人看的是证据不是结论）"


def test_escalation_path_is_never_gated():
    """★★★ 给"请求审批"加审批是循环的，而且正是 C-5 记的那个坑
    （模型交给人反而被罚）。"""
    assert derive_tier("approval.create_case", {}).tier is None


def test_huge_amount_is_never_automated():
    d = derive_tier("campaign.update_budget",
                    {"campaign_id": "C", "new_budget": NEVER_AUTOMATE_AMOUNT})
    assert d.tier == "D" and "永不自动" in d.reason


def test_amount_as_string_is_still_recognized():
    """★ 模型给的是 JSON 值，数字常以字符串到达 —— 认不出就把大额判成了小额
    （calendar `date + unknown` 那个坑的同族）。"""
    d = derive_tier("campaign.update_budget",
                    {"campaign_id": "C", "new_budget": str(NEVER_AUTOMATE_AMOUNT)})
    assert d.tier == "D"


def test_unparseable_amount_is_treated_as_most_cautious():
    """★ 认不出金额 ⇒ 按最谨慎处理（"我们不知道"一律保守，同 retrieval_unavailable）。"""
    d = derive_tier("campaign.update_budget", {"new_budget": "一千二"})
    assert d.tier == "D"


# ── ② 能升不能降 ──────────────────────────────────────────────────────────

def test_more_cautious_picks_the_least_autonomous():
    assert more_cautious("A", "C") == "C"
    assert more_cautious("C", "D") == "D"
    assert more_cautious("C", None) == "C"          # None = 没意见，被忽略
    assert more_cautious(None, None) is None


def test_declaration_can_only_tighten_never_loosen():
    """★★★ 这一条是整个设计的安全性所在：声明（或将来任何来源）**只能升不能降**。

    反面就是"模型自我授权" —— 被注入的模型第一件事就是给自己开 A 档（§27.2）。
    """
    derived = derive_tier("campaign.update_budget",
                          {"campaign_id": "C", "new_budget": 1}).tier   # C
    assert more_cautious(derived, "A") == "C", "声明 A 竟然放松了推导出的 C"
    assert more_cautious(derived, "D") == "D", "声明 D 应该能收紧"


# ── ③ 接线：收口真的用了它 ────────────────────────────────────────────────

class _Rec:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []
        self.audits: list[dict] = []

    async def emit(self, _db, *, org_id, run_id, kind, payload=None):   # noqa: ANN001
        self.events.append((kind, payload or {}))
        return len(self.events)

    async def audit(self, _db, *, org_id, run_id, action, object_key,   # noqa: ANN001
                    param_source, detail=None):
        self.audits.append({"action": action, "detail": detail or {}})


async def _never(**_kw):                                                # noqa: ANN003
    raise AssertionError("★ 这个写动作不该被执行到")


def test_gate_refuses_d_class_action_without_killing_the_run():
    """★★ D 档拒的是**这个动作**，不是整条 run —— 观测回到模型，
    它可以改用 `approval.create_case` 把事情交给人（硬终止 → 可恢复的观测）。"""
    rec = _Rec()

    async def _ob() -> bool:
        return False

    gate = ActionGate(db=None, tools=None,
                      bindings={"campaign.update_budget": ToolBinding(_never)},
                      org_id="o", run_id="r", over_budget=_ob,
                      emit=rec.emit, audit=rec.audit)
    out = asyncio.run(gate.invoke(
        tool="campaign.update_budget",
        arguments={"campaign_id": "C", "new_budget": NEVER_AUTOMATE_AMOUNT,
                   "client_request_id": "k"},
        ctx=DecisionContext(), param_source="model"))

    assert out.status == "failed"                    # 不是抛异常、不是杀 run
    assert "tier_d_never_automated" in (out.error or "")
    assert "approval.create_case" in out.observation["error"], "没告诉模型还有升级通道"
    assert rec.audits[-1]["action"] == "tier_d_refused"


# ── ④ 审批单要带上判定证据（要 PG）──────────────────────────────────────────

def _pg_available() -> bool:
    from syncopate.runtime.db import Database

    async def probe() -> bool:
        db = Database()
        try:
            await db.connect(max_size=2); await db.close(); return True
        except Exception:
            return False
    return asyncio.run(probe())


@__import__("pytest").mark.skipif(not _pg_available(), reason="需要 PostgreSQL")
def test_approval_case_carries_the_tier_reason():
    """★★ 档位不再由人选 ⇒ 审批单必须说清**为什么判成这一档**。

    设计原话：「人看的是证据不是结论」。少了这条，人在审批页上看到的
    就是一个光秃秃的字母，那等于把判断推给了一个不可追问的黑盒。
    """
    import uuid as _uuid

    from syncopate.runtime.db import Database, create_run
    from syncopate.runtime.worker import audit as w_audit, emit as w_emit

    org = f"org_{_uuid.uuid4().hex[:8]}"
    run = f"run_{_uuid.uuid4().hex[:8]}"

    async def go():
        db = Database()
        await db.connect(max_size=4)
        try:
            await create_run(db, org_id=org, run_id=run, user_message="x")

            async def _ob() -> bool:
                return False

            gate = ActionGate(db, tools=None,
                              bindings={"campaign.update_budget": ToolBinding(_never)},
                              org_id=org, run_id=run, over_budget=_ob,
                              emit=w_emit, audit=w_audit)
            out = await gate.invoke(
                tool="campaign.update_budget",
                arguments={"campaign_id": "CMP_1", "new_budget": 5,
                           "client_request_id": "k"},
                ctx=DecisionContext(), param_source="model")
            async with db.tx() as conn:
                row = await conn.fetchrow(
                    "SELECT evidence FROM approval_cases WHERE org_id=$1 AND run_id=$2",
                    org, run)
            return out, row
        finally:
            await db.close()

    out, row = asyncio.run(go())
    assert out.status == "halted", "小额写动作也该先停下问人（灰测默认档 C 的定义）"
    ev = row["evidence"]
    if isinstance(ev, str):
        import json
        ev = json.loads(ev)
    assert ev["tier"] == "C"
    assert "写动作" in ev["tier_reason"], f"审批单没写清判定理由：{ev}"
