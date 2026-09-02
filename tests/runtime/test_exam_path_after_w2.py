"""考场链路在 W2 改动之后仍然成立（09-03 逐条核）：
① 线上题面：context 为空 ⇒ 没有「当前投放任务」节、没有 account_id、没有 campaign 清单（与训练同形）
② 模型不填 account_id 调 campaign.list / risk.check_account ⇒ 收口按租户注入，工具真的查到东西
③ worker 进程里注册表已装载（注入靠 REGISTRY 判"谁需要"，没装载 = 静默不注入）
"""
from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from typing import Any

import pytest

from syncopate.core.contract import IS_V15
from syncopate.runtime.db import Database, create_run
from syncopate.runtime.platform import FakeAdPlatform
from syncopate.runtime.worker import Worker, WorkerConfig


def _pg_available() -> bool:
    async def probe() -> bool:
        db = Database()
        try:
            await db.connect(max_size=2); await db.close(); return True
        except Exception:
            return False
    return asyncio.run(probe())


pytestmark = [pytest.mark.skipif(not _pg_available(), reason="需要 PostgreSQL"),
              pytest.mark.skipif(not IS_V15, reason="v15 契约专有")]


@dataclass
class _P:
    kind: str
    final_answer: dict | None = None
    tool: str | None = None
    arguments: dict = field(default_factory=dict)
    rationale: str = ""
    thinking: str = ""
    param_source: str = "model"


class _Script:
    """先按剧本调工具（**不填 account_id**），再给终答。"""
    def __init__(self, calls):
        self.calls = list(calls)

    async def decide(self, **kw):
        if self.calls:
            t, a = self.calls.pop(0)
            return _P(kind="tool", tool=t, arguments=a)
        return _P(kind="final", final_answer={"text": "查完了。", "signal": None, "arguments": {}})


def test_registry_is_loaded_in_worker_process():
    import syncopate.runtime.worker  # noqa: F401
    from syncopate.core.tool_registry import REGISTRY
    assert REGISTRY.get("campaign.list") is not None, "worker 进程里注册表没装载 ⇒ 收口不知道谁要注入"
    assert "account_id" in REGISTRY.get("campaign.list").injected_params()


def test_production_prompt_has_no_identity_and_no_campaign_list():
    from syncopate.core.demo_context import demo_context
    from syncopate.runtime.decider import VllmDecider

    class _Tok:
        def encode(self, t, add_special_tokens=False): return list(range(len(t)))
        def decode(self, ids): return "x" * len(ids)
    d = VllmDecider.__new__(VllmDecider)
    d.tokenizer = _Tok(); d.context = demo_context(); d.answer_fields = []
    msgs = VllmDecider._messages(d, "CMP_2 能扩量吗", [], None)
    user = msgs[-1]["content"]
    assert "account_id" not in user and "在投 campaign" not in user and "当前投放任务" not in user
    assert user.startswith("当前时间：") and "用户请求：\nCMP_2 能扩量吗" in user
    # 工具 schema 里也没有 account_id（线上 decider 用的就是 REGISTRY.menu）
    from syncopate.core.tool_registry import REGISTRY
    for t in REGISTRY.menu(None):
        assert "account_id" not in t["function"]["parameters"].get("properties", {}), t["function"]["name"]


def test_worker_injects_account_when_model_omits_it():
    async def go(db):
        org = f"org_{uuid.uuid4().hex[:8]}"
        await create_run(db, org_id=org, run_id="r1", user_message="有哪些 campaign？风控怎么样")
        platform = FakeAdPlatform.from_fixture()          # demo 状态：7 条 campaign，账户 ACC_DEMO
        dec = _Script([("campaign.list", {"status": "ACTIVE"}), ("risk.check_account", {})])
        w = Worker(db, platform, config=WorkerConfig(org_id=org), decider=dec)
        assert await w.run_once() == "r1"
        async with db.tx() as conn:
            run = await conn.fetchrow("SELECT status FROM agent_runs WHERE org_id=$1 AND run_id='r1'", org)
            calls = await conn.fetch("SELECT tool, arguments, result FROM tool_calls WHERE org_id=$1 AND run_id='r1' ORDER BY id", org)
            obs = await conn.fetch("SELECT kind, payload FROM run_events WHERE org_id=$1 AND run_id='r1' AND kind LIKE 'tool.%' ORDER BY seq", org)
        return run, calls, obs

    async def main():
        db = Database(); await db.connect(max_size=5)
        try:
            return await go(db)
        finally:
            await db.close()
    run, calls, obs = asyncio.run(main())
    platform_account = FakeAdPlatform.from_fixture().account_id
    assert run["status"] == "succeeded", run["status"]
    tools = [c["tool"] for c in calls]
    assert tools[:2] == ["campaign.list", "risk.check_account"], tools
    blob = " ".join(str(o["payload"]) for o in obs) + " ".join(str(c["result"]) for c in calls)
    assert "unknown_tool" not in blob and "TypeError" not in blob and "missing" not in blob.lower(), blob[:400]
    assert all(str(o["payload"]).find("'ok': True") >= 0 for o in obs), blob[:400]
    # 记录的是**执行时**的参数：收口把租户账户注进去了（审计要看到真正执行的是谁的账户）
    assert all(c["arguments"].get("account_id") == platform_account for c in calls), [c["arguments"] for c in calls]
    assert "CMP_1" in blob or "campaigns" in blob, "campaign.list 没查到东西（注入没生效）：" + blob[:300]
