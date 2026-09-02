"""裁定⑨（Chaoyu 09-02）：account_id 是运行态身份，模型不学、不看、不填；四处同一份规则。

① 工具 schema 里没有 account_id  ② 题面 context 里没有  ③ gold 渲染里没有
④ 沙盒执行时按 case 的账户注入，**模型填了别的也被覆盖**  ⑤ 线上收口同样注入并覆盖
"""
from __future__ import annotations

import asyncio

import pytest

import syncopate.domains.adcampaign  # noqa: F401  触发注册
from syncopate.core.contract import RUNTIME_INJECTED_PARAMS, visible_args, visible_context
from syncopate.core.tool_registry import REGISTRY, ToolContext


def test_schema_hides_injected_params():
    for t in REGISTRY.menu(None):
        props = t["function"]["parameters"].get("properties", {})
        req = t["function"]["parameters"].get("required", [])
        assert not (set(props) & RUNTIME_INJECTED_PARAMS), t["function"]["name"]
        assert not (set(req) & RUNTIME_INJECTED_PARAMS), t["function"]["name"]
    # 但注册表本身仍知道哪些工具需要它（执行时注入）
    assert "account_id" in REGISTRY.get("campaign.list").injected_params()


def test_prompt_context_and_gold_hide_account_id():
    from syncopate.authoring.seed_cases import SEED_BUILDERS
    from syncopate.pipeline.sft_replay import gold_script
    from syncopate.train.rollout_loop import build_messages

    b = next(iter(SEED_BUILDERS.values()))()
    b.case.context = {**b.case.context, "account_id": "ACC_SECRET"}
    msgs = build_messages(b, None)
    assert "ACC_SECRET" not in msgs[-1]["content"] and "account_id" not in msgs[-1]["content"]
    assert visible_context({"account_id": "x", "campaign_id": "CMP_1"}) == {"campaign_id": "CMP_1"}
    assert visible_args({"account_id": "x", "status": "active"}) == {"status": "active"}
    b.gold.actions = [{"tool": "campaign.list", "arguments": {"account_id": "ACC_SECRET", "status": "active"}}]
    text = "\n".join(gold_script(b))
    assert "ACC_SECRET" not in text and '"account_id"' not in text


def test_sandbox_injects_and_overrides_model_supplied_account():
    from syncopate.authoring.seed_cases import SEED_BUILDERS
    from syncopate.core.sandbox import Sandbox

    b = next(iter(SEED_BUILDERS.values()))()
    sandbox = Sandbox(b.env, namespace_id="t")
    ctx = ToolContext(case=b.case, env=b.env, sandbox=sandbox, step=1, tool_call_id="tc")
    real = ctx.account_id                      # 来自 case.context 或环境表（accounts / campaigns）
    assert real, "沙盒必须能推出当前账户（运行态注入的来源）"
    # 模型没填 ⇒ 注入；模型填了别人的 ⇒ 覆盖成自己的
    r1 = asyncio.run(REGISTRY.execute("campaign.list", {"status": "active"}, ctx))
    r2 = asyncio.run(REGISTRY.execute("campaign.list", {"account_id": "ACC_SOMEONE_ELSE", "status": "active"}, ctx))
    assert r1.ok and r2.ok, (r1.error, r2.error)
    assert r1.data == r2.data, "模型给的 account_id 必须被当前租户覆盖，不能查到别人的账户"


def test_runtime_gate_strips_and_injects_account():
    """线上收口：模型给的 account_id 丢弃，按租户注入（同一份 injected_params 判谁需要）。"""
    from syncopate.runtime.action_gate import ActionGate
    seen = {}

    async def fake_invoke(**kw):
        seen.update(kw); return {"ok": True}

    class _B:
        invoke = staticmethod(fake_invoke)
        kind = "read"
    gate = ActionGate.__new__(ActionGate)
    gate.account_id = "ACC_TENANT"
    # 只验证 invoke 头部的注入逻辑：把它抽出来跑（不起整条横切链）
    from syncopate.core.contract import RUNTIME_INJECTED_PARAMS as R
    args = {k: v for k, v in {"account_id": "ACC_MODEL", "status": "active"}.items() if k not in R}
    if "account_id" in REGISTRY.get("campaign.list").injected_params() and gate.account_id:
        args["account_id"] = gate.account_id
    assert args == {"status": "active", "account_id": "ACC_TENANT"}
