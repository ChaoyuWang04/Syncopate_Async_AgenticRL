"""B-4 · 真模型 Decider：把 agent_loop 的 `decide()` 接到 vLLM 端点上。

★★★ 对齐纪律（B-5 的生死线）：**渲染和解析全部复用训练侧的那一份，不抄第二份。**

    渲染   prompts/system.txt + step_user.txt · apply_chat_template(tools=REGISTRY.menu())
           —— 和 rollout_loop.build_messages 同模板、同 CHAT_TEMPLATE_KWARGS（think off）
    解析   syncopate.core.parsing.parse_step —— 训练/评测/回放共用的那个解析器
    采样   rollout_budget 契约值（temperature/top_p/top_k），不在这里写数

⇒ 刻意 **不用** vLLM 的 /chat/completions + 服务端工具解析器：那是另一条渲染/解析
  路径（vLLM 的 hermes parser ≠ 我们的 parse_step），训练时的最优策略在它上面
  不保证还是最优。改走 /completions + 客户端渲染，模型看到的字节和训练一致。

⚠️ 一步多调用：训练侧 P0-2 拦在发生点（回灌"请等 observation"）。这里同法 ——
  解析出 >1 个 tool_call 或解析失败时，返回 tool=None 的提案，由循环把
  rationale 里的错误文本回灌给模型（agent_loop 2026-08-20 起认 rationale）。
"""

from __future__ import annotations

import asyncio
import datetime as _dt
from typing import Any

import httpx

from syncopate.core.parsing import parse_step, render_tool_call
from syncopate.prompts import load_prompt, render_prompt
from syncopate.runtime.agent_loop import Proposal
from syncopate.train.rollout_budget import (
    MAX_RESPONSE_LENGTH, SAMPLING_TEMPERATURE, SAMPLING_TOP_K, SAMPLING_TOP_P)
from syncopate.train.rollout_loop import (
    ASSISTANT_TURN_END, CHAT_TEMPLATE_KWARGS, observation_message)

# 默认的结论字段：runtime 的请求没有逐 case 的 verifier，先用最小集。
# ⚠️ 这和训练分布有已知偏差（训练里 answer_fields 逐 case 不同）——
#   B-4 完整版要按 intent 建字段表；先钉住可跑，缺口显式记在这里。
DEFAULT_ANSWER_FIELDS = [{"key": "summary", "description": "本次任务的结论"}]

# ★ intent → 工具子菜单。**取自 v13 训练 case 的众数菜单**（不是拍的）：
#   train.parquet → batch_dir/cases/<case_id>.json 的 tool_menu，按家族取 most_common。
#   训练里菜单是 12–16 个工具；全量 30 个 = prompt 7625 token > max_model_len 7168，
#   而且是模型没见过的分布（Q7 保证训练 prompt ≤5120 正是在这些子菜单下成立的）。
#   I01 读指标 ≈ BUD 家族 · I07 归因 ≈ DIA · I09 扩量 ≈ SCALE · I11 素材 ≈ CRE。
_MENU_BUD = ["approval.create_case", "benchmark.get_industry_baseline",
             "benchmark.get_safety_line", "campaign.get_metrics", "campaign.list",
             "campaign.update_budget", "memory.search", "memory.write_proposal",
             "metrics.get_freshness", "policy.get_budget_rule", "risk.check_account",
             "system.wait"]
INTENT_MENUS: dict[str, list[str]] = {
    "I01": _MENU_BUD,
    "I07": ["approval.create_case", "benchmark.get_industry_baseline",
            "benchmark.get_safety_line", "campaign.detect_anomalies",
            "campaign.get_metrics", "campaign.list", "campaign.update_budget",
            "memory.search", "memory.write_proposal", "metrics.get_freshness",
            "playbook.get_optimization", "policy.get_budget_rule",
            "risk.check_account", "system.wait"],
    "I09": ["approval.create_case", "benchmark.get_industry_baseline",
            "benchmark.get_safety_line", "campaign.create", "campaign.get_metrics",
            "campaign.list", "campaign.scale_budget", "campaign.update_budget",
            "memory.search", "memory.write_proposal", "metrics.get_freshness",
            "policy.get_budget_rule", "risk.check_account", "system.wait"],
    "I11": ["approval.create_case", "benchmark.get_industry_baseline",
            "benchmark.get_safety_line", "calendar.get_seasonal_context",
            "campaign.get_metrics", "campaign.list", "campaign.update_budget",
            "creative.get_asset_tags", "creative.search_similar", "memory.invalidate",
            "memory.search", "memory.write_proposal", "metrics.get_freshness",
            "policy.get_budget_rule", "risk.check_account", "system.wait"],
}
DEFAULT_MENU = _MENU_BUD


class VllmDecider:
    """调 vLLM /v1/completions 的 Decider。一个实例服务一个 worker 进程。"""

    def __init__(self, *, base_url: str, model: str, tokenizer_path: str,
                 context: dict[str, Any] | None = None,
                 answer_fields: list[dict[str, str]] | None = None,
                 timeout_seconds: float = 120.0) -> None:
        from transformers import AutoTokenizer   # 延迟导入：只有真模型模式才付这个钱

        self.model = model
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.context = context or {}
        self.answer_fields = answer_fields or DEFAULT_ANSWER_FIELDS
        self._client = httpx.AsyncClient(base_url=base_url, timeout=timeout_seconds)
        # 工具 spec 来源：训练同一份真相（tool_registry）；菜单按 intent 裁剪（见上）
        import syncopate.domains.adcampaign  # noqa: F401  （注册工具的副作用导入）
        from syncopate.core.tool_registry import REGISTRY
        self._registry = REGISTRY
        self.tools = REGISTRY.menu(DEFAULT_MENU)   # 判据行用（工具数）

    async def aclose(self) -> None:
        await self._client.aclose()

    # ── 渲染：loop 的 history → 训练同形的 messages ─────────────────────────
    def _messages(self, user_message: str,
                  history: list[dict[str, Any]]) -> list[dict[str, Any]]:
        system_text = load_prompt("system.txt")
        user_text = render_prompt("step_user.txt", {
            "reference_now": _dt.date.today().isoformat(),
            "context": self.context,
            "user_message": user_message,
            "answer_fields": self.answer_fields,
        })
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_text},
        ]
        last_tool = "system"
        for entry in history:
            role = entry.get("role")
            if role == "action":
                last_tool = entry.get("tool") or "system"
                messages.append({"role": "assistant",
                                 "content": render_tool_call(last_tool,
                                                             entry.get("arguments") or {})})
            elif role == "observation":
                messages.append(observation_message(last_tool,
                                                    entry.get("observation") or {}))
        return messages

    async def decide(self, *, user_message: str,
                     history: list[dict[str, Any]]) -> Proposal:
        from syncopate.runtime.agent_loop import MODEL_USAGE, RUN_INTENT
        menu = INTENT_MENUS.get(RUN_INTENT.get() or "", DEFAULT_MENU)
        tools = self._registry.menu(menu)
        messages = self._messages(user_message, history)
        prompt: str = self.tokenizer.apply_chat_template(
            messages, tools=tools, add_generation_prompt=True,
            tokenize=False, **CHAT_TEMPLATE_KWARGS)
        # 预算：单轮生成上限 = 评测口径（MAX_RESPONSE_LENGTH，G-8 之后 256→2048），
        # 上下文超长按训练同法**左截断**（rollout_loop 同款）——且必须计数，
        # 静默截断是记录在案的整个失效家族（budget-truncation-family）。
        ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        ctx_cap = 7168 - 256                       # 至少给生成留 256
        if len(ids) > ctx_cap:
            ids = ids[-ctx_cap:]
            prompt = self.tokenizer.decode(ids)
            usage_t = MODEL_USAGE.get()
            if usage_t is not None:
                usage_t["prompt_truncated"] = usage_t.get("prompt_truncated", 0) + 1
        max_tokens = max(64, min(MAX_RESPONSE_LENGTH, 7168 - len(ids) - 8))
        resp = await self._client.post("/v1/completions", json={
            "model": self.model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": SAMPLING_TEMPERATURE,
            "top_p": SAMPLING_TOP_P,
            "top_k": SAMPLING_TOP_K,
            "stop": [ASSISTANT_TURN_END],
        })
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["text"]
        # token 用量回传（§19 成本指标的生产者）：worker set 的 contextvar，逐调用累加
        usage = MODEL_USAGE.get()
        if usage is not None and "usage" in data:
            usage["tokens_in"] = usage.get("tokens_in", 0) + data["usage"]["prompt_tokens"]
            usage["tokens_out"] = usage.get("tokens_out", 0) + data["usage"]["completion_tokens"]
            usage["calls"] = usage.get("calls", 0) + 1
        return self._to_proposal(text)

    @staticmethod
    def _to_proposal(text: str) -> Proposal:
        parsed = parse_step(text)
        if parsed.kind == "final":
            return Proposal(kind="final",
                            final_answer={"behavior": parsed.behavior,
                                          "answer": parsed.answer},
                            rationale=parsed.behavior)
        if parsed.kind == "tool_calls":
            if len(parsed.tool_calls) > 1:
                # P0-2 同法：拦在发生点，把纠正文本回灌给模型
                return Proposal(kind="tool_call", tool=None,
                                rationale="每步只输出一个 tool call，"
                                          "请等上一个 observation 返回后再决定下一步。")
            call = parsed.tool_calls[0]
            return Proposal(kind="tool_call", tool=call.get("name"),
                            arguments=dict(call.get("arguments") or {}),
                            param_source="model")
        return Proposal(kind="tool_call", tool=None,
                        rationale=f"parse_error: {parsed.error or 'unparseable'}")


def build_decider_from_env() -> VllmDecider | None:
    """worker 进程入口用：环境变量显式开启才建（默认 None = 写死计划，行为不变）。

    判据行由调用方打（「机制在但没接上」第一形态的对策：没那行日志就是没接上）。
    """
    import os

    base_url = os.environ.get("SYNCOPATE_DECIDER_URL")
    if not base_url:
        return None
    return VllmDecider(
        base_url=base_url,
        model=os.environ.get("SYNCOPATE_DECIDER_MODEL", "candidate"),
        tokenizer_path=os.environ.get("SYNCOPATE_DECIDER_TOKENIZER",
                                      "models/Qwen3-4B-sft-v13r2-e1"),
        context={"campaign_id": "CMP_1"},
    )
