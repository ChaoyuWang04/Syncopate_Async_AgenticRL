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

import datetime as _dt
import json
import os
from typing import Any

import httpx

from syncopate.core.contract import IS_V15, TERMINAL_SIGNALS, visible_answer_fields
from syncopate.core.parsing import parse_step
from syncopate.core.parsing import render_tool_call as _render_tool_call_v14
from syncopate.core.parsing_v15 import render_tool_call as _render_tool_call_v15
from syncopate.core.contract import IS_V15 as _IS_V15
render_tool_call = _render_tool_call_v15 if _IS_V15 else _render_tool_call_v14   # 线格式随契约（Qwen3.5+ = XML）
from syncopate.core.parsing_v15 import parse_step_v15

# 部署侧 CoT 观察开关（Chaoyu 08-29）：与训练契约刻意分叉（F-5 同款先例）。
# 开 = chat 模板 enable_thinking，模型自决是否思考（v14.5 冷启过 40 条难例终答 think）。
# ⚠️ E27 警示在案：think-on 曾伴随 acted_when_should_not 上行——本开关仅观察用，
#   评测/考场路径不受影响（它们不设此环境变量）。
import os as _os
RUNTIME_THINKING = _os.environ.get("SYNCOPATE_RUNTIME_THINKING") == "1"
import re as _re
_THINK_RE = _re.compile(r"<think>(.*?)</think>", _re.S)
from syncopate.prompts import load_prompt, load_system_prompt, render_prompt
from syncopate.runtime.agent_loop import Proposal
from syncopate.train.rollout_budget import (
    MAX_RESPONSE_LENGTH, SAMPLING_TEMPERATURE, SAMPLING_TOP_K, SAMPLING_TOP_P)
from syncopate.train.rollout_loop import (
    ASSISTANT_TURN_END, CHAT_TEMPLATE_KWARGS, observation_message)

# 结论字段契约。★★ 2026-08-20 由 O-2a 探针实测定型（`22 §J-6`）：
#
#   A 只要 summary「本次任务的结论」  闲聊自然语言 **2** token（机器标签）· 任务 2/2 调工具
#   B 只要 reply「用人话说清楚」       闲聊 **21** token（真人话）· 任务 **1/2** 🔴
#     └ 失败样本："查一下 CMP_1 昨天的花费" → 不调工具，直接编「今日投放未启动」
#       ⇒ **只优化表达会把领域能力换走** —— `22 §J-3` 坑①的直接实证
#   C 两个并列（本档）                 闲聊 **10** token · 任务 2/2 调工具 ✅
#
# ⇒ 取 C：评分器要的机器可校验字段不丢，人要读的话也有。
# ⚠️ 剩余缺口（reply 质量仍不如裸底座）**才是 OPD 要修的那部分** —— 契约改不动它。
DEFAULT_ANSWER_FIELDS = [
    {"key": "summary", "description": "结论的机器可校验形式（简短标签或数值）"},
    {"key": "reply", "description": "给用户读的完整回复：一到三句自然语言，说清结论和依据"},
]

# ★★ 部署侧的上下文上限 —— **刻意与训练契约（5120+2048=7168）不同**。
#
# `rollout_budget.py` 里写着这一天的条款：「部署一旦硬性要求，就以部署为准，
# 三方再对齐一次」。多轮对话就是那个硬性要求：单任务 prompt 实测已到 4654
# （余量仅 466），历史一拼必撞左截断 —— 而 prompt 截断在本项目是元凶级前科
# （砍掉 system 规则书开头 ⇒ 行为枚举丢失 ⇒ 整轮 RL 白跑 + 一整套错误归因）。
#
# 为什么部署侧敢放宽而训练侧不能：训练要同时跑 8 条一组的采样 + 反传激活，
# 长度直接吃步速；服务侧只有前向，且 KV cache 是**分页按需分配**的，
# `max_model_len` 只是单序列上限不是预留。账：4B 每 token KV ≈144 KB ⇒
# 14336 顶格一条 ≈2 GB，KV 池 ~20 GB ⇒ 并发 8 条顶格也放得下。
# 且 prompt 里 ~4.2k 的 system+工具 schema 是**所有会话共享的 prefix**（cache 命中）。
# ⚠️ 改这个数必须同时改 vLLM 起服务的 `--max-model-len`（两边必须一致，
#   起服务命令在 `docs/syncopate/09 §0`）—— 服务端小于这里 = 400 报错。
RUNTIME_MAX_MODEL_LEN = int(os.environ.get("SYNCOPATE_RUNTIME_MAX_LEN", "18432"))   # 09-02：9216+8192 → 18432

# 一轮历史最多渲染多少 token 的结论（超了截断并计数——静默砍是禁的）
from syncopate.core.prior_turns import PRIOR_ANSWER_BUDGET, render_prior_messages  # noqa: E402

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

# ★★ 2026-08-20（Chaoyu）：**默认给全量 30 个工具，让模型自己选**，不做意图路由。
#
# 为什么以前不能：全量 prompt 7625 tok > 旧上限 7168 —— 是**预算**挡着，不是设计选择。
# 上限抬到 14336 之后账算得开（实测）：
#     全量工具 7625 + 历史 6×400 + 生成 2048 = 12073 < 14336
# ⚠️ 代价是**训练分布外**：训练 case 的菜单是 12–16 个工具，模型没见过 30 个的形状。
#   ⇒ 探针 `probe_full_menu.py` 量格式保持与选工具准确性；不合格就把
#     SYNCOPATE_TOOL_MENU=intent 打回按意图裁剪（INTENT_MENUS 因此保留不删）。
# ⚠️ 概括版（只给名字+首句，1049 tok）**不是选项**：不给参数 schema，
#   模型不知道怎么填参数 ⇒ 换来一堆 validation_failed。省的那点 token 靠
#   prefix cache 本来就免了（实测命中率 98.7%）。
FULL_MENU_MODE = os.environ.get("SYNCOPATE_TOOL_MENU", "full") != "intent"


class VllmDecider:
    """调 vLLM /v1/completions 的 Decider。一个实例服务一个 worker 进程。"""

    def __init__(self, *, base_url: str, model: str, tokenizer_path: str,
                 context: dict[str, Any] | None = None,
                 answer_fields: list[dict[str, str]] | None = None,
                 timeout_seconds: float | None = None) -> None:
        # ★ 09-02（K 线 29 号 D9）：模型调用超时不再写死——走 SYNCOPATE_DECIDER_TIMEOUT（秒），默认 120。
        #   think-on + 8192 生成预算下单次调用可能超过 120s；上限抬高时这里要能跟着改而不改代码。
        if timeout_seconds is None:
            timeout_seconds = float(os.environ.get("SYNCOPATE_DECIDER_TIMEOUT", "120"))
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
        # 判据行用（启动时打出来的工具数 —— 没这行就不知道到底给了模型几个工具）
        self.tools = REGISTRY.menu(None if FULL_MENU_MODE else DEFAULT_MENU)

    async def aclose(self) -> None:
        await self._client.aclose()

    # ── 渲染：loop 的 history → 训练同形的 messages ─────────────────────────
    def _prior_turn_messages(self, turns: list[dict]) -> list[dict[str, Any]]:
        """把会话里之前几轮渲染成 user/assistant 对 —— **与训练侧同一个函数**
        （`syncopate.core.prior_turns.render_prior_messages`，09-02 守则⑮对齐；此前这里是唯一实现，
        训练把历史折成题面文本，两边不同形）。只带问题 + 结论，不带工具步骤明细。
        """
        return render_prior_messages(turns, self.tokenizer, PRIOR_ANSWER_BUDGET)

    def _prior_as_inline_text(self, prior: list[dict]) -> str:
        """把历史折成**训练里那种**「[上一轮] …」文本（实验开关，默认关）。

        ★ 这是一次**受控实验**，不是产品决定：v15 的多轮在训练里是写在题面内的文本，
          在生产里是真消息对。两种形状哪个是主因，要量出来再定往哪边对齐（`25 §7㉝`）。
          开关 SYNCOPATE_PRIOR_INLINE=1；⛔ 默认必须是关的 —— 真实产品就是多轮消息。
        """
        lines = []
        for t in prior:
            lines.append(f"[上一轮] 用户：{t.get('user_message') or ''}")
            msgs = self._prior_turn_messages([t])
            lines.append(f"[上一轮] 助手：{msgs[-1]['content']}")
        return "\n".join(lines)

    def _messages(self, user_message: str,
                  history: list[dict[str, Any]],
                  prior: list[dict] | None = None) -> list[dict[str, Any]]:
        import os as _os

        inline = _os.environ.get("SYNCOPATE_PRIOR_INLINE") == "1"
        if inline and prior:
            user_message = self._prior_as_inline_text(prior) + "\n\n" + user_message
        system_text = load_system_prompt()
        user_text = render_prompt("step_user.txt", {
            "reference_now": _dt.date.today().isoformat(),
            "context": self.context,
            "user_message": user_message,
            # ★ 人话字段不进清单：清单是"要填的表"，人话不是表格里的一格
            "answer_fields": visible_answer_fields(self.answer_fields),
        })
        messages: list[dict[str, Any]] = [{"role": "system", "content": system_text}]
        # ★ 历史插在 system 之后、本轮任务之前 —— 本轮永远是最后一条 user，
        #   模型的"当前要办的事"不会被历史挤到中间去。
        if prior and not inline:
            messages += self._prior_turn_messages(prior)
        messages.append({"role": "user", "content": user_text})
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
        from syncopate.runtime.agent_loop import MODEL_USAGE, PRIOR_TURNS, RUN_INTENT
        # 默认全量（模型自己选）；SYNCOPATE_TOOL_MENU=intent 时退回按意图裁剪
        menu = None if FULL_MENU_MODE else INTENT_MENUS.get(RUN_INTENT.get() or "",
                                                            DEFAULT_MENU)
        tools = self._registry.menu(menu)
        messages = self._messages(user_message, history, PRIOR_TURNS.get())
        _kw = ({"enable_thinking": True} if RUNTIME_THINKING else CHAT_TEMPLATE_KWARGS)
        prompt: str = self.tokenizer.apply_chat_template(
            messages, tools=tools, add_generation_prompt=True,
            tokenize=False, **_kw)
        # 预算：单轮生成上限 = 评测口径（MAX_RESPONSE_LENGTH，G-8 之后 256→2048），
        # 上下文超长按训练同法**左截断**（rollout_loop 同款）——且必须计数，
        # 静默截断是记录在案的整个失效家族（budget-truncation-family）。
        # ⚠️ 护栏跟着 RUNTIME_MAX_MODEL_LEN 走：上限抬了而这条线不抬，
        #   历史照样被砍 = 白改（"机制在但没接上"的又一种形状）。
        ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        ctx_cap = RUNTIME_MAX_MODEL_LEN - 256      # 至少给生成留 256
        if len(ids) > ctx_cap:
            ids = ids[-ctx_cap:]
            prompt = self.tokenizer.decode(ids)
            usage_t = MODEL_USAGE.get()
            if usage_t is not None:
                usage_t["prompt_truncated"] = usage_t.get("prompt_truncated", 0) + 1
        max_tokens = max(64, min(MAX_RESPONSE_LENGTH,
                                 RUNTIME_MAX_MODEL_LEN - len(ids) - 8))
        # B-5 S4 · SLO 感知优先级（E33）：四个意图的 P95 预算差 36×（I01 5s vs I11 180s），
        # 引擎 FIFO 让最紧的陪最松的排队 ⇒ 高并发下 I01 先破线。按预算给引擎传
        # priority（vLLM --scheduling-policy priority 时生效，数值小者先跑；
        # fcfs 策略下该字段被忽略 = 向后兼容零风险）。只改调度顺序，不改任何产品语义。
        _prio = {"I01": -3, "I07": -2, "I09": -1}.get(RUN_INTENT.get() or "", 0)
        resp = await self._client.post("/v1/completions", json={
            "model": self.model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": SAMPLING_TEMPERATURE,
            "top_p": SAMPLING_TOP_P,
            "top_k": SAMPLING_TOP_K,
            "stop": [ASSISTANT_TURN_END],
            "priority": _prio,
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
        m = _THINK_RE.search(text)
        think = (m.group(1).strip() if m else "")
        if IS_V15:
            return VllmDecider._to_proposal_v15(text, think)
        parsed = parse_step(text)
        if parsed.kind == "final":
            return Proposal(kind="final",
                            final_answer={"behavior": parsed.behavior,
                                          "answer": parsed.answer},
                            rationale=parsed.behavior, thinking=think)
        if parsed.kind == "tool_calls":
            if len(parsed.tool_calls) > 1:
                # P0-2 同法：拦在发生点，把纠正文本回灌给模型
                return Proposal(kind="tool_call", tool=None,
                                rationale="每步只输出一个 tool call，"
                                          "请等上一个 observation 返回后再决定下一步。")
            call = parsed.tool_calls[0]
            return Proposal(kind="tool_call", tool=call.get("name"),
                            arguments=dict(call.get("arguments") or {}),
                            param_source="model", thinking=think)
        return Proposal(kind="tool_call", tool=None,
                        rationale=f"parse_error: {parsed.error or 'unparseable'}")


    @staticmethod
    def _to_proposal_v15(text: str, think: str) -> Proposal:
        """v15：行为是**显式动作**，runtime 直接拿信令去驱动状态机（N4）。

        ⚠️ 这里刻意**不复制**一份解析逻辑 —— 用的就是训练/评测那一份
        `parse_step_v15`（N5 一份契约）。runtime 另抄一份是本项目的老病
        （decider.py 抬头那段注释记的就是这件事）。
        """
        p = parse_step_v15(text)
        if p.kind == "signal":
            # 终止性信令 → 状态机触发器（defer 挂起复查 / clarify 等补充 / reject 终止）
            return Proposal(kind="final",
                            final_answer={"behavior": p.signal,
                                          "signal": p.signal,
                                          "arguments": dict(p.signal_args),
                                          "text": p.text},
                            rationale=p.signal, thinking=think)
        if p.kind == "final_text":
            # 纯自然语言终答；行为（tool_call / answer）由轨迹级推导，worker 掌握全程
            return Proposal(kind="final",
                            final_answer={"behavior": None, "text": p.text},
                            rationale="final_text", thinking=think)
        if p.kind == "tool_calls":
            if len(p.tool_calls) > 1:
                return Proposal(kind="tool_call", tool=None,
                                rationale="每步只输出一个 tool call，"
                                          "请等上一个 observation 返回后再决定下一步。")
            call = p.tool_calls[0]
            return Proposal(kind="tool_call", tool=call.get("name"),
                            arguments=dict(call.get("arguments") or {}),
                            param_source="model", thinking=think)
        return Proposal(kind="tool_call", tool=None,
                        rationale=f"parse_error: {p.error or 'unparseable'}")


def build_decider_from_env() -> VllmDecider | None:
    """worker 进程入口用：环境变量显式开启才建（默认 None = 写死计划，行为不变）。

    判据行由调用方打（「机制在但没接上」第一形态的对策：没那行日志就是没接上）。
    """
    import os

    base_url = os.environ.get("SYNCOPATE_DECIDER_URL")
    if not base_url:
        return None
    # ⛔ 2026-08-30：tokenizer 原本默认 "models/Qwen3-4B-sft-v13r2-e1" —— **v13 世代的模型**。
    #   起 v15 考场时忘了传，于是渲染用的是**另一个模型的分词器**，而且不报错。
    #   这正是「默认值指向了另一件事且不报错」那一形态（memory: project-mechanism-not-wired 第七形态），
    #   也是 eval_parallel 的 MODEL 早就立过规矩的同一类：**给错了宁可报错**。
    tok = os.environ.get("SYNCOPATE_DECIDER_TOKENIZER")
    if not tok:
        raise SystemExit(
            "🔴 设了 SYNCOPATE_DECIDER_URL 就必须显式给 SYNCOPATE_DECIDER_TOKENIZER —— "
            "它必须是**正在被 serve 的那个模型**的路径。\n"
            "   留默认值 = 用另一个模型的分词器渲染 prompt，且不会报错。")
    model = os.environ.get("SYNCOPATE_DECIDER_MODEL", "candidate")
    # ⛔ 2026-08-30：模型名对不上端点时，vLLM 对**每一条请求**回 404，而 worker 只把它
    #   记成一条 run.failed —— 考场里 109/277 条这么死的，读数看着"跑完了"。
    #   成因是上一轮的 worker 没停干净，它认的还是**上一个已经不在服务的模型名**，
    #   和新 worker 抢同一个队列。⇒ 启动时就核一次：名字不在端点的清单里，**直接不许起**。
    #   ⚠️ 端点没起来（连不上）是另一回事，只警告 —— 那是竞速，不是配错。
    try:
        import httpx as _hx

        served = [m["id"] for m in
                  _hx.get(f"{base_url.rstrip('/')}/v1/models", timeout=5).json()["data"]]
    except Exception as exc:                      # 连不上 / 还没起来
        print(f"[decider] ⚠️ 端点暂时问不到模型清单（{type(exc).__name__}），跳过核对", flush=True)
    else:
        if model not in served:
            raise SystemExit(
                f"🔴 SYNCOPATE_DECIDER_MODEL={model!r} 不在端点 {base_url} 服务的模型里：{served}\n"
                "   继续跑的话每条请求都会拿 404，而 worker 只会把它记成一条 run.failed。\n"
                "   最常见的原因：**上一轮的 worker 没停干净**，它和新 worker 在抢同一个队列。")
    print(f"[decider] base_url={base_url} model={model} tokenizer={tok}", flush=True)
    return VllmDecider(base_url=base_url, model=model, tokenizer_path=tok,
                       context=_demo_context())


def build_alt_deciders_from_env() -> dict[str, "VllmDecider"]:
    """dev mode 备选端点：SYNCOPATE_DECIDER_URL_SFT / _BASE（存在才建，各配 tokenizer）。"""
    import os

    out: dict[str, VllmDecider] = {}
    for tag, url_env, tok_env, tok_default in (
        ("sft", "SYNCOPATE_DECIDER_URL_SFT", "SYNCOPATE_DECIDER_TOKENIZER_SFT",
         "models/Qwen3-4B-sft-v14.5-epoch3"),
        ("base", "SYNCOPATE_DECIDER_URL_BASE", "SYNCOPATE_DECIDER_TOKENIZER_BASE",
         "models/Qwen3-4B"),
    ):
        url = os.environ.get(url_env)
        if url:
            out[tag] = VllmDecider(base_url=url, model="candidate",
                                   tokenizer_path=os.environ.get(tok_env, tok_default),
                                   context=_demo_context())
    return out


def _demo_context() -> dict[str, Any]:
    """渲染进 prompt 的「当前投放任务」上下文 —— 实现在 core/demo_context（训练建库共用同一份）。"""
    from syncopate.core.demo_context import demo_context
    return demo_context()
