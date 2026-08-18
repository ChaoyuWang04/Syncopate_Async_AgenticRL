"""模型输出解析：文本 -> 工具调用 或 最终结论。

Qwen3 一步的输出可能长这样：

    <think>用户要查 CPI，先调 get_metrics</think>
    <tool_call>
    {"name": "campaign.get_metrics", "arguments": {"campaign_id": "CMP_1024"}}
    </tool_call>

或者这样（终答）：

    ```json
    {"behavior": "tool_call", "answer": {"cpi": 2.10}}
    ```

解析器要宽容地吃下这两种，并且**把失败分类**——「格式崩了」和「内容错了」
是两种不同的问题：前者说明 SFT 不够，后者才是 RL 该修的。混在一起看不出训练卡在哪。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

# Qwen3 的思考块。RL 阶段是否允许 thinking 必须显式决定——
# 老师包里 SFT 侧硬编码 enable_thinking=False，RL 侧从不传（走模板默认 = 允许），
# 两阶段不一致（sft-truth-report T10）。我们统一剥离，并记录它出现过没有。
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
# ```json ... ``` 或裸 ``` ... ```
_CODE_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)

# ★ defer 是 M1 加的第四个行为，它是「过早决策」这个最贵错误的**正向对立面**。
# 没有这个标签，模型只能在「做」和「不做」之间选——学不会「等」。
# 而「等到 D7 再判」恰恰是这个业务里最常见的正确答案。
#
# ★★ answer 是 M5 加的第五个，对应负面数据 N1「不该调工具」
#（能力询问"你能改预算吗？"、闲聊、上下文惯性）。
#
# 为什么非得单开一个标签，不能复用现有的：
#   tool_call —— 语义是"我调查/执行完了，给出结论"。拿它装 N1，等于承认
#                **零工具调用也是合法的 tool_call**，其它所有意图的
#                「该查的没查」判据当场被撕开一个口子。
#   clarify   —— 语义是"信息不足要反问"，而 N1 根本不缺信息。
#   reject    —— 语义是"越权/离题要拒绝"，而 N1 是完全正当的问题。
# ⇒ 三个都装不下。设计文档 §27.1 把 N1 单列为一类，也是这个道理。
VALID_BEHAVIORS = {"tool_call", "clarify", "reject", "defer", "answer"}


@dataclass
class ParsedStep:
    """一步输出的解析结果。

    kind 三选一：
        "tool_calls"  模型要调工具
        "final"       模型给出了最终结论
        "error"       两者都没解析出来，或格式非法
    """

    kind: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    behavior: str = "tool_call"
    answer: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    had_thinking: bool = False
    raw_text: str = ""

    @property
    def ok(self) -> bool:
        return self.kind != "error"


def strip_thinking(text: str) -> tuple[str, bool]:
    """剥掉 <think> 块，返回 (剩余文本, 是否出现过 thinking)。"""
    stripped = _THINK_RE.sub("", text)
    return stripped, stripped != text


def _loads_tolerant(payload: str) -> Any:
    """宽容 JSON 解析：先直读，失败了再试着修常见毛病。

    小模型最常犯的两个：结尾多一个逗号、用单引号。我们修前者
    （后者不修——那是明确的格式错误，该让它掉分）。
    """
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        pass
    repaired = re.sub(r",\s*([}\]])", r"\1", payload)
    return json.loads(repaired)


def parse_tool_calls(text: str) -> list[dict[str, Any]]:
    """抽出全部 <tool_call> 块。

    ★ 只保留 name 和 arguments，**显式丢弃模型可能吐出的 id**。
    tool_call_id 是 runtime 的工程字段，必须由我们注入才能保证
    「第 k 步做了什么」可归因；让模型自己编 id 会把这条链打断。
    """
    calls: list[dict[str, Any]] = []
    for block in _TOOL_CALL_RE.findall(text):
        try:
            payload = _loads_tolerant(block)
        except json.JSONDecodeError:
            continue
        # ⛔ 2026-08-18：`_loads_tolerant` 可能返回**非对象**（模型吐出 `"foo"` / `[...]` / 数字都是合法 JSON）
        #    ⇒ 原来直接 `payload.get("name")` ⇒ **AttributeError 把整个 rollout 打崩**，
        #      进而拖垮一整跑（队列 T3 实测：3 分钟就死，3 处 RayTaskError）。
        #    ★ 解析器不该被模型的畸形输出打崩 —— 畸形应当变成**被扣分的行为**，不是崩溃。
        #    ⇒ 与本函数已有的风格一致：形状不对就丢弃这一条。
        if not isinstance(payload, dict):
            continue
        name = payload.get("name")
        if not isinstance(name, str) or not name:
            continue
        arguments = payload.get("arguments")
        if isinstance(arguments, str):        # 有的模型会把 arguments 再套一层字符串
            try:
                arguments = _loads_tolerant(arguments)
            except json.JSONDecodeError:
                arguments = {}
        calls.append({"name": name, "arguments": arguments if isinstance(arguments, dict) else {}})
    return calls


def parse_final_answer(text: str) -> tuple[str, dict[str, Any], str | None]:
    """从 ```json 代码块里抽最终结论，返回 (behavior, answer, error)。

    宽容度是有边界的：代码块必须存在、必须是合法 JSON、`answer` 必须是对象。
    这三条任一不满足就判 parse 失败 —— outcome 直接 0 分。
    这是有意的：格式是可以靠 SFT 学会的东西，不该给部分分。
    """
    blocks = _CODE_BLOCK_RE.findall(text)
    if not blocks:
        # 兜底：整段文本本身就是个 JSON 对象
        candidate = text.strip()
        if candidate.startswith("{") and candidate.endswith("}"):
            blocks = [candidate]
        else:
            return "tool_call", {}, "no_json_block"

    # 有多个代码块时取最后一个——模型有时会先举例再给真答案
    try:
        payload = _loads_tolerant(blocks[-1])
    except json.JSONDecodeError as exc:
        return "tool_call", {}, f"invalid_json: {exc.msg}"

    if not isinstance(payload, dict):
        return "tool_call", {}, "json_not_object"

    behavior = payload.get("behavior", "tool_call")
    if behavior not in VALID_BEHAVIORS:
        return "tool_call", {}, f"invalid_behavior: {behavior}"

    answer = payload.get("answer", {})
    if not isinstance(answer, dict):
        return behavior, {}, "answer_not_object"

    return behavior, answer, None


def parse_step(text: str) -> ParsedStep:
    """一步输出的完整解析。工具调用优先于终答。"""
    body, had_thinking = strip_thinking(text)

    tool_calls = parse_tool_calls(body)
    if tool_calls:
        return ParsedStep(kind="tool_calls", tool_calls=tool_calls,
                          had_thinking=had_thinking, raw_text=text)

    behavior, answer, error = parse_final_answer(body)
    if error is not None:
        return ParsedStep(kind="error", error=error, had_thinking=had_thinking, raw_text=text)

    return ParsedStep(kind="final", behavior=behavior, answer=answer,
                      had_thinking=had_thinking, raw_text=text)


# --------------------------------------------------------------------------
# 约束解码用的 JSON Schema
# --------------------------------------------------------------------------


def final_answer_schema(answer_keys: list[str]) -> dict[str, Any]:
    """终答的 JSON Schema，喂给 vLLM 的 guided_json（xgrammar）。

    ★ 这是选 JSON 而不是自定义标签的最大回报：一旦开了约束解码，
    「格式错误」这个失败模式**从根上消失**，format 就不用占 reward 权重了，
    全部权重可以给真正的任务能力。小模型最容易死在格式上，堵死这条能显著提升有效能力。
    """
    return {
        "type": "object",
        "properties": {
            "behavior": {"type": "string", "enum": sorted(VALID_BEHAVIORS)},
            "answer": {
                "type": "object",
                "properties": {key: {} for key in answer_keys},
                "required": answer_keys,
            },
        },
        "required": ["behavior", "answer"],
    }


def render_tool_call(name: str, arguments: dict[str, Any]) -> str:
    """反向渲染：构造 SFT 训练数据时，把 gold action 写成模型该输出的样子。"""
    payload = json.dumps({"name": name, "arguments": arguments}, ensure_ascii=False)
    return f"<tool_call>\n{payload}\n</tool_call>"


def render_final_answer(behavior: str, answer: dict[str, Any]) -> str:
    payload = json.dumps({"behavior": behavior, "answer": answer}, ensure_ascii=False, indent=2)
    return f"```json\n{payload}\n```"
