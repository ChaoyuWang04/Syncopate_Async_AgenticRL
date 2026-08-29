"""v15 解析器：文本 → 工具调用 / 信令 / 纯文本终答。

和 v14（`parsing.py`）的关系：**并存，不改它一个字**。
v14 那份要留着重放历史审计（`25 §3.3` 第 1 行），切换靠 `core/contract.py`。

三个通道（`25 §3.1`）：
    <think>…</think>                      思考段，解析时剥掉
    <tool_call>{name, arguments}</tool_call>  业务工具 或 session.* 信令
    其余纯文本                              终答段（给人读的话）

★ 行为推导是**轨迹级**的，不是单步级的 —— 这是最容易写错的地方：
    「有业务工具 + 纯文本收尾 → tool_call」这条，光看最后一步是判不出来的
    （最后一步只是一段纯文本，得知道**这条轨迹之前调过业务工具**）。
    ⇒ 单步解析只报 kind，行为由 `derive_behavior()` 在轨迹层算。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from syncopate.core.contract import REPORT_TOOL, SESSION_TOOL_NAMES, TERMINAL_SIGNALS

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
# v14 壳的指纹。v15 下出现 = 壳残留，必须被抓成错误而不是被宽容地吃掉。
_SHELL_RE = re.compile(r"```(?:json)?\s*\{.*?\"behavior\"\s*:.*?\}\s*```", re.DOTALL)

VALID_BEHAVIORS = frozenset({"tool_call", "clarify", "reject", "defer", "answer"})


@dataclass
class ParsedStepV15:
    """一步输出的解析结果。

    kind 四选一：
        "tool_calls"  调了业务工具（含 session.report）⇒ 轨迹继续
        "signal"      调了终止性 session.*（defer/clarify/reject）⇒ 轨迹结束
        "final_text"  纯文本终答 ⇒ 轨迹结束（行为由轨迹级推导）
        "error"       格式非法（空终答 / 坏 JSON / 壳残留 / 混合形态）
    """

    kind: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    signal: str | None = None            # defer / clarify / reject
    signal_args: dict[str, Any] = field(default_factory=dict)
    text: str = ""                       # 剥掉 think 与 tool_call 之后的纯文本
    error: str | None = None
    had_thinking: bool = False
    thinking_text: str = ""
    raw_text: str = ""

    @property
    def ok(self) -> bool:
        return self.kind != "error"


def strip_thinking(text: str) -> tuple[str, bool, str]:
    """剥掉 <think> 块，返回 (剩余文本, 是否出现过, 思考内容)。

    思考内容要**留下来**：N3「按需思考」的触发率就是数它非空的比例。
    """
    blocks = _THINK_RE.findall(text)
    stripped = _THINK_RE.sub("", text)
    inner = "\n".join(re.sub(r"^<think>|</think>$", "", b) for b in blocks)
    return stripped, bool(blocks), inner.strip()


def _loads_tolerant(payload: str) -> Any:
    """宽容 JSON：只修"结尾多一个逗号"。单引号不修 —— 那是明确的格式错误。"""
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        pass
    return json.loads(re.sub(r",\s*([}\]])", r"\1", payload))


def parse_tool_calls(text: str) -> tuple[list[dict[str, Any]], int]:
    """抽出全部 <tool_call>，返回 (调用列表, 畸形块数)。

    ⚠️ 与 v14 的差别：v14 把畸形块**静默丢弃**；v15 **数出来**并让上层判错。
    静默丢弃会让「模型吐了个坏 tool_call」看起来像「模型没调工具」——
    两者的修法完全不同（前者是格式问题，后者是决策问题）。
    """
    calls: list[dict[str, Any]] = []
    malformed = 0
    for block in _TOOL_CALL_RE.findall(text):
        try:
            payload = _loads_tolerant(block)
        except json.JSONDecodeError:
            malformed += 1
            continue
        if not isinstance(payload, dict):
            malformed += 1
            continue
        name = payload.get("name")
        if not isinstance(name, str) or not name:
            malformed += 1
            continue
        args = payload.get("arguments")
        if isinstance(args, str):
            try:
                args = _loads_tolerant(args)
            except json.JSONDecodeError:
                args = {}
        calls.append({"name": name, "arguments": args if isinstance(args, dict) else {}})
    return calls, malformed


def parse_step_v15(text: str) -> ParsedStepV15:
    """单步解析。工具调用优先于纯文本；终止性信令优先于业务工具。"""
    body, had_thinking, thinking = strip_thinking(text)
    calls, malformed = parse_tool_calls(body)
    residue = bool(_SHELL_RE.search(body))
    leftover = _TOOL_CALL_RE.sub("", body).strip()

    def _mk(kind: str, **kw):
        return ParsedStepV15(kind=kind, had_thinking=had_thinking,
                             thinking_text=thinking, raw_text=text, **kw)

    if malformed:
        return _mk("error", error=f"malformed_tool_call x{malformed}", tool_calls=calls)
    if residue:
        # 壳残留：v15 下这是契约回潮，必须报错而不是照着壳解析
        return _mk("error", error="shell_residue", tool_calls=calls)

    signals = [c for c in calls if c["name"] in TERMINAL_SIGNALS]
    if len(signals) > 1:
        return _mk("error", error="multi_terminal_signal", tool_calls=calls)
    if signals:
        others = [c for c in calls if c["name"] not in TERMINAL_SIGNALS]
        if others:
            # 混合形态：一轮里既发信令又调业务工具（`25 §6③`）
            return _mk("error", error="mixed_signal_and_tool", tool_calls=calls)
        s = signals[0]
        return _mk("signal", tool_calls=calls, signal=TERMINAL_SIGNALS[s["name"]],
                   signal_args=s["arguments"], text=leftover)
    if calls:
        return _mk("tool_calls", tool_calls=calls, text=leftover)
    if not leftover:
        # 空文本终答 —— v14 的老病（空头支票），v15 明确判错
        return _mk("error", error="empty_final_text")
    return _mk("final_text", text=leftover)


def derive_behavior(*, terminal: ParsedStepV15, business_tools_used: bool) -> str:
    """★ 轨迹级行为推导（`25 §3.1`）—— verifier 的行为闸从这里取值。

        调了终止性 session.*        → defer / clarify / reject
        纯文本收尾 + 调过业务工具    → tool_call（办了事）
        纯文本收尾 + 一个工具没调    → answer
    """
    if terminal.kind == "signal":
        assert terminal.signal in VALID_BEHAVIORS, terminal.signal
        return terminal.signal
    if terminal.kind == "final_text":
        return "tool_call" if business_tools_used else "answer"
    raise ValueError(f"非终止步不能推导行为：kind={terminal.kind}")


def render_signal(name: str, arguments: dict[str, Any]) -> str:
    """反向渲染（造 gold 用）。与 v14 的 render_tool_call 同形。"""
    assert name in SESSION_TOOL_NAMES, name
    payload = json.dumps({"name": name, "arguments": arguments}, ensure_ascii=False)
    return f"<tool_call>\n{payload}\n</tool_call>"


def render_report(fields: dict[str, Any]) -> str:
    """机器可核字段的结构化通道（承接 v14 的 answer_fields）。"""
    return render_signal(REPORT_TOOL, fields)
