"""OPD 的 prompt 渲染与 NL/工具分段（09-05 从 scripts/v16/probe_opd_divergence.py 搬入正规模块：训练入口不再 import 探针脚本）。

render_prompt_text：与 runtime decider 同一条渲染路径（system.txt + step_user.txt）；reference_now = 今天（与 decider 同，此前写死 2026-08-20），
context 空（裁定⑨：account_id 运行态注入，不进题面）。
segment_text：按 offset_mapping 把 v15 的 think / tool / 纯自然语言三段映射到 token；
只有纯自然语言终答可蒸。旧 JSON reply 白名单只属于 v14 历史，不再用于当前 OPD。
"""
from __future__ import annotations

import datetime as _dt

from syncopate.prompts import load_system_prompt, render_prompt
from syncopate.train.rollout_loop import CHAT_TEMPLATE_KWARGS


def render_prompt_text(tokenizer, user_message: str, tools, reference_now: str | None = None,
                       prior: list[dict] | None = None) -> str:
    """与 `decider._messages` 同一条渲染路径（不是运行时那条路径就不算量过）。"""
    from syncopate.core.contract import visible_answer_fields
    from syncopate.core.prior_turns import render_prior_messages

    messages = [
        {"role": "system", "content": load_system_prompt()},
    ]
    if prior:
        messages += render_prior_messages(prior, tokenizer)
    messages.append({"role": "user", "content": render_prompt("step_user.txt", {
        "reference_now": reference_now or _dt.date.today().isoformat(),
        "context": {},
        "user_message": user_message,
        "answer_fields": visible_answer_fields([]),
    })})
    return tokenizer.apply_chat_template(
        messages, tools=tools, add_generation_prompt=True, tokenize=False,
        **CHAT_TEMPLATE_KWARGS)


# `behavior` 的值是枚举（tool_call/answer/defer…），不是自然语言 —— 算格式段。
_ENUM_KEYS = {"behavior"}


def _char_labels(text: str, text_value_keys: set | None = None) -> list[str]:
    """字符级标注：tool / text / format。

    ⚠️ **先字符级再映射到 token**，不要逐 token 跑状态机 ——
      我第一版就是逐 token 的滑窗判断 ```，同一个围栏被连续几个 token 重复检测、
      状态来回翻转 ⇒ **一个 text 标签都没打出来**（判据又空了，第三次）。
    """
    labels: list[str] = ["format"] * len(text)
    i = 0
    in_tool = in_string = escape = is_value = False
    last_sig: str | None = None      # 上一个非空白的结构字符
    key_start = 0
    cur_key = ""
    while i < len(text):
        if not in_tool and text.startswith("<tool_call>", i):
            in_tool = True
        if in_tool:
            labels[i] = "tool"
            if text.startswith("</tool_call>", i):
                for j in range(i, min(len(text), i + 12)):
                    labels[j] = "tool"
                i += 12
                in_tool = False
                continue
            i += 1
            continue
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
                labels[i] = "format"
                if not is_value:                  # 刚闭合的是**键名**
                    cur_key = text[key_start:i]
                i += 1
                continue
            if text_value_keys is None:            # O-1 旧口径（存档）
                is_text = is_value and cur_key not in _ENUM_KEYS
            else:                                   # U 路白名单口径：只蒸指定键的值
                is_text = is_value and cur_key in text_value_keys
            labels[i] = "text" if is_text else "format"
        else:
            if ch == '"':
                in_string = True
                is_value = last_sig == ":"
                key_start = i + 1
            elif not ch.isspace():
                last_sig = ch
        i += 1
    return labels


def segment_tokens(tokens: list[str]) -> list[str]:
    """把字符级标注映射回 token（取该 token 跨度里的众数标签）。"""
    from collections import Counter

    text = "".join(tokens)
    ch = _char_labels(text)
    out, pos = [], 0
    for t in tokens:
        span = ch[pos:pos + len(t)]
        pos += len(t)
        out.append(Counter(span).most_common(1)[0][0] if span else "format")
    return out


def _mark_block(labels: list[str], text: str, opening: str, closing: str,
                label: str) -> None:
    """Mark all complete blocks and conservatively mask an unclosed block."""
    pos = 0
    while True:
        start = text.find(opening, pos)
        if start < 0:
            return
        end = text.find(closing, start + len(opening))
        stop = len(text) if end < 0 else end + len(closing)
        labels[start:stop] = [label] * (stop - start)
        pos = stop


def _looks_like_legacy_contract_shell(segment: str) -> bool:
    """v14 JSON/code-fence answers are not v15 natural-language targets."""
    import re

    stripped = segment.strip()
    if not stripped:
        return False
    if stripped.startswith("```"):
        return True
    if stripped[0] not in "[{":
        return False
    return bool(re.search(
        r'''["'](?:behavior|answer|reply|summary|signal|arguments|tool|name)["']\s*:''',
        stripped,
    ))


def v15_char_labels(text: str, *, implicit_think_open: bool = False) -> list[str]:
    """Label current-contract output as ``think`` / ``tool`` / ``text`` / ``format``.

    Qwen3.5's generation prompt already contains ``<think>\n``.  Generated tokens
    therefore commonly start with the reasoning body and only contain the closing
    ``</think>`` tag; that implicit-open case must be masked as carefully as a full
    think block.
    """
    labels = ["text"] * len(text)

    first_close = text.find("</think>")
    first_open = text.find("<think>")
    if first_close >= 0 and (first_open < 0 or first_close < first_open):
        stop = first_close + len("</think>")
        labels[:stop] = ["think"] * stop
    elif implicit_think_open and first_open < 0:
        # With Qwen thinking enabled, ``<think>\n`` is already in the generation
        # prompt.  If generation never closes it, the whole response is unfinished
        # reasoning, not a natural-language terminal answer.
        labels[:] = ["think"] * len(text)
    _mark_block(labels, text, "<think>", "</think>", "think")
    _mark_block(labels, text, "<tool_call>", "</tool_call>", "tool")
    # Defensive support for a malformed Qwen3.5 XML call missing its outer wrapper.
    _mark_block(labels, text, "<function=", "</function>", "tool")

    # Inspect each outside-block region separately.  A v14 JSON shell is not
    # partially blessed merely because its inner `reply` string sounds human.
    i = 0
    while i < len(text):
        if labels[i] != "text":
            i += 1
            continue
        j = i + 1
        while j < len(text) and labels[j] == "text":
            j += 1
        segment = text[i:j]
        if not segment.strip() or _looks_like_legacy_contract_shell(segment):
            labels[i:j] = ["format"] * (j - i)
        i = j
    return labels


def segment_text(tokenizer, text: str, *,
                 implicit_think_open: bool | None = None) -> tuple[list[int], list[str]]:
    """Current v15 segmenter, mapped with tokenizer offsets rather than token text.

    The old implementation still selected only a JSON ``reply`` value.  That was
    the v14 wire format; under v15 it inverted the objective by skipping correct
    plain answers and training only legacy shells.
    """
    from collections import Counter

    if implicit_think_open is None:
        implicit_think_open = bool(CHAT_TEMPLATE_KWARGS.get("enable_thinking"))
    enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    ch = v15_char_labels(text, implicit_think_open=implicit_think_open)
    labels = []
    for a, b in enc["offset_mapping"]:
        span = ch[a:b]
        labels.append(Counter(span).most_common(1)[0][0] if span else "format")
    return enc["input_ids"], labels
