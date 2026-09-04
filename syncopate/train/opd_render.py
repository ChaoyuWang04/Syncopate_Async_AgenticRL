"""OPD 的 prompt 渲染与 NL/工具分段（09-05 从 scripts/probe_opd_divergence.py 搬入正规模块：训练入口不再 import 探针脚本）。

render_prompt_text：与 runtime decider 同一条渲染路径（system.txt + step_user.txt）；reference_now = 今天（与 decider 同，此前写死 2026-08-20），
context 空（裁定⑨：account_id 运行态注入，不进题面）。
segment_text：按 offset_mapping 把字符级标注映射到 token（只有 reply 值可蒸）；segment_tokens 仅存档 O-1 口径。
"""
from __future__ import annotations

import datetime as _dt

from syncopate.prompts import load_prompt, render_prompt
from syncopate.train.rollout_loop import CHAT_TEMPLATE_KWARGS


def render_prompt_text(tokenizer, user_message: str, tools, reference_now: str | None = None) -> str:
    """与 `decider._messages` 同一条渲染路径（不是运行时那条路径就不算量过）。"""
    messages = [
        {"role": "system", "content": load_prompt("system.txt")},
        {"role": "user", "content": render_prompt("step_user.txt", {
            "reference_now": reference_now or _dt.date.today().isoformat(),
            "context": {},
            "user_message": user_message,
            "answer_fields": [{"key": "summary", "description": "本次任务的结论"}],
        })},
    ]
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


def segment_text(tokenizer, text: str) -> tuple[list[int], list[str]]:
    """★ 生产版分段（U 路 P0-3 修复，2026-08-28）：按 offset_mapping 把字符级标注
    映射到 token。⚠️ 老的 `segment_tokens` 对 BPE 表面形式（CJK 的 byte-level
    token 如 'ĠæŁ¥'）会整体错位——`"".join(tokens)` 不是原文，len(token)≠字符数，
    众数全塌 format ⇒ NL mask 恒空 ⇒ 蒸馏恒零（"机制在但没接上"候选，被 P0-3
    20 条抽检当场抓获）。训练侧一律用本函数；segment_tokens 仅存档 O-1 口径。"""
    from collections import Counter

    enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    ch = _char_labels(text, text_value_keys={"reply"})   # ★ 只有 reply 值可蒸（O-2a 语义：
    #   summary=机器值、tool/arguments/missing_field=工具段红线、rationale 先保守排除）
    labels = []
    for a, b in enc["offset_mapping"]:
        span = ch[a:b]
        labels.append(Counter(span).most_common(1)[0][0] if span else "format")
    return enc["input_ids"], labels
