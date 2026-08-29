#!/usr/bin/env python
"""v15 · R0 形态判定器 + 双臂评测（25 §4-R0 门槛①②a②b③④）。

    .venv/bin/python scripts/v15_r0_eval.py --certify      # 只跑负向认证（不吃 GPU）
    .venv/bin/python scripts/v15_r0_eval.py --arm shell --adapter <dir> --gpu 0
    .venv/bin/python scripts/v15_r0_eval.py --report       # 汇总两臂结果

★ 两臂「正确」的定义**不对称**，这是本判定器最容易写错的地方：
    壳臂 shell : 终答是 ```json{"behavior": X, ...}``` 且 X == 期望行为
    工具臂 tool: 期望 defer/clarify/reject ⇒ 必须调对应 session.* 工具
                 期望 answer               ⇒ 必须是纯文本终答（无任何 tool_call）

★ 判定器首用必做的两件（守则③⑬）：
    ⒜ 负向认证：手工构造错形态样本，证明它**会红**（--certify）
    ⒝ 机判结果人核抽样 ≥20 条（--report 会导出 sample_for_human_review.jsonl）
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

OUT = Path("_audit/v15_r0")
SESSION_NAMES = {"session.defer": "defer", "session.clarify": "clarify",
                 "session.reject": "reject"}

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_CODE_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def classify(text: str, arm: str) -> tuple[str, dict]:
    """输出文本 → (形态, 证据)。形态取值：
       defer / clarify / reject / answer / tool_call / shell:<behavior> / invalid_*

    刻意**不复用** core/parsing.py：那份是被测对象之一（v15 要改它），
    判定器必须独立实现，否则是拿被测物给自己打分。
    """
    body = _THINK_RE.sub("", text)
    calls = []
    for blk in _TOOL_CALL_RE.findall(body):
        try:
            p = json.loads(blk)
        except json.JSONDecodeError:
            calls.append({"name": "<unparseable>", "arguments": None})
            continue
        if isinstance(p, dict) and isinstance(p.get("name"), str):
            calls.append({"name": p["name"], "arguments": p.get("arguments")})
        else:
            calls.append({"name": "<malformed>", "arguments": None})

    sig = [c for c in calls if c["name"] in SESSION_NAMES]
    biz = [c for c in calls if c["name"] not in SESSION_NAMES and not c["name"].startswith("<")]
    bad = [c for c in calls if c["name"].startswith("<")]
    ev = {"n_tool_calls": len(calls), "signal": [c["name"] for c in sig],
          "business": [c["name"] for c in biz], "malformed": len(bad)}

    # 壳格式残留（两臂都要查：工具臂出现 = 契约没搬干净）
    shell_behavior = None
    for blk in _CODE_BLOCK_RE.findall(body):
        try:
            p = json.loads(blk)
        except json.JSONDecodeError:
            continue
        if isinstance(p, dict) and "behavior" in p:
            shell_behavior = p["behavior"]
    ev["shell_behavior"] = shell_behavior

    if bad:
        return "invalid_toolcall_syntax", ev
    if len(sig) > 1:
        return "invalid_multi_signal", ev
    if sig and shell_behavior is not None:
        return "invalid_mixed_shell_and_signal", ev
    if sig:
        return SESSION_NAMES[sig[0]["name"]], ev
    if arm == "shell" and shell_behavior is not None:
        return f"shell:{shell_behavior}", ev
    if shell_behavior is not None:              # 工具臂吐了壳 = 壳残留
        return f"shellresidue:{shell_behavior}", ev
    if biz:
        return "tool_call", ev
    if body.strip():
        return "answer", ev
    return "invalid_empty", ev


def is_correct(shape: str, expected: str, arm: str) -> bool:
    """★ 两臂定义不对称。"""
    if arm == "shell":
        return shape == f"shell:{expected}"
    if expected in ("defer", "clarify", "reject"):
        return shape == expected
    if expected == "answer":
        return shape == "answer"
    return shape == expected


def signal_syntax_ok(text: str) -> bool | None:
    """门槛②a：信令调用的 schema 合法性。None = 本条没有信令调用，不计入。"""
    body = _THINK_RE.sub("", text)
    req = {"session.defer": {"reason", "recheck_after_days"},
           "session.clarify": {"question", "missing_fields"},
           "session.reject": {"reason_code", "explanation"}}
    found = False
    for blk in _TOOL_CALL_RE.findall(body):
        try:
            p = json.loads(blk)
        except json.JSONDecodeError:
            return False
        if not isinstance(p, dict) or p.get("name") not in req:
            continue
        found = True
        args = p.get("arguments")
        if not isinstance(args, dict) or not req[p["name"]].issubset(args):
            return False
        if p["name"] == "session.reject" and args.get("reason_code") not in (
                "out_of_scope", "unauthorized", "policy"):
            return False
        if p["name"] == "session.defer" and not isinstance(
                args.get("recheck_after_days"), int):
            return False
    return True if found else None


# ── 负向认证：每条都必须被判成"不对"，否则判定器本身是坏的 ──────────────
NEGATIVE_CASES = [
    ("工具臂：只用人话说了等一等，没调工具",
     "tool", "defer", "这个数据还太新，建议再观察三天看看。", "answer"),
    ("工具臂：调错了信令（该 defer 却 reject）",
     "tool", "defer", '<tool_call>\n{"name": "session.reject", "arguments": '
     '{"reason_code": "policy", "explanation": "不行"}}\n</tool_call>', "reject"),
    ("工具臂：壳格式回潮",
     "tool", "defer", '```json\n{"behavior": "defer", "answer": {}}\n```',
     "shellresidue:defer"),
    ("工具臂：同时调两个信令",
     "tool", "defer", '<tool_call>\n{"name": "session.defer", "arguments": {}}\n</tool_call>'
     '<tool_call>\n{"name": "session.clarify", "arguments": {}}\n</tool_call>',
     "invalid_multi_signal"),
    ("工具臂：tool_call 里是坏 JSON",
     "tool", "defer", "<tool_call>\n{name: session.defer,,}\n</tool_call>",
     "invalid_toolcall_syntax"),
    ("工具臂：空输出",
     "tool", "answer", "   ", "invalid_empty"),
    ("壳臂：标签贴错",
     "shell", "defer", '```json\n{"behavior": "answer", "answer": {}}\n```', "shell:answer"),
    ("壳臂：没给壳，只有人话",
     "shell", "defer", "建议再观察三天。", "answer"),
]
# ★ 刻意分开的一类：门槛① 与 ②a **故意会给出不同判定**，两个读数各管一件事。
#   ① 行为表达正确率 = 「动作选对了没有」；②a 信令语法合法率 = 「参数填对了没有」。
#   合并成一条会丢掉信息（分不清"选错动作"和"选对了但填马虎"），
#   分开则 ②a 的 ≥99% 底线正好防住「①赢得很空洞」。
#   ⚠️ 这条最初被我误编进负向清单、认证当场判红——**保留它并断言两个读数都对**，
#      而不是删掉它让认证变绿（那就是"为了达标放宽判据"）。
SPLIT_CASES = [
    ("信令选对但缺必填参数：① 应判对 · ②a 应判红", "tool", "defer",
     '<tool_call>\n{"name": "session.defer", "arguments": {"reason": "太新"}}\n</tool_call>',
     "defer", True, False),
]
POSITIVE_CASES = [
    ("工具臂：正确 defer", "tool", "defer",
     '<tool_call>\n{"name": "session.defer", "arguments": {"reason": "数据太新", '
     '"recheck_after_days": 3}}\n</tool_call>', "defer"),
    ("工具臂：正确 answer（纯文本）", "tool", "answer", "ROAS 是广告花的钱换回多少收入。", "answer"),
    ("壳臂：正确 defer", "shell", "defer",
     '```json\n{"behavior": "defer", "answer": {"recheck_after_days": 3}}\n```', "shell:defer"),
]


def certify() -> int:
    print("═══ 负向认证：以下每条都必须判『不对』 ═══")
    bad = 0
    for name, arm, exp, text, want_shape in NEGATIVE_CASES:
        shape, _ = classify(text, arm)
        ok = is_correct(shape, exp, arm)
        shape_ok = shape == want_shape
        flag = "✅会红" if not ok else "🔴 没红（判定器有病）"
        sm = "" if shape_ok else f"  ⚠️形态判成 {shape}，预期 {want_shape}"
        print(f"  {flag}  {name}{sm}")
        bad += int(ok) + int(not shape_ok)
    print("\n═══ 正向对照：以下每条都必须判『对』 ═══")
    for name, arm, exp, text, want_shape in POSITIVE_CASES:
        shape, _ = classify(text, arm)
        ok = is_correct(shape, exp, arm) and shape == want_shape
        print(f"  {'✅判对' if ok else '🔴 判错 → ' + shape}  {name}")
        bad += int(not ok)
    print("\n═══ ①与②a 分工认证（两个读数必须给出不同判定）═══")
    for name, arm, exp, text, want_shape, want_ok, want_syn in SPLIT_CASES:
        shape, _ = classify(text, arm)
        ok = is_correct(shape, exp, arm)
        syn = signal_syntax_ok(text)
        good = (shape == want_shape) and (ok is want_ok) and (syn is want_syn)
        print(f"  {'✅' if good else '🔴'} {name} → 形态={shape} ①判{'对' if ok else '错'} "
              f"②a判{'过' if syn else '红'}")
        bad += int(not good)

    print("\n═══ 语法闸②a 单独认证 ═══")
    syn = [("缺 recheck_after_days", '<tool_call>\n{"name":"session.defer","arguments":'
            '{"reason":"x"}}\n</tool_call>', False),
           ("reason_code 不在枚举内", '<tool_call>\n{"name":"session.reject","arguments":'
            '{"reason_code":"whatever","explanation":"x"}}\n</tool_call>', False),
           ("recheck_after_days 是字符串", '<tool_call>\n{"name":"session.defer","arguments":'
            '{"reason":"x","recheck_after_days":"三天"}}\n</tool_call>', False),
           ("完全合法", '<tool_call>\n{"name":"session.defer","arguments":'
            '{"reason":"x","recheck_after_days":3}}\n</tool_call>', True)]
    for name, text, want in syn:
        got = signal_syntax_ok(text)
        ok = got is want
        print(f"  {'✅' if ok else '🔴'} {name}: 判 {got}（应为 {want}）")
        bad += int(not ok)
    print()
    if bad:
        print(f"🔴 判定器负向认证不通过（{bad} 项异常）—— 不许拿它去判真数据")
        return 1
    print("✅ 判定器负向认证通过：错形态全部会红、对形态全部判对、语法闸会红")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--certify", action="store_true")
    args = ap.parse_args()
    if args.certify:
        return certify()
    ap.error("生成与汇总在 --certify 通过后接入（见 R0 runner）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
