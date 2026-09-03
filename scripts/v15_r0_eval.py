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
import collections
import json
import re
from pathlib import Path
from syncopate.core.model_paths import TEST_TOKENIZER, STUDENT_MODEL, TEACHER_MODEL

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


# ── prompt 构造：必须和训练时**逐字同源** ────────────────────────────────
SCAFFOLD_CASE = "FRESH_0002"       # OOD 题用的固定脚手架（只换 user_message）


def build_prompts(arm: str, rows: list[dict], tok, bundles) -> list[str]:
    """两臂的 prompt：A 臂原样；B 臂把 session 工具注入 <tools> 段。

    ⚠️ B 臂的注入方式必须和 `v15_r0_build.make_rows` **一模一样**（字符串替换
    `</tools>`），否则训练分布与评测分布不一致 —— R0 结论就作废了。
    spec 本身由 `assert_spec_frozen()` 守着（08-29 实案：顺手加两个 description 就会漂）。
    """
    import json as _json

    from syncopate.core.contract import SESSION_TOOL_SPECS
    from syncopate.core.tool_registry import REGISTRY
    from syncopate.train.rollout_loop import build_messages

    inject = "\n".join(_json.dumps(t["function"], ensure_ascii=False)
                        for t in SESSION_TOOL_SPECS)
    scaffold = bundles[SCAFFOLD_CASE]
    out = []
    for r in rows:
        bd = bundles.get(r["case_id"], scaffold)
        msgs = build_messages(bd, bd.case.tool_menu)
        # 分布外题：脚手架不变，只把用户那句话换掉
        msgs = [dict(m) for m in msgs]
        for m in reversed(msgs):
            if m["role"] == "user":
                m["content"] = r["user_message"]
                break
        text = tok.apply_chat_template(
            msgs, tools=REGISTRY.menu(bd.case.tool_menu),
            add_generation_prompt=True, tokenize=False, enable_thinking=False)
        if arm == "tool":
            assert "</tools>" in text, r["case_id"]
            text = text.replace("</tools>", inject + "\n</tools>", 1)
        out.append(text)
    return out


def generate(arm: str, adapter: str, gpu: int, max_turns: int = 8) -> dict:
    """跑一臂：80 道题走**完整多轮 rollout**，再判终止形态。

    ⛔ 首版做的是**单轮生成**，结果两臂在全部 7 道分布外 defer 题上都判 0% ——
       查原文才发现两臂都**正确地先调业务工具**（campaign.list / get_metrics），
       而 defer 的 gold 本来就要先查数据成熟度再决定等不等。
       单轮生成把这正确的第一步判成"形态错"，−2.5pp 的主假说读数**整个作废**。
       （「归因之前先查输入」第 N 次兑现；仪器坏了不能拿它宣布假说失败。）
    ⇒ 本版用**训练/评测那一份** `run_rollout`（N5 一份契约，不另抄一个循环），
       并发跑 80 条，靠一个批处理器把同一轮的请求合并成一次 vLLM 调用。
    """
    import asyncio
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    from pathlib import Path as P

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    import syncopate.domains.adcampaign  # noqa: F401
    from syncopate.domains.adcampaign import build_domain
    from syncopate.pipeline.split import load_bundles
    from syncopate.train.rollout_loop import RolloutConfig, run_rollout

    rows = ([dict(json.loads(l), kind_set="indist")
             for l in open("data/v15_r0/test_indist.jsonl")] +
            [dict(json.loads(l), kind_set="ood")
             for l in open("data/v15_r0/test_ood.jsonl")])
    tok = AutoTokenizer.from_pretrained(STUDENT_MODEL)
    bundles = load_bundles(P("data/batches/v13"))
    reg = build_domain().registry
    reg.latency_scale = 0.0

    llm = LLM(model="models/Qwen3-4B-sft-v14.5-epoch3", enable_lora=True,
              max_lora_rank=32, max_model_len=8192, gpu_memory_utilization=0.85,
              disable_log_stats=True)
    lora = LoRARequest(arm, 1, adapter)
    sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=512)

    # ── 批处理器：把同一轮里各条 rollout 的请求攒成一次 vLLM 调用 ──────────
    pending: list[tuple[list[int], asyncio.Future]] = []
    lock = asyncio.Lock()

    async def flusher(stop: asyncio.Event):
        while not stop.is_set():
            await asyncio.sleep(0.15)
            async with lock:
                batch, pending[:] = list(pending), []
            if not batch:
                continue
            outs = llm.generate([{"prompt_token_ids": ids} for ids, _ in batch],
                                sp, lora_request=lora, use_tqdm=False)
            for (_, fut), o in zip(batch, outs):
                if not fut.done():
                    fut.set_result(list(o.outputs[0].token_ids))

    async def gen(prompt_ids, sampling_params):
        fut = asyncio.get_running_loop().create_future()
        async with lock:
            pending.append((list(prompt_ids), fut))
        return await fut

    async def drive():
        stop = asyncio.Event()
        task = asyncio.create_task(flusher(stop))
        try:
            async def one(r):
                bd = bundles.get(r["case_id"], bundles[SCAFFOLD_CASE])
                # 分布外题：脚手架 case 不变，只把用户那句话换掉
                bd = _with_user_message(bd, r["user_message"])
                out = await run_rollout(bd, registry=reg, tokenizer=tok, generate=gen,
                                        config=RolloutConfig(max_assistant_turns=max_turns),
                                        rollout_id="r0", run_id=arm)
                return out
            return await asyncio.gather(*[one(r) for r in rows])
        finally:
            stop.set()
            await asyncio.sleep(0)
            task.cancel()

    outs = asyncio.run(drive())

    recs = []
    for r, out in zip(rows, outs):
        # ★ 必须用**原始**终答文本：final_text 在 v15 下已被剥掉 <tool_call>，
        #   拿它分类会让信令统计恒为 0（08-29 实案，`25 §7⑧`⒟）。
        text = out.trajectory.final_raw_text or out.trajectory.final_text or ""
        shape, ev = classify(text, arm)
        # 轨迹级：调过业务工具后以纯文本收尾 ⇒ tool_call（不是 answer）
        if shape == "answer" and out.trajectory.business_actions:
            shape = "tool_call"
        ev["business_tools"] = [a.name for a in out.trajectory.business_actions]
        # 对照读数：判定器（独立实现）vs 轨迹推导（被测实现）。两者不一致要人核。
        ev["derived_behavior"] = out.trajectory.behavior
        ev["truncated"] = out.trajectory.truncated
        recs.append({**r, "arm": arm, "adapter": adapter, "text": text,
                     "shape": shape, "evidence": ev,
                     "correct": is_correct(shape, r["behavior"], arm),
                     "syntax_ok": signal_syntax_ok(text)})
    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"{arm}_{P(adapter).name}"
    (OUT / f"gen_{tag}.jsonl").write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in recs))
    return summarize(recs, tag)


def _with_user_message(bundle, message: str):
    """换掉 case 的用户那句话，其余（工具菜单/沙盒/世界状态）全不动。"""
    import copy
    if bundle.case.user_message == message:
        return bundle
    b = copy.deepcopy(bundle)
    b.case.user_message = message
    return b


def summarize(recs: list[dict], tag: str) -> dict:
    def rate(sub):
        return round(sum(r["correct"] for r in sub) / max(1, len(sub)), 4)

    ind = [r for r in recs if r["kind_set"] == "indist"]
    ood = [r for r in recs if r["kind_set"] == "ood"]
    syn = [r["syntax_ok"] for r in recs if r["syntax_ok"] is not None]
    by_beh = {}
    for b in ("defer", "clarify", "reject", "answer"):
        s = [r for r in ood if r["behavior"] == b]
        if s:
            by_beh[b] = rate(s)
    # ②b 语义读数：形态不对，但人话里表达了该行为（Chaoyu 的"自然语言也该得分"）
    KW = {"defer": ["再观察", "等", "过几天", "还太新", "不够成熟", "复查"],
          "clarify": ["请问", "哪条", "补充", "需要知道", "是哪"],
          "reject": ["无法", "不能", "超出", "越权", "不支持"]}
    sem = 0
    for r in ood:
        if r["correct"] or r["behavior"] not in KW:
            continue
        if any(k in r["text"] for k in KW[r["behavior"]]):
            sem += 1
    return {"tag": tag, "n": len(recs),
            "indist_correct": rate(ind), "ood_correct": rate(ood),
            "ood_by_behavior": by_beh,
            "signal_syntax_ok": round(sum(syn) / max(1, len(syn)), 4) if syn else None,
            "signal_calls_seen": len(syn),
            "semantic_but_wrong_shape_ood": sem,
            "shapes": dict(collections.Counter(r["shape"] for r in recs))}


def selfcheck_on_gold(arm: str, per: int = 6) -> int:
    """★ 闭环自检：把 **gold 轨迹**喂给判定器，形态必须 100% 判对。

    这是"判据能不能对自己失败"的正向那一半（负向认证是 --certify）。
    ⛔ 2026-08-29 教训：R0 评测连着三次读数作废，第三次的根因是
       判定器拿的是**被剥掉 <tool_call> 的** final_text ⇒ 信令统计恒为 0，
       而 --certify 只用手写字符串测过判定器，从没端到端喂过一条真轨迹。
       ⇒ 负向认证 + 闭环自检**两个都要**：一个证明它会红，一个证明它认得出对的。
    """
    import asyncio
    from pathlib import Path as P

    from transformers import AutoTokenizer

    from syncopate.domains.adcampaign import build_domain
    from syncopate.pipeline.sft_replay import _ScriptedEngine, gold_script
    from syncopate.pipeline.split import load_bundles
    from syncopate.train.rollout_loop import RolloutConfig, run_rollout

    tok = AutoTokenizer.from_pretrained(STUDENT_MODEL)
    reg = build_domain().registry
    reg.latency_scale = 0.0
    by: dict[str, list] = {}
    for b in load_bundles(P("data/batches/v13")).values():
        if b.gold:
            by.setdefault(b.verifier.expected_behavior, []).append(b)
    ok = tot = 0
    bad = []
    for beh in sorted(by):
        for b in by[beh][:per]:
            out = asyncio.run(run_rollout(
                b, registry=reg, tokenizer=tok,
                generate=_ScriptedEngine(tok, gold_script(b)),
                config=RolloutConfig(max_assistant_turns=14),
                rollout_id="sc", run_id="selfcheck"))
            raw = out.trajectory.final_raw_text or ""
            shape, _ = classify(raw, arm)
            if shape == "answer" and out.trajectory.business_actions:
                shape = "tool_call"
            good = is_correct(shape, beh, arm)
            tot += 1
            ok += int(good)
            if not good:
                bad.append((b.case_id, beh, shape))
    print(f"★ 闭环自检（gold 轨迹形态判定）: {ok}/{tot} = {ok/max(1,tot):.1%}  应为 100%")
    for x in bad[:8]:
        print("    ✗", x)
    return 0 if ok == tot else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--certify", action="store_true")
    ap.add_argument("--arm", choices=["shell", "tool"])
    ap.add_argument("--adapter")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()
    if args.certify:
        return certify()
    if args.selfcheck:
        return selfcheck_on_gold(args.arm or "tool")
    if args.arm and args.adapter:
        if certify() != 0:
            return 1
        if selfcheck_on_gold(args.arm) != 0:
            print("🔴 闭环自检没过 —— 判定器认不出 gold，不许拿它去判模型输出")
            return 1
        s = generate(args.arm, args.adapter, args.gpu)
        print(json.dumps(s, ensure_ascii=False, indent=2))
        (OUT / f"summary_{s['tag']}.json").write_text(
            json.dumps(s, ensure_ascii=False, indent=2))
        return 0
    ap.error("要么 --certify，要么 --arm/--adapter")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
