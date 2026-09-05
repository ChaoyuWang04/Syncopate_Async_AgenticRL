#!/usr/bin/env python
"""v15 · R1 门槛⑤ 判分对拍（`25 §R1`）：同一批 gold 轨迹，新旧两路判分**逐 case 同分**。

    .venv/bin/python scripts/v15_r1_score_crosscheck.py [--n 120]

判据的立意（`25 §3.3`）：换契约换的是**取数来源**，不是**比对逻辑**。
所以同一条 gold 走两套契约，reward / 子分 / cap 应当**一模一样**。
不一样的地方就是契约悄悄改变了判分——必须逐条查明，而不是调判据去迁就。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from collections import Counter
from pathlib import Path

OUT = Path("_audit/v15_r1")


def gold_script_v15(bundle) -> list[str]:
    """R2 语义迁移器的最小原型：v14 gold → v15 契约的每步文本。"""
    from syncopate.core.parsing import render_tool_call
    from syncopate.core.parsing_v15 import render_report, render_signal

    steps = [render_tool_call(a["tool"], a.get("arguments", {})) for a in bundle.gold.actions]
    beh = bundle.verifier.expected_behavior
    fa = dict(bundle.gold.final_answer or {})
    if beh == "defer":
        steps.append(render_signal("session.defer", {
            "reason": "数据还不足以支撑结论",
            "recheck_after_days": int(fa.get("recheck_after_days", 5) or 5)}))
    elif beh == "clarify":
        mf = fa.get("missing_field") or "campaign_id"
        steps.append(render_signal("session.clarify",
                                   {"question": f"请补充 {mf} 后我再继续。", "missing_fields": [mf]}))
    elif beh == "reject":
        rr = {"unauthorized": "unauthorized", "policy": "policy"}.get(
            fa.get("reject_reason"), "out_of_scope")
        steps.append(render_signal("session.reject",
                                   {"reason_code": rr, "explanation": "无法执行该请求。"}))
    else:
        # tool_call / answer：机器字段走 session.report，再用一句人话收尾
        if fa:
            steps.append(render_report(fa))
        steps.append("已经处理完了，结论如上。")
    return steps


async def score_one(bundle, *, tokenizer, registry, v15: bool):
    from syncopate.core.verifier_engine import score_trajectory
    from syncopate.domains.adcampaign import build_domain
    from syncopate.domains.adcampaign.policies import compute_decision, score_policy
    from syncopate.pipeline.sft_replay import _ScriptedEngine, gold_script
    from syncopate.train.rollout_loop import RolloutConfig, run_rollout

    script = gold_script_v15(bundle) if v15 else gold_script(bundle)
    out = await run_rollout(
        bundle, registry=registry, tokenizer=tokenizer,
        generate=_ScriptedEngine(tokenizer, script),
        config=RolloutConfig(max_assistant_turns=max(12, len(script) + 4)),
        rollout_id="x", run_id="crosscheck")
    res = score_trajectory(bundle, out.trajectory, out.sandbox,
                           policy_scorer=score_policy, decision_fn=compute_decision)
    return {"reward": res.reward, "subscores": res.subscores,
            "caps": sorted(h.name for h in res.cap_hits),
            "behavior": out.trajectory.behavior,
            "num_steps": out.trajectory.num_steps,
            "parse_ok": out.trajectory.parse_ok}


def score_batch(case_ids: list[str], v15: bool) -> dict:
    """★ 在**独立子进程**里跑一个契约。

    ⚠️ 不用 importlib.reload：契约值在 import 时被多个模块吃进去
    （sft_replay / rollout_loop / verifier / 域注册），只 reload 其中几个会留下
    **混合状态**，制造假红 —— 实测过一次（v15 下 CHAT 被误判成 0.25，
    干净进程里其实是 1.0）。子进程 + 起手就设环境变量才是真实跑法。
    """
    import os
    import subprocess
    import sys
    code = f"""
import asyncio, json, sys
sys.path.insert(0, ".")
from pathlib import Path
from transformers import AutoTokenizer
from syncopate.domains.adcampaign import build_domain
from syncopate.domains.adcampaign.policies import compute_decision, score_policy
from syncopate.pipeline.split import load_bundles
from syncopate.pipeline.sft_replay import gold_script, _ScriptedEngine
from syncopate.train.rollout_loop import RolloutConfig, run_rollout
from syncopate.core.verifier_engine import score_trajectory
from syncopate.core.model_paths import TEST_TOKENIZER, STUDENT_MODEL, TEACHER_MODEL
tok = AutoTokenizer.from_pretrained(STUDENT_MODEL)
reg = build_domain().registry; reg.latency_scale = 0.0
bundles = load_bundles(Path("data/batches/v13"))
out = {{}}
for cid in {case_ids!r}:
    b = bundles[cid]
    script = gold_script(b)
    o = asyncio.run(run_rollout(b, registry=reg, tokenizer=tok,
                                generate=_ScriptedEngine(tok, script),
                                config=RolloutConfig(max_assistant_turns=max(12, len(script) + 4)),
                                rollout_id="x", run_id="cc"))
    r = score_trajectory(b, o.trajectory, o.sandbox,
                         policy_scorer=score_policy, decision_fn=compute_decision)
    out[cid] = {{"reward": r.reward, "subscores": r.subscores,
                "caps": sorted(h.name for h in r.cap_hits),
                "behavior": o.trajectory.behavior,
                "num_steps": o.trajectory.num_business_steps}}
print("@@JSON@@" + json.dumps(out))
"""
    env = dict(os.environ, SYNCOPATE_CONTRACT="v15" if v15 else "v14")
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    if "@@JSON@@" not in p.stdout:
        raise SystemExit(f"🔴 子进程失败:\n{p.stdout[-2000:]}\n{p.stderr[-2000:]}")
    return json.loads(p.stdout.split("@@JSON@@", 1)[1].strip())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150)
    args = ap.parse_args()

    from syncopate.pipeline.split import load_bundles
    allb = [b for b in load_bundles(Path("data/batches/v13")).values() if b.gold]
    # ★ 按行为分层取样：不分层的话前 N 条全是 tool_call，
    #   CHAT 的 reply 这条路径根本测不到（判据覆盖不到的分支 = 没有判据）。
    by: dict[str, list] = {}
    for b in allb:
        by.setdefault(b.verifier.expected_behavior, []).append(b)
    per = max(1, args.n // max(1, len(by)))
    ids = [b.case_id for beh in sorted(by) for b in by[beh][:per]]
    print("分层取样:", {k: min(per, len(v)) for k, v in sorted(by.items())})

    a = score_batch(ids, v15=False)
    b2 = score_batch(ids, v15=True)

    rows = [{"case_id": c, "v14": a[c], "v15": b2[c]} for c in ids if c in a and c in b2]
    same = [r for r in rows if abs(r["v14"]["reward"] - r["v15"]["reward"]) < 1e-9]
    diff = [r for r in rows if r not in same]
    sub_diff = Counter()
    for r in diff:
        for k in r["v14"]["subscores"]:
            if abs(r["v14"]["subscores"][k] - r["v15"]["subscores"][k]) > 1e-9:
                sub_diff[k] += 1
    beh_ok = sum(1 for r in rows if r["v14"]["behavior"] == r["v15"]["behavior"])
    cap_ok = sum(1 for r in rows if r["v14"]["caps"] == r["v15"]["caps"])

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "score_crosscheck.json").write_text(
        json.dumps({"n": len(rows), "diff": diff[:40]}, ensure_ascii=False, indent=2))

    print("════ R1 门槛⑤ 判分对拍（同一批 gold，新旧两路判分，各自独立子进程）════")
    print(f"样本 {len(rows)} 条（要求 ≥100）")
    print(f"逐 case 同分   : {len(same)}/{len(rows)} = {len(same)/max(1,len(rows)):.2%}"
          f"   门槛 =100%  {'✅' if not diff else '🔴'}")
    print(f"行为推导一致   : {beh_ok}/{len(rows)} = {beh_ok/max(1,len(rows)):.2%}")
    print(f"cap 命中一致   : {cap_ok}/{len(rows)} = {cap_ok/max(1,len(rows)):.2%}")
    if sub_diff:
        print(f"不同分的子分   : {dict(sub_diff)}")
        for r in diff[:6]:
            print(f"  {r['case_id']:12s} v14 r={r['v14']['reward']:.4f} caps={r['v14']['caps']} | "
                  f"v15 r={r['v15']['reward']:.4f} caps={r['v15']['caps']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
