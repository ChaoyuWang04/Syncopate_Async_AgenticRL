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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from syncopate.domains.adcampaign import build_domain
    from syncopate.pipeline.split import load_bundles
    tok = AutoTokenizer.from_pretrained("models/Qwen3-4B")
    registry = build_domain().registry
    registry.latency_scale = 0.0
    bundles = [b for b in load_bundles(Path("data/batches/v13")).values() if b.gold]
    bundles = bundles[: args.n]

    rows, errs = [], []
    for b in bundles:
        try:
            os.environ["SYNCOPATE_CONTRACT"] = "v14"
            import importlib, syncopate.core.contract as C
            importlib.reload(C)
            a = asyncio.run(score_one(b, tokenizer=tok, registry=registry, v15=False))
            os.environ["SYNCOPATE_CONTRACT"] = "v15"
            importlib.reload(C)
            import syncopate.train.rollout_loop as RL
            importlib.reload(RL)
            bb = asyncio.run(score_one(b, tokenizer=tok, registry=registry, v15=True))
        except Exception as exc:
            errs.append({"case_id": b.case_id, "error": f"{type(exc).__name__}: {exc}"})
            continue
        finally:
            os.environ["SYNCOPATE_CONTRACT"] = "v14"
            import importlib, syncopate.core.contract as C2
            importlib.reload(C2)
            import syncopate.train.rollout_loop as RL2
            importlib.reload(RL2)
        rows.append({"case_id": b.case_id, "expected": b.verifier.expected_behavior,
                     "v14": a, "v15": bb})

    same_reward = [r for r in rows if abs(r["v14"]["reward"] - r["v15"]["reward"]) < 1e-9]
    diff = [r for r in rows if r not in same_reward]
    sub_diff = Counter()
    for r in diff:
        for k in r["v14"]["subscores"]:
            if abs(r["v14"]["subscores"][k] - r["v15"]["subscores"][k]) > 1e-9:
                sub_diff[k] += 1
    beh_ok = sum(1 for r in rows if r["v14"]["behavior"] == r["v15"]["behavior"])

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "score_crosscheck.json").write_text(json.dumps(
        {"n": len(rows), "errors": errs, "diff": diff[:40]}, ensure_ascii=False, indent=2))

    print("════ R1 门槛⑤ 判分对拍（同一批 gold，新旧两路判分）════")
    print(f"样本 {len(rows)} 条（要求 ≥100）  解析失败 {len(errs)} 条")
    print(f"逐 case 同分: {len(same_reward)}/{len(rows)} = "
          f"{len(same_reward)/max(1,len(rows)):.2%}   门槛 =100%  "
          f"{'✅' if len(diff) == 0 else '🔴'}")
    print(f"行为推导一致: {beh_ok}/{len(rows)} = {beh_ok/max(1,len(rows)):.2%}")
    if sub_diff:
        print(f"不同分的子分分布: {dict(sub_diff)}")
        print("前 5 条明细：")
        for r in diff[:5]:
            print(f"  {r['case_id']:12s} 期望={r['expected']:9s} "
                  f"v14 r={r['v14']['reward']:.4f} steps={r['v14']['num_steps']} | "
                  f"v15 r={r['v15']['reward']:.4f} steps={r['v15']['num_steps']}")
    if errs:
        print(f"错误样例: {errs[:3]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
