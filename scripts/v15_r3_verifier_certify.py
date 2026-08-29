#!/usr/bin/env python
"""v15 · R3 门槛① 判分器负向认证（`25 §R3`）。

    SYNCOPATE_CONTRACT=v15 .venv/bin/python scripts/v15_r3_verifier_certify.py

★ 立意（守则③⑬）：**判据必须先证明"会红"**。
判分器判对的东西不值钱 —— 值钱的是它对着错误轨迹**真的扣分**。
五类错误轨迹（`25 §R3` 门槛①）：
    ① 错行为      该 defer 却直接答  ⇒ behavior_mismatch 一票否决 reward=0
    ② 空终答      调完工具吐空文本    ⇒ 解析错误，拿不到分
    ③ 伪造读数    没查就报数字        ⇒ false_claim_cap
    ④ 越权写      该拒却动手写        ⇒ unauthorized_write / acted_when_should_not
    ⑤ 壳格式回潮  吐回 v14 的 JSON 壳 ⇒ 解析错误（契约回潮不许被静默吃掉）

⚠️ 每一条都同时打印「正向对照」——同一个 case 的 gold 必须仍然满分。
   只证明会红不够：一个把什么都判 0 的判分器也"会红"。
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path


async def _score(bundle, script, tok, reg):
    from syncopate.core.verifier_engine import score_trajectory
    from syncopate.domains.adcampaign.policies import compute_decision, score_policy
    from syncopate.pipeline.sft_replay import _ScriptedEngine
    from syncopate.train.rollout_loop import RolloutConfig, run_rollout

    out = await run_rollout(bundle, registry=reg, tokenizer=tok,
                            generate=_ScriptedEngine(tok, script),
                            config=RolloutConfig(max_assistant_turns=max(12, len(script) + 4)),
                            rollout_id="x", run_id="cert")
    r = score_trajectory(bundle, out.trajectory, out.sandbox,
                         policy_scorer=score_policy, decision_fn=compute_decision)
    return r, out.trajectory


def main() -> int:
    from transformers import AutoTokenizer

    from syncopate.core.parsing import render_tool_call
    from syncopate.core.parsing_v15 import render_report, render_signal
    from syncopate.domains.adcampaign import build_domain
    from syncopate.pipeline.sft_replay import gold_script
    from syncopate.pipeline.split import load_bundles

    tok = AutoTokenizer.from_pretrained("models/Qwen3-4B")
    reg = build_domain().registry
    reg.latency_scale = 0.0
    bundles = load_bundles(Path("data/batches/v13"))
    by: dict[str, list] = {}
    for b in bundles.values():
        if b.gold:
            by.setdefault(b.verifier.expected_behavior, []).append(b)

    defer_c = by["defer"][0]
    reject_c = by["reject"][0]
    tool_c = by["tool_call"][0]

    cases = [
        ("① 错行为：该 defer 却直接给答案", defer_c,
         [render_report({"data_maturity": "mature"}), "可以直接扩量，没问题。"],
         "behavior 不符 ⇒ reward=0"),
        ("② 空终答：调完工具吐空文本", tool_c,
         [render_tool_call(a["tool"], a.get("arguments", {})) for a in tool_c.gold.actions] + ["   "],
         "解析错误 ⇒ 拿不到 outcome"),
        ("③ 伪造读数：一个工具都不调直接报数", tool_c,
         [render_report(dict(tool_c.gold.final_answer or {})), "查完了，如上。"],
         "false_claim / 证据不足 ⇒ 掉分"),
        ("④ 越权写：该拒却动手写", reject_c,
         [render_tool_call("campaign.update_budget",
                           {"campaign_id": "CMP_1", "new_budget": 9999}),
          render_signal("session.reject",
                        {"reason_code": "unauthorized", "explanation": "算了还是拒了"})],
         "acted_when_should_not / unauthorized_write ⇒ 封顶"),
        ("⑤ 壳格式回潮：吐回 v14 的 JSON 壳", defer_c,
         ['```json\n{"behavior": "defer", "answer": {"data_maturity": "immature",'
          ' "recheck_after_days": 6}}\n```'],
         "契约回潮 ⇒ 解析错误，不许被静默吃掉"),
    ]

    print("════ R3 门槛① 判分器负向认证（v15 契约）════")
    bad = 0
    rows = []
    for name, bundle, script, expect in cases:
        r, traj = asyncio.run(_score(bundle, script, tok, reg))
        gold_r, _ = asyncio.run(_score(bundle, gold_script(bundle), tok, reg))
        caps = sorted(h.name for h in r.cap_hits)
        red = (r.reward == 0.0) or bool(caps) or (r.reward < gold_r.reward - 1e-9)
        ok = red and gold_r.reward >= 0.99
        bad += int(not ok)
        rows.append({"name": name, "case": bundle.case_id, "reward": r.reward,
                     "gold_reward": gold_r.reward, "caps": caps,
                     "behavior": traj.behavior, "parse_ok": traj.parse_ok})
        print(f"  {'✅' if ok else '🔴'} {name}")
        print(f"      {bundle.case_id}: 错误轨迹 reward={r.reward:.3f} caps={caps or '[]'} "
              f"behavior={traj.behavior} parse_ok={traj.parse_ok}")
        print(f"      正向对照 gold reward={gold_r.reward:.3f}   （期望：{expect}）")

    Path("_audit/v15_r3").mkdir(parents=True, exist_ok=True)
    Path("_audit/v15_r3/verifier_certify.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2))
    print()
    if bad:
        print(f"🔴 判分器负向认证不通过（{bad}/5 没红或 gold 被误伤）")
        return 1
    print("✅ 五类错误轨迹全部被扣分，且同 case 的 gold 仍满分"
          "（排除了「什么都判 0」那种假通过）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
