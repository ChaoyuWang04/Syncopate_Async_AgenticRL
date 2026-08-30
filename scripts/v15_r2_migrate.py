#!/usr/bin/env python
"""v15 · R2 门槛① —— v13 419 行的**语义冻结**迁移核对（`25 §R2`）。

    .venv/bin/python scripts/v15_r2_migrate.py [--out _audit/v15_r2/migrate_419.json]

⛔ 逐字节冻结在换壳后不可能（终答整段换了形态）⇒ 改**语义冻结**：同一批 gold
在 v14 / v15 两个契约开关下各回放一遍，四项逐条断言必须全等。

★ 全量 419，**不取样**。R1 的教训：判据覆盖不到的分支 = 那个分支没有判据
（前 120 条全是 tool_call ⇒ 120/120 假绿，分层后当场掉到 40%）。

★ 两个契约必须跑在**两个进程**里：contract.py 在 import 期读环境变量，
  同一进程内切不了（这不是缺陷，是"唯一真相来源"的代价）。
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

CODE = r"""
import asyncio, json
from pathlib import Path
from transformers import AutoTokenizer
from syncopate.domains.adcampaign import build_domain
from syncopate.pipeline.sft_replay import _ScriptedEngine, gold_script
from syncopate.pipeline.split import load_bundles
from syncopate.train.rollout_loop import RolloutConfig, run_rollout

tok = AutoTokenizer.from_pretrained("models/Qwen3-4B")
reg = build_domain().registry
reg.latency_scale = 0.0
bundles = {k: v for k, v in load_bundles(Path("data/batches/v13")).items() if v.gold}
import os as _os
if _os.environ.get("MIGRATE_SCOPE") == "frozen419":
    import pandas as _pd
    keep = set(_pd.read_parquet("data/sft/v13/train.parquet").case_id)
    bundles = {k: v for k, v in bundles.items() if k in keep}
out = {}
for cid, b in bundles.items():
    script = gold_script(b)
    o = asyncio.run(run_rollout(b, registry=reg, tokenizer=tok,
                                generate=_ScriptedEngine(tok, script),
                                config=RolloutConfig(max_assistant_turns=max(12, len(script) + 4)),
                                rollout_id="m", run_id="migrate"))
    t = o.trajectory
    out[cid] = {
        "behavior": t.behavior,
        "expected_behavior": b.verifier.expected_behavior,
        "actions": [[a.name, a.arguments] for a in t.business_actions],
        "final_answer": t.final_answer,
        "final_text": t.final_text,
        # 判分器真正核对的字段（required 且 value_source != "any"）——门槛①⒞ 的口径
        "machine_keys": sorted(f.key for f in b.verifier.required_answer_fields
                               if f.value_source != "any"),
        "parse_ok": t.parse_ok,
        "truncated": t.truncated,
    }
print("@@JSON@@" + json.dumps(out, ensure_ascii=False, sort_keys=True))
"""


def replay(v15: bool, scope: str) -> dict:
    env = dict(os.environ, SYNCOPATE_CONTRACT="v15" if v15 else "v14", MIGRATE_SCOPE=scope)
    if v15:
        env["SYNCOPATE_THINK"] = "1"          # v15 默认 think-on（§3.2 修法 A）
    p = subprocess.run([sys.executable, "-c", CODE], capture_output=True, text=True, env=env)
    if "@@JSON@@" not in p.stdout:
        raise SystemExit(f"🔴 子进程失败:\n{p.stdout[-3000:]}\n{p.stderr[-3000:]}")
    return json.loads(p.stdout.split("@@JSON@@", 1)[1].strip())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="_audit/v15_r2/migrate_419.json")
    ap.add_argument("--scope", choices=["frozen419", "all"], default="frozen419",
                    help="frozen419=data/sft/v13/train.parquet 的 419 个冻结 case（门槛①口径）；"
                         "all=data/batches/v13 全部 gold case（更宽的扫描）")
    args = ap.parse_args()

    print(f"回放 v14 契约…（scope={args.scope}）", flush=True)
    a = replay(v15=False, scope=args.scope)
    print("回放 v15 契约…", flush=True)
    b = replay(v15=True, scope=args.scope)

    ids = sorted(set(a) | set(b))
    bad = {"case_set": [], "actions": [], "final_answer": [], "behavior": []}
    dropped: list[dict] = []
    for cid in ids:
        if cid not in a or cid not in b:
            bad["case_set"].append(cid)
            continue
        x, y = a[cid], b[cid]
        if x["actions"] != y["actions"]:
            bad["actions"].append({"case_id": cid, "v14": x["actions"], "v15": y["actions"]})
        # ⒞ 门槛口径 = **判分器真正核对的字段**（required 且 value_source != "any"）。
        #   ⚠️ 这个口径不是我放宽的，是 25 §R2① 原文「终答**机器字段**（report 参数）」；
        #   把 gold 里那些**从没被判分器读过**的额外字段也算进来，量的是另一件事 ⇒ 单独统计。
        mk = x.get("machine_keys") or []
        fx = {k: (x["final_answer"] or {}).get(k) for k in mk}
        fy = {k: (y["final_answer"] or {}).get(k) for k in mk}
        if fx != fy:
            bad["final_answer"].append({"case_id": cid, "keys": mk, "v14": fx, "v15": fy})
        # 额外观测（不进门槛）：v14 终答里有、v15 report 里没有的**非判分字段**去哪了
        extra = {k: v for k, v in (x["final_answer"] or {}).items()
                 if k not in mk and k != "summary" and k not in (y["final_answer"] or {})}
        if extra:
            txt = (y.get("final_text") or "")
            survived = {k: (str(v) in txt) for k, v in extra.items()}
            dropped.append({"case_id": cid, "extra": extra, "in_v15_prose": survived})
        if x["behavior"] != y["behavior"]:
            bad["behavior"].append({"case_id": cid, "v14": x["behavior"], "v15": y["behavior"],
                                    "expected": x["expected_behavior"]})

    n = len(ids)
    print("\n════ R2 门槛① 语义冻结（全量，不取样）════")
    rows = [("⒜ case 集合一致", n - len(bad["case_set"]), n),
            ("⒝ 业务工具动作序逐条一致", n - len(bad["actions"]), n),
            ("⒞ 终答机器字段逐字段一致", n - len(bad["final_answer"]), n),
            ("⒟ 行为推导与 v14 一致", n - len(bad["behavior"]), n)]
    failed = 0
    for label, ok, tot in rows:
        good = ok == tot
        failed += int(not good)
        print(f"  {label:26s}: {ok}/{tot}   门槛 ={tot}/{tot}   {'✅' if good else '🔴'}")
    # ── 额外观测：非判分字段的去向（不进门槛，但必须被看见）──────────────
    if dropped:
        nkeys = sum(len(d["extra"]) for d in dropped)
        alive = sum(sum(d["in_v15_prose"].values()) for d in dropped)
        from collections import Counter
        top = Counter(k for d in dropped for k in d["extra"]).most_common(6)
        print(f"\n  ⓘ 非判分字段（v14 终答有、v15 report 无）：{len(dropped)} case / {nkeys} 字段；"
              f"其中 {alive} 个的值仍出现在 v15 人话终答里（{alive/max(1,nkeys):.0%}）")
        print(f"    最常见: {top}")
        print("    ⚠️ 这些字段判分器从没读过（不在 required_answer_fields）⇒ 不影响分数；"
              "但它们是模型以前会写出来的结构化细节 —— 要不要保留由 Chaoyu 定。")
    for k, v in bad.items():
        for item in v[:5]:
            print(f"    ✗ {k}: {item if isinstance(item, str) else json.dumps(item, ensure_ascii=False)[:300]}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"n": n, "scope": args.scope, "counts": {k: len(v) for k, v in bad.items()},
         "bad": bad, "dropped_nonscored_fields": dropped},
        ensure_ascii=False, indent=2))
    print(f"产物 → {args.out}")
    return failed


if __name__ == "__main__":
    raise SystemExit(main())
