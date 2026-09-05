"""E-think（26 §W4′）：CoT 开/关 A/B 的判读——读两份 eval_local 审计 json，按预注册的 J1–J6 出表。

    python -m syncopate.evaluation.think_ab_report _audit/v16/eval/think_off.json _audit/v16/eval/think_on.json [--out _audit/v16/eval/think_ab.md]

J1 总均分配对差 B−A（bootstrap 95% CI）· J2 区分度子集（合并均分 ∈[0.2,0.8]）配对差 · J3 零梯度构成 · J4 有效性
（B 臂 trunc_tokens ≤10%、parse_ok 差 ≤5pp）· J5 语言（B 臂 think cjk<0.5 占比；两臂终答 cjk 中位差）· J6 成本（步数、墙钟）。
决策规则照 26 原文，不在这里改。
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path


def _ci(diffs, n=4000, seed=0):
    rng = random.Random(seed); m = len(diffs)
    if m == 0:
        return (float("nan"), float("nan"))
    bs = sorted(statistics.mean(rng.choice(diffs) for _ in range(m)) for _ in range(n))
    return (bs[int(0.025 * n)], bs[int(0.975 * n)])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("off"); ap.add_argument("on")
    ap.add_argument("--out", default="_audit/v16/eval/think_ab.md")
    a = ap.parse_args()
    A = json.load(open(a.off)); B = json.load(open(a.on))
    ra = {r["case_id"]: r for r in A["rows"]}; rb = {r["case_id"]: r for r in B["rows"]}
    ids = sorted(set(ra) & set(rb))
    L = [f"# E-think · CoT 开/关 A/B 判读", "", f"A={a.off}（{len(ra)} 题）· B={a.on}（{len(rb)} 题）· 配对 {len(ids)} 题 · 每题 {A['gen']['samples_per_case']}/{B['gen']['samples_per_case']} 样本", ""]
    d = [rb[c]["reward"] - ra[c]["reward"] for c in ids]
    lo, hi = _ci(d)
    ma, mb = statistics.mean(ra[c]["reward"] for c in ids), statistics.mean(rb[c]["reward"] for c in ids)
    L += [f"## J1 总均分：A {ma:.3f} · B {mb:.3f} · Δ(B−A) {mb-ma:+.3f}  95%CI [{lo:+.3f}, {hi:+.3f}]", ""]
    disc = [c for c in ids if 0.2 <= (ra[c]["reward"] + rb[c]["reward"]) / 2 <= 0.8]
    dd = [rb[c]["reward"] - ra[c]["reward"] for c in disc]
    lo2, hi2 = _ci(dd)
    wins = sum(x > 0.01 for x in dd); losses = sum(x < -0.01 for x in dd)
    L += [f"## J2 区分度子集（合并均分∈[0.2,0.8]）：n={len(disc)}{'（<30 ⇒ 不可判）' if len(disc) < 30 else ''} · Δ(B−A) {statistics.mean(dd) if dd else float('nan'):+.3f}  95%CI [{lo2:+.3f}, {hi2:+.3f}] · B 胜/负/平 {wins}/{losses}/{len(dd)-wins-losses}", ""]
    def zg(rows):
        live = sum(r["reward_std"] > 0.01 for r in rows); sat = sum(r["reward_std"] <= 0.01 and r["reward"] > 0.9 for r in rows)
        dead = sum(r["reward_std"] <= 0.01 and r["reward"] < 0.15 for r in rows); stuck = len(rows) - live - sat - dead
        return live, sat, dead, stuck
    za, zb = zg([ra[c] for c in ids]), zg([rb[c] for c in ids])
    L += [f"## J3 零梯度构成（有梯度/饱和/全灭/卡死）：A {za} · B {zb} · 有梯度 Δ {zb[0]-za[0]:+d}（{(zb[0]-za[0])/max(1,len(ids)):+.1%} 题）", ""]
    tb = statistics.mean(rb[c].get("trunc_tokens", 0) for c in ids); ta = statistics.mean(ra[c].get("trunc_tokens", 0) for c in ids)
    pa = statistics.mean(ra[c]["parse_ok"] for c in ids); pb = statistics.mean(rb[c]["parse_ok"] for c in ids)
    j4 = tb <= 0.10 and abs(pa - pb) <= 0.05
    L += [f"## J4 有效性：B 臂 trunc_tokens {tb:.1%}（A {ta:.1%}）· parse_ok A {pa:.1%} / B {pb:.1%} ⇒ {'✅' if j4 else '🔴 结论无效'}", ""]
    def med(xs): xs = [x for x in xs if x is not None]; return statistics.median(xs) if xs else float("nan")
    tk = [rb[c].get("think_cjk_med") for c in ids if rb[c].get("think_steps_mean", 0) > 0]
    th_en = sum(1 for x in tk if x is not None and x < 0.5) / max(1, len([x for x in tk if x is not None]))
    rep_a = med(ra[c].get("reply_cjk_med") for c in ids); rep_b = med(rb[c].get("reply_cjk_med") for c in ids)
    L += [f"## J5 语言：B 臂有思考的题 {len(tk)} · think cjk<0.5 占 {th_en:.0%} · think 字数中位 {med(rb[c].get('think_chars_med') for c in ids):.0f} · 终答 cjk 中位 A {rep_a:.2f} / B {rep_b:.2f}（差 {rep_b-rep_a:+.2f}）", ""]
    sa = statistics.median(ra[c]["num_steps"] for c in ids); sb = statistics.median(rb[c]["num_steps"] for c in ids)
    L += [f"## J6 成本：步数中位 A {sa:.1f} / B {sb:.1f}（墙钟看探针 json 的 wall_secs）", ""]
    # 按族
    fams = sorted({c.split('_')[0] for c in ids})
    L += ["## 按族（A 均分 / B 均分 / Δ / n）", "", "| 族 | A | B | Δ | n |", "|---|---|---|---|---|"]
    for f in fams:
        cs = [c for c in ids if c.startswith(f + "_")]
        xa = statistics.mean(ra[c]["reward"] for c in cs); xb = statistics.mean(rb[c]["reward"] for c in cs)
        L.append(f"| {f} | {xa:.3f} | {xb:.3f} | {xb-xa:+.3f} | {len(cs)} |")
    L += ["", "## 决策规则（26 原文）：J4 过 ∧（J2 B 显著更好 ∨ J3 有梯度 +≥10%）∧ J1 CI 上界 > −0.05 ⇒ 默认开 CoT、不设语言闸",
          f"- J2 B 显著更好：{'✅' if len(disc) >= 30 and lo2 > 0 else '✗'} · J3 有梯度 +≥10%：{'✅' if (zb[0]-za[0]) >= 0.10*len(ids) else '✗'} · J1 CI 上界 > −0.05：{'✅' if hi > -0.05 else '✗'} · J4：{'✅' if j4 else '✗'}"]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); Path(a.out).write_text("\n".join(L))
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
