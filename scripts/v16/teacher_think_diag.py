"""v16 · 27B 教师原始思考画像（09-04 先量后动，守则⑤；Chaoyu 放行的诊断，不改任何已注册阈值）。

问题：run16 CoT 蒸馏采样 892 步只命中 12（1%），行为类探针 0/20×3。过滤链（900 token 内要有 </think> · 中文占比 ≥0.5 ·
首动作 == gold）每个丢弃原因都是静默 continue，分不出是哪一条在拦。

做法：从难例池取 --n 个 case，每 case 取 ≤3 个可采样步（首/中/末），每步向教师要 --samples 条思考，
**max_tokens 放到 --max-tokens（默认 4096）**只为量出真实长度；然后对每条样本同时算出
「按现行 900 上限会不会写完」「中文占比」「首动作是否等于 gold」，即现行链上每一道闸的通过率。

预注册判读（跑前写死，跑完不改）：
  · closed_within_900_rate < 50%  ⇒ 900 token 上限是主拦截（27B 想得比 8B 长）
  · cjk_below_0.5_rate    > 50%  ⇒ 语言闸是主拦截（教师用英文思考）
  · 两者都不成立而 action_match_rate（写完的样本里）< 30% ⇒ 教师/gold 不一致，问题在题不在闸
读数：_audit/v16/teacher_think_diag.json（聚合 + 逐样本）· _audit/v16/teacher_think_diag.md（原样 8 条给 Chaoyu 看）

    SYNCOPATE_CONTRACT=v15 SYNCOPATE_THINK=1 python scripts/v16/teacher_think_diag.py --teacher http://127.0.0.1:8210/v1 --n 20
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import json
import re
from collections import Counter
from pathlib import Path

import httpx

from syncopate.core.model_paths import STUDENT_MODEL, TEST_TOKENIZER
from syncopate.pipeline.split import DEFAULT_BATCH_DIR, DEFAULT_SPLIT_DIR


def _q(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else 0


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="http://127.0.0.1:8210/v1")
    ap.add_argument("--n", type=int, default=20, help="难例 case 数")
    ap.add_argument("--samples", type=int, default=4, help="每步采样条数")
    ap.add_argument("--max-tokens", type=int, default=4096, help="只为量真实长度；现行链的 900 上限另算")
    ap.add_argument("--steps-per-case", type=int, default=3)
    ap.add_argument("--out", default="_audit/v16")
    ap.add_argument("--arm", default="base", help="base=原样；zh_prefix=<think> 后加中文引子（量「教师能不能用中文想」，不改任何闸）")
    ap.add_argument("--zh-prefix", default="好的，我用中文把这一步想清楚。", help="zh_prefix 臂的引子（计入 think 文本）")
    args = ap.parse_args()

    from syncopate.pipeline import build_sft as B
    from syncopate.pipeline.cot_prompt import explicit_hard_prompt
    from transformers import AutoTokenizer
    from syncopate.domains.adcampaign import build_domain
    from syncopate.pipeline.build_dataset import build_sft_row
    from syncopate.pipeline.split import load_bundles
    from syncopate.core.parsing_v15 import parse_tool_calls

    tok_path = STUDENT_MODEL if Path(STUDENT_MODEL, "tokenizer.json").exists() else TEST_TOKENIZER
    tok = AutoTokenizer.from_pretrained(tok_path)
    reg = build_domain().registry; reg.latency_scale = 0.0
    bundles = load_bundles(Path(DEFAULT_BATCH_DIR))
    sft_ids = json.load(open(f"{DEFAULT_SPLIT_DIR}/sft_cases.json"))["case_ids"]
    cands = [c for c in sft_ids if c.split("_")[0] in B.HARD_FAMILIES and c in bundles]
    cands.sort(key=lambda c: -len(bundles[c].gold.actions))
    B.rng.shuffle(cands)
    cands = cands[: args.n]
    print(f"[diag] 难例池取 {len(cands)} case（族 {sorted(B.HARD_FAMILIES)}）· tokenizer={tok_path}")

    def first_action(text: str):
        if "<tool_call>" in text:
            calls, _ = parse_tool_calls(text)
            return calls[0]["name"] if calls else None
        return "__text__"

    samples: list[dict] = []
    sem = asyncio.Semaphore(48)
    async with httpx.AsyncClient(timeout=600) as client:
        async def one(ctx: str, want: str, cid: str, si: int):
            async with sem:
                B._SEED[0] += 1
                lead = args.zh_prefix if args.arm == "zh_prefix" else ""
                r = await client.post(f"{args.teacher}/completions", json={
                    "model": "t", "prompt": ctx + "<think>\n" + lead, "max_tokens": args.max_tokens,
                    "seed": B._SEED[0], "temperature": 0.7, "top_p": 0.95})
                r.raise_for_status()
                j = r.json()["choices"][0]
                g = j["text"]; fin = j.get("finish_reason")
            g = lead + g          # 引子算进 think（建库若采用此臂，成行的 think 也会含它）
            closed = "</think>" in g
            think, post = (g.split("</think>", 1) if closed else (g, ""))
            think = think.strip()
            n_tok_think = len(tok.encode(think, add_special_tokens=False))
            cjk = len(re.findall(r"[一-鿿]", think)) / max(1, len(think))
            n_seg = len([p for p in re.split(r"\n\s*\n|\n", think) if p.strip()])
            got = first_action(post) if closed else None
            # 现行链（THINK_MAX_TOKENS=900 · THINK_MAX_CHARS · THINK_MAX_SEGS · cjk≥0.5 · 首动作==gold）逐闸判
            within_900 = closed and (n_tok_think + 2) <= B.THINK_MAX_TOKENS   # 名字沿用，实际按当前 THINK_MAX_TOKENS 判
            gates = {"closed_within_900": within_900, "chars_ok": len(think) <= B.THINK_MAX_CHARS,
                     "segs_ok": n_seg <= B.THINK_MAX_SEGS, "cjk_ok": cjk >= 0.5, "action_ok": got == want}
            return {"case_id": cid, "step": si, "want": want, "got": got, "closed": closed, "finish_reason": fin,
                    "think_tokens": n_tok_think, "think_chars": len(think), "cjk": round(cjk, 3), "n_seg": n_seg,
                    "gates": gates, "pass_current_chain": all(v for k, v in gates.items() if k != "cjk_ok"),   # 09-04：语言不入链
                    "think_head": think[:600], "post_head": post[:300]}

        jobs = []
        for cid in cands:
            b = copy.deepcopy(bundles[cid])
            b.gold.final_answer = dict(b.gold.final_answer or {})
            b.gold.final_answer.setdefault("reply", "（诊断占位：终答人话不参与本次采样）")
            b.case.user_message = explicit_hard_prompt(b.case.user_message, cid)
            base = await build_sft_row(b, tokenizer=tok, registry=reg, index=0, split="train", config=None)
            full = tok.decode(list(base["input_ids"])[:base["total_length"]])
            segs = full.split(B.ASST)
            n_steps = len(segs) - 1
            eligible = [i for i in range(n_steps) if first_action(segs[i + 1]) != "session.report"]
            if not eligible:
                continue
            pick = sorted({eligible[0], eligible[len(eligible) // 2], eligible[-1]})[: args.steps_per_case]
            for si in pick:
                ctx = B.ASST.join(segs[: si + 1]) + B.ASST + "\n"
                want = first_action(segs[si + 1])
                jobs += [one(ctx, want, cid, si) for _ in range(args.samples)]
        print(f"[diag] 发出 {len(jobs)} 条采样（max_tokens={args.max_tokens}）", flush=True)
        res = await asyncio.gather(*jobs, return_exceptions=True)
    errs = [r for r in res if isinstance(r, Exception)]
    samples = [r for r in res if not isinstance(r, Exception)]
    n = len(samples)
    closed = [s for s in samples if s["closed"]]
    agg = {
        "n_samples": n, "n_errors": len(errs), "err_examples": [repr(e)[:200] for e in errs[:3]],
        "max_tokens_used": args.max_tokens, "current_THINK_MAX_TOKENS": B.THINK_MAX_TOKENS,
        "closed_rate_at_max_tokens": round(len(closed) / max(1, n), 3),
        "closed_within_900_rate": round(sum(s["gates"]["closed_within_900"] for s in samples) / max(1, n), 3),
        "think_tokens_p50_p90_max": [_q([s["think_tokens"] for s in samples], .5), _q([s["think_tokens"] for s in samples], .9),
                                     max([s["think_tokens"] for s in samples] or [0])],
        "think_chars_p50_p90": [_q([s["think_chars"] for s in samples], .5), _q([s["think_chars"] for s in samples], .9)],
        "cjk_p50": _q([s["cjk"] for s in samples], .5),
        "cjk_below_0.5_rate": round(sum(not s["gates"]["cjk_ok"] for s in samples) / max(1, n), 3),
        "action_match_rate_among_closed": round(sum(s["gates"]["action_ok"] for s in closed) / max(1, len(closed)), 3),
        "pass_current_chain_rate": round(sum(s["pass_current_chain"] for s in samples) / max(1, n), 3),
        "mismatch_top": Counter(f"{s['want']}->{s['got']}" for s in closed if not s["gates"]["action_ok"]).most_common(8),
        "finish_reasons": dict(Counter(s["finish_reason"] for s in samples)),
    }
    # 预注册判读
    verdict = []
    if agg["closed_within_900_rate"] < 0.5: verdict.append("900 token 上限是主拦截（27B 想得比 8B 长）")
    if agg["cjk_below_0.5_rate"] > 0.5: verdict.append("语言闸是主拦截（教师非中文思考）")
    if not verdict and agg["action_match_rate_among_closed"] < 0.3: verdict.append("教师/gold 不一致：问题在题不在闸")
    if not verdict: verdict.append("单闸都不显著；看 pass_current_chain_rate 与 mismatch_top")
    agg["verdict_preregistered"] = verdict
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    agg["arm"] = args.arm
    suffix = "" if args.arm == "base" else f"_{args.arm}"
    json.dump({"agg": agg, "samples": samples}, open(out / f"teacher_think_diag{suffix}.json", "w"), ensure_ascii=False, indent=1)
    md = ["# 27B 教师原始思考画像（v16 诊断）", "", "```", json.dumps(agg, ensure_ascii=False, indent=1), "```", ""]
    for s in samples[:8]:
        md += [f"## {s['case_id']} step{s['step']} · want={s['want']} got={s['got']} · closed={s['closed']} · tokens={s['think_tokens']} · cjk={s['cjk']}",
               "", "```", s["think_head"], "…", "--- post ---", s["post_head"], "```", ""]
    (out / f"teacher_think_diag{suffix}.md").write_text("\n".join(md))
    print("[diag] " + json.dumps(agg, ensure_ascii=False))
    print(f"[diag] 判读（预注册）：{verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
