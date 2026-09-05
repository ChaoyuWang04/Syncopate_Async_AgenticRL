"""检查教师对 reject/defer/clarify 三类行为的思考是否仍选中正确动作。

任一行为低于 70% 返回 2：smoke/observe 可以留下读数后继续，candidate/strict
必须阻止下一阶段。完整结果写到显式 ``--out``，不再落进旧版本目录。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import httpx


async def async_main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20, help="每种行为抽几条 case")
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--teacher", default="http://127.0.0.1:8211/v1")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    from transformers import AutoTokenizer

    from syncopate.core.model_paths import build_tokenizer_path
    from syncopate.domains.adcampaign import build_domain
    from syncopate.pipeline import build_sft as builder
    from syncopate.pipeline.build_dataset import build_sft_row
    from syncopate.pipeline.split import (
        DATA_VERSION,
        DEFAULT_BATCH_DIR,
        DEFAULT_SPLIT_DIR,
        load_split_bundles,
    )

    out = args.out or Path(f"_audit/{DATA_VERSION}/behavior_think_probe.json")
    tok = AutoTokenizer.from_pretrained(build_tokenizer_path())
    registry = build_domain().registry
    registry.latency_scale = 0.0
    bundles = load_split_bundles(Path(DEFAULT_BATCH_DIR), Path(DEFAULT_SPLIT_DIR), "sft")
    by: dict[str, list] = defaultdict(list)
    for bundle in bundles.values():
        behavior = bundle.verifier.expected_behavior
        if bundle.gold and behavior in ("reject", "defer", "clarify"):
            by[behavior].append(bundle)

    result: dict[str, dict] = {}
    async with httpx.AsyncClient(timeout=180) as client:

        async def one_think(context: str):
            builder._SEED[0] += 1
            response = await client.post(f"{args.teacher}/completions", json={
                "model": "t",
                "prompt": context + "<think>\n",
                "max_tokens": builder.THINK_MAX_TOKENS,
                "seed": builder._SEED[0],
                "temperature": 0.7,
                "top_p": 0.95,
            })
            response.raise_for_status()
            return response.json()["choices"][0]["text"]

        for behavior, rows in sorted(by.items()):
            builder.rng.shuffle(rows)
            hit = tried = 0
            kept: list[dict] = []
            drop: Counter = Counter()
            for bundle in rows[: args.n]:
                base = await build_sft_row(
                    bundle, tokenizer=tok, registry=registry, index=0,
                    split="train", config=None,
                )
                full = tok.decode(list(base["input_ids"])[: base["total_length"]])
                segments = full.split(builder.ASST)
                want = f"session.{behavior}"
                for index in range(1, len(segments)):
                    segment = segments[index]
                    if f'"name": "{want}"' not in segment and f"<function={want}>" not in segment:
                        continue
                    context = builder.ASST.join(segments[:index]) + builder.ASST + "\n"
                    tried += 1
                    generations = await asyncio.gather(
                        *[one_think(context) for _ in range(args.samples)],
                        return_exceptions=True,
                    )
                    accepted = None
                    for generation in generations:
                        if isinstance(generation, Exception):
                            drop["exception"] += 1
                            continue
                        if "</think>" not in generation:
                            drop["no_close_think"] += 1
                            continue
                        think, post = generation.split("</think>", 1)
                        think = think.strip()
                        n_segments = len(
                            [part for part in re.split(r"\n\s*\n|\n", think) if part.strip()]
                        )
                        from syncopate.core.parsing_v15 import parse_tool_calls

                        calls, _ = parse_tool_calls(post) if "<tool_call>" in post else ([], 0)
                        name = calls[0]["name"] if calls else None
                        if not think:
                            drop["empty_think"] += 1
                        elif len(think) > builder.THINK_MAX_CHARS:
                            drop["too_long_chars"] += 1
                        elif n_segments > builder.THINK_MAX_SEGS:
                            drop["too_many_segs"] += 1
                        elif name != want:
                            drop["action_mismatch"] += 1
                            drop[f"mismatch:{want}->{name}"] += 1
                        else:
                            drop["hit"] += 1
                            accepted = think
                            break
                    if accepted:
                        hit += 1
                        kept.append({"case_id": bundle.case_id, "think": accepted})
                    break
            rate = hit / max(1, tried)
            passed = tried > 0 and rate >= 0.70
            result[behavior] = {
                "tried": tried,
                "hit": hit,
                "rate": round(rate, 3),
                "pass": passed,
                "examples": kept[:3],
                "drop": dict(drop),
            }
            print(
                f"[behavior-think] {behavior:8s} {hit}/{tried} = {rate:.0%}  "
                f"{'✅' if passed else '🔴'}"
            )
            print(f"[behavior-diag] {behavior:8s} 丢弃原因 {dict(drop)}")

    expected = {"reject", "defer", "clarify"}
    missing = sorted(expected - set(result))
    if missing:
        print(f"[behavior-think] 🔴 没有可测样本：{missing}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"[behavior-think] → {out}")
    return 0 if not missing and all(result[name]["pass"] for name in expected) else 2


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(async_main(argv))


if __name__ == "__main__":
    raise SystemExit(main())
