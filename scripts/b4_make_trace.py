#!/usr/bin/env python
"""E32 S0.1 · 真实形状 trace 数据集（B-4 压测的第二条轨，cache/路由的唯一合法尺子）。

    .venv/bin/python scripts/b4_make_trace.py                # 默认 512 条，seed 42
    .venv/bin/python scripts/b4_make_trace.py --n 512 --seed 42

来源：checkpoints/grpo/*/rollout_dumps/*.jsonl 的 `input`（真实完整 prompt：system prompt
+ case 上下文 + 多轮工具观察）与 `output`（真实解码长度）。⛔ 产物只进 _audit/，不进 HF/上游
（Chaoyu 08-28 裁定④）。

设计要点：
- 按 case 家族分层采样（比例保持），seed 固定 ⇒ 可复现；
- 输出顺序按 (run, file, line) 排 ⇒ 保留"同 case 的 8 条 rollout 相邻下发"的真实到达结构
  （dispatched.jsonl 实测同 case 背靠背），这正是前缀亲和路由要吃的形状；
- 每条带 max_tokens = 真实输出 token 数（cap 2048）⇒ 重放时解码长度分布与生产一致；
- 统计头落 _audit/b4_trace_stats.json：prompt/output token 分位数、全局公共前缀长度
  （= prefix cache 可命中上限）、每请求 KV 字节账（S3 PD 账的输入，72 KB/token fp8）。
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
from collections import Counter, defaultdict

sys.path.insert(0, ".")

MODEL_DIR = "models/Qwen3-4B-sft-v13r2-e1"
OUT_TRACE = "_audit/b4_trace.jsonl"
OUT_STATS = "_audit/b4_trace_stats.json"
KV_BYTES_PER_TOKEN_FP8 = 2 * 36 * 8 * 128  # K+V × 层 × KV头 × head_dim × 1B（config.json 实读）


def pct(sorted_vals: list[int], q: float) -> int:
    return sorted_vals[min(len(sorted_vals) - 1, int(q * len(sorted_vals)))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-tokens-cap", type=int, default=2048)
    args = ap.parse_args()

    # ── 1 · 收全量候选（只留有 input/output 的行）────────────────────────────
    records: list[dict] = []
    files = sorted(glob.glob("checkpoints/grpo/*/rollout_dumps/*.jsonl"))
    if not files:
        print("🔴 没找到 rollout_dumps，路径前提变了"); return 1
    for path in files:
        run = path.split("/")[2]
        fidx = int(os.path.basename(path).split(".")[0])
        with open(path) as f:
            for lidx, line in enumerate(f):
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                inp, out = d.get("input") or "", d.get("output") or ""
                if not inp or not out:
                    continue
                cid = d.get("case_id") or "?"
                records.append({
                    "prompt": inp, "output": out, "case_id": cid,
                    "family": cid.split("_")[0],
                    "_ord": (run, fidx, lidx),
                })
    print(f"候选 {len(records)} 条 / {len(files)} 文件")

    # ── 2 · 家族分层采样（比例保持，seed 固定）──────────────────────────────
    rng = random.Random(args.seed)
    by_fam: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_fam[r["family"]].append(r)
    total = len(records)
    sampled: list[dict] = []
    for fam, pool in sorted(by_fam.items()):
        k = max(1, round(args.n * len(pool) / total))
        sampled.extend(rng.sample(pool, min(k, len(pool))))
    rng.shuffle(sampled)           # 家族拼接顺序打掉
    sampled = sampled[: args.n]
    sampled.sort(key=lambda r: r["_ord"])   # 还原真实到达相邻性（同 case 背靠背）

    # ── 3 · tokenize（真实 token 数；max_tokens = 真实输出长度）────────────────
    from transformers import AutoTokenizer  # noqa: E402  （慢 import 放采样后）
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    p_lens, o_lens = [], []
    for r in sampled:
        r["prompt_tokens"] = len(tok(r["prompt"]).input_ids)
        r["output_tokens"] = len(tok(r["output"]).input_ids)
        r["max_tokens"] = min(r["output_tokens"], args.max_tokens_cap)
        p_lens.append(r["prompt_tokens"]); o_lens.append(r["output_tokens"])

    # 全局公共前缀 = prefix cache 可命中上限（chars → tokens）
    common = sampled[0]["prompt"]
    for r in sampled[1:]:
        while not r["prompt"].startswith(common):
            common = common[:-256] if len(common) > 256 else os.path.commonprefix([common, r["prompt"]])
    common_tokens = len(tok(common).input_ids)

    # ── 4 · 落盘 ────────────────────────────────────────────────────────────
    os.makedirs("_audit", exist_ok=True)
    with open(OUT_TRACE, "w") as f:
        for i, r in enumerate(sampled):
            f.write(json.dumps({
                "idx": i, "case_id": r["case_id"], "family": r["family"],
                "prompt_tokens": r["prompt_tokens"], "max_tokens": r["max_tokens"],
                "prompt": r["prompt"],
            }, ensure_ascii=False) + "\n")

    p_lens.sort(); o_lens.sort()
    stats = {
        "n": len(sampled), "seed": args.seed, "sources": sorted({r["_ord"][0] for r in sampled}),
        "families": dict(Counter(r["family"] for r in sampled)),
        "prompt_tokens": {q: pct(p_lens, float(q)) for q in ("0.1", "0.5", "0.9", "0.99")},
        "output_tokens": {q: pct(o_lens, float(q)) for q in ("0.1", "0.5", "0.9", "0.99")},
        "common_prefix_tokens": common_tokens,
        "kv_fp8_mb_per_req": {
            "p50": round(pct(p_lens, 0.5) * KV_BYTES_PER_TOKEN_FP8 / 2**20, 1),
            "p90": round(pct(p_lens, 0.9) * KV_BYTES_PER_TOKEN_FP8 / 2**20, 1),
        },
    }
    with open(OUT_STATS, "w") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"✅ {OUT_TRACE} / {OUT_STATS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
