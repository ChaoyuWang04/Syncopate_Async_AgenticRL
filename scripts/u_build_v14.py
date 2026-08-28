#!/usr/bin/env python
"""U 路 P2 · v14 SFT 数据集构建（`24 §4-P2`）。

    .venv/bin/python scripts/u_build_v14.py     # → data/sft/v14/{train,val}.parquet

构成（seed 固定，可复现）：
  ① v13 train 419 行原样继承（影子链已验的资产不重造）
  ② 多轮 L2 行（P1 红旗正枪）：查询类 case 的第二轮指代追问——历史前缀+追问进
     user_message，gold=[get_metrics(同 campaign), 终答(summary+reply)]，
     **走 build_sft_sample 真回放**（工具真执行=回放断言天然满足）
  ③ 闲聊行（离线蒸馏）：chat_gold.jsonl（教师文本+契约装壳），直接 tokenize
  ④ CoT 行：cot_traces.jsonl（8B think 末答对才留），单步案 ctx+<think>…</think>+gold
     re-tokenize，mask=prompt 0 / 生成段 1
  val = v13 val 84 行原样（SFT 选点用）。
范围注记：L3/L4 多轮行刻意不造——该行为归 P3 RL（gold 派生要碰写权限语义，
  教错比不教危险）；C-6（RL val 并入）属 RL 数据，在 P3 前的 rl v14 生成时做。
"""

from __future__ import annotations

import asyncio
import copy
import json
import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, ".")
rng = random.Random(1409)

METRIC_ASKS = [("消耗", "spend_7d"), ("安装量", "installs_7d"), ("ROAS", "roas_d7"),
               ("CPI", "cpi"), ("点击率", "ctr"), ("频次", "frequency")]
REF_FORMS = ["它的{m}呢？", "这条计划的{m}呢", "那{m}怎么样", "顺便看下它的{m}"]


async def build_l2_rows(tokenizer, registry, max_rows: int = 200) -> list[dict]:
    from syncopate.pipeline.build_dataset import build_sft_row
    from syncopate.pipeline.split import load_bundles
    bundles = load_bundles(Path("data/batches/v13"))
    split_ids = json.load(open("data/splits/v13/splits.json")) if \
        Path("data/splits/v13/splits.json").exists() else None
    rows, skipped = [], 0
    cands = []
    for cid, b in bundles.items():
        if b.gold and b.gold.actions and \
                b.gold.actions[0]["tool"] == "campaign.get_metrics" and \
                b.case.context.get("campaign_id"):
            cands.append(b)
    rng.shuffle(cands)
    print(f"L2 候选 case {len(cands)}")
    for b in cands:
        if len(rows) >= max_rows:
            break
        camp = b.case.context["campaign_id"]
        mname, mkey = METRIC_ASKS[len(rows) % len(METRIC_ASKS)]
        ask = rng.choice(REF_FORMS).format(m=mname)
        b2 = copy.deepcopy(b)
        prev_sum = (b.gold.final_answer or {}).get("summary") or \
                   (b.gold.final_answer or {}).get("conclusion") or "已给出结论"
        b2.case.user_message = (
            f"[上一轮] 用户：{b.case.user_message}\n"
            f"[上一轮] 助手：{str(prev_sum)[:120]}\n\n{ask}")
        b2.case.case_id = f"{b.case_id}_MT2"
        b2.gold.actions = [{"tool": "campaign.get_metrics",
                            "arguments": {"campaign_id": camp}}]
        b2.gold.final_answer = {
            "summary": f"{camp} {mkey} 已查",
            "reply": f"{camp} 的{mname}指标已经取到，详见上方数据；如需对比或进一步分析随时说。"}
        try:
            cfg = None
            row = await build_sft_row(b2, tokenizer=tokenizer, registry=registry,
                                      index=90000 + len(rows), split="train",
                                      config=cfg)
        except Exception as e:  # noqa: BLE001
            skipped += 1
            if skipped <= 3:
                print(f"  ⚠️ {b.case_id} 回放失败：{str(e)[:120]}")
            continue
        row["bucket"] = "multiturn"
        rows.append(row)
    print(f"L2 行 {len(rows)}（回放失败丢 {skipped}——显式计数）")
    return rows


def build_chat_rows(tokenizer, start_idx: int) -> list[dict]:
    sys.path.insert(0, "scripts")
    from probe_opd_divergence import render_prompt_text
    rows = []
    for i, x in enumerate(open("data/u_route/chat_gold.jsonl")):
        d = json.loads(x)
        prompt = render_prompt_text(tokenizer, d["prompt"], tools=None)
        ids_p = tokenizer(prompt, add_special_tokens=False).input_ids
        ids_g = tokenizer(d["gold_json"] + "<|im_end|>",
                          add_special_tokens=False).input_ids
        rows.append({
            "case_id": f"CHATG_{i:04d}", "input_ids": ids_p + ids_g,
            "loss_mask": [0] * len(ids_p) + [1] * len(ids_g),
            "prompt_length": len(ids_p), "total_length": len(ids_p) + len(ids_g),
            "supervised_tokens": len(ids_g), "split": "train",
            "index": start_idx + i, "signal_class": "graded",
            "behavior": "answer", "bucket": "chat_distill",
        })
    print(f"闲聊行 {len(rows)}")
    return rows


def build_cot_rows(tokenizer, start_idx: int) -> list[dict]:
    ASST = "<|im_start|>assistant"
    df = pd.concat([pd.read_parquet("data/sft/v13/train.parquet"),
                    pd.read_parquet("data/sft/v13/val.parquet")])
    by_id = {r.case_id: r for _, r in df.iterrows()}
    rows = []
    for i, x in enumerate(open("data/u_route/cot_traces.jsonl")):
        d = json.loads(x)
        src = by_id.get(d["case_id"])
        if src is None:
            continue
        ids = list(src.input_ids)
        full = tokenizer.decode(ids[: src.total_length])
        cut = full.rfind(ASST)
        head_end = full.find("\n", cut) + 1
        ctx = full[:head_end]
        new_tail = f"<think>\n{d['think']}\n</think>\n\n{d['gold_tail']}"
        ids_p = tokenizer(ctx, add_special_tokens=False).input_ids
        ids_g = tokenizer(new_tail, add_special_tokens=False).input_ids
        rows.append({
            "case_id": f"{d['case_id']}_COT{d['sample_idx']}",
            "input_ids": ids_p + ids_g,
            "loss_mask": [0] * len(ids_p) + [1] * len(ids_g),
            "prompt_length": len(ids_p), "total_length": len(ids_p) + len(ids_g),
            "supervised_tokens": len(ids_g), "split": "train",
            "index": start_idx + i, "signal_class": "graded",
            "behavior": d["behavior"], "bucket": "cot_distill",
        })
    print(f"CoT 行 {len(rows)}")
    return rows


async def main() -> int:
    from transformers import AutoTokenizer
    from syncopate.domains.adcampaign import build_domain
    tokenizer = AutoTokenizer.from_pretrained("models/Qwen3-4B")
    registry = build_domain().registry
    registry.latency_scale = 0.0

    t13 = pd.read_parquet("data/sft/v13/train.parquet")
    v13 = pd.read_parquet("data/sft/v13/val.parquet")
    l2 = await build_l2_rows(tokenizer, registry)
    chat = build_chat_rows(tokenizer, 91000)
    cot = build_cot_rows(tokenizer, 92000)
    new = pd.DataFrame(l2 + chat + cot)
    train = pd.concat([t13, new], ignore_index=True)
    out = Path("data/sft/v14"); out.mkdir(parents=True, exist_ok=True)
    train.to_parquet(out / "train.parquet")
    v13.to_parquet(out / "val.parquet")
    manifest = {
        "version": "v14", "seed": 1409,
        "sources": {"v13_train": len(t13), "l2_multiturn": len(l2),
                    "chat_distill": len(chat), "cot_distill": len(cot)},
        "total": len(train),
        "supervised_tokens": int(train.supervised_tokens.sum()),
        "behavior_counts": train.behavior.value_counts().to_dict(),
        "bucket_counts": train.bucket.value_counts().to_dict(),
    }
    json.dump(manifest, open(out / "manifest.json", "w"), ensure_ascii=False, indent=2)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    # 健全性断言：所有新行监督 token > 0、mask 与长度一致
    for r in l2 + chat + cot:
        assert r["supervised_tokens"] > 0, r["case_id"]
        assert len(r["input_ids"]) == len(r["loss_mask"]) == r["total_length"], r["case_id"]
    print("✅ v14 构建完成，健全性断言全过")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
