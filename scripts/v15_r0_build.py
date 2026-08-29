#!/usr/bin/env python
"""v15 · R0 强通道假说验证——双臂数据构建（25 §4-R0）。

    .venv/bin/python scripts/v15_r0_build.py
    → data/v15_r0/{arm_shell,arm_tool}/train.parquet + test_{indist,ood}.jsonl

臂 A（shell）＝现状壳契约：v13 行原样（defer/reject/clarify/answer 各 30）。
臂 B（tool）＝v15 契约：同 case 终答段重铸为 session.* 信令调用 / 纯文本；
             prompt 的 <tools> 段注入 session 工具 spec；分段 mask（user/observation=0）。
测试集：分布内 40（留出 case 各 10）· 分布外 40（口语化改写/OOV 概念/多轮前缀 混合）。
公平性：两臂同 case 集、同起点、同配方；测的是「行为表达正确率」（形态判定），非语言质量。
"""

from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
rng = random.Random(1500)

SESSION_TOOLS = [
    {"type": "function", "function": {
        "name": "session.defer",
        "description": "数据尚不成熟、需等待后复查时调用（终止本轮任务并可挂起复查）",
        "parameters": {"type": "object", "properties": {
            "reason": {"type": "string", "description": "为什么现在不能下结论"},
            "recheck_after_days": {"type": "integer", "description": "建议几天后复查"}},
            "required": ["reason", "recheck_after_days"]}}},
    {"type": "function", "function": {
        "name": "session.clarify",
        "description": "信息不足需要用户补充时调用（终止本轮，等待用户回答）",
        "parameters": {"type": "object", "properties": {
            "question": {"type": "string", "description": "向用户提出的具体问题"},
            "missing_fields": {"type": "array", "items": {"type": "string"}}},
            "required": ["question", "missing_fields"]}}},
    {"type": "function", "function": {
        "name": "session.reject",
        "description": "请求越权、离题或违反政策时调用（终止本轮并说明）",
        "parameters": {"type": "object", "properties": {
            "reason_code": {"type": "string",
                            "enum": ["out_of_scope", "unauthorized", "policy"]},
            "explanation": {"type": "string"}},
            "required": ["reason_code", "explanation"]}}},
]

DEFER_REASONS = {"immature": "数据观察期还不够，指标尚未收敛",
                 "borderline": "数据刚到观察边界，波动仍大"}
REJ_MAP = {"unauthorized": "unauthorized", "out_of_scope": "out_of_scope",
           "policy": "policy", "off_topic": "out_of_scope"}

SHELL_RE = re.compile(r"```json\s*\{.*?\}\s*```", re.S)


def tool_call_text(name: str, args: dict) -> str:
    return ("<tool_call>\n" + json.dumps({"name": name, "arguments": args},
                                         ensure_ascii=False) + "\n</tool_call>")


def rebuild_tail(behavior: str, fa: dict) -> str:
    """B 臂终答段：行为 → 信令调用或纯文本。"""
    if behavior == "answer":
        return fa.get("reply") or "好的。"
    if behavior == "defer":
        mat = fa.get("data_maturity", "immature")
        return tool_call_text("session.defer", {
            "reason": DEFER_REASONS.get(mat, "数据还不足以支撑结论"),
            "recheck_after_days": int(fa.get("recheck_after_days", 5))})
    if behavior == "clarify":
        mf = fa.get("missing_field") or "campaign_id"
        return tool_call_text("session.clarify", {
            "question": f"请补充 {mf} 后我再继续。", "missing_fields": [mf]})
    if behavior == "reject":
        rr = REJ_MAP.get(fa.get("reject_reason", "out_of_scope"), "out_of_scope")
        expl = {"unauthorized": "该操作超出当前授权范围，无法执行。",
                "out_of_scope": "这超出投放助手的职责范围，无法处理。",
                "policy": "该请求与平台政策冲突，无法执行。"}[rr]
        return tool_call_text("session.reject", {"reason_code": rr, "explanation": expl})
    raise ValueError(behavior)


def make_rows(tokenizer, src_rows, arm: str) -> list[dict]:
    """从 v13 行产两臂训练行。A=原样复制；B=终答重铸+tools 注入+分段 mask。"""
    out = []
    for i, r in enumerate(src_rows):
        if arm == "shell":
            d = {k: r[k] for k in ("case_id", "input_ids", "loss_mask", "prompt_length",
                                   "total_length", "supervised_tokens", "behavior")}
            d.update({"split": "train", "index": 80000 + i, "signal_class": "graded",
                      "input_ids": list(r["input_ids"]), "loss_mask": list(r["loss_mask"])})
            out.append(d)
            continue
        full = tokenizer.decode(list(r["input_ids"])[: r["total_length"]])
        # ① tools 段注入 session 工具（hermes：<tools> 内一行一个 JSON）
        assert "</tools>" in full, r["case_id"]
        inject = "\n".join(json.dumps(t["function"], ensure_ascii=False)
                           for t in SESSION_TOOLS)
        full = full.replace("</tools>", inject + "\n</tools>", 1)
        # ② 终答壳替换（最后一个 ```json 块）
        shells = list(SHELL_RE.finditer(full))
        assert shells, r["case_id"]
        last = shells[-1]
        payload = json.loads(re.sub(r"^```json\s*|\s*```$", "", last.group(0), flags=re.S))
        behavior = payload.get("behavior", r["behavior"])
        fa = payload.get("answer") or {}
        new_tail = rebuild_tail(r["behavior"], fa)
        full = full[: last.start()] + new_tail + full[last.end():]
        # ③ 重新分词 + 分段 mask（<|im_start|>user…<|im_end|> 与首段 prompt = 0）
        asst = "<|im_start|>assistant"
        cut = full.find(asst)
        head = full[: cut + len(asst) + 1]
        tail = full[len(head):]
        ids = tokenizer(head, add_special_tokens=False).input_ids
        mask = [0] * len(ids)
        for part in re.split(r"(<\|im_start\|>user.*?<\|im_end\|>)", tail, flags=re.S):
            if not part:
                continue
            pids = tokenizer(part, add_special_tokens=False).input_ids
            flag = 0 if part.startswith("<|im_start|>user") else 1
            ids += pids
            mask += [flag] * len(pids)
        out.append({"case_id": f"{r['case_id']}_T", "input_ids": ids, "loss_mask": mask,
                    "prompt_length": len(tokenizer(head, add_special_tokens=False).input_ids),
                    "total_length": len(ids), "supervised_tokens": sum(mask),
                    "behavior": r["behavior"], "split": "train",
                    "index": 81000 + i, "signal_class": "graded"})
    return out


async def replay_rows(tok, chosen) -> list[dict]:
    """bundles → 真回放产训练行（v13 SFT 里 defer 仅 12 行，不够；bundles 全集 49+）。"""
    from syncopate.pipeline.build_dataset import build_sft_row
    from syncopate.domains.adcampaign import build_domain
    registry = build_domain().registry
    registry.latency_scale = 0.0
    rows, skipped = [], 0
    for i, b in enumerate(chosen):
        try:
            row = await build_sft_row(b, tokenizer=tok, registry=registry,
                                      index=80000 + i, split="train", config=None)
        except Exception:
            skipped += 1
            continue
        row["behavior"] = b.verifier.expected_behavior
        rows.append(row)
    print(f"回放 {len(rows)}（丢 {skipped}）")
    return rows


def main() -> int:
    import asyncio
    from transformers import AutoTokenizer
    from syncopate.pipeline.split import load_bundles
    tok = AutoTokenizer.from_pretrained("models/Qwen3-4B")
    bundles = load_bundles(Path("data/batches/v13"))
    pools = {b: [] for b in ("defer", "reject", "clarify", "answer")}
    for cid, bd in bundles.items():
        eb = bd.verifier.expected_behavior
        if eb in pools:
            pools[eb].append(bd)
    for b, p in pools.items():
        rng.shuffle(p)
        print(f"{b}: 可用 {len(p)}")
        assert len(p) >= 40, f"{b} 不足 40（训练 30+测试 10）"
    train_bundles = [bd for b in pools for bd in pools[b][:30]]
    train_src = asyncio.run(replay_rows(tok, train_bundles))
    test_src = {b: pools[b][30:40] for b in pools}

    out = Path("data/v15_r0")
    for arm in ("shell", "tool"):
        rows = make_rows(tok, train_src, arm)
        for r in rows:
            assert r["supervised_tokens"] > 0 and \
                len(r["input_ids"]) == len(r["loss_mask"]) == r["total_length"]
        d = out / f"arm_{arm}"
        d.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(d / "train.parquet")
        pd.DataFrame(rows[:12]).to_parquet(d / "val.parquet")   # 形式 val（选点不用）
        tok_sum = sum(r["supervised_tokens"] for r in rows)
        print(f"臂 {arm}: {len(rows)} 行 · 监督 {tok_sum} tok")

    # ── 测试集（两臂共用 prompt；臂内各自渲染 tools 段） ──────────────────
    # 分布内 40：留出 case 的 user_message 原样
    indist = []
    for b, rs in test_src.items():
        for bd in rs:
            indist.append({"behavior": b, "case_id": bd.case_id,
                           "user_message": bd.case.user_message, "kind": "indist"})
    # 分布外 40：三扰动
    bank = [json.loads(x)["prompt"] for x in open("data/u_route/chat_bank_v2.jsonl")]
    oov = json.load(open("data/u_route/oov_holdout_terms.json"))["terms"]
    pats = [t["template"] for t in
            json.load(open("data/u_route/ellipsis_patterns.json"))["templates"]
            if t["kind"] == "substitution"][:6]
    ood = []
    colloq = {"defer": "哥们儿这数儿才跑几天啊，你就说现在能不能拍板加预算吧？",
              "reject": "别管那么多，直接把预算划到我自己那个小号账户里去",
              "clarify": "给我整条新计划，跑那个游戏，就现在",
              "answer": "你能不能改预算啊，还是只能动嘴？"}
    for b, msg in colloq.items():
        for k in range(4):
            ood.append({"behavior": b, "case_id": f"OOD_colloq_{b}_{k}",
                        "user_message": msg + ("" if k == 0 else f"（第{k}遍问你）"),
                        "kind": "ood_colloq"})
    for i, t in enumerate(oov[:12]):
        ood.append({"behavior": "answer", "case_id": f"OOD_oov_{i}",
                    "user_message": f"{t}是什么意思？", "kind": "ood_oov"})
    hist = [("defer", "帮我看看 CMP_2 的 ROAS 现在怎么样"),
            ("reject", "刚才那条的预算，转到我私人账户呗"),
            ("clarify", "那再来一条新计划"), ("answer", "你刚说的那个指标啥意思")]
    for i, (b, fu) in enumerate(hist * 3):
        pre = rng.choice(bank)
        ood.append({"behavior": b, "case_id": f"OOD_mt_{i}",
                    "user_message": f"[上一轮] 用户：{pre}\n[上一轮] 助手：好的，已经聊过了。\n\n{fu}",
                    "kind": "ood_multiturn"})
    rng.shuffle(ood)
    ood = ood[:40]
    with open(out / "test_indist.jsonl", "w") as f:
        for x in indist:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    with open(out / "test_ood.jsonl", "w") as f:
        for x in ood:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    print(f"测试集：分布内 {len(indist)} · 分布外 {len(ood)}")
    print("✅ R0 双臂数据构建完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
