#!/usr/bin/env python
"""v15 · R0/R1/R2 开工前探针（25 §4 各门槛的实测依据）。

    .venv/bin/python scripts/v15_probes.py            → _audit/v15_probes/results.json

六个探针，每个都对应 25 号里一条**写在测量之前**的门槛，用来把猜的数字换成实测的：

  P1  空 think 块落在监督段还是提示段        → 24 §7「CoT 被压死」的归因是否成立
  P2  cot_hard 行内部的 think 空/非空比例    → CoT 数据自身有没有被空块稀释
  P3  chat 模板 think-off/on 的结构差异      → 修法可行性的机制依据
  P4  think-on 是否破坏 SFT/RL 序列同构      → 修法的最大风险项（走 pytest，见 run_all）
  P5  session.* 工具 schema 的 prompt 增量   → 25 §R1④（原写 ≤400，未经测量）
  P6  v15 行为推导器 vs v14 旧标签 对拍上限  → 25 §R1②（原写 ≥99%，未经测量）
"""
from __future__ import annotations

import asyncio
import collections
import glob
import json
import re
import statistics
from pathlib import Path

OUT = Path("_audit/v15_probes")
# ★ spec 从契约模块取，不留副本（守则⑨：这里根本不该有第二份）
from syncopate.core.contract import SESSION_TOOL_SPECS as SESSION_TOOLS  # noqa: E402
from syncopate.core.model_paths import TEST_TOKENIZER, STUDENT_MODEL, TEACHER_MODEL
from syncopate.pipeline.split import DEFAULT_BATCH_DIR, DEFAULT_SPLIT_DIR, DEFAULT_SFT_DIR, DEFAULT_RL_DIR

REPORT_TOOL = [{"type": "function", "function": {
    "name": "session.report",
    "description": "给出本轮结论里机器需要核对的结构化字段",
    "parameters": {"type": "object", "properties": {k: {"type": "string"} for k in
                   ["decision", "approved_budget", "cited_clause_id",
                    "review_status", "reason_code", "risk_level"]},
                   "required": ["decision"]}}}]


def _supervised_text(tok, ids, mask, want=1):
    return tok.decode([t for t, m in zip(ids, mask) if m == want], skip_special_tokens=False)


def p1_p2_think_placement(tok) -> dict:
    """think 块落在监督段还是提示段 + cot 行内部的空/非空比例。"""
    import pandas as pd
    d = pd.read_parquet("data/sft/v14_5/train.parquet")
    sup_rows = prompt_rows = nonempty_rows = 0
    blk_empty = blk_nonempty = 0
    buckets = collections.Counter()
    for _, r in d.iterrows():
        ids, m = list(r["input_ids"]), list(r["loss_mask"])
        sup, unsup = _supervised_text(tok, ids, m, 1), _supervised_text(tok, ids, m, 0)
        if "<think>" in unsup:
            prompt_rows += 1
        blocks = re.findall(r"<think>(.*?)</think>", sup, re.S)
        if not blocks:
            continue
        sup_rows += 1
        buckets[r["bucket"]] += 1
        ne = [b for b in blocks if b.strip()]
        blk_empty += len(blocks) - len(ne)
        blk_nonempty += len(ne)
        if ne:
            nonempty_rows += 1
    return {
        "rows_total": len(d),
        "rows_think_in_prompt_segment": prompt_rows,
        "rows_think_in_supervised_segment": sup_rows,
        "rows_with_nonempty_supervised_think": nonempty_rows,
        "supervised_blocks_empty": blk_empty,
        "supervised_blocks_nonempty": blk_nonempty,
        "supervised_empty_share": round(blk_empty / max(1, blk_empty + blk_nonempty), 4),
        "buckets_of_supervised_think_rows": dict(buckets),
    }


def p3_template_shape(tok) -> dict:
    """think-off 会把 think 块在生成提示里**预先关闭**；think-on 不会。"""
    msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}]
    out = {}
    for flag in (False, True):
        t = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                    tokenize=False, enable_thinking=flag)
        out[f"enable_thinking={flag}"] = t[-48:]
    return out


def p5_session_tool_budget(tok) -> dict:
    """session 工具 schema 的 prompt token 增量（训练口径=真实 case 菜单）。"""
    from syncopate.core.tool_registry import REGISTRY
    import syncopate.domains.adcampaign  # noqa: F401  注册工具
    from syncopate.pipeline.split import load_bundles
    from syncopate.train.rollout_budget import MAX_PROMPT_LENGTH
    from syncopate.train.rollout_loop import build_messages

    bundles = load_bundles(Path(DEFAULT_BATCH_DIR))
    bases, deltas = [], []
    for _, bd in list(bundles.items())[:120]:
        msgs = build_messages(bd, bd.case.tool_menu)
        tools = REGISTRY.menu(bd.case.tool_menu)
        f = lambda t: len(tok.apply_chat_template(  # noqa: E731
            msgs, tools=t, add_generation_prompt=True, tokenize=True, enable_thinking=False))
        bases.append(f(tools))
        deltas.append(f(tools + SESSION_TOOLS + REPORT_TOOL) - bases[-1])
    after = [b + d for b, d in zip(bases, deltas)]
    full = len(tok.apply_chat_template(
        [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}],
        tools=REGISTRY.menu(None) + SESSION_TOOLS + REPORT_TOOL,
        add_generation_prompt=True, tokenize=True, enable_thinking=False))
    return {
        "n_cases": len(bases),
        "prompt_before_median": statistics.median(bases),
        "prompt_before_max": max(bases),
        "delta_median": statistics.median(deltas),
        "delta_max": max(deltas),
        "prompt_after_max": max(after),
        "max_prompt_length": MAX_PROMPT_LENGTH,
        "headroom_after": MAX_PROMPT_LENGTH - max(after),
        "deploy_regime_full_menu_prompt": full,
    }


def p6_behavior_derivation_crosscheck() -> dict:
    """v15 形态推导器在 v14 考场 run 上能达到的**上限**一致率。

    v14 的 run 里不存在 session.* 调用 ⇒ 旧标签 defer/clarify/reject 的行
    **物理上不可能**被推导出来。所以对拍只在 {tool_call, answer} 子集上有意义。
    """
    cm = collections.Counter()
    for f in sorted(glob.glob("logs/u_route/run_v145*.jsonl")):
        for row in map(json.loads, open(f)):
            for t in row.get("turns", []):
                b = t.get("behavior") or "(none)"
                cm[(b, "tool_call" if (t.get("tools") or []) else "answer")] += 1
    total = sum(cm.values())
    mappable = {k: v for k, v in cm.items() if k[0] in ("tool_call", "answer")}
    signal = sum(v for k, v in cm.items() if k[0] in ("defer", "clarify", "reject"))
    ok = sum(v for (b, d), v in mappable.items() if b == d)
    return {
        "turns_total": total,
        "turns_signal_behaviors_unmappable": signal,
        "turns_mappable": sum(mappable.values()),
        "mappable_agreement": round(ok / max(1, sum(mappable.values())), 4),
        "naive_agreement_all_turns": round(
            sum(v for (b, d), v in cm.items() if b == d) / max(1, total), 4),
        "cross_tab": {f"{b}->{d}": v for (b, d), v in sorted(cm.items(), key=lambda x: -x[1])},
    }


def p7_constrained_decoding_wired() -> dict:
    """约束解码到底接上了没有——`final_answer_schema` 的调用点数。"""
    import subprocess
    hits = subprocess.run(
        ["grep", "-rn", "final_answer_schema", "--include=*.py", "syncopate/", "scripts/", "tests/"],
        capture_output=True, text=True).stdout.strip().splitlines()
    call_sites = [h for h in hits
                  if "def final_answer_schema" not in h and '"""' not in h
                  and not h.startswith("scripts/v15_probes.py")]   # 排除本探针自身
    return {"definition_found": bool(hits), "call_sites": len(call_sites), "lines": call_sites}


def main() -> None:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(STUDENT_MODEL)
    results = {
        "P1_P2_think_placement": p1_p2_think_placement(tok),
        "P3_template_shape": p3_template_shape(tok),
        "P5_session_tool_budget": p5_session_tool_budget(tok),
        "P6_behavior_derivation": p6_behavior_derivation_crosscheck(),
        "P7_constrained_decoding_wired": p7_constrained_decoding_wired(),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(json.dumps(results, ensure_ascii=False, indent=2))
    print("\n★ P4（think-on 是否破坏 SFT/RL 同构）走 pytest，不在本脚本内：")
    print("  SYNCOPATE_THINK=1 .venv/bin/python -m pytest tests/train/test_rollout_loop.py "
          "-k 'token_identical or full_render_differs'")


if __name__ == "__main__":
    main()
