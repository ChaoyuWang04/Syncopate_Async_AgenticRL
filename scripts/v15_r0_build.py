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
import os
import random
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
rng = random.Random(1500)

# ★ spec 从契约模块取，不留副本（守则⑨）。⚠️ R0 双臂数据是用**这份 spec 的当时值**
# 冻结出来的；日后改 spec 不会回改已冻结的 parquet —— 重建 R0 数据才会生效。
from syncopate.core.contract import SESSION_TOOL_SPECS as SESSION_TOOLS

# ★ 冻结指纹：data/v15_r0/ 是用下面这个 spec 哈希建出来的。改了 spec 而不重建数据 ⇒
#   评测 prompt 与训练 prompt 不一致，R0 结论作废。判据写在发生点，不靠人记得检查。
FROZEN_SPEC_SHA = "6a5c2fd5a8868a10"


def _spec_fingerprint() -> str:
    import hashlib
    blob = json.dumps(SESSION_TOOLS, ensure_ascii=False, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def assert_spec_frozen() -> None:
    got = _spec_fingerprint()
    if got != FROZEN_SPEC_SHA:
        raise SystemExit(
            f"🔴 session 工具 spec 变了（{got} ≠ 冻结值 {FROZEN_SPEC_SHA}）。\n"
            "   data/v15_r0/ 是用旧 spec 冻的 ⇒ 要么改回 spec，要么重建 R0 数据并更新本常量。\n"
            "   （2026-08-29 实案：顺手加两个 description 就会走到这里，见 25 §7）")

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


async def make_rows(tokenizer, chosen, arm: str) -> list[dict]:
    """两臂各走**同一条**构建路径 `build_sft_sample`，只切契约开关。

    ⛔ 2026-08-29 重建的原因（R0 第三次作废的根因，`25 §7⑧`）：
      旧版是"拿 v14 的行 → 解码 → 字符串注入 session 工具 → 重新分词"。
      `full.replace("</tools>", inject, 1)` 替换的是**第一个** `</tools>` ——
      而那个在 Qwen3 模板的样板句
      「You are provided with function signatures within <tools></tools> XML tags:」里。
      于是 session 工具被塞进了**说明句**，不是真正的工具区。
      ⇒ 拿「工具不在强通道标准位置」的数据去验证「强通道更好」，等于把假说自己拆了。
    ⇒ 现在两臂都由 `build_sft_sample` 产出（=评测用的同一个 run_rollout + menu），
      训练 prompt 与评测 prompt **构造上就不可能不一致**（N5 一份契约）。
    """
    from syncopate.core.tool_registry import REGISTRY
    from syncopate.pipeline.sft_replay import build_sft_sample
    import syncopate.domains.adcampaign  # noqa: F401
    REGISTRY.latency_scale = 0.0

    rows = []
    for i, b in enumerate(chosen):
        smp = await build_sft_sample(b, tokenizer=tokenizer, registry=REGISTRY)
        rows.append({"case_id": f"{b.case_id}_T" if arm == "tool" else b.case_id,
                     "input_ids": smp.input_ids, "loss_mask": smp.loss_mask,
                     "prompt_length": smp.prompt_length,
                     "total_length": smp.total_length,
                     "supervised_tokens": smp.supervised_tokens,
                     "behavior": b.verifier.expected_behavior,
                     "split": "train", "index": (81000 if arm == "tool" else 80000) + i,
                     "signal_class": "graded"})
    return rows


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
    test_src = {b: pools[b][30:40] for b in pools}

    out = Path("data/v15_r0")
    arm = os.environ.get("SYNCOPATE_CONTRACT", "v14")
    arm_name = "tool" if arm == "v15" else "shell"
    rows = asyncio.run(make_rows(tok, train_bundles, arm_name))
    for r in rows:
        assert r["supervised_tokens"] > 0 and \
            len(r["input_ids"]) == len(r["loss_mask"]) == r["total_length"]
    d = out / f"arm_{arm_name}"
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(d / "train.parquet")
    pd.DataFrame(rows[:12]).to_parquet(d / "val.parquet")
    print(f"臂 {arm_name}: {len(rows)} 行 · 监督 {sum(r['supervised_tokens'] for r in rows)} tok")

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
