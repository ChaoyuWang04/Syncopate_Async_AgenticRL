#!/usr/bin/env python3
"""训练数据体检：**开训之前**把"看起来像数据、其实是模板"的东西照出来。

⛔ 为什么要有这个文件（Chaoyu 2026-08-30）：
  「我们不能每次都花了超级长的时间走完完整的训练之后再来返工来做检查。」
  已经付过两次学费，两次都是**同一种病**、都靠事后考场才发现：
    · ㉖ 41.8% 的行以同一句「已经按上面的结果处理完了。」收尾（v13 压舱 gold 没有 reply）
    · ㉗ 150 行把人话塞进 session.report 再原样抄一遍（reply 被当成机器字段）
  两次的共同形状是：**多样性塌在某一个通道里**，而当时的闸只量另一个通道。
  ⇒ 这里按「模型实际会读到/被监督到的每一段文本」逐通道量，不挑通道。

用法：
    python scripts/v15_data_audit.py data/sft/v15/train.parquet [--json out.json]

判据（超了就红，退出码 1）见 GATES。⚠️ 阈值都写了**为什么是这个数**；
改阈值要先问"新口径在旧产物里存不存在"（守则：换契约会让旧单位标定的数字同时失效）。
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer
from syncopate.core.model_paths import TEST_TOKENIZER, STUDENT_MODEL, TEACHER_MODEL

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ── 通道切分 ────────────────────────────────────────────────────────────────
_TOOLCALL = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)   # 线格式无关：JSON 或 Qwen3.5 XML 都截出来，解析交给 parsing_v15
_THINK = re.compile(r"<think>(.*?)</think>", re.S)
_ASSIST = re.compile(r"<\|im_start\|>assistant\n(.*?)<\|im_end\|>", re.S)
_USER = re.compile(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", re.S)

# 归一化：把数字/ID/日期抹掉，剩下的就是"句式"。
# ★ 这是本文件的核心手法 —— 逐字重复很容易躲过（换个数字就不同了），
#   句式重复躲不过。㉖ 那次如果按句式量，第一眼就会看见 41.8%。
_NORM = [(re.compile(r"\d+(\.\d+)?"), "§N"),
         (re.compile(r"(CMP|ACC|CRE|CASE|ASSET)_[A-Za-z0-9]+"), "§ID"),
         (re.compile(r"[A-Z]{2,}_[A-Z0-9_]+"), "§ID")]


def norm(t: str) -> str:
    for pat, rep in _NORM:
        t = pat.sub(rep, t)
    return re.sub(r"\s+", " ", t).strip()


def split_row(text: str) -> dict:
    """一行训练样本 → 各通道文本。"""
    out = {"user": [], "think": [], "calls": [], "prose": []}
    out["user"] = [u.strip() for u in _USER.findall(text)
                   if "<tool_response>" not in u]
    for seg in _ASSIST.findall(text):
        for th in _THINK.findall(seg):
            if th.strip():
                out["think"].append(th.strip())
        body = _THINK.sub("", seg).strip()
        from syncopate.core.parsing_v15 import parse_tool_calls
        calls, _malformed = parse_tool_calls(body)      # [{"name", "arguments"}]，JSON/XML 两种线格式都认
        out["calls"].extend(calls)
        rest = _TOOLCALL.sub("", body).strip()
        if rest:
            out["prose"].append(rest)
    return out


# 概念追问的问法壳。⚠️ 「什么是毛利？」「毛利又是什么」「那毛利呢」是**同一个问题**，
#   同一句定义回答它们是对的 —— 把问法当成不同题面，会把正确的行为报成"预设答案"。
#   （08-30 实测：16 项误报全是这一种。判据要量的是"答案与题面无关"，不是"字面不同"。）
_ASK_SHELL = ["再说说", "什么是", "又是什么", "是什么意思", "是什么", "那", "呢"]


def qkey(u: str) -> str:
    """题面 → "在问什么"。只脱概念追问的壳；正经任务题面原样保留。"""
    lines = [x for x in u.strip().splitlines() if x.strip()]
    last = lines[-1] if lines else ""
    k = re.sub(r"[\s？?。，、]", "", last)
    for pat in _ASK_SHELL:
        k = k.replace(pat, "")
    return k or last


def top_share(items: list[str]) -> tuple[float, str, int]:
    if not items:
        return 0.0, "", 0
    c = collections.Counter(items).most_common(1)[0]
    return c[1] / len(items), c[0], c[1]


# ── 判据 ────────────────────────────────────────────────────────────────────
GATES = {
    # 逐字重复：同一句话原样出现在很多行 ⇒ 模型学"背这句"。
    # 2% 的来历：v14.5 实测健康值 0.3–1.1%，㉖ 那次是 41.8% —— 2% 把两者分得开。
    "prose_verbatim_top": 0.02,
    # 句式重复：抹掉数字/ID 之后还一样 ⇒ 模板。比逐字松，因为同一任务族本来就像。
    # 8% 的来历：健康桶实测 3–6%（同族任务收尾方式本就接近），㉖ 归一化后 43%。
    "prose_pattern_top": 0.08,
    # 一个答案服务多少个**不同的题面** —— "预设答案"最直接的形状。
    "answer_serves_prompts": 3,
    # 信令自由文本：常量模板会长成"万能出口"（㉘：模型背下那句 reject 当退路）
    "signal_top": 0.35,
    # 题面自己也不许是同一句（否则等于同一道题刷了很多遍）
    "user_verbatim_top": 0.02,
    # think 段重复 ⇒ 教的是"想的样子"不是"怎么想"
    "think_verbatim_top": 0.10,
}


def audit(path: Path, model: str = STUDENT_MODEL) -> dict:
    tok = AutoTokenizer.from_pretrained(model)
    df = pd.read_parquet(path)
    rows = []
    for _, r in df.iterrows():
        text = tok.decode(list(r["input_ids"])[:r["total_length"]])
        ch = split_row(text)
        ch["bucket"] = r.get("bucket", "?")
        ch["case_id"] = r.get("case_id", "?")
        rows.append(ch)

    from syncopate.core.contract import PROSE_FIELDS

    report = {"file": str(path), "rows": len(rows), "buckets": {}, "findings": []}
    add = report["findings"].append

    # ---- 全局：人话通道 / 机器通道 / 信令通道 ----
    prose_all, prose_by_bucket = [], collections.defaultdict(list)
    user_all = []
    think_all = []
    sig_free, report_args, prose_in_report = [], [], []
    answer_to_prompts = collections.defaultdict(set)
    for ch in rows:
        last = ch["prose"][-1] if ch["prose"] else ""
        if last:
            prose_all.append(last)
            prose_by_bucket[ch["bucket"]].append(last)
            q = ch["user"][0] if ch["user"] else ""
            answer_to_prompts[last].add(qkey(q))
        user_all += ch["user"][:1]
        think_all += ch["think"]
        for c in ch["calls"]:
            name, args = c.get("name", ""), (c.get("arguments") or {})
            if name == "session.report":
                report_args.append(json.dumps(args, ensure_ascii=False, sort_keys=True))
                for k in set(args) & set(PROSE_FIELDS):
                    prose_in_report.append((ch["case_id"], k))
            elif name.startswith("session."):
                for k in ("explanation", "question", "reason"):
                    if args.get(k):
                        sig_free.append(str(args[k]))

    def measure(label, items, gate_key, extra=""):
        share, top, n = top_share(items)
        uniq = len(set(items))
        line = (f"{label:22} 条数 {len(items):5} 唯一 {uniq:5} "
                f"最高频 {share:6.1%} ×{n:<4}{extra}")
        bad = share > GATES[gate_key] and len(items) >= 20
        report["buckets"][label] = {"n": len(items), "unique": uniq,
                                    "top_share": share, "top": top[:60]}
        print(("🔴 " if bad else "   ") + line)
        if top:
            print(f"{'':25}最高频文本 = {top[:70]!r}")
        if bad:
            add({"gate": gate_key, "label": label, "share": share, "top": top[:120]})
        return bad

    print(f"\n=== {path} · {len(rows)} 行 ===\n")
    print("── 人话通道（终答）──")
    measure("全部终答·逐字", prose_all, "prose_verbatim_top")
    measure("全部终答·句式", [norm(p) for p in prose_all], "prose_pattern_top")
    for b in sorted(prose_by_bucket):
        if len(prose_by_bucket[b]) >= 20:
            measure(f"  {b}·句式", [norm(p) for p in prose_by_bucket[b]],
                    "prose_pattern_top")

    print("\n── 预设答案（同一个答案服务了几个不同题面）──")
    multi = [(a, len(qs)) for a, qs in answer_to_prompts.items()
             if len(qs) >= GATES["answer_serves_prompts"]]
    multi.sort(key=lambda x: -x[1])
    if multi:
        print(f"🔴 {len(multi)} 个答案各自服务了 ≥{GATES['answer_serves_prompts']} 个不同题面")
        for a, n in multi[:5]:
            print(f"     ×{n:<4} {a[:70]!r}")
        add({"gate": "answer_serves_prompts", "count": len(multi),
             "worst": [[a[:120], n] for a, n in multi[:5]]})
    else:
        print("   无（每个答案基本只对应自己的题面）")

    print("\n── 机器通道（session.report）──")
    if prose_in_report:
        print(f"🔴 人话字段进了 report：{len(prose_in_report)} 处 "
              f"（字段 {sorted({k for _, k in prose_in_report})}）")
        add({"gate": "prose_in_report", "count": len(prose_in_report)})
    else:
        print("   人话字段进 report：0 ✓")
    share, top, n = top_share(report_args)
    print(f"   report 参数组 {len(report_args)} 条 · 唯一 {len(set(report_args))} · "
          f"最高频 {share:.1%} ×{n}")

    print("\n── 信令自由文本 ──")
    measure("信令 explanation/question/reason", sig_free, "signal_top")

    print("\n── 题面 / think ──")
    measure("题面（首轮 user）·逐字", user_all, "user_verbatim_top")
    measure("题面·句式", [norm(u) for u in user_all], "prose_pattern_top")
    if think_all:
        measure("think 段·逐字", think_all, "think_verbatim_top")
    else:
        print("   think 段：0（think-off 数据）")

    print("\n── 收尾句式（末 12 字）──")
    tails = [p[-12:] for p in prose_all if len(p) >= 12]
    measure("终答末 12 字", tails, "prose_verbatim_top")

    ok = not report["findings"]
    print("\n" + ("✅ 体检通过" if ok else f"🔴 体检发现 {len(report['findings'])} 项"))
    report["ok"] = ok
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("parquet")
    ap.add_argument("--json")
    ap.add_argument("--model", default=STUDENT_MODEL)
    a = ap.parse_args()
    rep = audit(Path(a.parquet), a.model)
    if a.json:
        Path(a.json).write_text(json.dumps(rep, ensure_ascii=False, indent=1))
    sys.exit(0 if rep["ok"] else 1)
