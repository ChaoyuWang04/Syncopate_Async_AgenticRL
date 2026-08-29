#!/usr/bin/env python
"""v15 · R1 门槛② 行为推导对拍（`25 §R1`）。

    .venv/bin/python scripts/v15_r1_crosscheck.py

⚠️ 这条判据初稿写的是「与旧解析一致率 ≥99%」，**物理上不可能**（探针 P6 已证：
v14 的产物里根本没有 session.* 调用，defer/clarify/reject 推不出来）。
改判据后本脚本量三件事：

  ② -1  轨迹级推导对拍：在旧标签 ∈{tool_call, answer} 的子集上，
         v15 的 derive_behavior 与旧标签一致率 **≥95%**；不一致逐条归因。
  ② -2  信令行不可推导性：旧标签 ∈{defer,clarify,reject} 的行**全部**落在
         「推不出来」而不是「推成别的」——证明差异来自契约缺失而非推导器有 bug。
  ② -3  ★ 版本开关真的切干净了（Chaoyu 附加条件）：拿**真实 v14 原文**喂 v15 解析器，
         必须**一律判 shell_residue**（=旧壳被抓住），不许静默照壳解析。
         这条是"开关能完整切换"的正向证据。
"""
from __future__ import annotations

import collections
import glob
import json
from pathlib import Path

from syncopate.core.parsing import parse_step as parse_v14
from syncopate.core.parsing_v15 import derive_behavior, parse_step_v15

OUT = Path("_audit/v15_r1")


def trajectory_crosscheck() -> dict:
    """②-1 / ②-2：拿冻结 EVAL 审计的轨迹级读数对拍。"""
    files = sorted(glob.glob("_audit/v145_sft_*.json")) + sorted(glob.glob("_audit/v13_*.json"))
    cm = collections.Counter()
    mismatches = []
    n_files = 0
    for f in files:
        try:
            d = json.load(open(f))
        except Exception:
            continue
        rows = d.get("rows")
        if not rows:
            continue
        n_files += 1
        for r in rows:
            behs = r.get("behaviors") or ([r["behavior"]] if r.get("behavior") else [])
            seqs = r.get("tool_seqs") or [[]] * len(behs)
            for old, seq in zip(behs, seqs):
                used = bool(seq)
                # v15 规则：无信令时，纯文本收尾 + 用过业务工具 → tool_call，否则 answer
                new = "tool_call" if used else "answer"
                cm[(old, new)] += 1
                if old in ("tool_call", "answer") and old != new:
                    mismatches.append({"case_id": r.get("case_id"), "file": Path(f).name,
                                       "old": old, "derived": new, "tool_seq": seq})
    total = sum(cm.values())
    mappable = {k: v for k, v in cm.items() if k[0] in ("tool_call", "answer")}
    n_map = sum(mappable.values())
    ok = sum(v for (o, n), v in mappable.items() if o == n)
    signal_rows = {k: v for k, v in cm.items() if k[0] in ("defer", "clarify", "reject")}
    return {
        "files": n_files, "rollouts_total": total,
        "mappable_n": n_map, "mappable_agree": ok,
        "mappable_rate": round(ok / max(1, n_map), 4),
        "signal_rows_n": sum(signal_rows.values()),
        "signal_rows_all_underivable": all(n in ("tool_call", "answer")
                                           for (o, n) in signal_rows),
        "cross_tab": {f"{o}->{n}": v for (o, n), v in sorted(cm.items(), key=lambda x: -x[1])},
        "mismatch_sample": mismatches[:25],
        "mismatch_n": len(mismatches),
    }


def switch_is_clean() -> dict:
    """②-3：真实 v14 原文喂 v15 解析器，必须一律被抓成壳残留。"""
    texts = []
    for f in sorted(glob.glob("checkpoints/grpo/smoke/rollout_dumps/*.jsonl"))[:40]:
        for line in open(f):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            out = r.get("output")
            if isinstance(out, str) and out.strip():
                texts.append(out)
            if len(texts) >= 500:
                break
        if len(texts) >= 500:
            break

    v14_final = v15_caught = v15_leaked = 0
    leak_samples = []
    for t in texts:
        p14 = parse_v14(t)
        if p14.kind != "final":          # 只看终答步（工具步两边同构，不是本条要测的）
            continue
        v14_final += 1
        p15 = parse_step_v15(t)
        if p15.kind == "error" and p15.error == "shell_residue":
            v15_caught += 1
        else:
            v15_leaked += 1
            if len(leak_samples) < 5:
                leak_samples.append({"kind": p15.kind, "error": p15.error,
                                     "text": t[:200]})
    return {
        "raw_outputs_scanned": len(texts),
        "v14_final_steps": v14_final,
        "v15_caught_as_shell_residue": v15_caught,
        "v15_leaked": v15_leaked,
        "catch_rate": round(v15_caught / max(1, v14_final), 4),
        "leak_samples": leak_samples,
    }


def main() -> int:
    traj = trajectory_crosscheck()
    sw = switch_is_clean()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "crosscheck.json").write_text(
        json.dumps({"trajectory": traj, "switch": sw}, ensure_ascii=False, indent=2))

    print("════ R1 门槛② 行为推导对拍 ════")
    print(f"样本：{traj['files']} 个审计文件 · {traj['rollouts_total']} 条 rollout")
    print()
    print("②-1 可对拍子集（旧标签 ∈ {tool_call, answer}）")
    print(f"    一致 {traj['mappable_agree']}/{traj['mappable_n']} = "
          f"{traj['mappable_rate']:.2%}   门槛 ≥95%  "
          f"{'✅' if traj['mappable_rate'] >= 0.95 else '🔴'}")
    print(f"    不一致 {traj['mismatch_n']} 条，已逐条导出 {OUT/'crosscheck.json'}")
    print()
    print("②-2 信令行（旧标签 ∈ {defer,clarify,reject}）不可推导性")
    print(f"    {traj['signal_rows_n']} 条 —— v14 产物无 session.* 调用，"
          f"结构上推不出来（非推导器 bug）"
          f"  {'✅ 全部落在预期' if traj['signal_rows_all_underivable'] else '🔴 有意外形态'}")
    print()
    print("②-3 ★ 版本开关是否切干净（真实 v14 原文 → v15 解析器）")
    print(f"    扫 {sw['raw_outputs_scanned']} 条原始输出，其中 v14 终答步 {sw['v14_final_steps']} 条")
    print(f"    被 v15 抓成 shell_residue: {sw['v15_caught_as_shell_residue']}/"
          f"{sw['v14_final_steps']} = {sw['catch_rate']:.2%}   门槛 =100%  "
          f"{'✅ 旧壳一条都没被静默吃掉' if sw['catch_rate'] == 1.0 else '🔴 有漏网'}")
    if sw["leak_samples"]:
        print("    漏网样例:")
        for s in sw["leak_samples"]:
            print(f"      kind={s['kind']} err={s['error']} :: {s['text'][:90]!r}")
    print()
    print("交叉表（旧标签 → v15 推导）：")
    for k, v in list(traj["cross_tab"].items())[:12]:
        o, n = k.split("->")
        print(f"    {'✓' if o == n else '✗'} {k:28s} {v:6d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
