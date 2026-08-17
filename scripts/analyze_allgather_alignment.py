#!/usr/bin/env python3
"""A14 · 从 NCCL 的 TUNING 日志里统计「分块字节数不被 16 整除」的**按字节加权**占比。

背景：E18 §10 证明了机制 —— NCCL 的 Simple kernel 按 16 字节（128 位）向量化访存，
**每 rank 分块字节数只要不能被 16 整除，就整段退化成标量路径，all_gather 掉 12×**。
但那是在微基准上证明的。**还没证明 verl 的 ZeRO-3 真的撞在上面** ⇒ 这就是 A14。

⚠️⚠️ **必须按字节加权，不能按调用次数** —— 小张量再多也解释不了 6.02×。
   （一万次 128 字节的错位调用，加起来还不到一次 22 MB 的零头。）

用法：
    NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=TUNING python -m syncopate.train.launch_rl … > run.log
    python scripts/analyze_allgather_alignment.py run.log --json out.json

判据（跑之前写死，见 run_batch2_gpu.sh 的 P1）：
    按字节加权的 %16!=0 占比 > 80%  ⇒ 因果链闭合，接 A15（决定给谁提 upstream issue）
    < 20%                          ⇒ 6.02× 另有原因，E18 §10 还差一环
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

# NCCL INFO 行形如：
#   node:pid:tid [0] NCCL INFO AllGather: 22369620 Bytes -> Algo 1 proto 2 time ...
_OP_RE = re.compile(r"NCCL INFO (\w+): (\d+) Bytes")
# 它自己打的选择行：`-> Algo X proto Y`（不依赖我们对成本模型表的解读，见 E18 §10.1）
# ⚠️ NCCL 这里**有两种输出格式**：数字（`Algo 1 proto 2`）和名字（`Algo RING proto SIMPLE`）。
#    2026-08-17 第一版只认数字 ⇒ 「选择」列全空，差点写成「本次没拿到协议选择」。
#    ★ 又一次：**判据为空，先怀疑解析器，再怀疑现象。**
_CHOICE_RE = re.compile(r"(\w+): \d+ Bytes -> Algo (\w+) proto (\w+)")

ALGO = {0: "Tree", 1: "Ring", 2: "CollNetDirect", 3: "CollNetChain", 4: "NVLS", 5: "NVLSTree", 6: "PAT"}
PROTO = {0: "LL", 1: "LL128", 2: "Simple"}


def analyze(path: Path) -> dict:
    per_op: dict[str, dict] = defaultdict(
        lambda: {"calls": 0, "bytes": 0, "misaligned_calls": 0, "misaligned_bytes": 0,
                 "sizes": defaultdict(int)}
    )
    choices: dict[str, set] = defaultdict(set)

    with path.open(errors="replace") as fh:
        for line in fh:
            if "NCCL INFO" not in line or " Bytes" not in line:
                continue
            m = _OP_RE.search(line)
            if not m:
                continue
            op, nbytes = m.group(1), int(m.group(2))
            e = per_op[op]
            e["calls"] += 1
            e["bytes"] += nbytes
            e["sizes"][nbytes] += 1
            if nbytes % 16 != 0:
                e["misaligned_calls"] += 1
                e["misaligned_bytes"] += nbytes
            c = _CHOICE_RE.search(line)
            if c:
                a, pr = c.group(2), c.group(3)
                a = ALGO.get(int(a), a) if a.isdigit() else a
                pr = PROTO.get(int(pr), pr) if pr.isdigit() else pr
                choices[c.group(1)].add(f"{a}+{pr}")

    out = {}
    for op, e in per_op.items():
        top = sorted(e["sizes"].items(), key=lambda kv: -kv[0] * kv[1])[:8]
        out[op] = {
            "calls": e["calls"],
            "total_bytes": e["bytes"],
            "misaligned_share_by_bytes": round(e["misaligned_bytes"] / e["bytes"], 4) if e["bytes"] else None,
            "misaligned_share_by_calls": round(e["misaligned_calls"] / e["calls"], 4) if e["calls"] else None,
            "algo_proto_chosen": sorted(choices.get(op, [])),
            "top_sizes_by_volume": [
                {"bytes": b, "calls": n, "mod16": b % 16, "volume": b * n} for b, n in top
            ],
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("log", type=Path)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    if not args.log.exists():
        print(f"没有这个文件：{args.log}", file=sys.stderr)
        return 2

    res = analyze(args.log)
    if not res:
        print(f"⚠️ {args.log} 里没有 `NCCL INFO <Op>: N Bytes` 行 —— "
              f"是不是忘了 NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=TUNING？", file=sys.stderr)
        return 1

    print(f"# {args.log}")
    print(f"  {'算子':<16}{'调用数':>9}{'总字节':>16}{'★%16!=0(按字节)':>18}{'(按次数)':>11}  选择")
    for op, v in sorted(res.items(), key=lambda kv: -kv[1]["total_bytes"]):
        print(f"  {op:<16}{v['calls']:>9}{v['total_bytes']:>16,}"
              f"{(v['misaligned_share_by_bytes'] or 0) * 100:>17.1f}%"
              f"{(v['misaligned_share_by_calls'] or 0) * 100:>10.1f}%"
              f"  {','.join(v['algo_proto_chosen']) or '-'}")

    ag = res.get("AllGather")
    if ag and ag["misaligned_share_by_bytes"] is not None:
        share = ag["misaligned_share_by_bytes"]
        print(f"\n  ★ AllGather 按字节加权的错位占比 = {share * 100:.1f}%")
        if share > 0.8:
            print("  ⇒ **判据 P1 通过**：因果链闭合，verl 的 ZeRO-3 确实撞在 16 字节对齐悬崖上。"
                  "\n     下一步 A15：决定给 NCCL（kernel 整段退化）还是 FSDP2（padding 不管字节对齐）提 issue。")
        elif share < 0.2:
            print("  ⇒ **判据 P1 被推翻**：6.02× 另有原因，E18 §10 的因果链还差一环。"
                  "\n     ⛔ 不许再把「6.02× 由对齐造成」写成定论 —— 按 README §4 补一段四行的推翻记录。")
        else:
            print("  ⇒ 落在 20–80% 的灰区：对齐**解释了一部分但不是全部**，"
                  "要按分块尺寸分档再看（top_sizes_by_volume）。")
        print("  前几大体量的分块（按 体量=字节×次数 排）：")
        for s in ag["top_sizes_by_volume"]:
            flag = "🔴错位" if s["mod16"] else "🟢对齐"
            print(f"    {s['bytes']:>14,} B × {s['calls']:>5} 次  %16={s['mod16']:<3} {flag}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(res, indent=2))
        print(f"  → {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
