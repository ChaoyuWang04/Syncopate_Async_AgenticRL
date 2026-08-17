#!/usr/bin/env python3
"""E01 / A5 · 从 nsys 的 sqlite 里把「一步的时间去哪了」拆开。

★ 为什么不用 `nsys stats` 的现成报告就完事：
它给的是**全局**的 kernel 排行，而我们要回答的是两条 track 共用的那个问题 ——
**每张卡上、每个进程各自忙了多久、忙在什么类型的活上、剩下的时间在等谁**。
`trainer` 和 `rollout` 在同一份 trace 里，不按进程拆开就分不出 update_actor 和 gen。

⚠️ 本项目没有 NVTX 阶段标注（verl 默认不打）⇒ **不能**把 GPU 时间直接切成
update_actor / old_log_prob / ref 三段。能给的是「算子类型 × 进程 × 卡」的分解，
以及**kernel 级的占空比**（比 nvidia-smi 采样精确一个量级）。
⇒ 想要真正的阶段归属，得给 verl 打 NVTX（记在 E01 §8）。

用法：
    python scripts/analyze_nsys_step.py logs/nsys/e01_rl_v13.sqlite --json _audit/infra/e01.json
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

# 算子归类。顺序即优先级（先匹配先算）。
CATEGORIES = [
    ("nccl",       re.compile(r"nccl|ncclDevKernel|AllGather|ReduceScatter|AllReduce", re.I)),
    ("attention",  re.compile(r"flash|attention|paged|unified_attention", re.I)),
    ("gemm",       re.compile(r"gemm|cutlass|cublas|matmul|s16816|wmma", re.I)),
    ("norm_softmax", re.compile(r"softmax|layer_norm|rms_norm|layernorm", re.I)),
    ("elementwise", re.compile(r"elementwise|vectorized|copy|cast|fill|add|mul", re.I)),
    ("reduce",     re.compile(r"reduce|sum|argmax|topk|sort", re.I)),
    ("quant_moe",  re.compile(r"quant|dequant|moe|expert", re.I)),
]


def classify(name: str) -> str:
    for label, rx in CATEGORIES:
        if rx.search(name):
            return label
    return "other"


def analyze(db: Path) -> dict:
    con = sqlite3.connect(str(db))
    q = lambda s, *a: list(con.execute(s, *a))

    # 采样窗口：用 kernel 的首尾时间戳（比 session 的 start/stop 更贴近"真在采"的那段）
    t0, t1 = q("SELECT MIN(start), MAX(end) FROM CUPTI_ACTIVITY_KIND_KERNEL")[0]
    window_ns = (t1 - t0) if t0 is not None else 0

    procs = {r[0]: r[2] for r in q("SELECT globalPid, Pid, name FROM PROCESSES")}
    pid_of = {r[0]: r[1] for r in q("SELECT globalPid, Pid FROM PROCESSES")}

    # ★ 关键的一张表：kernel 时间 × 进程 × 卡 × 算子类别
    rows = q("""
        SELECT k.globalPid, k.deviceId, s.value AS name, SUM(k.end - k.start), COUNT(*)
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN StringIds s ON s.id = k.demangledName
        GROUP BY k.globalPid, k.deviceId, s.value
    """)

    by_proc: dict = defaultdict(lambda: defaultdict(lambda: {"ns": 0, "calls": 0}))
    by_device: dict = defaultdict(int)
    top_kernels: dict = defaultdict(int)
    for gpid, dev, name, ns, n in rows:
        cat = classify(name or "")
        key = (procs.get(gpid, "?"), pid_of.get(gpid, gpid), dev)
        by_proc[key][cat]["ns"] += ns
        by_proc[key][cat]["calls"] += n
        by_device[dev] += ns
        top_kernels[name] += ns

    # memcpy（H2D/D2H 在这台机器上尤其要紧：没有 P2P，卡间都绕主机）
    memcpy = q("""
        SELECT k.deviceId, SUM(k.end - k.start), SUM(k.bytes), COUNT(*)
        FROM CUPTI_ACTIVITY_KIND_MEMCPY k GROUP BY k.deviceId
    """)

    out = {
        "db": str(db),
        "window_seconds": round(window_ns / 1e9, 3),
        "devices": {},
        "processes": {},
        "top_kernels": [
            {"name": n, "seconds": round(ns / 1e9, 3)}
            for n, ns in sorted(top_kernels.items(), key=lambda kv: -kv[1])[:15]
        ],
        "memcpy": [
            {"device": d, "seconds": round(ns / 1e9, 3), "gigabytes": round((b or 0) / 1e9, 3),
             "calls": n}
            for d, ns, b, n in memcpy
        ],
    }
    for dev, ns in sorted(by_device.items()):
        out["devices"][str(dev)] = {
            "kernel_seconds": round(ns / 1e9, 3),
            # ★ kernel 级占空比：这段窗口里这张卡真正在跑 kernel 的比例
            "busy_share": round(ns / window_ns, 4) if window_ns else None,
        }
    for (pname, pid, dev), cats in sorted(by_proc.items(), key=lambda kv: -sum(
            c["ns"] for c in kv[1].values())):
        total = sum(c["ns"] for c in cats.values())
        out["processes"][f"{pname}(pid={pid})@gpu{dev}"] = {
            "kernel_seconds": round(total / 1e9, 3),
            "busy_share": round(total / window_ns, 4) if window_ns else None,
            "by_category": {
                k: {"seconds": round(v["ns"] / 1e9, 3),
                    "share": round(v["ns"] / total, 4) if total else None,
                    "calls": v["calls"]}
                for k, v in sorted(cats.items(), key=lambda kv: -kv[1]["ns"])
            },
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("db", type=Path)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()
    if not args.db.exists():
        print(f"没有 {args.db} —— 先用 `nsys stats` 或 `nsys export` 生成 sqlite", file=sys.stderr)
        return 2

    res = analyze(args.db)
    print(f"# {res['db']}   采样窗口 {res['window_seconds']} s")
    print("\n  ★ 每张卡的 kernel 级占空比（这段窗口里真在跑 kernel 的比例）")
    for dev, v in res["devices"].items():
        print(f"    GPU{dev}   {v['kernel_seconds']:>8.2f} s   忙 {v['busy_share'] * 100:>5.1f}%")

    print("\n  ★ 按进程 × 卡（trainer 与 rollout 就是靠这个分开的）")
    for name, v in res["processes"].items():
        cats = "  ".join(f"{k} {vv['share'] * 100:.0f}%" for k, vv in
                         list(v["by_category"].items())[:5])
        print(f"    {name:<44}{v['kernel_seconds']:>8.2f} s  忙 {v['busy_share'] * 100:>5.1f}%   {cats}")

    print("\n  ★ 最贵的 kernel")
    for k in res["top_kernels"][:8]:
        print(f"    {k['seconds']:>8.2f} s  {k['name'][:110]}")

    if res["memcpy"]:
        print("\n  ★ memcpy（无 P2P ⇒ 卡间数据都绕主机内存）")
        for m in res["memcpy"]:
            print(f"    GPU{m['device']}  {m['seconds']:>7.2f} s  {m['gigabytes']:>8.2f} GB  {m['calls']} 次")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(res, indent=2, ensure_ascii=False))
        print(f"\n  → {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
