#!/usr/bin/env python3
"""E01 / A5 · 从 nsys 的 sqlite 里把「一步的时间去哪了」拆开。

★ 为什么不用 `nsys stats` 的现成报告就完事：
它给的是**全局**的 kernel 排行，而我们要回答的是两条 track 共用的那个问题 ——
**每张卡上、每个进程各自忙了多久、忙在什么类型的活上、剩下的时间在等谁**。
`trainer` 和 `rollout` 在同一份 trace 里，不按进程拆开就分不出 update_actor 和 gen。

两种模式：
  ① 无 NVTX（旧 trace）：给「算子类型 × 进程 × 卡」的分解 + **kernel 级占空比**
     （比 nvidia-smi 采样精确一个量级）。
  ② 🆕 有 NVTX（`launch_rl --nvtx` 跑出来的）：**按阶段归属** kernel 时间
     —— update_actor / old_log_prob / ref / gen / param_sync 各占多少，**这才是 A5 的正题**。
     verl 自带的 `marked_timer` 名字里有 marker、函数体里一个都没有 ⇒ 我们自己打（verl_patches）。

用法：
    python scripts/infra/analyze_nsys_step.py logs/nsys/e01_rl_v13.sqlite --json _audit/infra/e01.json
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


def phase_attribution(con, window: tuple[int, int]) -> dict:
    """★ A5 的正题：把 kernel 时间按 **NVTX 阶段** 归属。

    ⚠️ 两个进程的事：range 打在 **trainer driver**（阶段边界在那里），
    kernel 跑在 **WorkerDict**（另一个进程）⇒ 只能靠 nsys 的**同一条时间轴**对齐，
    按时间区间归属，不是按进程。
    ⚠️ range 会嵌套（`step` 套着 `gen`/`update_actor`…）⇒ 取**最内层**那个，否则重复计。
    """
    import bisect

    rows = list(con.execute("""
        SELECT s.value, e.start, e.end FROM NVTX_EVENTS e
        JOIN StringIds s ON s.id = e.textId
        WHERE s.value LIKE 'syncopate/%' AND e.end IS NOT NULL
    """))
    if not rows:
        # 有些版本把文本直接写在 text 列
        rows = list(con.execute("""
            SELECT text, start, end FROM NVTX_EVENTS
            WHERE text LIKE 'syncopate/%' AND end IS NOT NULL
        """))
    if not rows:
        return {"error": "trace 里没有 syncopate/* 的 NVTX range —— "
                         "是不是没传 --nvtx，或者补丁只在 driver 生效？"}

    ranges = sorted(((int(a), int(b), t) for t, a, b in rows), key=lambda r: r[0])
    starts = [r[0] for r in ranges]

    per_phase: dict[str, dict] = defaultdict(lambda: {"ns": 0, "kernels": 0})
    unattributed = {"ns": 0, "kernels": 0}
    for kstart, kend in con.execute(
            "SELECT start, end FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE start >= ? AND end <= ?",
            window):
        j = bisect.bisect_right(starts, kstart) - 1
        best = None
        # 往回找有限几个（嵌套深度有限），取**包住它且最短**的那个 = 最内层
        for i in range(j, max(-1, j - 12), -1):
            a, b, t = ranges[i]
            if a <= kstart and kend <= b and (best is None or (b - a) < (best[1] - best[0])):
                best = ranges[i]
        if best is None:
            unattributed["ns"] += kend - kstart
            unattributed["kernels"] += 1
        else:
            e = per_phase[best[2].split("/", 1)[1]]
            e["ns"] += kend - kstart
            e["kernels"] += 1

    total = sum(v["ns"] for v in per_phase.values()) + unattributed["ns"]
    return {
        "n_ranges": len(ranges),
        "by_phase": {k: {"seconds": round(v["ns"] / 1e9, 3),
                         "share": round(v["ns"] / total, 4) if total else None,
                         "kernels": v["kernels"]}
                     for k, v in sorted(per_phase.items(), key=lambda kv: -kv[1]["ns"])},
        "unattributed": {"seconds": round(unattributed["ns"] / 1e9, 3),
                         "share": round(unattributed["ns"] / total, 4) if total else None,
                         "kernels": unattributed["kernels"]},
    }


def gap_analysis(con, top: int = 15) -> dict:
    """★ 空泡：**GPU 上两个 kernel 之间的间隙**，按进程分、按总时长排。

    这是「图形界面用眼睛看时间轴」那件事的可计算版本：
    把每个进程的 kernel 按开始时间排好，算相邻之间的空档，
    再按「空档长度」分档统计。**GUI 看得见的东西，这里都能算出来** ——
    差别只在 GUI 适合开放式探索，这个适合量化与两次跑对比。

    ⚠️ 一个 kernel 结束到下一个开始之间的空档，成因可能是：
    CPU 侧还没提交（launch 慢 / Python 慢）、在等别的流、在等 memcpy、或者在等别的进程。
    **本函数只回答「有多少、多长、在谁身上」，不回答「为什么」** —— 后者要看那段时间里
    CPU 侧在跑什么 API（`CUPTI_ACTIVITY_KIND_RUNTIME`），排成下一步。
    """
    procs = {r[0]: (r[1], r[2]) for r in con.execute("SELECT globalPid, Pid, name FROM PROCESSES")}
    out = {}
    for gpid, (pid, pname) in procs.items():
        rows = list(con.execute(
            "SELECT start, end FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE globalPid=? ORDER BY start",
            (gpid,)))
        if len(rows) < 2:
            continue
        span = rows[-1][1] - rows[0][0]
        busy = sum(e - s for s, e in rows)
        gaps = []
        prev_end = rows[0][1]
        for s, e in rows[1:]:
            if s > prev_end:
                gaps.append(s - prev_end)
            prev_end = max(prev_end, e)
        gaps.sort(reverse=True)
        total_gap = sum(gaps)
        # 分档：看空泡是「很多小的」还是「少数大的」——两者的解法完全不同
        buckets = {"<10µs": 0, "10–100µs": 0, "0.1–1ms": 0, "1–10ms": 0, ">10ms": 0}
        for g in gaps:
            us = g / 1e3
            k = ("<10µs" if us < 10 else "10–100µs" if us < 100 else
                 "0.1–1ms" if us < 1000 else "1–10ms" if us < 10000 else ">10ms")
            buckets[k] += g
        out[f"{pname}(pid={pid})"] = {
            "span_s": round(span / 1e9, 3),
            "busy_s": round(busy / 1e9, 3),
            "busy_share": round(busy / span, 4) if span else None,
            "gap_s": round(total_gap / 1e9, 3),
            "n_gaps": len(gaps),
            "top_gaps_ms": [round(g / 1e6, 3) for g in gaps[:top]],
            "gap_seconds_by_size": {k: round(v / 1e9, 3) for k, v in buckets.items()},
        }
    return out


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
    out["phases"] = phase_attribution(con, (t0, t1)) if t0 is not None else {"error": "无 kernel"}
    out["gaps"] = gap_analysis(con)
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
    print("\n  ⚠️ 按 deviceId 聚合（**会把多张卡叠成一张**：Ray 给每个 worker 只设一张卡，")
    print("     每个进程眼里自己那张就是 0 ⇒ 下面这个百分比可能 >100%，只作合计用，别当占空比读）")
    for dev, v in res["devices"].items():
        print(f"    dev{dev}（合计）  {v['kernel_seconds']:>8.2f} s   {v['busy_share'] * 100:>6.1f}%"
              f"   ← 想看单卡占空比请看下面**按进程**那张表")

    print("\n  ★ 按进程 × 卡（trainer 与 rollout 就是靠这个分开的）")
    for name, v in res["processes"].items():
        cats = "  ".join(f"{k} {vv['share'] * 100:.0f}%" for k, vv in
                         list(v["by_category"].items())[:5])
        print(f"    {name:<44}{v['kernel_seconds']:>8.2f} s  忙 {v['busy_share'] * 100:>5.1f}%   {cats}")

    ph = res.get("phases", {})
    if ph.get("by_phase"):
        print(f"\n  ★★ 按 NVTX 阶段归属（{ph['n_ranges']} 个 range）—— **这才是 A5 的正题**")
        for k, v in ph["by_phase"].items():
            print(f"    {k:<22}{v['seconds']:>9.2f} s{(v['share'] or 0) * 100:>8.1f}%   {v['kernels']} 个 kernel")
        u = ph["unattributed"]
        print(f"    {'（不在任何 range 内）':<22}{u['seconds']:>9.2f} s{(u['share'] or 0) * 100:>8.1f}%"
              f"   {u['kernels']} 个 kernel   ← 大多是 rollout 侧（它不在 trainer 的 range 里）")
    elif ph.get("error"):
        print(f"\n  ⚠️ 阶段归属不可用：{ph['error']}")

    if res.get("gaps"):
        print("\n  ★★ 空泡（每个进程的 kernel 之间的空档）—— 「有多少、多长、在谁身上」")
        print(f"    {'进程':<34}{'跨度':>8}{'忙':>8}{'空':>8}{'忙占比':>8}   空泡按长度分档（秒）")
        for k, v in sorted(res["gaps"].items(), key=lambda kv: -kv[1]["gap_s"]):
            dist = " ".join(f"{kk}:{vv}" for kk, vv in v["gap_seconds_by_size"].items() if vv > 0.01)
            print(f"    {k:<34}{v['span_s']:>8.1f}{v['busy_s']:>8.1f}{v['gap_s']:>8.1f}"
                  f"{(v['busy_share'] or 0) * 100:>7.1f}%   {dist}")
        print("    ⚠️ 本节只答「有多少空泡」，不答「为什么」——后者要把空档和 CPU 侧的 API 调用对齐")

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
