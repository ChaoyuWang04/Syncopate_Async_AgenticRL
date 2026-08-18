#!/usr/bin/env python
"""删大文件之前，先把「还有分析价值的那点东西」提出来。

    python scripts/extract_ckpt_fingerprint.py <actor_dir> [--keep-lora]

产出（都放在同一个 actor 目录里，加起来 KB 级 / 带 --keep-lora 时 ~250 MB/rank）：
    rank_fingerprint.json      每个 rank 的逐层 LoRA 范数 + 逐对相对差 + 优化器状态差
    model_lora_only_rank<N>.pt 仅 --keep-lora：每个 rank 的 LoRA 权重（全量分片的 3%）

★ 为什么要这一步（2026-08-18）

E21（三个 rank 各训各的）的证据**就藏在这些 27 GB 的文件里**。
直接删 = 把证据一起删掉；全留 = 磁盘撑爆（曾经因此丢过一次最终 ckpt）。
⇒ **把证据从"一堆权重"变成"一组数"**，然后才能安心删。

⚠️ 本脚本**只写不删**。删除动作单独做，这样出错时还有得救。
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="提取 ckpt 的跨 rank 指纹")
    ap.add_argument("actor_dir", type=Path)
    ap.add_argument("--keep-lora", action="store_true", help="额外把每个 rank 的 LoRA 权重单独存一份")
    ap.add_argument("--layers", type=int, default=6, help="逐层范数取前几层（全量太长）")
    args = ap.parse_args(argv)

    import torch

    actor = args.actor_dir
    shards = sorted(actor.glob("model_world_size_*_rank_*.pt"))
    if not shards:
        print(f"[跳过] {actor} 没有全量分片")
        return 0

    # ⚠️ 记「这一跑是什么时候跑的」，不是「指纹是什么时候提的」——
    #    否则下游按 mtime 排序会把最新提取的当成最新的跑（踩过）。
    run_dir = actor.parent.parent
    marker = run_dir / "dispatched.jsonl"
    fp: dict = {"actor_dir": str(actor), "n_ranks": len(shards),
                "run_mtime": (marker.stat().st_mtime if marker.exists() else None),
                "ranks": {}, "pairwise": {}}
    loras: dict[int, dict] = {}
    for i, sh in enumerate(shards):
        sd = torch.load(sh, map_location="cpu", weights_only=False)
        lora = {k: v for k, v in sd.items() if "lora_" in k}
        loras[i] = lora
        keys = sorted(lora)
        fp["ranks"][str(i)] = {
            "file": sh.name,
            "n_lora_tensors": len(lora),
            "global_norm": float(torch.cat([v.float().flatten() for v in lora.values()]).norm()),
            "per_layer_norm": {k: float(lora[k].float().norm()) for k in keys[: args.layers]},
        }
        if args.keep_lora:
            out = actor / f"model_lora_only_rank{i}.pt"
            torch.save(lora, out)
            print(f"  · 存 {out.name}  {out.stat().st_size / 1024**2:.0f} MB")
        del sd

    # ★ 判据本身：跨 rank 逐位相同吗？不同的话差多少？
    for i, j in itertools.combinations(range(len(shards)), 2):
        ki = sorted(loras[i])
        diff = [k for k in ki if not torch.equal(loras[i][k], loras[j][k])]
        a = torch.cat([loras[i][k].float().flatten() for k in ki])
        b = torch.cat([loras[j][k].float().flatten() for k in ki])
        fp["pairwise"][f"{i}-{j}"] = {
            "n_tensors_differing": len(diff),
            "n_tensors_total": len(ki),
            "relative_diff": float((a - b).norm() / max(a.norm().item(), 1e-12)),
            "identical": len(diff) == 0,
        }

    # 优化器状态（E21 的静态证据里 exp_avg_sq 相对差 99%）
    opts = sorted(actor.glob("optim_world_size_*_rank_*.pt"))
    if len(opts) >= 2:
        def flat(path):
            o = torch.load(path, map_location="cpu", weights_only=False)
            st = o.get("state", o)
            out = {}
            if isinstance(st, dict):
                for k, v in st.items():
                    if isinstance(v, dict):
                        for kk, vv in v.items():
                            if torch.is_tensor(vv) and vv.numel() > 1:
                                out[f"{k}.{kk}"] = vv.float()
            return out
        f0, f1 = flat(opts[0]), flat(opts[1])
        common = sorted(set(f0) & set(f1))[:60]
        if common:
            bad = [k for k in common if not torch.equal(f0[k], f1[k])]
            a = torch.cat([f0[k].flatten() for k in common])
            b = torch.cat([f1[k].flatten() for k in common])
            fp["optimizer_0_vs_1"] = {
                "n_sampled": len(common),
                "n_differing": len(bad),
                "relative_diff": float((a - b).norm() / max(a.norm().item(), 1e-12)),
                "identical": len(bad) == 0,
            }

    out = actor / "rank_fingerprint.json"
    out.write_text(json.dumps(fp, ensure_ascii=False, indent=1), encoding="utf-8")
    out = actor / "rank_fingerprint.json"
    out.write_text(json.dumps(fp, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[OK] {out}")
    for k, v in fp["pairwise"].items():
        if v["identical"]:
            print(f"   rank{k}: ✅ 逐位相同")
        else:
            print(f"   rank{k}: 🔴 {v['n_tensors_differing']}/{v['n_tensors_total']} 不同, "
                  f"相对差 {v['relative_diff']:.4f}")
    o = fp.get("optimizer_0_vs_1")
    if o:
        tag = "✅ 一致" if o["identical"] else f"🔴 {o['n_differing']}/{o['n_sampled']} 不同, 相对差 {o['relative_diff']:.4f}"
        print(f"   优化器 rank0-1: {tag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
