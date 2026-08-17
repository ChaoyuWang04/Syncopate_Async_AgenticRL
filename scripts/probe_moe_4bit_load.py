#!/usr/bin/env python3
"""A9 · 4bit MoE 的加载路径：绕开 bnb 逐层量化造成的显存碎片。

★ 需求从哪来（README §7 · A1 的「新挖出的坑」）：
A1 证明 Qwen3-30B-A3B 4bit 能跑（15.6 GB、LoRA 30.1 M、前反向通过），
**但加载时 bnb 逐层量化造成严重碎片**：权重只有 13.32 GB，却有
**17.43 GB reserved-but-unallocated**，直接 OOM。
当时靠 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 解掉 ——
**而它在真训练路径上用不了**：和 vLLM colocate 的内存池冲突
（`AssertionError: Expandable segments are not compatible with memory pool`，
pytorch#147851，`launch_rl` 里专门 pop 掉这个环境变量）。

⇒ **A2 真跑之前必须换一条加载路径。** 本探针比较三条：

    ① 现状     在线 4bit 量化（bnb 逐层）                        —— 复现碎片
    ② 预量化   先量化一次 → `save_pretrained` → **从盘直接加载**   —— 期望无碎片
    ③ 低内存   `low_cpu_mem_usage` + `max_memory` 限额             —— 兜底

★ 判据（跑之前写死）：
  P1  ①会出现 reserved − allocated > **3 GB** 的碎片（复现 A1 那个现象）
  P2  ②的碎片 < **1 GB**，且**不需要 expandable_segments** ⇒ A2 的加载路径就用它
  P3  若②也碎，说明碎片来自 bnb 的**权重布局**而不是量化过程 ⇒ 得换量化后端（A2 要重新设计）

用法：
    python scripts/probe_moe_4bit_load.py --stage online   --json logs/a9_online.json
    python scripts/probe_moe_4bit_load.py --stage save     --out models/Qwen3-30B-A3B-nf4
    python scripts/probe_moe_4bit_load.py --stage preload  --pre models/Qwen3-30B-A3B-nf4 \\
        --json logs/a9_preload.json
⚠️ **三个阶段必须分进程跑**：同一进程里量化完再加载，测到的碎片是两者叠加的。
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

MODEL = "models/Qwen3-30B-A3B-Instruct-2507"


def mem() -> dict:
    import torch

    a = torch.cuda.memory_allocated() / 2**30
    r = torch.cuda.memory_reserved() / 2**30
    return {"allocated_gb": round(a, 3), "reserved_gb": round(r, 3),
            "fragment_gb": round(r - a, 3)}


def report(tag: str, t0: float) -> dict:
    m = mem()
    m["seconds"] = round(time.perf_counter() - t0, 1)
    print(f"  {tag:<28} 已分配 {m['allocated_gb']:6.2f} GB · 预留 {m['reserved_gb']:6.2f} GB"
          f" · **碎片 {m['fragment_gb']:6.2f} GB** · {m['seconds']}s")
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["online", "save", "preload"], required=True)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--pre", default="models/Qwen3-30B-A3B-nf4")
    ap.add_argument("--out", default="models/Qwen3-30B-A3B-nf4")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    # ⚠️ 判据的一部分：**不许**开 expandable_segments —— 它在真训练路径上用不了，
    #    开着测出来的「不碎」是假的（A1 就是这么被骗过一次的）。
    # （`save` 是离线的一次性动作，不进训练路径 ⇒ 它可以开；测量的那两个阶段不许开）
    if args.stage != "save" and os.environ.get("PYTORCH_CUDA_ALLOC_CONF"):
        print(f"⛔ 检测到 PYTORCH_CUDA_ALLOC_CONF={os.environ['PYTORCH_CUDA_ALLOC_CONF']}"
              f" —— 本探针必须在**没有**它的情况下跑，否则结论不成立")
        return 2

    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    out: dict = {"stage": args.stage, "model": args.model}
    t0 = time.perf_counter()
    quant = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
    )

    if args.stage == "online":
        print(f"# ① 在线 4bit 量化（复现 A1 的碎片）")
        model = AutoModelForCausalLM.from_pretrained(
            args.model, quantization_config=quant, dtype=torch.bfloat16, device_map={"": 0})
        out["after_load"] = report("在线量化后", t0)
        out["verdict"] = ("🔴 复现了碎片（>3 GB）" if out["after_load"]["fragment_gb"] > 3
                          else "⚠️ 没复现碎片 —— A1 的现象可能依赖别的条件，先别改 A2 的设计")
        del model

    elif args.stage == "save":
        print(f"# ② 量化一次并存盘 → {args.out}")
        # 这一步**允许**开 expandable_segments（它是离线的一次性动作，不进训练路径）
        model = AutoModelForCausalLM.from_pretrained(
            args.model, quantization_config=quant, dtype=torch.bfloat16, device_map={"": 0})
        out["after_quant"] = report("量化后", t0)
        Path(args.out).mkdir(parents=True, exist_ok=True)
        model.save_pretrained(args.out, safe_serialization=True)
        out["saved_to"] = args.out
        out["after_save"] = report("存盘后", t0)

    else:  # preload
        print(f"# ③ 从**预量化**的盘上直接加载：{args.pre}")
        model = AutoModelForCausalLM.from_pretrained(
            args.pre, dtype=torch.bfloat16, device_map={"": 0})
        out["after_load"] = report("预量化加载后", t0)
        frag = out["after_load"]["fragment_gb"]
        out["verdict"] = ("✅ P2 成立：预量化加载几乎不碎（<1 GB）⇒ A2 的加载路径就用它"
                          if frag < 1 else
                          "🔴 P3：预量化也碎 ⇒ 碎片来自 bnb 的权重布局，不是量化过程 ⇒ A2 要重新设计")
        print(f"  ⇒ {out['verdict']}")
        del model

    torch.cuda.empty_cache()
    out["after_empty_cache"] = mem()
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(out, indent=2, ensure_ascii=False))
        print(f"  → {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
