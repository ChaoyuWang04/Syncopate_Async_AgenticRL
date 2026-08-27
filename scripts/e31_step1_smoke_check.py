"""E31 第 1 步验收②（48 步冒烟）：从训练日志提取全部可量化判据，落 JSON 工件。

四个量化指标（E31 第 1 步②）+ 八判据（06 §1.5 起跑后清单的可程序化子集 + E31 判据行）：
  kl ≤ 1.5×kl_floor · 序列 IS 截断(frac_high+low) ≤ 0.10 · ESS/N ≥ 0.85 · steps ≥ 48
  ① [pool] 动态分池启用   ② [agent-loop] 下发记账   ③ [lora-probe] 非空(step≥1)
  ④ [sync-payload] 第2次起 lora_>0   ⑤ kl 全程回落地板   ⑥ prompt clip_ratio 恒 0
  ⑦ aborted_ratio 恒 0   ⑧ [unified-fp8] vLLM+trainer 两条判据行都在（机制接上的直接证据）
另报不设门槛的观察值：s/gstep（覆盖数按 global_step 差分实测，守则④）、新增 UserWarning 数。

用法：.venv/bin/python scripts/e31_step1_smoke_check.py <训练日志> [--out logs/e31/step1_smoke.json]
     [--require-inner]（第 3 步：内层两条判据行也必须在）[--kl-mult 1.5]（第 3 步用 2.0）
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def grab(text: str, key: str) -> list[float]:
    return [float(m) for m in re.findall(re.escape(key) + r":([0-9.eE+-]+)", text)]


def main() -> int:
    log = Path(sys.argv[1])
    out = Path(sys.argv[sys.argv.index("--out") + 1]) if "--out" in sys.argv \
        else ROOT / "logs/e31/step1_smoke.json"
    text = log.read_text()

    kl = grab(text, "rollout_corr/kl")
    frac_hi = grab(text, "rollout_corr/rollout_is_seq_fraction_high")
    frac_lo = grab(text, "rollout_corr/rollout_is_seq_fraction_low")
    ess = grab(text, "rollout_corr/rollout_is_eff_sample_size")
    clip = grab(text, "prompt_length/clip_ratio")
    aborted = grab(text, "response/aborted_ratio")
    gsteps = grab(text, "training/global_step")
    step_s = grab(text, "timing_s/step")

    floor = json.loads((ROOT / "logs/e31/kl_floor_bf16.json").read_text())["median"]
    kl_mult = float(sys.argv[sys.argv.index("--kl-mult") + 1]) if "--kl-mult" in sys.argv else 1.5
    lora_pushes = len(re.findall(r"\[sync-payload\] 本次同步推出去：.*lora_ ([1-9]\d*) 个", text))

    eight = {
        "pool_动态分池": "[pool] 动态分池启用" in text,
        "agent-loop_下发记账": "[agent-loop] 下发记账 ✓" in text,
        "lora-probe_非空": bool(re.search(r"\[lora-probe\] step=[1-9]\d* engine\.list_loras\(\)=\[\d", text)),
        "sync-payload_lora>0": lora_pushes >= 1,
        "kl_回落地板": bool(kl) and max(kl) <= kl_mult * floor,
        "prompt_clip_ratio_恒0": bool(clip) and max(clip) == 0.0,
        "aborted_恒0": bool(aborted) and max(aborted) == 0.0,
        "unified-fp8_两侧判据行": ("[unified-fp8] vLLM lm_head MXFP8 已生效" in text
                                   and "[unified-fp8] trainer lm_head MXFP8 已生效" in text),
    }
    if "--require-inner" in sys.argv:
        eight["unified-fp8_内层两侧判据行"] = (
            "[unified-fp8] vLLM 内层 MXFP8 已生效" in text
            and "[unified-fp8] trainer 内层 MXFP8 已生效" in text)

    # s/gstep：覆盖数按相邻行 global_step 差分实测（单行日志答不了，守则④）
    per_gstep = [s / d for s, d in
                 zip(step_s[1:], [b - a for a, b in zip(gsteps, gsteps[1:])]) if d > 0]

    report = {
        "log": str(log),
        "kl_values": kl, "kl_floor": floor,
        "seq_truncation_frac_max": max((h + l) for h, l in zip(frac_hi, frac_lo)) if frac_hi else 1.0,
        "ess_min": min(ess) if ess else 0.0,
        "steps_completed": int(max(gsteps)) + 1 if gsteps else 0,
        "eight_criteria": eight,
        "s_per_gstep": [round(v, 2) for v in per_gstep],
        "user_warnings": text.count("UserWarning"),
        "lora_push_count": lora_pushes,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    ok = (bool(kl) and max(kl) <= kl_mult * floor
          and report["seq_truncation_frac_max"] <= 0.10
          and report["ess_min"] >= 0.85
          and report["steps_completed"] >= 48
          and all(eight.values()))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print("✅ 冒烟验收全过" if ok else "❌ 有判据没过（看上面 false/超阈的项）")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
