"""量「模型到底动了多少」：‖ΔW‖/‖W‖。

    python -m syncopate.train.weight_shift --base models/Qwen3-4B --adapter checkpoints/sft/v13/epoch1

★★★ 为什么这个数值得单独一个模块

M7 那次跑完 150 步、零错误、曲线好看，结论却是「什么都没测出」。
根因是这一个数：**‖ΔW‖/‖W‖ = 0.0093%**（正常 LoRA 微调 0.5%–5%）。
**loss 降了、指标好看、而权重几乎没动** —— 它不在任何常规面板上。

⚠️⚠️ **口径必须钉死，否则两次跑的数不可比。**

2026-08-17 第一版在 SFT 里图省事，算的是「可训练参数自己的位移比」：

    ‖Δθ_trainable‖ / ‖θ_trainable(初始)‖        ← 🔴 **错的**

错在 **LoRA 的 B 矩阵初始化为零**：分母里只有 A，而 B 从 0 长起来
⇒ 这个比值既不反映"基座被改了多少"，也**不能和 M7 的 0.0093% 相比**。
第一个 epoch 就报了 15.99%，看着像"训过头"，其实是尺子的问题。

⇒ **正确口径：LoRA 实际叠加到基座上的那个增量，比基座本身。**

    ΔW_eff = (alpha / r) · B @ A          每个被适配的层
    ratio  = ‖ΔW_eff‖_F / ‖W_base‖_F      **只对被适配的层算**，
                                          不把没动过的层塞进分母稀释掉

★ 只算被适配的层，是因为把全模型放进分母会让这个数随 target_modules 的选择漂移
  —— 换一批 target_modules，同样的训练强度会得出不同的"位移"。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


def lora_effective_shift(base_dir: Path, adapter_dir: Path) -> dict[str, Any]:
    """返回 {ratio, per_layer, n_layers, alpha, r}。"""
    from safetensors.torch import load_file

    cfg = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
    r = int(cfg["r"])
    alpha = float(cfg["lora_alpha"])
    scale = alpha / r

    weights_path = adapter_dir / "adapter_model.safetensors"
    lora = load_file(str(weights_path))

    # 把 A / B 按层配对。peft 的命名形如
    #   base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight
    pairs: dict[str, dict[str, torch.Tensor]] = {}
    for key, tensor in lora.items():
        if ".lora_A" in key:
            pairs.setdefault(key.split(".lora_A")[0], {})["A"] = tensor
        elif ".lora_B" in key:
            pairs.setdefault(key.split(".lora_B")[0], {})["B"] = tensor

    # 基座权重：只加载被适配到的那些层
    index_path = base_dir / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
    else:
        index = {k: "model.safetensors" for k in []}

    shard_cache: dict[str, dict[str, torch.Tensor]] = {}

    def base_weight(prefix: str) -> torch.Tensor | None:
        # base_model.model.model.layers.0... -> model.layers.0...
        name = prefix.replace("base_model.model.", "") + ".weight"
        shard = index.get(name)
        if shard is None:
            return None
        if shard not in shard_cache:
            shard_cache[shard] = load_file(str(base_dir / shard))
        return shard_cache[shard].get(name)

    num_sq = 0.0
    den_sq = 0.0
    per_layer: list[tuple[str, float]] = []
    for prefix, ab in sorted(pairs.items()):
        if "A" not in ab or "B" not in ab:
            continue
        base = base_weight(prefix)
        if base is None:
            continue
        delta = (ab["B"].float() @ ab["A"].float()) * scale
        d = float(torch.linalg.matrix_norm(delta))
        w = float(torch.linalg.matrix_norm(base.float()))
        num_sq += d * d
        den_sq += w * w
        per_layer.append((prefix.replace("base_model.model.model.layers.", "L"), d / max(w, 1e-12)))

    ratio = (num_sq ** 0.5) / max(den_sq ** 0.5, 1e-12)
    return {"ratio": ratio, "n_layers": len(per_layer), "alpha": alpha, "r": r,
            "per_layer": per_layer}


def verdict(ratio: float) -> str:
    """红线见 docs/syncopate/14-sft-health-metrics.md。区间是**文献量级 + 我们自己两次实测**。"""
    pct = ratio * 100
    if pct < 0.1:
        return f"🔴 {pct:.4f}% —— 几乎没动，多半白训（M7 那次是 0.0093%）"
    if pct < 0.5:
        return f"🟡 {pct:.4f}% —— 偏小，能不能推动看评测"
    if pct <= 5.0:
        return f"✅ {pct:.4f}% —— 正常 LoRA 区间"
    if pct <= 10.0:
        return f"🟡 {pct:.4f}% —— 偏大，留意输出熵有没有塌"
    return f"🔴 {pct:.4f}% —— 过大，熵大概率塌了，接 GRPO 会探索不动"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--top", type=int, default=8, help="打印位移最大的前几层")
    args = ap.parse_args()

    out = lora_effective_shift(args.base, args.adapter)
    print(f"适配层 {out['n_layers']} 个 · lora_alpha={out['alpha']} r={out['r']}")
    print(f"‖ΔW_eff‖/‖W_base‖ = {verdict(out['ratio'])}")
    print("\n位移最大的层：")
    for name, v in sorted(out["per_layer"], key=lambda kv: -kv[1])[:args.top]:
        print(f"  {name:52} {v*100:7.3f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
