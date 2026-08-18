#!/usr/bin/env python3
"""量一个 RL ckpt「到底把模型推了多远」——直接读 verl 的分片 .pt，不需要先导出 adapter。

★ 为什么需要它（2026-08-18，因 E17 的 KL 讨论而写）：
E17 的 B 臂把 KL 约束**整个关掉**换了 12.7% 的吞吐。要判断这么做行不行，
**只看任务分是不够的** —— 任务分对「模型漂了多远」是瞎的，而两种失败长得一模一样：
    ① 真的训飞了（漂很远）
    ② 尺子太粗（根本测不出影响）
⇒ 必须**直接量位移**。口径抄 `weight_shift.py`（那份的教训写得很清楚）：

    ΔW_eff = (alpha / r) · B @ A       每个被适配的层实际叠加到基座上的增量
    ratio  = ‖ΔW_eff‖_F / ‖W_base‖_F   **只对被适配的层算**，别用没动过的层稀释分母

⚠️ 不要图省事去算「可训练参数自己的位移比」：LoRA 的 B 初始化为 0，
   那个比值既不反映基座被改了多少，也不能跨跑比较（weight_shift.py 的 §口径 记着这次踩坑）。

用法：
    python scripts/rl_ckpt_drift.py checkpoints/grpo/b16_ref_on_60/global_step_13/actor
    python scripts/rl_ckpt_drift.py <A> <B>        # 两个一起报，直接对比
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import torch

from syncopate.train.ckpt_guards import assert_ranks_identical



def drift(actor_dir: Path) -> dict:
    meta = json.loads((actor_dir / "lora_train_meta.json").read_text())
    scale = meta["lora_alpha"] / meta["r"]
    assert_ranks_identical(actor_dir)
    sd = torch.load(actor_dir / "model_world_size_3_rank_0.pt", map_location="cpu", weights_only=False)
    per_layer, num, den = [], 0.0, 0.0
    for k in sd:
        if not k.endswith("lora_A.default.weight"):
            continue
        base_key = k.replace("lora_A.default.weight", "base_layer.weight")
        b_key = k.replace("lora_A.default.weight", "lora_B.default.weight")
        if base_key not in sd or b_key not in sd:
            continue
        A = sd[k].float(); B = sd[b_key].float(); W = sd[base_key].float()
        dW = (scale * (B @ A))
        n, d = dW.norm().item(), W.norm().item()
        per_layer.append((k.replace("base_model.model.model.layers.", "L").replace(
            ".lora_A.default.weight", ""), n / d if d else 0.0))
        num += n ** 2; den += d ** 2
    ratio = (num ** 0.5) / (den ** 0.5) if den else 0.0
    per_layer.sort(key=lambda x: -x[1])
    return {"dir": str(actor_dir), "scale": scale, "n_adapted_layers": len(per_layer),
            "drift_ratio": ratio, "top_layers": per_layer[:5], "bottom_layers": per_layer[-3:]}

def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__); return 2
    for p in sys.argv[1:]:
        r = drift(Path(p))
        print(f"\n# {r['dir']}")
        print(f"  被适配的层 {r['n_adapted_layers']} 个 · alpha/r = {r['scale']}")
        print(f"  ★ ‖ΔW_eff‖/‖W_base‖ = **{r['drift_ratio']*100:.4f}%**"
              f"   （M7 那次是 0.0093% = 几乎没动；正常 LoRA 微调 0.5–5%）")
        print(f"    动得最多的层：{[(n, round(v*100,4)) for n, v in r['top_layers']]}")
        print(f"    动得最少的层：{[(n, round(v*100,4)) for n, v in r['bottom_layers']]}")
    return 0

if __name__ == "__main__": raise SystemExit(main())
