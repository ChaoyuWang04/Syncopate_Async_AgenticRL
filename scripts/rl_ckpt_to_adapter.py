#!/usr/bin/env python3
"""把 verl 的 RL ckpt 抽成 PEFT 格式的 LoRA adapter —— 评测链路的那一步。

★ 为什么需要它（2026-08-18）：
verl 存的是**分片的全量 state_dict**（`model_world_size_3_rank_*.pt`，每个 8.5 GB，
其中 97% 是和基座逐字节相同的冻结权重）。而 `eval_local` / `merge_adapter` 要的是
**PEFT 目录**（`adapter_config.json` + `adapter_model.safetensors`）。
中间这一步此前没有脚本，导致「跑完的模型没法评测」——
而任务级尺子（B5）正是这条线唯一的成功判据（主线 17 §4.3）。

★ 键名差异（实测比对过主线产出的 adapter）：
    verl ckpt   ...q_proj.lora_A.**default**.weight
    PEFT 期望   ...q_proj.lora_A.weight              ← 少一个 `.default`

⚠️ DDP（`fsdp_size=1`）下每个 rank 持有完整副本 ⇒ 只读 rank_0 就够。
   若将来用分片（fsdp_size>1），这里要改成合并所有 rank。**已加断言。**

用法：
    python scripts/rl_ckpt_to_adapter.py checkpoints/grpo/<exp>/global_step_N/actor \\
        --out models/adapters/<name>
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch

from syncopate.train.ckpt_guards import assert_ranks_identical
from safetensors.torch import save_file

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("actor_dir", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    meta = json.loads((a.actor_dir / "lora_train_meta.json").read_text())
    shards = sorted(a.actor_dir.glob("model_world_size_*_rank_*.pt"))
    assert shards, f"没找到分片：{a.actor_dir}"
    ws = int(shards[0].name.split("world_size_")[1].split("_")[0])
    sd = torch.load(shards[0], map_location="cpu", weights_only=False)

    lora = {k.replace(".default.weight", ".weight"): v.contiguous()
            for k, v in sd.items() if "lora_" in k}
    assert lora, "这个 ckpt 里没有 lora_ 键 —— 是全参微调？"
    # ⚠️ DDP 下每个 rank 是完整副本；分片 / 梯度不同步时这个假设都不成立。
    # ★★ 就是这一句在 2026-08-18 炸出了 E21（三个 rank 各训各的）。
    #    它现在提成了共用函数 —— 因为当时另外两个读 ckpt 的脚本都没有这句话，
    #    而"保护性逻辑只写在一条路径上"正是本项目记过的失效形状。
    n_checked = assert_ranks_identical(a.actor_dir, sample=len(lora))
    if n_checked:
        print(f"  ✅ 校验：{n_checked} 个 LoRA 张量在 rank_0 / rank_1 上逐位相同（DDP 副本）")

    tm = sorted({k.split(".")[-4] for k in lora if k.endswith("lora_A.weight")})
    cfg = {"task_type": "CAUSAL_LM", "peft_type": "LORA", "auto_mapping": None,
           "base_model_name_or_path": None, "revision": None, "inference_mode": False,
           "r": meta["r"], "lora_alpha": meta["lora_alpha"], "lora_dropout": 0.0,
           "fan_in_fan_out": False, "bias": "none", "target_modules": tm,
           "modules_to_save": None, "init_lora_weights": True, "layers_to_transform": None,
           "layers_pattern": None, "rank_pattern": {}, "alpha_pattern": {}}
    a.out.mkdir(parents=True, exist_ok=True)
    save_file(lora, a.out / "adapter_model.safetensors")
    (a.out / "adapter_config.json").write_text(json.dumps(cfg, indent=4))
    print(f"  {a.actor_dir}\n  ⇒ {a.out}  ({len(lora)} 个张量 · r={meta['r']} · "
          f"alpha={meta['lora_alpha']} · target={len(tm)} 类)")
    return 0

if __name__ == "__main__": raise SystemExit(main())
