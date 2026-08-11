"""把 SFT 的 LoRA 合并进基座权重，产出 RL 的起点模型。

★★★ 为什么要合并，而不是用 verl 的 `lora_adapter_path`

verl 确实能从已有 adapter 继续训（`transformer_impl.py:_build_lora_module`，
`PeftModel.from_pretrained(..., is_trainable=True)`）。但它同时决定了 reference policy：

    ref_in_actor = lora_rank > 0 or lora_adapter_path is not None

用 LoRA 时 verl 算 reference 的办法是**把 adapter 关掉**——关掉之后得到的是**基座**。
于是 KL 项会一直把模型往「没学过业务的那个基座」拉，把 SFT 学到的东西往回拽。
手册 §24③ 说的正是这件事：**reference 必须指向 SFT ckpt，不是 base。**

合并之后：

    新基座 = SFT 模型   ⇒ RL 的起点对了
    关掉 adapter = SFT  ⇒ reference 也对了

一次合并同时解决两件事，而且不依赖 verl 那条 adapter 加载路径的细节行为。

代价是磁盘上多一份 8GB 的模型。相对于「跑两小时 RL 结果起点是错的」，这个代价可以忽略。

    python -m syncopate.train.merge_adapter \
        --base models/Qwen3-4B --adapter checkpoints/sft/v3_ctrl/epoch1 \
        --out models/Qwen3-4B-sft-e1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="把 LoRA 合并进基座")
    parser.add_argument("--base", default="models/Qwen3-4B")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base_path = str((ROOT / args.base).resolve())
    adapter_path = str((ROOT / args.adapter).resolve())
    out_dir = ROOT / args.out

    print(f"[合并] 基座 {base_path}")
    print(f"       adapter {adapter_path}")
    # 用 bf16 加载并合并：RL 侧 FSDP 也是 bf16 持参，保持一致，
    # 免得合并时用 fp32、训练时降到 bf16 又引入一次舍入差异
    model = AutoModelForCausalLM.from_pretrained(base_path, dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir, safe_serialization=True)
    AutoTokenizer.from_pretrained(base_path).save_pretrained(out_dir)
    # 留个来源记录：合并出来的模型看起来和基座一模一样，不写清楚来源，
    # 下一个窗口没法分辨 models/ 下哪个是哪个
    (out_dir / "syncopate_provenance.json").write_text(json.dumps({
        "base": args.base, "adapter": args.adapter,
        "note": "SFT LoRA 已合并进权重。RL 在此之上挂新 LoRA，"
                "关掉 adapter 得到的 reference 就是 SFT 本身（手册 §24③）。",
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[OK] -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
