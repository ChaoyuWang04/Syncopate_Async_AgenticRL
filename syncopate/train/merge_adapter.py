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



def _assert_merge_landed(base_path: str, adapter_path: str, merged_model, max_resid: float = 0.5) -> None:
    """判据：合并后的权重 − 基座权重 ≈ adapter 的 ΔW_eff。

    量两件事（**必须分开看**）：
        保真残差 = ‖kept − Δ‖ / ‖Δ‖      ← 判据。方向对不对
        幅度比   = ‖kept‖ / ‖Δ‖          ← 参考。只看它会得出错误的安心结论
    """
    import glob as _glob

    import torch
    from safetensors import safe_open

    cfg = json.loads(Path(adapter_path, "adapter_config.json").read_text())
    scale = cfg["lora_alpha"] / cfg["r"]
    files = sorted(_glob.glob(str(Path(adapter_path) / "*.safetensors")))
    if not files:
        print("[合并] ⚠️ adapter 不是 safetensors，跳过合并校验")
        return
    sd = merged_model.state_dict()
    with safe_open(files[0], framework="pt") as fh:
        ka = next((k for k in sorted(fh.keys()) if k.endswith("lora_A.weight")), None)
        if ka is None:
            print("[合并] ⚠️ adapter 里没有 lora_A，跳过合并校验")
            return
        A = fh.get_tensor(ka).float()
        B = fh.get_tensor(ka.replace("lora_A", "lora_B")).float()
    delta = scale * (B @ A)
    name = ka.replace("base_model.model.", "").replace(".lora_A.weight", ".weight")
    if name not in sd:
        print(f"[合并] ⚠️ 权重里找不到 {name}，跳过合并校验")
        return
    merged_w = sd[name].float().cpu()
    base_w = None
    for f in sorted(Path(base_path).glob("*.safetensors")):
        with safe_open(f, framework="pt") as fh:
            if name in fh.keys():
                base_w = fh.get_tensor(name).float()
                break
    if base_w is None:
        print(f"[合并] ⚠️ 基座里找不到 {name}，跳过合并校验")
        return
    kept = merged_w - base_w
    if kept.norm().item() == 0.0:
        raise SystemExit(f"🔴 合并后 {name} 与基座**逐位相同** ⇒ 增量根本没进权重。")
    resid = ((kept - delta).norm() / delta.norm()).item()
    mag = (kept.norm() / delta.norm()).item()
    rel = (delta.norm() / base_w.norm()).item()
    print(f"[合并] 校验 {name}: Δ 占基座 {rel * 100:.4f}% · 保真残差 {resid:.2f} · 幅度比 {mag:.2f}")
    if resid > max_resid:
        raise SystemExit(
            f"🔴 保真残差 {resid:.2f} > {max_resid} ⇒ 增量太小，被 bf16 存储的舍入噪声淹没了。\n"
            "   **不要合并这一级的增量**，保持 adapter 形态；\n"
            "   （RL 一轮的增量典型是 0.05% 量级，残差 ~0.87 —— 见 docs/syncopate/18 §3.3）"
        )


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

    # ★★ 2026-08-18：合并之后必须验一遍「增量真的落进权重了」。
    # 实测过的坑：**损失来自存储精度，不是累加精度** —— 在 fp32 里相加再存 bf16 一样丢。
    #   Δ 占基座 0.42%（SFT 一遍）⇒ 保真残差 0.36，可用
    #   Δ 占基座 0.056%（RL 一轮）⇒ 保真残差 0.87，**合并等于毁掉它**
    # ⇒ 这个断言不是防打字错，是防「小增量被 bf16 静默吃掉」。
    _assert_merge_landed(base_path, adapter_path, model)

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
